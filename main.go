package main

import (
	"context"
	"errors"
	"flag"
	"io"
	"log"
	"os"
	"strings"
	"sync" // Added for concurrent testing
	"time"

	"github.com/joho/godotenv"
	"github.com/pkoukk/tiktoken-go"
	openai "github.com/sashabaranov/go-openai"
)

// ProviderConfig holds all info for one API provider
type ProviderConfig struct {
	Name    string
	BaseURL string
	APIKey  string
	Model   string
}

// testProviderMetrics runs a full benchmark test against a single provider.
// It is designed to be run as a goroutine.
func testProviderMetrics(config ProviderConfig, tke *tiktoken.Tiktoken, wg *sync.WaitGroup) {
	// Defer wg.Done() if this is part of a concurrent group
	if wg != nil {
		defer wg.Done()
	}

	log.Printf("--- Testing: %s (%s) ---", config.Name, config.Model)

	// 5. Configure the OpenAI Client
	clientConfig := openai.DefaultConfig(config.APIKey)
	clientConfig.BaseURL = config.BaseURL
	client := openai.NewClientWithConfig(clientConfig)

	// 6. Define the request
	prompt := "You are a helpful assistant. Please write a short, 150-word story about a curious robot exploring an ancient, overgrown library on a forgotten planet."
	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleUser,
			Content: prompt,
		},
	}

	req := openai.ChatCompletionRequest{
		Model:     config.Model,
		Messages:  messages,
		MaxTokens: 512,
		Stream:    true,
	}

	// 7. Execute the stream and measure metrics
	startTime := time.Now() // ---- START TIMER
	var firstTokenTime time.Time
	var fullResponseContent strings.Builder

	ctx := context.Background()
	stream, err := client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		log.Printf(" Error creating stream for %s: %v", config.Name, err)
		return
	}
	defer stream.Close() // IMPORTANT: Always close the stream

	log.Printf("[%s] ... Request sent. Waiting for stream ...", config.Name)

	for {
		response, err := stream.Recv()

		// Check for end of stream
		if errors.Is(err, io.EOF) {
			log.Printf("[%s] ... Stream complete.", config.Name)
			break
		}

		if err != nil {
			log.Printf("Stream error for %s: %v", config.Name, err)
			return
		}

		// Get the content from the first choice
		content := response.Choices[0].Delta.Content

		// Check if this is the first chunk with actual text
		if content != "" && firstTokenTime.IsZero() {
			firstTokenTime = time.Now() // ---- TTFT METRIC
			log.Printf("[%s] ... First token received!", config.Name)
		}

		// Append the content to our builder
		if content != "" {
			fullResponseContent.WriteString(content)
		}
	}

	endTime := time.Now() // ---- E2E METRIC

	// --- 8. Calculate and Print Results ---

	if firstTokenTime.IsZero() {
		log.Printf("Error for %s: Did not receive any content from the API.", config.Name)
		return
	}

	// Get accurate token count
	fullResponse := fullResponseContent.String()
	tokenList := tke.Encode(fullResponse, nil, nil)
	completionTokens := len(tokenList)

	if completionTokens == 0 {
		log.Printf("Error for %s: Received response with 0 tokens.", config.Name)
		return
	}

	// 1. End-to-End Latency
	e2eLatency := endTime.Sub(startTime)

	// 2. Time to First Token (TTFT)
	ttft := firstTokenTime.Sub(startTime)

	// 3. Throughput (Tokens per Second)
	// This is (Total Tokens - 1) / (Time from first token to last token)
	generationTime := e2eLatency - ttft
	var throughput float64

	if generationTime.Seconds() <= 0 {
		// Handle edge case where generation is too fast or only 1 token
		throughput = 0.0
	} else {
		throughput = (float64(completionTokens) - 1.0) / generationTime.Seconds()
	}

	// --- Print Results (use log for thread-safety) ---
	log.Println("==============================================")
	log.Printf("   LLM Metrics for: %s", config.Name)
	log.Printf("   Model: %s", config.Model)
	log.Printf("   Total Output Tokens: %d", completionTokens)
	log.Println("----------------------------------------------")
	log.Printf("   ⚡ End-to-End Latency: %v", e2eLatency)
	log.Printf("   ⚡ Latency (TTFT):     %v", ttft)
	log.Printf("   ⚡ Throughput (Tokens/sec): %.2f tokens/s", throughput)
	log.Println("==============================================")
	// Uncomment to see the full response
	// log.Printf("[%s] Full Response:\n%s\n", config.Name, fullResponse)
}

func main() {
	// --- Define Provider static info ---
	providerBaseURLs := map[string]string{
		"generic": "https://openrouter.ai/api/v1", // Default, can be overridden by --url
		"nim":     "https://integrate.api.nvidia.com/v1",
		"nahcrof": "https://ai.nahcrof.com/v2",
		"novita":  "https://api.novita.ai/openai",
		"nebius":  "https://api.tokenfactory.nebius.com/v1",
		"minimax": "https://api.minimax.io/v1",
	}

	// 1. Load .env file (if it exists)
	if err := godotenv.Load(); err != nil {
		log.Println("Note: .env file not found, reading from system environment.")
	}

	// 2. Parse Command-Line Flags
	providerName := flag.String("provider", "", "Specific provider to test (e.g., nim, novita). If empty, tests 'generic' provider.")
	testAll := flag.Bool("all", false, "Test all configured providers concurrently.")
	flagGenericURL := flag.String("url", "", "Override Base URL for 'generic' provider (default: https://openrouter.ai/api/v1)")
	flagGenericModel := flag.String("model", "", "Model name for 'generic' provider (required if --provider is not set)")
	flag.Parse()

	// 3. Initialize Tokenizer
	tke, err := tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		log.Fatalf("Error getting tokenizer: %v\n(You might need to run: go get github.com/pkoukk/tiktoken-go)", err)
	}

	// 4. Build Full Provider Config Map from .env and flags
	allProviderConfigs := make(map[string]ProviderConfig)

	// Generic Provider (uses --url and --model flags)
	genericBaseURL := *flagGenericURL
	if genericBaseURL == "" {
		genericBaseURL = providerBaseURLs["generic"]
	}
	allProviderConfigs["generic"] = ProviderConfig{
		Name:    "generic",
		BaseURL: genericBaseURL,
		APIKey:  os.Getenv("OAI_API_KEY"),
		Model:   *flagGenericModel,
	}

	// NIM Provider
	allProviderConfigs["nim"] = ProviderConfig{
		Name:    "nim",
		BaseURL: providerBaseURLs["nim"],
		APIKey:  os.Getenv("NIM_API_KEY"),
		Model:   os.Getenv("NIM_MODEL"),
	}

	// NAHCROF Provider
	allProviderConfigs["nahcrof"] = ProviderConfig{
		Name:    "nahcrof",
		BaseURL: providerBaseURLs["nahcrof"],
		APIKey:  os.Getenv("NAHCROF_API_KEY"),
		Model:   os.Getenv("NAHCROF_MODEL"),
	}

	// NovitaAI Provider
	allProviderConfigs["novita"] = ProviderConfig{
		Name:    "novita",
		BaseURL: providerBaseURLs["novita"],
		APIKey:  os.Getenv("NOVITA_API_KEY"),
		Model:   os.Getenv("NOVITA_MODEL"),
	}

	// NebiusAI Provider
	allProviderConfigs["nebius"] = ProviderConfig{
		Name:    "nebius",
		BaseURL: providerBaseURLs["nebius"],
		APIKey:  os.Getenv("NEBIUS_API_KEY"),
		Model:   os.Getenv("NEBIUS_MODEL"),
	}

	// MiniMax Provider
	allProviderConfigs["minimax"] = ProviderConfig{
		Name:    "minimax",
		BaseURL: providerBaseURLs["minimax"],
		APIKey:  os.Getenv("MINIMAX_API_KEY"),
		Model:   os.Getenv("MINIMAX_MODEL"),
	}

	// 5. Select Providers to Test based on flags
	providersToTest := []ProviderConfig{}

	if *testAll {
		log.Println("--- Testing all configured providers... ---")
		for name, config := range allProviderConfigs {
			if config.APIKey != "" && config.Model != "" {
				providersToTest = append(providersToTest, config)
			} else {
				// Don't log generic provider if not set, it's optional
				if name != "generic" {
					log.Printf("... Skipping '%s': APIKey or Model not configured in .env\n", name)
				}
			}
		}
		// Check generic provider separately for --all
		genConfig := allProviderConfigs["generic"]
		if genConfig.APIKey != "" && genConfig.Model != "" {
			log.Println("... 'generic' provider is configured, but will be skipped. Use --provider=generic or no flags to test it.")
		}

	} else if *providerName != "" {
		log.Printf("--- Testing single provider: '%s' ---\n", *providerName)
		config, ok := allProviderConfigs[*providerName]
		if !ok {
			log.Fatalf("Error: Provider '%s' not recognized.", *providerName)
		}
		if config.APIKey == "" || config.Model == "" {
			log.Fatalf("Error: Provider '%s' is not configured. (Missing APIKey/Model in .env or --model flag for generic)", *providerName)
		}
		providersToTest = append(providersToTest, config)
	} else {
		// Default: test "generic" provider
		log.Println("--- Testing default 'generic' provider... ---")
		config := allProviderConfigs["generic"]
		if config.APIKey == "" {
			log.Fatal("Error: OAI_API_KEY not set for 'generic' provider.")
		}
		if config.Model == "" {
			log.Fatal("Error: --model flag is required for 'generic' provider.")
		}
		providersToTest = append(providersToTest, config)
	}

	if len(providersToTest) == 0 {
		log.Fatal("No providers configured or selected to test.")
	}

	// 6. Run Tests
	var wg sync.WaitGroup
	for _, provider := range providersToTest {
		if *testAll {
			// Run all tests concurrently
			wg.Add(1)
			go testProviderMetrics(provider, tke, &wg)
		} else {
			// Run a single test sequentially
			testProviderMetrics(provider, tke, nil)
		}
	}

	// Wait for all concurrent tests to finish
	if *testAll {
		wg.Wait()
		log.Println("--- All provider tests complete. ---")
	}
}
