package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
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

// TestResult holds the benchmark results for a provider
type TestResult struct {
	Provider         string        `json:"provider"`
	Model            string        `json:"model"`
	Timestamp        time.Time     `json:"timestamp"`
	E2ELatency       time.Duration `json:"e2e_latency_ms"`
	TTFT             time.Duration `json:"ttft_ms"`
	Throughput       float64       `json:"throughput_tokens_per_sec"`
	CompletionTokens int           `json:"completion_tokens"`
	Success          bool          `json:"success"`
	Error            string        `json:"error,omitempty"`
}

// testProviderMetrics runs a full benchmark test against a single provider.
// It is designed to be run as a goroutine.
func testProviderMetrics(config ProviderConfig, tke *tiktoken.Tiktoken, wg *sync.WaitGroup, logDir, resultsDir string, results *[]TestResult, resultsMutex *sync.Mutex) {
	// Defer wg.Done() if this is part of a concurrent group
	if wg != nil {
		defer wg.Done()
	}

	// Create log file for this provider
	timestamp := time.Now().Format("20060102-150405")
	logFile, err := os.Create(filepath.Join(logDir, fmt.Sprintf("%s-%s.log", config.Name, timestamp)))
	if err != nil {
		log.Printf("Error creating log file for %s: %v", config.Name, err)
		return
	}
	defer logFile.Close()

	// Create a logger for this provider that writes to both stdout and file
	providerLogger := log.New(io.MultiWriter(os.Stdout, logFile), "", log.LstdFlags)

	providerLogger.Printf("--- Testing: %s (%s) ---", config.Name, config.Model)

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

	// Add timeout context to prevent indefinite hangs (2 minutes)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	stream, err := client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		providerLogger.Printf("Error creating stream for %s: %v", config.Name, err)
		// Save error result
		result := TestResult{
			Provider:  config.Name,
			Model:     config.Model,
			Timestamp: time.Now(),
			Success:   false,
			Error:     err.Error(),
		}
		saveResult(resultsDir, result)
		appendResult(results, resultsMutex, result)
		return
	}
	defer stream.Close() // IMPORTANT: Always close the stream

	providerLogger.Printf("[%s] ... Request sent. Waiting for stream ...", config.Name)

	for {
		response, err := stream.Recv()

		// Check for end of stream
		if errors.Is(err, io.EOF) {
			providerLogger.Printf("[%s] ... Stream complete.", config.Name)
			break
		}

		if err != nil {
			errMsg := err.Error()
			if ctx.Err() == context.DeadlineExceeded {
				errMsg = "Timeout: stream took longer than 5 minutes"
			}
			providerLogger.Printf("Stream error for %s: %v", config.Name, errMsg)
			// Save error result
			result := TestResult{
				Provider:  config.Name,
				Model:     config.Model,
				Timestamp: time.Now(),
				Success:   false,
				Error:     errMsg,
			}
			saveResult(resultsDir, result)
			appendResult(results, resultsMutex, result)
			return
		}

		// Check if Choices array is empty (some APIs send empty chunks)
		if len(response.Choices) == 0 {
			providerLogger.Printf("[%s] ... Received empty chunk (no Choices)", config.Name)
			continue
		}

		// Get the content from the first choice
		content := response.Choices[0].Delta.Content

		// Check if this is the first chunk with actual text
		if content != "" && firstTokenTime.IsZero() {
			firstTokenTime = time.Now() // ---- TTFT METRIC
			providerLogger.Printf("[%s] ... First token received!", config.Name)
		}

		// Append the content to our builder
		if content != "" {
			fullResponseContent.WriteString(content)
		}
	}

	endTime := time.Now() // ---- E2E METRIC

	// --- 8. Calculate and Print Results ---

	if firstTokenTime.IsZero() {
		providerLogger.Printf("Error for %s: Did not receive any content from the API.", config.Name)
		// Save error result
		result := TestResult{
			Provider:  config.Name,
			Model:     config.Model,
			Timestamp: time.Now(),
			Success:   false,
			Error:     "No content received from API",
		}
		saveResult(resultsDir, result)
		appendResult(results, resultsMutex, result)
		return
	}

	// Get accurate token count
	fullResponse := fullResponseContent.String()
	tokenList := tke.Encode(fullResponse, nil, nil)
	completionTokens := len(tokenList)

	if completionTokens == 0 {
		providerLogger.Printf("Error for %s: Received response with 0 tokens.", config.Name)
		// Save error result
		result := TestResult{
			Provider:  config.Name,
			Model:     config.Model,
			Timestamp: time.Now(),
			Success:   false,
			Error:     "Received 0 tokens",
		}
		saveResult(resultsDir, result)
		appendResult(results, resultsMutex, result)
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

	// --- Print Results (use providerLogger for thread-safety) ---
	providerLogger.Println("==============================================")
	providerLogger.Printf("   LLM Metrics for: %s", config.Name)
	providerLogger.Printf("   Model: %s", config.Model)
	providerLogger.Printf("   Total Output Tokens: %d", completionTokens)
	providerLogger.Println("----------------------------------------------")
	providerLogger.Printf("   End-to-End Latency: %v", e2eLatency)
	providerLogger.Printf("   Latency (TTFT):     %v", ttft)
	providerLogger.Printf("   Throughput (Tokens/sec): %.2f tokens/s", throughput)
	providerLogger.Println("==============================================")
	// Uncomment to see the full response
	// providerLogger.Printf("[%s] Full Response:\n%s\n", config.Name, fullResponse)

	// Save successful result
	result := TestResult{
		Provider:         config.Name,
		Model:            config.Model,
		Timestamp:        time.Now(),
		E2ELatency:       e2eLatency,
		TTFT:             ttft,
		Throughput:       throughput,
		CompletionTokens: completionTokens,
		Success:          true,
	}
	saveResult(resultsDir, result)
	appendResult(results, resultsMutex, result)
}

// appendResult safely appends a result to the shared results slice
func appendResult(results *[]TestResult, mutex *sync.Mutex, result TestResult) {
	if results != nil && mutex != nil {
		mutex.Lock()
		*results = append(*results, result)
		mutex.Unlock()
	}
}

// saveResult saves the test result to a JSON file
func saveResult(resultsDir string, result TestResult) {
	timestamp := result.Timestamp.Format("20060102-150405")
	filename := filepath.Join(resultsDir, fmt.Sprintf("%s-%s.json", result.Provider, timestamp))

	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		log.Printf("Error marshaling result for %s: %v", result.Provider, err)
		return
	}

	if err := os.WriteFile(filename, data, 0644); err != nil {
		log.Printf("Error writing result file for %s: %v", result.Provider, err)
		return
	}

	log.Printf("Result saved: %s", filename)
}

// generateMarkdownReport creates a summary report of all test results
func generateMarkdownReport(resultsDir string, results []TestResult, sessionTimestamp string) error {
	filename := filepath.Join(resultsDir, "REPORT.md")

	var report strings.Builder
	report.WriteString("# LLM API Speed Test Results\n\n")
	report.WriteString(fmt.Sprintf("**Test Session:** %s\n\n", sessionTimestamp))
	report.WriteString("---\n\n")

	// Summary statistics
	successful := 0
	failed := 0
	for _, r := range results {
		if r.Success {
			successful++
		} else {
			failed++
		}
	}

	report.WriteString("## Summary\n\n")
	report.WriteString(fmt.Sprintf("- **Total Providers Tested:** %d\n", len(results)))
	report.WriteString(fmt.Sprintf("- **Successful:** %d\n", successful))
	report.WriteString(fmt.Sprintf("- **Failed:** %d\n\n", failed))

	// Successful results table
	if successful > 0 {
		report.WriteString("## Successful Tests\n\n")
		report.WriteString("| Provider | Model | E2E Latency | TTFT | Throughput | Tokens |\n")
		report.WriteString("|----------|-------|-------------|------|------------|--------|\n")

		for _, r := range results {
			if r.Success {
				report.WriteString(fmt.Sprintf("| %s | %s | %v | %v | %.2f tok/s | %d |\n",
					r.Provider,
					r.Model,
					r.E2ELatency,
					r.TTFT,
					r.Throughput,
					r.CompletionTokens))
			}
		}
		report.WriteString("\n")
	}

	// Failed results
	if failed > 0 {
		report.WriteString("## Failed Tests\n\n")
		report.WriteString("| Provider | Model | Error |\n")
		report.WriteString("|----------|-------|-------|\n")

		for _, r := range results {
			if !r.Success {
				report.WriteString(fmt.Sprintf("| %s | %s | %s |\n",
					r.Provider,
					r.Model,
					r.Error))
			}
		}
		report.WriteString("\n")
	}

	// Leaderboard (sorted by throughput)
	if successful > 0 {
		report.WriteString("## Performance Leaderboard\n\n")
		report.WriteString("### By Throughput (Tokens/sec)\n\n")

		// Sort by throughput
		successfulResults := make([]TestResult, 0)
		for _, r := range results {
			if r.Success {
				successfulResults = append(successfulResults, r)
			}
		}

		// Simple bubble sort by throughput descending
		for i := 0; i < len(successfulResults); i++ {
			for j := i + 1; j < len(successfulResults); j++ {
				if successfulResults[j].Throughput > successfulResults[i].Throughput {
					successfulResults[i], successfulResults[j] = successfulResults[j], successfulResults[i]
				}
			}
		}

		report.WriteString("| Rank | Provider | Throughput | TTFT | E2E Latency |\n")
		report.WriteString("|------|----------|------------|------|-------------|\n")

		for i, r := range successfulResults {
			report.WriteString(fmt.Sprintf("| %d | %s | %.2f tok/s | %v | %v |\n",
				i+1,
				r.Provider,
				r.Throughput,
				r.TTFT,
				r.E2ELatency))
		}
		report.WriteString("\n")

		// Sort by TTFT
		report.WriteString("### By Time to First Token (TTFT)\n\n")

		for i := 0; i < len(successfulResults); i++ {
			for j := i + 1; j < len(successfulResults); j++ {
				if successfulResults[j].TTFT < successfulResults[i].TTFT {
					successfulResults[i], successfulResults[j] = successfulResults[j], successfulResults[i]
				}
			}
		}

		report.WriteString("| Rank | Provider | TTFT | Throughput | E2E Latency |\n")
		report.WriteString("|------|----------|------|------------|-------------|\n")

		for i, r := range successfulResults {
			report.WriteString(fmt.Sprintf("| %d | %s | %v | %.2f tok/s | %v |\n",
				i+1,
				r.Provider,
				r.TTFT,
				r.Throughput,
				r.E2ELatency))
		}
		report.WriteString("\n")

		// Sort by E2E Latency
		report.WriteString("### By End-to-End Latency\n\n")

		for i := 0; i < len(successfulResults); i++ {
			for j := i + 1; j < len(successfulResults); j++ {
				if successfulResults[j].E2ELatency < successfulResults[i].E2ELatency {
					successfulResults[i], successfulResults[j] = successfulResults[j], successfulResults[i]
				}
			}
		}

		report.WriteString("| Rank | Provider | E2E Latency | TTFT | Throughput |\n")
		report.WriteString("|------|----------|-------------|------|------------|\n")

		for i, r := range successfulResults {
			report.WriteString(fmt.Sprintf("| %d | %s | %v | %v | %.2f tok/s |\n",
				i+1,
				r.Provider,
				r.E2ELatency,
				r.TTFT,
				r.Throughput))
		}
		report.WriteString("\n")
	}

	report.WriteString("---\n\n")
	report.WriteString(fmt.Sprintf("*Report generated at %s*\n", time.Now().Format("2006-01-02 15:04:05")))

	if err := os.WriteFile(filename, []byte(report.String()), 0644); err != nil {
		return fmt.Errorf("error writing report: %v", err)
	}

	log.Printf("Report generated: %s", filename)
	return nil
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

	// 3. Create session-based folder structure
	sessionTimestamp := time.Now().Format("20060102-150405")
	sessionDir := filepath.Join("results", fmt.Sprintf("session-%s", sessionTimestamp))
	logDir := filepath.Join(sessionDir, "logs")
	resultsDir := sessionDir

	if err := os.MkdirAll(logDir, 0755); err != nil {
		log.Fatalf("Error creating logs directory: %v", err)
	}

	if err := os.MkdirAll(resultsDir, 0755); err != nil {
		log.Fatalf("Error creating results directory: %v", err)
	}

	log.Printf("Session folder: %s/", sessionDir)
	log.Printf("Logs will be saved to: %s/", logDir)
	log.Printf("Results will be saved to: %s/", resultsDir)

	// 4. Initialize Tokenizer
	tke, err := tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		log.Fatalf("Error getting tokenizer: %v\n(You might need to run: go get github.com/pkoukk/tiktoken-go)", err)
	}

	// 5. Build Full Provider Config Map from .env and flags
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
	var results []TestResult
	var resultsMutex sync.Mutex

	for _, provider := range providersToTest {
		if *testAll {
			// Run all tests concurrently
			wg.Add(1)
			go testProviderMetrics(provider, tke, &wg, logDir, resultsDir, &results, &resultsMutex)
		} else {
			// Run a single test sequentially
			testProviderMetrics(provider, tke, nil, logDir, resultsDir, &results, &resultsMutex)
		}
	}

	// Wait for all concurrent tests to finish
	if *testAll {
		wg.Wait()
		log.Println("--- All provider tests complete. ---")
	}

	// Generate markdown report
	log.Println("Generating summary report...")
	if err := generateMarkdownReport(resultsDir, results, sessionTimestamp); err != nil {
		log.Printf("Warning: Failed to generate report: %v", err)
	}

	log.Printf("All tests complete. Results saved to: %s/", sessionDir)
}
