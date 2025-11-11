// Package main implements an LLM API speed testing tool.
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

// ProviderConfig holds all info for one API provider.
type ProviderConfig struct {
	Name    string
	BaseURL string
	APIKey  string
	Model   string
}

// TestResult holds the benchmark results for a provider.
type TestResult struct {
	Provider         string        `json:"provider"`
	Model            string        `json:"model"`
	Timestamp        time.Time     `json:"timestamp"`
	E2ELatency       time.Duration `json:"e2eLatencyMs"`
	TTFT             time.Duration `json:"ttftMs"`
	Throughput       float64       `json:"throughputTokensPerSec"`
	CompletionTokens int           `json:"completionTokens"`
	Success          bool          `json:"success"`
	Error            string        `json:"error,omitempty"`
	Mode             string        `json:"mode"`
}

// TestMode represents the type of test being performed.
type TestMode string

const (
	// ModeStreaming represents streaming mode testing.
	ModeStreaming TestMode = "streaming"
	// ModeToolCalling represents tool-calling mode testing.
	ModeToolCalling TestMode = "tool-calling"
	// ModeMixed represents mixed mode testing (both streaming and tool-calling).
	ModeMixed TestMode = "mixed"
	// NotAvailable is a constant for unavailable metrics.
	NotAvailable = "N/A"
)

// formatDuration formats a duration as decimal seconds.
func formatDuration(d time.Duration) string {
	return fmt.Sprintf("%.3fs", d.Seconds())
}

var saveResponses bool

// singleTestRun performs one test run and returns metrics or error.
func singleTestRun(ctx context.Context, config ProviderConfig, tke *tiktoken.Tiktoken, providerLogger *log.Logger) (e2e, ttft time.Duration, throughput float64, tokens int, response string, err error) {
	// Configure the OpenAI Client
	clientConfig := openai.DefaultConfig(config.APIKey)
	clientConfig.BaseURL = config.BaseURL
	client := openai.NewClientWithConfig(clientConfig)

	// Define the request
	prompt := "You are a helpful assistant. Please write a short, 150-word story about a curious robot exploring " +
		"an ancient, overgrown library on a forgotten planet."
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

	// Execute the stream and measure metrics
	startTime := time.Now()
	var firstTokenTime time.Time
	var fullResponseContent strings.Builder

	stream, streamErr := client.CreateChatCompletionStream(ctx, req)
	if streamErr != nil {
		return 0, 0, 0, 0, "", fmt.Errorf("error creating stream: %w", streamErr)
	}
	defer func() {
		if closeErr := stream.Close(); closeErr != nil {
			providerLogger.Printf("[%s] Warning: Failed to close stream: %v", config.Name, closeErr)
		}
	}()

	providerLogger.Printf("[%s] ... Request sent. Waiting for stream ...", config.Name)

	chunkCount := 0
	nonEmptyChunks := 0
	reasoningChunks := 0

	for {
		response, recvErr := stream.Recv()

		// Check for end of stream
		if errors.Is(recvErr, io.EOF) {
			providerLogger.Printf("[%s] ... Stream complete. Received %d chunks (%d content, %d reasoning)",
				config.Name, chunkCount, nonEmptyChunks, reasoningChunks)
			break
		}

		if recvErr != nil {
			if ctx.Err() == context.DeadlineExceeded {
				return 0, 0, 0, 0, "", fmt.Errorf("timeout exceeded")
			}
			return 0, 0, 0, 0, "", fmt.Errorf("stream error: %w", recvErr)
		}

		chunkCount++

		// Check if Choices array is empty
		if len(response.Choices) == 0 {
			// Log occasionally for debugging (every 100 chunks), not every single one
			if chunkCount%100 == 0 {
				providerLogger.Printf("[%s] ... Chunk %d: Empty Choices array (diagnostic: ID=%s, Model=%s)",
					config.Name, chunkCount, response.ID, response.Model)
			}
			continue
		}

		delta := response.Choices[0].Delta

		// Get both regular content and reasoning content (for thinking models)
		content := delta.Content
		reasoningContent := delta.ReasoningContent

		// Check if this is the first chunk with actual text (either type)
		if (content != "" || reasoningContent != "") && firstTokenTime.IsZero() {
			firstTokenTime = time.Now()
			if reasoningContent != "" {
				providerLogger.Printf("[%s] ... First token received (reasoning)! (chunk %d, len=%d)",
					config.Name, chunkCount, len(reasoningContent))
			} else {
				providerLogger.Printf("[%s] ... First token received! (chunk %d, len=%d)",
					config.Name, chunkCount, len(content))
			}
		}

		// Append both types of content
		if content != "" {
			nonEmptyChunks++
			fullResponseContent.WriteString(content)
		}
		if reasoningContent != "" {
			reasoningChunks++
			fullResponseContent.WriteString(reasoningContent)
		}
	}

	endTime := time.Now()

	if firstTokenTime.IsZero() {
		return 0, 0, 0, 0, "", fmt.Errorf("no content received from API (received %d chunks)", chunkCount)
	}

	// Get accurate token count
	fullResponse := fullResponseContent.String()
	tokenList := tke.Encode(fullResponse, nil, nil)
	completionTokens := len(tokenList)

	providerLogger.Printf(
		"[%s] ... Total content length: %d bytes, %d tokens",
		config.Name, len(fullResponse), completionTokens)

	if completionTokens == 0 {
		return 0, 0, 0, 0, "", fmt.Errorf("received 0 tokens (content length: %d bytes)", len(fullResponse))
	}

	// Calculate metrics
	e2eLatency := endTime.Sub(startTime)
	ttftLatency := firstTokenTime.Sub(startTime)
	generationTime := e2eLatency - ttftLatency

	var throughputVal float64
	if generationTime.Seconds() <= 0 {
		throughputVal = 0.0
	} else {
		throughputVal = (float64(completionTokens) - 1.0) / generationTime.Seconds()
	}

	return e2eLatency, ttftLatency, throughputVal, completionTokens, fullResponse, nil
}

// singleToolCallRun performs one tool-calling test run and returns metrics or error.
func singleToolCallRun(ctx context.Context, config ProviderConfig, tke *tiktoken.Tiktoken, providerLogger *log.Logger) (e2e, ttft time.Duration, throughput float64, tokens int, response string, err error) {
	// Configure the OpenAI Client
	clientConfig := openai.DefaultConfig(config.APIKey)
	clientConfig.BaseURL = config.BaseURL
	client := openai.NewClientWithConfig(clientConfig)

	// Define a weather tool
	tools := []openai.Tool{
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "get_weather",
				Description: "Get the current weather in a given location",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
							"type":        "string",
							"description": "The city and state, e.g. San Francisco, CA",
						},
						"unit": map[string]interface{}{
							"type": "string",
							"enum": []string{"celsius", "fahrenheit"},
						},
					},
					"required": []string{"location"},
				},
			},
		},
	}

	prompt := "What's the weather like in San Francisco, Tokyo, and London? Please check all three cities and " +
		"tell me which one has the best weather for outdoor activities today."
	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleUser,
			Content: prompt,
		},
	}

	req := openai.ChatCompletionRequest{
		Model:     config.Model,
		Messages:  messages,
		Tools:     tools,
		MaxTokens: 512,
		Stream:    true,
	}

	// Execute the stream and measure metrics
	startTime := time.Now()
	var firstTokenTime time.Time
	var fullResponseContent strings.Builder

	stream, streamErr := client.CreateChatCompletionStream(ctx, req)
	if streamErr != nil {
		return 0, 0, 0, 0, "", fmt.Errorf("error creating stream: %w", streamErr)
	}
	defer func() {
		if closeErr := stream.Close(); closeErr != nil {
			providerLogger.Printf("[%s] Warning: Failed to close stream: %v", config.Name, closeErr)
		}
	}()

	providerLogger.Printf("[%s] ... Tool calling request sent. Waiting for stream ...", config.Name)

	chunkCount := 0
	nonEmptyChunks := 0
	reasoningChunks := 0
	toolCallChunks := 0

	for {
		response, recvErr := stream.Recv()

		// Check for end of stream
		if errors.Is(recvErr, io.EOF) {
			providerLogger.Printf(
				"[%s] ... Tool calling stream complete. Received %d chunks (%d content, %d reasoning, %d tool)",
				config.Name, chunkCount, nonEmptyChunks, reasoningChunks, toolCallChunks)
			break
		}

		if recvErr != nil {
			if ctx.Err() == context.DeadlineExceeded {
				return 0, 0, 0, 0, "", fmt.Errorf("timeout exceeded")
			}
			return 0, 0, 0, 0, "", fmt.Errorf("stream error: %w", recvErr)
		}

		chunkCount++

		// Check if Choices array is empty
		if len(response.Choices) == 0 {
			// Log occasionally for debugging (every 100 chunks), not every single one
			if chunkCount%100 == 0 {
				providerLogger.Printf("[%s] ... Chunk %d: Empty Choices array (diagnostic: ID=%s, Model=%s)",
					config.Name, chunkCount, response.ID, response.Model)
			}
			continue
		}

		delta := response.Choices[0].Delta

		// Check for first token (content, reasoning, or tool call)
		hasContent := delta.Content != ""
		hasReasoningContent := delta.ReasoningContent != ""
		hasToolCall := len(delta.ToolCalls) > 0

		if (hasContent || hasReasoningContent || hasToolCall) && firstTokenTime.IsZero() {
			firstTokenTime = time.Now()
			switch {
			case hasReasoningContent:
				providerLogger.Printf(
					"[%s] ... First token received (reasoning, tool-calling)! (chunk %d)", config.Name, chunkCount)
			case hasToolCall:
				providerLogger.Printf("[%s] ... First token received (tool-call)! (chunk %d)", config.Name, chunkCount)
			default:
				providerLogger.Printf("[%s] ... First token received (tool-calling)! (chunk %d)", config.Name, chunkCount)
			}
		}

		// Append content if present
		if hasContent {
			nonEmptyChunks++
			fullResponseContent.WriteString(delta.Content)
		}

		// Append reasoning content if present
		if hasReasoningContent {
			reasoningChunks++
			fullResponseContent.WriteString(delta.ReasoningContent)
		}

		// Append tool call information as text for token counting
		if hasToolCall {
			toolCallChunks++
			for _, toolCall := range delta.ToolCalls {
				if toolCall.Function.Name != "" {
					fullResponseContent.WriteString(toolCall.Function.Name)
				}
				if toolCall.Function.Arguments != "" {
					fullResponseContent.WriteString(toolCall.Function.Arguments)
				}
			}
		}
	}

	endTime := time.Now()

	if firstTokenTime.IsZero() {
		return 0, 0, 0, 0, "", fmt.Errorf("no content received from API (received %d chunks)", chunkCount)
	}

	// Get accurate token count
	fullResponse := fullResponseContent.String()
	tokenList := tke.Encode(fullResponse, nil, nil)
	completionTokens := len(tokenList)

	providerLogger.Printf(
		"[%s] ... Total content length: %d bytes, %d tokens",
		config.Name, len(fullResponse), completionTokens)

	if completionTokens == 0 {
		return 0, 0, 0, 0, "", fmt.Errorf("received 0 tokens (content length: %d bytes)", len(fullResponse))
	}

	// Calculate metrics
	e2eLatency := endTime.Sub(startTime)
	ttftLatency := firstTokenTime.Sub(startTime)
	generationTime := e2eLatency - ttftLatency

	var throughputVal float64
	if generationTime.Seconds() <= 0 {
		throughputVal = 0.0
	} else {
		throughputVal = (float64(completionTokens) - 1.0) / generationTime.Seconds()
	}

	return e2eLatency, ttftLatency, throughputVal, completionTokens, fullResponse, nil
}

// testProviderMetrics runs a full benchmark test against a single provider.
// It runs 3 iterations and reports averaged results, with a 2-minute total timeout.
func testProviderMetrics(config ProviderConfig, tke *tiktoken.Tiktoken, wg *sync.WaitGroup, logDir, resultsDir string, results *[]TestResult, resultsMutex *sync.Mutex, mode TestMode) {
	// Defer wg.Done() if this is part of a concurrent group
	if wg != nil {
		defer wg.Done()
	}

	// Create log file for this provider
	timestamp := time.Now().Format("20060102-150405")
	logFile, err := os.Create(filepath.Clean(filepath.Join(logDir, fmt.Sprintf("%s-%s.log", config.Name, timestamp))))
	if err != nil {
		log.Printf("Error creating log file for %s: %v", config.Name, err)
		return
	}
	defer func() {
		if closeErr := logFile.Close(); closeErr != nil {
			log.Printf("Warning: Failed to close log file: %v", closeErr)
		}
	}()

	// Create a logger for this provider that writes to both stdout and file
	providerLogger := log.New(io.MultiWriter(os.Stdout, logFile), "", log.LstdFlags)

	modeStr := string(mode)
	providerLogger.Printf("--- Testing: %s (%s) - Mode: %s - Running 3 concurrent iterations ---",
		config.Name, config.Model, modeStr)

	// Create 5-minute timeout context for all runs (reasoning models can be slow)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	// Determine which modes to run based on mode parameter
	var modesToRun []TestMode
	if mode == ModeMixed {
		modesToRun = []TestMode{ModeStreaming, ModeToolCalling}
	} else {
		modesToRun = []TestMode{mode}
	}

	// Run 3 iterations per mode
	const iterationsPerMode = 3
	type runResult struct {
		e2e        time.Duration
		ttft       time.Duration
		throughput float64
		tokens     int
		err        error
		runNum     int
		mode       TestMode
	}

	totalRuns := len(modesToRun) * iterationsPerMode
	resultsChan := make(chan runResult, totalRuns)
	var runWg sync.WaitGroup

	// Launch concurrent workers for each mode
	runNum := 1
	for _, testMode := range modesToRun {
		for i := 1; i <= iterationsPerMode; i++ {
			runWg.Add(1)
			go func(currentRunNum int, currentMode TestMode) {
				defer runWg.Done()
				providerLogger.Printf("[%s] Run %d/%d (%s) starting", config.Name, currentRunNum, totalRuns, currentMode)

				var e2e, ttft time.Duration
				var throughput float64
				var tokens int
				var runErr error
				var responseContent string

				// Execute the appropriate test based on mode
				if currentMode == ModeToolCalling {
					e2e, ttft, throughput, tokens, responseContent, runErr = singleToolCallRun(ctx, config, tke, providerLogger)
				} else {
					e2e, ttft, throughput, tokens, responseContent, runErr = singleTestRun(ctx, config, tke, providerLogger)
				}

				// Save response if flag is enabled
				if saveResponses && runErr == nil && responseContent != "" {
					responseFile := filepath.Clean(filepath.Join(logDir,
						fmt.Sprintf("%s-run%d-%s-response.txt", config.Name, currentRunNum, currentMode)))
					if err := os.WriteFile(responseFile, []byte(responseContent), 0600); err != nil {
						providerLogger.Printf("[%s] Warning: Failed to save response for run %d: %v",
							config.Name, currentRunNum, err)
					}
				}

				if runErr != nil {
					providerLogger.Printf("[%s] Run %d (%s) failed: %v", config.Name, currentRunNum, currentMode, runErr)
				} else {
					providerLogger.Printf("[%s] Run %d (%s) complete: E2E=%s TTFT=%s Throughput=%.2f tok/s",
						config.Name, currentRunNum, currentMode, formatDuration(e2e), formatDuration(ttft), throughput)
				}

				resultsChan <- runResult{
					e2e:        e2e,
					ttft:       ttft,
					throughput: throughput,
					tokens:     tokens,
					err:        runErr,
					runNum:     currentRunNum,
					mode:       currentMode,
				}
			}(runNum, testMode)
			runNum++
		}
	}

	// Close channel after all workers complete
	go func() {
		runWg.Wait()
		close(resultsChan)
	}()

	// Collect results from all workers
	var e2eSum, ttftSum time.Duration
	var throughputSum float64
	var tokensSum int
	successfulRuns := 0
	var firstError error

	for result := range resultsChan {
		if result.err == nil {
			e2eSum += result.e2e
			ttftSum += result.ttft
			throughputSum += result.throughput
			tokensSum += result.tokens
			successfulRuns++
		} else if firstError == nil {
			firstError = result.err
		}
	}

	if successfulRuns == 0 {
		providerLogger.Printf("[%s] All runs failed", config.Name)
		// Save error result
		result := TestResult{
			Provider:  config.Name,
			Model:     config.Model,
			Timestamp: time.Now(),
			Success:   false,
			Error:     firstError.Error(),
			Mode:      modeStr,
		}
		saveResult(resultsDir, result)
		appendResult(results, resultsMutex, result)
		return
	}

	// Calculate averages
	avgE2E := e2eSum / time.Duration(successfulRuns)
	avgTTFT := ttftSum / time.Duration(successfulRuns)
	avgThroughput := throughputSum / float64(successfulRuns)
	avgTokens := tokensSum / successfulRuns

	// Print averaged results
	providerLogger.Println("==============================================")
	providerLogger.Printf("   LLM Metrics for: %s (averaged over %d run(s))", config.Name, successfulRuns)
	providerLogger.Printf("   Model: %s", config.Model)
	providerLogger.Printf("   Mode: %s", modeStr)
	providerLogger.Printf("   Avg Output Tokens: %d", avgTokens)
	providerLogger.Println("----------------------------------------------")
	providerLogger.Printf("   End-to-End Latency: %s", formatDuration(avgE2E))
	providerLogger.Printf("   Latency (TTFT):     %s", formatDuration(avgTTFT))
	providerLogger.Printf("   Throughput (Tokens/sec): %.2f tokens/s", avgThroughput)
	providerLogger.Println("==============================================")

	// Save successful result
	result := TestResult{
		Provider:         config.Name,
		Model:            config.Model,
		Timestamp:        time.Now(),
		E2ELatency:       avgE2E,
		TTFT:             avgTTFT,
		Throughput:       avgThroughput,
		CompletionTokens: avgTokens,
		Success:          true,
		Mode:             modeStr,
	}
	saveResult(resultsDir, result)
	appendResult(results, resultsMutex, result)
}

// appendResult safely appends a result to the shared results slice.
func appendResult(results *[]TestResult, mutex *sync.Mutex, result TestResult) {
	if results != nil && mutex != nil {
		mutex.Lock()
		*results = append(*results, result)
		mutex.Unlock()
	}
}

// saveResult saves the test result to a JSON file.
func saveResult(resultsDir string, result TestResult) {
	timestamp := result.Timestamp.Format("20060102-150405")
	filename := filepath.Join(resultsDir, fmt.Sprintf("%s-%s.json", result.Provider, timestamp))

	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		log.Printf("Error marshaling result for %s: %v", result.Provider, err)
		return
	}

	if err := os.WriteFile(filename, data, 0600); err != nil {
		log.Printf("Error writing result file for %s: %v", result.Provider, err)
		return
	}

	log.Printf("Result saved: %s", filename)
}

// generateMarkdownReport creates a summary report of all test results.
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
		report.WriteString("| Provider | Model | Mode | E2E Latency | TTFT | Throughput | Tokens |\n")
		report.WriteString("|----------|-------|------|-------------|------|------------|--------|\n")

		for _, r := range results {
			if r.Success {
				report.WriteString(fmt.Sprintf("| %s | %s | %s | %s | %s | %.2f tok/s | %d |\n",
					r.Provider,
					r.Model,
					r.Mode,
					formatDuration(r.E2ELatency),
					formatDuration(r.TTFT),
					r.Throughput,
					r.CompletionTokens))
			}
		}
		report.WriteString("\n")
	}

	// Failed results
	if failed > 0 {
		report.WriteString("## Failed Tests\n\n")
		report.WriteString("| Provider | Model | Mode | Error |\n")
		report.WriteString("|----------|-------|------|-------|\n")

		for _, r := range results {
			if !r.Success {
				report.WriteString(fmt.Sprintf("| %s | %s | %s | %s |\n",
					r.Provider,
					r.Model,
					r.Mode,
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
			report.WriteString(fmt.Sprintf("| %d | %s | %.2f tok/s | %s | %s |\n",
				i+1,
				r.Provider,
				r.Throughput,
				formatDuration(r.TTFT),
				formatDuration(r.E2ELatency)))
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
			report.WriteString(fmt.Sprintf("| %d | %s | %s | %.2f tok/s | %s |\n",
				i+1,
				r.Provider,
				formatDuration(r.TTFT),
				r.Throughput,
				formatDuration(r.E2ELatency)))
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
			report.WriteString(fmt.Sprintf("| %d | %s | %s | %s | %.2f tok/s |\n",
				i+1,
				r.Provider,
				formatDuration(r.E2ELatency),
				formatDuration(r.TTFT),
				r.Throughput))
		}
		report.WriteString("\n")
	}

	report.WriteString("---\n\n")
	report.WriteString(fmt.Sprintf("*Report generated at %s*\n", time.Now().Format("2006-01-02 15:04:05")))

	if err := os.WriteFile(filename, []byte(report.String()), 0600); err != nil {
		return fmt.Errorf("error writing report: %w", err)
	}

	log.Printf("Report generated: %s", filename)
	return nil
}

// DiagnosticSummary holds the aggregated results from a diagnostic run.
type DiagnosticSummary struct {
	Provider      string         `json:"provider"`
	Model         string         `json:"model"`
	Mode          string         `json:"mode"`
	Timestamp     time.Time      `json:"timestamp"`
	TotalRequests int            `json:"totalRequests"`
	Successful    int            `json:"successful"`
	Failed        int            `json:"failed"`
	AvgE2ELatency time.Duration  `json:"avgE2eLatency"`
	AvgTTFT       time.Duration  `json:"avgTtft"`
	AvgThroughput float64        `json:"avgThroughput"`
	AvgTokens     int            `json:"avgTokens"`
	Errors        map[string]int `json:"errors,omitempty"`
}

// diagnosticMode runs continuous testing with 10 workers for 90 seconds.
// Makes requests every 15 seconds, with 30-second timeout per request.
// Workers stop starting new requests when insufficient time remains (5s grace period).
// Expected: 4 requests per worker (at 0s, 15s, 30s, 45s) for a total of 40 requests.
func diagnosticMode(config ProviderConfig, tke *tiktoken.Tiktoken, logDir, resultsDir string, mode TestMode, wg *sync.WaitGroup, results *[]DiagnosticSummary, resultsMutex *sync.Mutex) {
	if wg != nil {
		defer wg.Done()
	}
	timestamp := time.Now().Format("20060102-150405")
	logFileName := filepath.Clean(filepath.Join(logDir, fmt.Sprintf("%s-diagnostic-%s.log", config.Name, timestamp)))
	logFile, err := os.Create(logFileName)
	if err != nil {
		log.Printf("Error creating diagnostic log file for %s: %v", config.Name, err)
		return
	}
	defer func() {
		if closeErr := logFile.Close(); closeErr != nil {
			log.Printf("Warning: Failed to close log file: %v", closeErr)
		}
	}()

	providerLogger := log.New(io.MultiWriter(os.Stdout, logFile), "", log.LstdFlags)
	providerLogger.Printf("=== DIAGNOSTIC MODE: %s (%s) - Mode: %s ===", config.Name, config.Model, mode)
	providerLogger.Printf("Running 10 workers for 90 seconds with requests every 15 seconds")
	providerLogger.Printf("Timeout per request: 30 seconds")

	// Create a 90-second timeout for the entire diagnostic session
	sessionStartTime := time.Now()
	sessionDuration := 90 * time.Second
	sessionCtx, sessionCancel := context.WithTimeout(context.Background(), sessionDuration)
	defer sessionCancel()

	// Define timeout constants
	const requestTimeout = 30 * time.Second
	const gracePeriod = 5 * time.Second

	// Metrics tracking
	type diagnosticResult struct {
		workerID   int
		reqNum     int
		e2e        time.Duration
		ttft       time.Duration
		throughput float64
		tokens     int
		err        error
		mode       TestMode
		response   string
	}

	resultsChan := make(chan diagnosticResult, 1000)
	var workerWg sync.WaitGroup

	// Start 10 workers
	const numWorkers = 10
	for workerID := 1; workerID <= numWorkers; workerID++ {
		workerWg.Add(1)
		go func(id int) {
			defer workerWg.Done()
			reqNum := 0

			// Create ticker for requests every 15 seconds
			ticker := time.NewTicker(15 * time.Second)
			defer ticker.Stop()

			// Make first request immediately
			for {
				reqNum++

				// Create timeout context for this request
				reqCtx, reqCancel := context.WithTimeout(sessionCtx, requestTimeout)

				providerLogger.Printf("[Worker %d] Request #%d starting", id, reqNum)

				var e2e, ttft time.Duration
				var throughput float64
				var tokens int
				var reqErr error
				var responseContent string

				// Determine which test function to use based on mode
				var testMode TestMode
				switch mode {
				case ModeMixed:
					// Alternate between streaming and tool-calling in mixed mode
					if reqNum%2 == 1 {
						testMode = ModeStreaming
						e2e, ttft, throughput, tokens, responseContent, reqErr = singleTestRun(reqCtx, config, tke, providerLogger)
					} else {
						testMode = ModeToolCalling
						e2e, ttft, throughput, tokens, responseContent, reqErr = singleToolCallRun(reqCtx, config, tke, providerLogger)
					}
				case ModeToolCalling:
					testMode = ModeToolCalling
					e2e, ttft, throughput, tokens, responseContent, reqErr = singleToolCallRun(reqCtx, config, tke, providerLogger)
				case ModeStreaming:
					testMode = ModeStreaming
					e2e, ttft, throughput, tokens, responseContent, reqErr = singleTestRun(reqCtx, config, tke, providerLogger)
				default:
					testMode = ModeStreaming
					e2e, ttft, throughput, tokens, responseContent, reqErr = singleTestRun(reqCtx, config, tke, providerLogger)
				}

				reqCancel()

				// Save response if flag is enabled
				if saveResponses && reqErr == nil && responseContent != "" {
					responseFile := filepath.Clean(filepath.Join(logDir,
						fmt.Sprintf("%s-worker%d-req%d-%s-response.txt", config.Name, id, reqNum, testMode)))
					if err := os.WriteFile(responseFile, []byte(responseContent), 0600); err != nil {
						providerLogger.Printf("[Worker %d] Warning: Failed to save response for request #%d: %v",
							id, reqNum, err)
					}
				}

				if reqErr != nil {
					providerLogger.Printf("[Worker %d] Request #%d (%s) failed: %v", id, reqNum, testMode, reqErr)
				} else {
					providerLogger.Printf("[Worker %d] Request #%d (%s) success: E2E=%s TTFT=%s Throughput=%.2f tok/s Tokens=%d",
						id, reqNum, testMode, formatDuration(e2e), formatDuration(ttft), throughput, tokens)
				}

				resultsChan <- diagnosticResult{
					workerID:   id,
					reqNum:     reqNum,
					e2e:        e2e,
					ttft:       ttft,
					throughput: throughput,
					tokens:     tokens,
					err:        reqErr,
					mode:       testMode,
					response:   responseContent,
				}

				// Wait for next tick or session end
				select {
				case <-sessionCtx.Done():
					providerLogger.Printf("[Worker %d] Session ended, completed %d requests", id, reqNum)
					return
				case <-ticker.C:
					// Check if there's enough time remaining before starting the next request
					elapsed := time.Since(sessionStartTime)
					timeRemaining := sessionDuration - elapsed

					// Skip new requests if insufficient time remains
					if timeRemaining < requestTimeout+gracePeriod {
						providerLogger.Printf(
							"[Worker %d] Stopping - insufficient time remaining for next request (%.1fs left, need %.1fs)",
							id, timeRemaining.Seconds(), (requestTimeout + gracePeriod).Seconds())
						providerLogger.Printf("[Worker %d] Completed %d requests", id, reqNum)
						return
					}
					// Continue to next request
				}
			}
		}(workerID)
	}

	// Wait for all workers to complete
	go func() {
		workerWg.Wait()
		close(resultsChan)
	}()

	// Collect and aggregate results
	var successCount, failureCount int
	var totalE2E, totalTTFT time.Duration
	var totalThroughput float64
	var totalTokens int
	errors := make(map[string]int)

	for result := range resultsChan {
		if result.err != nil {
			failureCount++
			errors[result.err.Error()]++
		} else {
			successCount++
			totalE2E += result.e2e
			totalTTFT += result.ttft
			totalThroughput += result.throughput
			totalTokens += result.tokens
		}
	}

	// Print summary
	providerLogger.Println("")
	providerLogger.Println("========================================")
	providerLogger.Println("   DIAGNOSTIC MODE SUMMARY")
	providerLogger.Println("========================================")
	providerLogger.Printf("Provider: %s", config.Name)
	providerLogger.Printf("Model: %s", config.Model)
	providerLogger.Printf("Mode: %s", mode)
	providerLogger.Printf("Total Requests: %d", successCount+failureCount)
	providerLogger.Printf("Successful: %d", successCount)
	providerLogger.Printf("Failed: %d", failureCount)

	if successCount > 0 {
		avgE2E := totalE2E / time.Duration(successCount)
		avgTTFT := totalTTFT / time.Duration(successCount)
		avgThroughput := totalThroughput / float64(successCount)
		avgTokens := totalTokens / successCount

		providerLogger.Println("--------------------------------------")
		providerLogger.Printf("Average E2E Latency: %s", formatDuration(avgE2E))
		providerLogger.Printf("Average TTFT: %s", formatDuration(avgTTFT))
		providerLogger.Printf("Average Throughput: %.2f tokens/s", avgThroughput)
		providerLogger.Printf("Average Tokens: %d", avgTokens)
	}

	if len(errors) > 0 {
		providerLogger.Println("--------------------------------------")
		providerLogger.Println("Errors encountered:")
		for errMsg, count := range errors {
			providerLogger.Printf("  - %s (x%d)", errMsg, count)
		}
	}

	providerLogger.Println("========================================")

	// Create diagnostic summary
	summary := DiagnosticSummary{
		Provider:      config.Name,
		Model:         config.Model,
		Mode:          string(mode),
		Timestamp:     time.Now(),
		TotalRequests: successCount + failureCount,
		Successful:    successCount,
		Failed:        failureCount,
	}

	if successCount > 0 {
		summary.AvgE2ELatency = totalE2E / time.Duration(successCount)
		summary.AvgTTFT = totalTTFT / time.Duration(successCount)
		summary.AvgThroughput = totalThroughput / float64(successCount)
		summary.AvgTokens = totalTokens / successCount
	}

	if len(errors) > 0 {
		summary.Errors = errors
	}

	// Save diagnostic summary to JSON
	summaryFile := filepath.Join(resultsDir, fmt.Sprintf("%s-diagnostic-summary-%s.json", config.Name, timestamp))
	data, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		providerLogger.Printf("Warning: Failed to marshal diagnostic summary: %v", err)
	} else {
		if err := os.WriteFile(summaryFile, data, 0600); err != nil {
			providerLogger.Printf("Warning: Failed to write diagnostic summary: %v", err)
		} else {
			providerLogger.Printf("Diagnostic summary saved: %s", summaryFile)
		}
	}

	// Append to results slice if provided
	if results != nil && resultsMutex != nil {
		resultsMutex.Lock()
		*results = append(*results, summary)
		resultsMutex.Unlock()
	}
}

// generateDiagnosticReport creates a markdown report for diagnostic mode results.
func generateDiagnosticReport(resultsDir string, results []DiagnosticSummary, sessionTimestamp string) error {
	filename := filepath.Join(resultsDir, "DIAGNOSTIC-REPORT.md")

	var report strings.Builder
	report.WriteString("# LLM API Diagnostic Mode Results\n\n")
	report.WriteString(fmt.Sprintf("**Test Session:** %s\n\n", sessionTimestamp))
	report.WriteString("**Test Duration:** 90 seconds per provider\n")
	report.WriteString("**Workers:** 10 concurrent workers\n")
	report.WriteString("**Request Frequency:** Every 15 seconds per worker\n")
	report.WriteString("**Timeout:** 30 seconds per request\n\n")
	report.WriteString("---\n\n")

	// Summary statistics
	totalProviders := len(results)
	var totalRequests, totalSuccessful, totalFailed int
	for _, r := range results {
		totalRequests += r.TotalRequests
		totalSuccessful += r.Successful
		totalFailed += r.Failed
	}

	report.WriteString("## Summary\n\n")
	report.WriteString(fmt.Sprintf("- **Providers Tested:** %d\n", totalProviders))
	report.WriteString(fmt.Sprintf("- **Total Requests:** %d\n", totalRequests))
	report.WriteString(fmt.Sprintf("- **Successful:** %d (%.1f%%)\n",
		totalSuccessful, 100.0*float64(totalSuccessful)/float64(totalRequests)))
	report.WriteString(fmt.Sprintf("- **Failed:** %d (%.1f%%)\n\n",
		totalFailed, 100.0*float64(totalFailed)/float64(totalRequests)))

	// Detailed results table
	if len(results) > 0 {
		report.WriteString("## Detailed Results\n\n")
		report.WriteString("| Provider | Model | Mode | Total Requests | Success | Failed | Avg E2E |" +
			" Avg TTFT | Avg Throughput |\n")
		report.WriteString("|----------|-------|------|----------------|---------|--------|---------|" +
			"----------|----------------|\n")

		for _, r := range results {
			successRate := fmt.Sprintf("%d/%d", r.Successful, r.TotalRequests)
			failRate := fmt.Sprintf("%d", r.Failed)
			avgE2E := NotAvailable
			avgTTFT := NotAvailable
			avgThroughput := NotAvailable

			if r.Successful > 0 {
				avgE2E = formatDuration(r.AvgE2ELatency)
				avgTTFT = formatDuration(r.AvgTTFT)
				avgThroughput = fmt.Sprintf("%.2f tok/s", r.AvgThroughput)
			}

			report.WriteString(fmt.Sprintf("| %s | %s | %s | %d | %s | %s | %s | %s | %s |\n",
				r.Provider,
				r.Model,
				r.Mode,
				r.TotalRequests,
				successRate,
				failRate,
				avgE2E,
				avgTTFT,
				avgThroughput))
		}
		report.WriteString("\n")
	}

	// Performance Leaderboard
	successfulResults := make([]DiagnosticSummary, 0)
	for _, r := range results {
		if r.Successful > 0 {
			successfulResults = append(successfulResults, r)
		}
	}

	if len(successfulResults) > 0 {
		report.WriteString("## Performance Leaderboard\n\n")
		report.WriteString("### By Throughput (Tokens/sec)\n\n")

		// Sort by throughput
		for i := 0; i < len(successfulResults); i++ {
			for j := i + 1; j < len(successfulResults); j++ {
				if successfulResults[j].AvgThroughput > successfulResults[i].AvgThroughput {
					successfulResults[i], successfulResults[j] = successfulResults[j], successfulResults[i]
				}
			}
		}

		report.WriteString("| Rank | Provider | Throughput | TTFT | E2E Latency | Success Rate |\n")
		report.WriteString("|------|----------|------------|------|-------------|-------------|\n")

		for i, r := range successfulResults {
			successRate := fmt.Sprintf("%.1f%%", 100.0*float64(r.Successful)/float64(r.TotalRequests))
			report.WriteString(fmt.Sprintf("| %d | %s | %.2f tok/s | %s | %s | %s |\n",
				i+1,
				r.Provider,
				r.AvgThroughput,
				formatDuration(r.AvgTTFT),
				formatDuration(r.AvgE2ELatency),
				successRate))
		}
		report.WriteString("\n")

		// Sort by TTFT
		report.WriteString("### By Time to First Token (TTFT)\n\n")

		for i := 0; i < len(successfulResults); i++ {
			for j := i + 1; j < len(successfulResults); j++ {
				if successfulResults[j].AvgTTFT < successfulResults[i].AvgTTFT {
					successfulResults[i], successfulResults[j] = successfulResults[j], successfulResults[i]
				}
			}
		}

		report.WriteString("| Rank | Provider | TTFT | Throughput | E2E Latency | Success Rate |\n")
		report.WriteString("|------|----------|------|------------|-------------|-------------|\n")

		for i, r := range successfulResults {
			successRate := fmt.Sprintf("%.1f%%", 100.0*float64(r.Successful)/float64(r.TotalRequests))
			report.WriteString(fmt.Sprintf("| %d | %s | %s | %.2f tok/s | %s | %s |\n",
				i+1,
				r.Provider,
				formatDuration(r.AvgTTFT),
				r.AvgThroughput,
				formatDuration(r.AvgE2ELatency),
				successRate))
		}
		report.WriteString("\n")
	}

	// Error Analysis
	hasErrors := false
	for _, r := range results {
		if len(r.Errors) > 0 {
			hasErrors = true
			break
		}
	}

	if hasErrors {
		report.WriteString("## Error Analysis\n\n")

		for _, r := range results {
			if len(r.Errors) > 0 {
				report.WriteString(fmt.Sprintf("### %s Errors\n\n", r.Provider))
				report.WriteString("| Error | Count |\n")
				report.WriteString("|-------|-------|\n")

				for errMsg, count := range r.Errors {
					report.WriteString(fmt.Sprintf("| %s | %d |\n", errMsg, count))
				}
				report.WriteString("\n")
			}
		}
	}

	report.WriteString("---\n\n")
	report.WriteString(fmt.Sprintf("*Report generated at %s*\n", time.Now().Format("2006-01-02 15:04:05")))

	if err := os.WriteFile(filename, []byte(report.String()), 0600); err != nil {
		return fmt.Errorf("error writing diagnostic report: %w", err)
	}

	log.Printf("Diagnostic report generated: %s", filename)
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
	providerName := flag.String("provider", "",
		"Specific provider to test (e.g., nim, novita). If empty, tests 'generic' provider.")
	testAll := flag.Bool("all", false, "Test all configured providers concurrently.")
	flagGenericURL := flag.String("url", "",
		"Override Base URL for 'generic' provider (default: https://openrouter.ai/api/v1)")
	flagGenericModel := flag.String("model", "",
		"Model name for 'generic' provider (required if --provider is not set)")
	toolCalling := flag.Bool("tool-calling", false, "Use tool calling mode instead of regular streaming")
	mixed := flag.Bool("mixed", false, "Run both streaming and tool-calling modes (3 runs each)")
	diagnostic := flag.Bool("diagnostic", false,
		"Run diagnostic mode: 10 workers making requests every 15s for 1 minute with 30s timeout")
	flagSaveResponses := flag.Bool("save-responses", false, "Save all API responses to log files")
	flag.Parse()

	// Set global flag for saving responses
	saveResponses = *flagSaveResponses

	// 3. Create session-based folder structure
	sessionTimestamp := time.Now().Format("20060102-150405")
	sessionDir := filepath.Join("results", fmt.Sprintf("session-%s", sessionTimestamp))
	logDir := filepath.Join(sessionDir, "logs")
	resultsDir := sessionDir

	if err := os.MkdirAll(logDir, 0750); err != nil {
		log.Fatalf("Error creating logs directory: %v", err)
	}

	if err := os.MkdirAll(resultsDir, 0750); err != nil {
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

	switch {
	case *testAll:
		log.Println("--- Testing all configured providers... ---")
		for name, config := range allProviderConfigs {
			if config.APIKey != "" && config.Model != "" {
				providersToTest = append(providersToTest, config)
			} else if name != "generic" {
				// Don't log generic provider if not set, it's optional
				log.Printf("... Skipping '%s': APIKey or Model not configured in .env\n", name)
			}
		}
		// Check generic provider separately for --all
		genConfig := allProviderConfigs["generic"]
		if genConfig.APIKey != "" && genConfig.Model != "" {
			log.Println("... 'generic' provider is configured, but will be skipped. " +
				"Use --provider=generic or no flags to test it.")
		}
	case *providerName != "":
		log.Printf("--- Testing single provider: '%s' ---\n", *providerName)
		config, ok := allProviderConfigs[*providerName]
		if !ok {
			log.Fatalf("Error: Provider '%s' not recognized.", *providerName)
		}
		if config.APIKey == "" || config.Model == "" {
			log.Fatalf("Error: Provider '%s' is not configured. "+
				"(Missing APIKey/Model in .env or --model flag for generic)", *providerName)
		}
		providersToTest = append(providersToTest, config)
	default:
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

	// Determine test mode
	var testMode TestMode
	switch {
	case *mixed:
		testMode = ModeMixed
		log.Println("Test mode: Mixed (streaming + tool-calling)")
	case *toolCalling:
		testMode = ModeToolCalling
		log.Println("Test mode: Tool-calling)")
	default:
		testMode = ModeStreaming
		log.Println("Test mode: Streaming")
	}

	// 6. Run Tests
	if *diagnostic {
		// Run diagnostic mode
		log.Println("=== RUNNING IN DIAGNOSTIC MODE ===")

		var diagnosticResults []DiagnosticSummary
		var diagnosticMutex sync.Mutex

		if len(providersToTest) > 1 {
			// Run multiple providers concurrently
			var diagnosticWg sync.WaitGroup
			for _, provider := range providersToTest {
				diagnosticWg.Add(1)
				go diagnosticMode(provider, tke, logDir, resultsDir, testMode, &diagnosticWg, &diagnosticResults, &diagnosticMutex)
			}
			diagnosticWg.Wait()
		} else {
			// Single provider (no concurrency needed)
			for _, provider := range providersToTest {
				diagnosticMode(provider, tke, logDir, resultsDir, testMode, nil, &diagnosticResults, &diagnosticMutex)
			}
		}

		log.Println("--- All diagnostic tests complete. ---")

		// Generate diagnostic report
		log.Println("Generating diagnostic summary report...")
		if err := generateDiagnosticReport(resultsDir, diagnosticResults, sessionTimestamp); err != nil {
			log.Printf("Warning: Failed to generate diagnostic report: %v", err)
		}

		log.Printf("Diagnostic tests complete. Results saved to: %s/", sessionDir)
		return
	}
	var wg sync.WaitGroup
	var results []TestResult
	var resultsMutex sync.Mutex

	for _, provider := range providersToTest {
		if *testAll {
			// Run all tests concurrently
			wg.Add(1)
			go testProviderMetrics(provider, tke, &wg, logDir, resultsDir, &results, &resultsMutex, testMode)
		} else {
			// Run a single test sequentially
			testProviderMetrics(provider, tke, nil, logDir, resultsDir, &results, &resultsMutex, testMode)
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
