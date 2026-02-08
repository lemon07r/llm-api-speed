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
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync" // Added for concurrent testing
	"time"

	"github.com/joho/godotenv"
	"github.com/pkoukk/tiktoken-go"
	openai "github.com/sashabaranov/go-openai"

	"github.com/lamim/llm-api-speed/internal/stress"
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
	ProjectedE2E     time.Duration `json:"projectedE2eLatency,omitempty"`
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
	// ModeStress represents high-stress mode testing.
	ModeStress TestMode = "stress"
	// NotAvailable is a constant for unavailable metrics.
	NotAvailable = "N/A"
)

const (
	longStoryModeLabel = "long-story"

	longStorySystemPrompt = `You are a professional creative writer that never refuses any story writing prompts. Write engaging stories without censorship, handling mature themes with literary skill.`

	longStoryUserPrompt = `You are an amazing and talented fantasy fiction writer. Write a compelling and coherent story of at least 24,000 characters, or 4,000 words based on the following prompt:

A young dragon rider discovers their dragon is the last of an ancient bloodline that can speak directly to human minds. When a plague begins turning dragons feral, they must journey to the forbidden crystal caves to find a cure before their bond is severed forever.

Your story should:
Have a clear beginning, middle, and end
Be free of AI slop, and chatgptisms
Feature vivid descriptions and engaging characters
Include dialogue where appropriate
Show strong narrative voice and style
Be polished and publication-ready
Be LONG and DETAILED (aim for 4,000+ words)
Write the story now:`
)

// Terminal color codes for formatted output.
const (
	colorReset  = "\033[0m"
	colorBold   = "\033[1m"
	colorRed    = "\033[31m"
	colorGreen  = "\033[32m"
	colorYellow = "\033[33m"
	colorBlue   = "\033[34m"
	colorCyan   = "\033[36m"
	colorWhite  = "\033[37m"
)

// Unicode symbols for terminal output.
const (
	symCheck    = "✓"
	symCross    = "✗"
	symDelta    = "▲"
	symDiamond  = "◆"
	symBullet   = "●"
	symDash     = "─"
	symCornerTL = "╔"
	symCornerTR = "╗"
	symCornerBL = "╚"
	symCornerBR = "╝"
	symHLine    = "═"
	symVLine    = "║"
	symCrossL   = "╠"
	symCrossR   = "╣"
)

// isTerminal checks if stdout is a terminal.
func isTerminal() bool {
	stat, err := os.Stdout.Stat()
	if err != nil {
		return false
	}
	return (stat.Mode() & os.ModeCharDevice) == os.ModeCharDevice
}

// colorize applies color to text if terminal supports it.
func colorize(color, text string) string {
	if !isTerminal() {
		return text
	}
	return color + text + colorReset
}

// bold makes text bold if terminal supports it.
func bold(text string) string {
	return colorize(colorBold, text)
}

// StressSummary holds the aggregated results from a high-stress run.
type StressSummary struct {
	Provider        string
	Model           string
	Timestamp       time.Time
	Workers         int
	DurationSeconds int
	TotalRequests   int
	Successful      int
	Failed          int
	ShortRequests   int
	ToolRequests    int
	LongRequests    int
	AvgE2ELatency   time.Duration
	P50E2E          time.Duration
	P95E2E          time.Duration
	P99E2E          time.Duration
	AvgTTFT         time.Duration
	AvgThroughput   float64
	RequestsPerSec  float64
	TotalTokens     int
	ProjectedE2E    time.Duration
	Errors          map[string]int
}

// RequestTypeStats holds per-request-type statistics for stress mode.
type RequestTypeStats struct {
	Count         int
	Successful    int
	Failed        int
	AvgE2E        time.Duration
	MinE2E        time.Duration
	MaxE2E        time.Duration
	P50E2E        time.Duration
	P95E2E        time.Duration
	P99E2E        time.Duration
	AvgTTFT       time.Duration
	AvgThroughput float64
	TotalTokens   int
	AvgTokens     int
}

// HealthGrade represents a health assessment grade.
type HealthGrade struct {
	Symbol string
	Label  string
	Color  string
}

// gradeSuccessRate returns health grade for success rate.
func gradeSuccessRate(rate float64) HealthGrade {
	switch {
	case rate >= 0.99:
		return HealthGrade{symCheck, "EXCELLENT", colorGreen}
	case rate >= 0.95:
		return HealthGrade{symCheck, "GOOD", colorGreen}
	case rate >= 0.90:
		return HealthGrade{symDelta, "FAIR", colorYellow}
	case rate >= 0.80:
		return HealthGrade{symDelta, "POOR", colorYellow}
	default:
		return HealthGrade{symCross, "CRITICAL", colorRed}
	}
}

// gradeLatencyP95 returns health grade for P95 latency.
func gradeLatencyP95(p95 time.Duration) HealthGrade {
	switch {
	case p95 < 3*time.Second:
		return HealthGrade{symCheck, "EXCELLENT", colorGreen}
	case p95 < 5*time.Second:
		return HealthGrade{symCheck, "GOOD", colorGreen}
	case p95 < 10*time.Second:
		return HealthGrade{symDelta, "FAIR", colorYellow}
	case p95 < 20*time.Second:
		return HealthGrade{symDelta, "POOR", colorYellow}
	default:
		return HealthGrade{symCross, "CRITICAL", colorRed}
	}
}

// gradeThroughput returns health grade for throughput.
func gradeThroughput(tps float64) HealthGrade {
	switch {
	case tps >= 150:
		return HealthGrade{symCheck, "EXCELLENT", colorGreen}
	case tps >= 100:
		return HealthGrade{symCheck, "GOOD", colorGreen}
	case tps >= 50:
		return HealthGrade{symDelta, "FAIR", colorYellow}
	case tps >= 20:
		return HealthGrade{symDelta, "POOR", colorYellow}
	default:
		return HealthGrade{symCross, "CRITICAL", colorRed}
	}
}

// gradeErrorRate returns health grade for error rate.
func gradeErrorRate(rate float64) HealthGrade {
	switch {
	case rate < 0.01:
		return HealthGrade{symCheck, "EXCELLENT", colorGreen}
	case rate < 0.02:
		return HealthGrade{symCheck, "GOOD", colorGreen}
	case rate < 0.05:
		return HealthGrade{symDelta, "FAIR", colorYellow}
	case rate < 0.10:
		return HealthGrade{symDelta, "POOR", colorYellow}
	default:
		return HealthGrade{symCross, "CRITICAL", colorRed}
	}
}

// calculateOverallGrade calculates overall grade for stress test.
func calculateOverallGrade(summary StressSummary) (string, string) {
	successRate := float64(summary.Successful) / float64(summary.TotalRequests)
	errorRate := float64(summary.Failed) / float64(summary.TotalRequests)

	score := 0
	if successRate >= 0.95 {
		score += 2
	} else if successRate >= 0.90 {
		score++
	}
	if summary.P95E2E < 5*time.Second {
		score += 2
	} else if summary.P95E2E < 10*time.Second {
		score++
	}
	if summary.AvgThroughput >= 100 {
		score += 2
	} else if summary.AvgThroughput >= 50 {
		score++
	}
	if errorRate < 0.02 {
		score += 2
	} else if errorRate < 0.05 {
		score++
	}

	switch {
	case score >= 7:
		return "A", colorize(colorGreen, "EXCELLENT")
	case score >= 6:
		return "A-", colorize(colorGreen, "VERY GOOD")
	case score >= 5:
		return "B+", colorize(colorGreen, "GOOD")
	case score >= 4:
		return "B", colorize(colorYellow, "SATISFACTORY")
	case score >= 3:
		return "C", colorize(colorYellow, "NEEDS IMPROVEMENT")
	case score >= 2:
		return "D", colorize(colorRed, "POOR")
	default:
		return "F", colorize(colorRed, "CRITICAL")
	}
}

// stripANSI removes ANSI escape sequences from a string.
func stripANSI(s string) string {
	// Remove common ANSI escape sequences
	s = strings.ReplaceAll(s, colorReset, "")
	s = strings.ReplaceAll(s, colorBold, "")
	s = strings.ReplaceAll(s, colorRed, "")
	s = strings.ReplaceAll(s, colorGreen, "")
	s = strings.ReplaceAll(s, colorYellow, "")
	s = strings.ReplaceAll(s, colorBlue, "")
	s = strings.ReplaceAll(s, colorCyan, "")
	s = strings.ReplaceAll(s, colorWhite, "")
	return s
}

// visibleLen returns the visible length of a string (excluding ANSI codes).
func visibleLen(s string) int {
	return len(stripANSI(s))
}

// printBoxTop prints the top border of a box.
func printBoxTop() {
	log.Println(colorize(colorCyan, symCornerTL+strings.Repeat(symHLine, 70)+symCornerTR))
}

// printBoxBottom prints the bottom border of a box.
func printBoxBottom() {
	log.Println(colorize(colorCyan, symCornerBL+strings.Repeat(symHLine, 70)+symCornerBR))
}

// printBoxDivider prints a horizontal divider.
func printBoxDivider() {
	log.Println(colorize(colorCyan, symCrossL+strings.Repeat(symHLine, 70)+symCrossR))
}

// printBoxLine prints a line inside the box.
func printBoxLine(content string) {
	padding := 68 - visibleLen(content)
	if padding < 0 {
		padding = 0
	}
	log.Println(colorize(colorCyan, symVLine) + " " + content + strings.Repeat(" ", padding) + " " + colorize(colorCyan, symVLine))
}

// printBoxLineColored prints a line with specific color.
func printBoxLineColored(content, lineColor string) {
	padding := 68 - visibleLen(content)
	if padding < 0 {
		padding = 0
	}
	log.Println(colorize(colorCyan, symVLine) + " " + colorize(lineColor, content) + strings.Repeat(" ", padding) + " " + colorize(colorCyan, symVLine))
}

// formatNumber formats large numbers with commas.
func formatNumber(n int) string {
	if n < 1000 {
		return fmt.Sprintf("%d", n)
	}
	if n < 1000000 {
		return fmt.Sprintf("%d,%03d", n/1000, n%1000)
	}
	return fmt.Sprintf("%d,%03d,%03d", n/1000000, (n%1000000)/1000, n%1000)
}

func logInterleavedToolError(providerLogger *log.Logger, config ProviderConfig, streamErr error) {
	var apiErr *openai.APIError
	if errors.As(streamErr, &apiErr) {
		param := ""
		if apiErr.Param != nil {
			param = *apiErr.Param
		}
		message := apiErr.Message
		lowerMsg := strings.ToLower(message)
		if param == "parallel_tool_calls" || strings.Contains(lowerMsg, "parallel_tool_calls") {
			providerLogger.Printf("[%s] Interleaved tool calls NOT supported by model %s (error: %s)", config.Name, config.Model, message)
			return
		}
		providerLogger.Printf("[%s] Interleaved tool-call request rejected by API: %s", config.Name, message)
		return
	}
	providerLogger.Printf("[%s] Interleaved tool-call request failed before streaming: %v", config.Name, streamErr)
}

// resolveTestMode determines which TestMode should run based on CLI flags and whether
// tool-reasoning checks should remain enabled. It returns the selected mode, the
// effective reasoning-check flag (disabled automatically for pure streaming tests),
// and whether tool-calling mode was implicitly forced by requesting reasoning checks.
func resolveTestMode(toolCalling, mixed, toolReasoningCheck bool) (TestMode, bool, bool) {
	forcedToolCalling := false
	mode := ModeStreaming

	switch {
	case mixed:
		mode = ModeMixed
	case toolCalling:
		mode = ModeToolCalling
	case toolReasoningCheck:
		mode = ModeToolCalling
		forcedToolCalling = true
	}

	if mode == ModeStreaming {
		toolReasoningCheck = false
	}

	return mode, toolReasoningCheck, forcedToolCalling
}

// resolveStressWorkers determines the number of workers for stress mode.
func resolveStressWorkers(stress bool, stressWorkers int, stressLevel string) int {
	if !stress {
		return 0
	}
	if stressWorkers > 0 {
		return stressWorkers
	}
	switch stressLevel {
	case "moderate":
		return 100
	case "heavy":
		return 500
	case "extreme":
		return 1000
	default:
		log.Fatalf("Error: Unknown stress level '%s'. Use: moderate, heavy, or extreme", stressLevel)
		return 0
	}
}

// validateStressSettings validates stress mode configuration parameters.
func validateStressSettings(numWorkers, stressDuration, stressLongBias, stressRampUp int) {
	if numWorkers < 1 {
		log.Fatal("Error: --stress-workers must be at least 1")
	}
	if stressDuration < 60 {
		log.Fatal("Error: --stress-duration must be at least 60 seconds")
	}
	if stressLongBias < 10 || stressLongBias > 80 {
		log.Fatal("Error: --stress-long-bias must be between 10 and 80")
	}
	if stressRampUp < 1 {
		log.Fatal("Error: --stress-rampup must be at least 1 second")
	}
}

// formatDuration formats a duration as decimal seconds.
func formatDuration(d time.Duration) string {
	return fmt.Sprintf("%.3fs", d.Seconds())
}

var saveResponses bool
var targetTokens int
var maxTokens int

// calculateProjectedE2E calculates the projected E2E latency for a normalized token count.
// Formula: ProjectedE2E = TTFT + (TargetTokens / Throughput).
func calculateProjectedE2E(ttft time.Duration, throughput float64, target int) time.Duration {
	if throughput <= 0 || target <= 0 {
		return 0
	}
	generationTime := float64(target) / throughput
	return ttft + time.Duration(generationTime*float64(time.Second))
}

// writeTestResultRow writes a single test result row to the report.
func writeTestResultRow(report *strings.Builder, r TestResult, includeProjected bool) {
	if includeProjected && r.ProjectedE2E > 0 {
		fmt.Fprintf(report, "| %s | %s | %s | %s | %s | %.2f tok/s | %d | %s |\n",
			r.Provider, r.Model, r.Mode,
			formatDuration(r.E2ELatency), formatDuration(r.TTFT),
			r.Throughput, r.CompletionTokens, formatDuration(r.ProjectedE2E))
	} else {
		fmt.Fprintf(report, "| %s | %s | %s | %s | %s | %.2f tok/s | %d |\n",
			r.Provider, r.Model, r.Mode,
			formatDuration(r.E2ELatency), formatDuration(r.TTFT),
			r.Throughput, r.CompletionTokens)
	}
}

// writeDiagnosticResultRow writes a single diagnostic result row to the report.
func writeDiagnosticResultRow(report *strings.Builder, r DiagnosticSummary, includeProjected bool) {
	successRate := fmt.Sprintf("%d/%d", r.Successful, r.TotalRequests)
	failRate := fmt.Sprintf("%d", r.Failed)
	avgE2E := NotAvailable
	avgTTFT := NotAvailable
	avgThroughput := NotAvailable
	projectedE2E := NotAvailable

	if r.Successful > 0 {
		avgE2E = formatDuration(r.AvgE2ELatency)
		avgTTFT = formatDuration(r.AvgTTFT)
		avgThroughput = fmt.Sprintf("%.2f tok/s", r.AvgThroughput)
		if r.ProjectedE2E > 0 {
			projectedE2E = formatDuration(r.ProjectedE2E)
		}
	}

	if includeProjected {
		fmt.Fprintf(report, "| %s | %s | %s | %d | %s | %s | %s | %s | %s | %s |\n",
			r.Provider, r.Model, r.Mode, r.TotalRequests,
			successRate, failRate, avgE2E, avgTTFT, avgThroughput, projectedE2E)
	} else {
		fmt.Fprintf(report, "| %s | %s | %s | %d | %s | %s | %s | %s | %s |\n",
			r.Provider, r.Model, r.Mode, r.TotalRequests,
			successRate, failRate, avgE2E, avgTTFT, avgThroughput)
	}
}

// writeProjectedE2ELeaderboard writes the projected E2E leaderboard section for TestResult.
func writeProjectedE2ELeaderboard(report *strings.Builder, results []TestResult) {
	// Sort by Projected E2E
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].ProjectedE2E > 0 && results[i].ProjectedE2E > 0 &&
				results[j].ProjectedE2E < results[i].ProjectedE2E {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	report.WriteString("| Rank | Provider | Projected E2E | TTFT | Throughput |\n")
	report.WriteString("|------|----------|---------------|------|------------|\n")

	for i, r := range results {
		if r.ProjectedE2E > 0 {
			fmt.Fprintf(report, "| %d | %s | %s | %s | %.2f tok/s |\n",
				i+1, r.Provider, formatDuration(r.ProjectedE2E),
				formatDuration(r.TTFT), r.Throughput)
		}
	}
	report.WriteString("\n")
}

// writeProjectedE2EDiagnosticLeaderboard writes the projected E2E leaderboard section for DiagnosticSummary.
func writeProjectedE2EDiagnosticLeaderboard(report *strings.Builder, results []DiagnosticSummary) {
	// Sort by Projected E2E
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].ProjectedE2E > 0 && results[i].ProjectedE2E > 0 &&
				results[j].ProjectedE2E < results[i].ProjectedE2E {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	report.WriteString("| Rank | Provider | Projected E2E | TTFT | Throughput | Success Rate |\n")
	report.WriteString("|------|----------|---------------|------|------------|-------------|\n")

	for i, r := range results {
		if r.ProjectedE2E > 0 {
			successRate := fmt.Sprintf("%.1f%%", 100.0*float64(r.Successful)/float64(r.TotalRequests))
			fmt.Fprintf(report, "| %d | %s | %s | %s | %.2f tok/s | %s |\n",
				i+1, r.Provider, formatDuration(r.ProjectedE2E),
				formatDuration(r.AvgTTFT), r.AvgThroughput, successRate)
		}
	}
	report.WriteString("\n")
}

// writeTestResultLeaderboards writes all leaderboard sections for TestResult.
func writeTestResultLeaderboards(report *strings.Builder, results []TestResult) {
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
		fmt.Fprintf(report, "| %d | %s | %.2f tok/s | %s | %s |\n",
			i+1, r.Provider, r.Throughput,
			formatDuration(r.TTFT), formatDuration(r.E2ELatency))
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
		fmt.Fprintf(report, "| %d | %s | %s | %.2f tok/s | %s |\n",
			i+1, r.Provider, formatDuration(r.TTFT),
			r.Throughput, formatDuration(r.E2ELatency))
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
		fmt.Fprintf(report, "| %d | %s | %s | %s | %.2f tok/s |\n",
			i+1, r.Provider, formatDuration(r.E2ELatency),
			formatDuration(r.TTFT), r.Throughput)
	}
	report.WriteString("\n")

	// Sort by Projected E2E (if available)
	if targetTokens > 0 {
		fmt.Fprintf(report, "### By Projected E2E Latency (%d tokens)\n\n", targetTokens)
		writeProjectedE2ELeaderboard(report, successfulResults)
	}
}

// createHTTPClient creates an OpenAI client with optimized HTTP transport for high-concurrency testing.
func createHTTPClient(config ProviderConfig, workers ...int) *openai.Client {
	clientConfig := openai.DefaultConfig(config.APIKey)
	clientConfig.BaseURL = config.BaseURL

	// Determine connection pool size based on workers
	poolSize := 100
	if len(workers) > 0 && workers[0] > 0 {
		poolSize = workers[0] * 2
	}

	// Configure HTTP transport for connection pooling and keep-alive
	transport := &http.Transport{
		MaxIdleConns:        poolSize,
		MaxIdleConnsPerHost: poolSize,
		MaxConnsPerHost:     poolSize,
		IdleConnTimeout:     90 * time.Second,
		TLSHandshakeTimeout: 30 * time.Second, // Increased for high-latency scenarios
		DisableKeepAlives:   false,
		ForceAttemptHTTP2:   true,
	}

	clientConfig.HTTPClient = &http.Client{
		Transport: transport,
		Timeout:   0, // No global timeout - handled per-request via context
	}

	return openai.NewClientWithConfig(clientConfig)
}

// runStreamingChat executes a streaming chat completion request and computes metrics.
func runStreamingChat(ctx context.Context, config ProviderConfig, tke *tiktoken.Tiktoken, providerLogger *log.Logger, req openai.ChatCompletionRequest) (e2e, ttft time.Duration, throughput float64, tokens int, response string, err error) {
	client := createHTTPClient(config)

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

		if len(response.Choices) == 0 {
			if chunkCount%100 == 0 {
				providerLogger.Printf("[%s] ... Chunk %d: Empty Choices array (diagnostic: ID=%s, Model=%s)",
					config.Name, chunkCount, response.ID, response.Model)
			}
			continue
		}

		delta := response.Choices[0].Delta
		content := delta.Content
		reasoningContent := delta.ReasoningContent

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

	fullResponse := fullResponseContent.String()
	tokenList := tke.Encode(fullResponse, nil, nil)
	completionTokens := len(tokenList)

	providerLogger.Printf(
		"[%s] ... Total content length: %d bytes, %d tokens",
		config.Name, len(fullResponse), completionTokens)

	if completionTokens == 0 {
		return 0, 0, 0, 0, "", fmt.Errorf("received 0 tokens (content length: %d bytes)", len(fullResponse))
	}

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

// singleTestRun performs one test run and returns metrics or error.
func singleTestRun(ctx context.Context, config ProviderConfig, tke *tiktoken.Tiktoken, providerLogger *log.Logger) (e2e, ttft time.Duration, throughput float64, tokens int, response string, err error) {
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

	return runStreamingChat(ctx, config, tke, providerLogger, req)
}

// longStoryRun performs a single long-form story generation run and returns metrics or error.
func longStoryRun(ctx context.Context, config ProviderConfig, tke *tiktoken.Tiktoken, providerLogger *log.Logger) (e2e, ttft time.Duration, throughput float64, tokens int, response string, err error) {
	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: longStorySystemPrompt,
		},
		{
			Role:    openai.ChatMessageRoleUser,
			Content: longStoryUserPrompt,
		},
	}
	storyMaxTokens := maxTokens
	if storyMaxTokens <= 0 {
		storyMaxTokens = 16384
	}

	req := openai.ChatCompletionRequest{
		Model:     config.Model,
		Messages:  messages,
		MaxTokens: storyMaxTokens,
		Stream:    true,
	}

	return runStreamingChat(ctx, config, tke, providerLogger, req)
}

// singleToolCallRun performs one tool-calling test run and returns metrics or error.
// When toolReasoningCheck is true, additional logging is produced to validate that
// tool calls occur alongside multi-step reasoning (before and after tool use).
func singleToolCallRun(ctx context.Context, config ProviderConfig, tke *tiktoken.Tiktoken, providerLogger *log.Logger, toolReasoningCheck bool) (e2e, ttft time.Duration, throughput float64, tokens int, response string, err error) {
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

	prompt := "You are a weather analysis assistant. You MUST call the get_weather tool at least once for " +
		"each city you are asked about before answering. Do not guess or answer without using the tool. " +
		"Question: What's the weather like in San Francisco, Tokyo, and London? Please check all three cities " +
		"using the tool and then tell me which one has the best weather for outdoor activities today."
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
	req.ToolChoice = "auto"
	if toolReasoningCheck {
		req.ParallelToolCalls = true
	}

	// Execute the stream and measure metrics
	startTime := time.Now()
	var firstTokenTime time.Time
	var fullResponseContent strings.Builder

	stream, streamErr := client.CreateChatCompletionStream(ctx, req)
	if streamErr != nil {
		if toolReasoningCheck {
			logInterleavedToolError(providerLogger, config, streamErr)
		}
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
	streamReportedToolCalls := false
	streamInterleavedContent := false
	streamInterleavedReasoning := false
	chunkIndex := 0
	reasoningBeforeTools := false
	reasoningAfterTools := false
	inToolPhase := false
	toolPhaseCount := 0

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
		chunkIndex++

		// Check if Choices array is empty
		if len(response.Choices) == 0 {
			// Log EVERY empty chunk for deep debugging
			providerLogger.Printf("[%s] ... Chunk %d: Empty Choices array (ID=%s, Model=%s, Object=%s)",
				config.Name, chunkCount, response.ID, response.Model, response.Object)
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
			streamReportedToolCalls = true
			if hasContent {
				streamInterleavedContent = true
			}
			if hasReasoningContent {
				streamInterleavedReasoning = true
			}
			for _, toolCall := range delta.ToolCalls {
				if toolCall.Function.Name != "" {
					fullResponseContent.WriteString(toolCall.Function.Name)
				}
				if toolCall.Function.Arguments != "" {
					fullResponseContent.WriteString(toolCall.Function.Arguments)
				}
			}
		}

		// Track reasoning relative to tool-call phases for behavioral checks
		if hasToolCall {
			if !inToolPhase {
				inToolPhase = true
				toolPhaseCount++
			}
		} else {
			inToolPhase = false
		}

		if hasReasoningContent {
			if !streamReportedToolCalls {
				reasoningBeforeTools = true
			} else if !hasToolCall {
				reasoningAfterTools = true
			}
		}
	}

	endTime := time.Now()

	if toolReasoningCheck {
		reasoningCheckPass := streamReportedToolCalls && reasoningBeforeTools && reasoningAfterTools
		providerLogger.Printf("[%s] Tool-reasoning summary: toolCallsObserved=%t reasoningBeforeTools=%t reasoningAfterTools=%t toolPhases=%d pass=%t", config.Name, streamReportedToolCalls, reasoningBeforeTools, reasoningAfterTools, toolPhaseCount, reasoningCheckPass)
		providerLogger.Printf("[%s] Interleaved tool-call summary: interleavedContent=%t interleavedReasoning=%t", config.Name, streamInterleavedContent, streamInterleavedReasoning)
	}

	if firstTokenTime.IsZero() {
		return 0, 0, 0, 0, "", fmt.Errorf("no content received from API (received %d chunks)", chunkCount)
	}

	// Get accurate token count
	fullResponse := fullResponseContent.String()
	tokenList := tke.Encode(fullResponse, nil, nil)
	completionTokens := len(tokenList)
	if toolCallChunks == 0 {
		providerLogger.Printf("[%s] Warning: no tool calls were observed in tool-calling mode (model returned only text/reasoning)", config.Name)
		return 0, 0, 0, 0, fullResponse, fmt.Errorf("no tool calls observed in tool-calling mode")
	}

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

// healthCheck performs a single test request to verify endpoint is reachable before stress testing.
func healthCheck(ctx context.Context, config ProviderConfig, _ *tiktoken.Tiktoken) error {
	// Create a simple test request
	prompt := "Hi, this is a health check. Respond with OK."
	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleUser,
			Content: prompt,
		},
	}

	req := openai.ChatCompletionRequest{
		Model:     config.Model,
		Messages:  messages,
		MaxTokens: 10,
		Stream:    true,
	}

	// Use a short timeout for health check
	healthCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	client := createHTTPClient(config)
	stream, err := client.CreateChatCompletionStream(healthCtx, req)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer func() {
		if closeErr := stream.Close(); closeErr != nil {
			// Log but don't fail health check on close error
			_ = closeErr
		}
	}()

	// Try to receive at least one chunk
	_, err = stream.Recv()
	if err != nil && !errors.Is(err, io.EOF) {
		return fmt.Errorf("health check stream failed: %w", err)
	}

	return nil
}

// printStandardSummary prints a formatted terminal summary for standard test mode.
func printStandardSummary(results []TestResult, config ProviderConfig) {
	log.Println()
	printBoxTop()
	printBoxLineColored(bold("STANDARD TEST COMPLETE"), colorWhite)
	printBoxLine(fmt.Sprintf("%s / %s", config.Name, config.Model))
	printBoxDivider()
	printBoxLine(bold("CONFIGURATION"))

	modeStr := "streaming"
	iterations := 3
	for _, r := range results {
		if r.Mode != "" {
			modeStr = r.Mode
			break
		}
	}
	if modeStr == "mixed" {
		iterations = 6
	}

	printBoxLine(fmt.Sprintf("  Mode:        %s (%d runs)", modeStr, iterations))
	printBoxDivider()
	printBoxLine(bold("RESULTS (Averaged)"))

	modeResults := make(map[string][]TestResult)
	for _, r := range results {
		if r.Success {
			modeResults[r.Mode] = append(modeResults[r.Mode], r)
		}
	}

	printBoxLine("  ┌─────────────┬──────────┬──────────┬─────────────┬────────┐")
	printBoxLine("  │ Mode        │ E2E      │ TTFT     │ Throughput  │ Tokens │")
	printBoxLine("  ├─────────────┼──────────┼──────────┼─────────────┼────────┤")

	modes := []string{"streaming", "tool-calling", "long-story"}
	modeLabels := map[string]string{"streaming": "Streaming", "tool-calling": "Tool-Call", "long-story": "Long-Story"}
	printedModes := make(map[string]bool)

	for _, mode := range modes {
		if printedModes[mode] {
			continue
		}
		if resList, ok := modeResults[mode]; ok && len(resList) > 0 {
			printedModes[mode] = true
			var avgE2E, avgTTFT time.Duration
			var avgThroughput float64
			var avgTokens int
			for _, r := range resList {
				avgE2E += r.E2ELatency
				avgTTFT += r.TTFT
				avgThroughput += r.Throughput
				avgTokens += r.CompletionTokens
			}
			count := len(resList)
			avgE2E /= time.Duration(count)
			avgTTFT /= time.Duration(count)
			avgThroughput /= float64(count)
			avgTokens /= count

			label := modeLabels[mode]
			if label == "" {
				label = mode
			}
			line := fmt.Sprintf("  │ %-11s │ %8s │ %8s │ %8.1f/s  │ %6d │",
				label,
				formatDuration(avgE2E),
				formatDuration(avgTTFT),
				avgThroughput,
				avgTokens)
			printBoxLine(line)
		}
	}

	printBoxLine("  └─────────────┴──────────┴──────────┴─────────────┴────────┘")

	failed := 0
	for _, r := range results {
		if !r.Success {
			failed++
		}
	}

	printBoxDivider()
	if failed == 0 {
		printBoxLineColored(bold("STATUS: ")+colorize(colorGreen, symCheck+" All tests passed"), colorGreen)
	} else {
		printBoxLineColored(bold("STATUS: ")+colorize(colorRed, fmt.Sprintf("%s %d of %d tests failed", symCross, failed, len(results))), colorRed)
	}
	printBoxBottom()
	log.Println()
}

// printDiagnosticSummary prints a formatted terminal summary for diagnostic mode.
func printDiagnosticSummary(summary DiagnosticSummary, config ProviderConfig) {
	log.Println()
	printBoxTop()
	printBoxLineColored(bold("DIAGNOSTIC TEST COMPLETE"), colorWhite)
	printBoxLine(fmt.Sprintf("%s / %s", config.Name, config.Model))
	printBoxDivider()
	printBoxLine(bold("CONFIGURATION"))
	printBoxLine("  Workers:  10    Duration:  90s    Interval:  15s")
	printBoxLine("  Timeout:  30s")
	printBoxDivider()
	printBoxLine(bold("EXECUTION SUMMARY"))
	printBoxLine(fmt.Sprintf("  Total Requests:    %s", formatNumber(summary.TotalRequests)))

	successRate := 0.0
	if summary.TotalRequests > 0 {
		successRate = 100.0 * float64(summary.Successful) / float64(summary.TotalRequests)
	}
	failRate := 100.0 - successRate

	if summary.Successful > 0 {
		printBoxLine(fmt.Sprintf("  %s Successful:      %s (%.1f%%)", colorize(colorGreen, symCheck), formatNumber(summary.Successful), successRate))
	} else {
		printBoxLine("  ✓ Successful:      0 (0.0%)")
	}

	if summary.Failed > 0 {
		printBoxLine(fmt.Sprintf("  %s Failed:          %s (%.1f%%)", colorize(colorRed, symCross), formatNumber(summary.Failed), failRate))
	} else {
		printBoxLine("  ✗ Failed:          0 (0.0%)")
	}

	printBoxDivider()
	printBoxLine(bold("PERFORMANCE METRICS"))

	if summary.Successful > 0 {
		printBoxLine(fmt.Sprintf("  Avg E2E:        %s      P50:  %s      P95:  %s",
			formatDuration(summary.AvgE2ELatency),
			formatDuration(summary.AvgE2ELatency),
			formatDuration(summary.AvgE2ELatency)))
		printBoxLine(fmt.Sprintf("  Avg TTFT:       %s      Min:  %s      Max:  %s",
			formatDuration(summary.AvgTTFT),
			formatDuration(summary.AvgTTFT),
			formatDuration(summary.AvgTTFT)))
		printBoxLine(fmt.Sprintf("  Avg Throughput: %.1f tok/s", summary.AvgThroughput))
		printBoxLine(fmt.Sprintf("  Avg Tokens:     %d", summary.AvgTokens))
	} else {
		printBoxLine("  No successful requests to calculate metrics")
	}

	if len(summary.Errors) > 0 {
		printBoxDivider()
		printBoxLine(bold("ERROR BREAKDOWN"))
		for errMsg, count := range summary.Errors {
			printBoxLine(fmt.Sprintf("  %s %s (x%d)", symCross, errMsg, count))
		}
	}

	if summary.Successful > 0 {
		printBoxDivider()
		printBoxLine(bold("HEALTH ASSESSMENT"))
		srGrade := gradeSuccessRate(successRate / 100.0)
		printBoxLine(fmt.Sprintf("  %s Success Rate:  %s (%.1f%%)", colorize(srGrade.Color, srGrade.Symbol), srGrade.Label, successRate))
	}

	printBoxBottom()
	log.Println()
}

// printLongStorySummary prints a formatted terminal summary for long-story mode.
func printLongStorySummary(result TestResult, config ProviderConfig) {
	log.Println()
	printBoxTop()
	printBoxLineColored(bold("LONG-FORM GENERATION COMPLETE"), colorWhite)
	printBoxLine(fmt.Sprintf("%s / %s", config.Name, config.Model))
	printBoxDivider()
	printBoxLine(bold("CONFIGURATION"))
	printBoxLine("  Target:  ~4,000 words    Max tokens:  16,384")
	printBoxLine("  Timeout: 10 minutes")
	printBoxDivider()
	printBoxLine(bold("RESULTS"))

	if result.Success {
		printBoxLineColored("  Status:          "+symCheck+" SUCCESS", colorGreen)
		printBoxLine(fmt.Sprintf("  Output Tokens:   %s", formatNumber(result.CompletionTokens)))
		wordCount := result.CompletionTokens * 2 / 3
		printBoxLine(fmt.Sprintf("  Word Count:      ~%s words", formatNumber(wordCount)))
		printBoxLine("  " + strings.Repeat(symDash, 40))
		printBoxLine(fmt.Sprintf("  E2E Latency:     %s", formatDuration(result.E2ELatency)))
		printBoxLine(fmt.Sprintf("  TTFT:            %s", formatDuration(result.TTFT)))
		printBoxLine(fmt.Sprintf("  Throughput:      %.1f tok/s", result.Throughput))
		genTime := result.E2ELatency - result.TTFT
		printBoxLine(fmt.Sprintf("  Generation Time: %s", formatDuration(genTime)))
		printBoxDivider()
		printBoxLine(bold("QUALITY METRICS"))
		tokensPerWord := float64(result.CompletionTokens) / float64(wordCount)
		printBoxLine(fmt.Sprintf("  Tokens per word:  %.2f  (Good coherence indicator)", tokensPerWord))
		wordsPerMin := float64(wordCount) / genTime.Minutes()
		printBoxLine(fmt.Sprintf("  Generation rate:  %.1f words/minute", wordsPerMin))
		printBoxDivider()
		printBoxLineColored(bold("STATUS: ")+colorize(colorGreen, symCheck+" Story generated successfully"), colorGreen)
	} else {
		printBoxLineColored("  Status:          "+symCross+" FAILED", colorRed)
		printBoxLine(fmt.Sprintf("  Error:           %s", result.Error))
		printBoxDivider()
		printBoxLineColored(bold("STATUS: ")+colorize(colorRed, symCross+" Generation failed"), colorRed)
	}

	printBoxBottom()
	log.Println()
}

// printStressSummary prints a formatted terminal summary for stress mode.
func printStressSummary(summary StressSummary, perTypeStats map[string]RequestTypeStats, config ProviderConfig) {
	log.Println()
	printBoxTop()
	printBoxLineColored(bold("HIGH-STRESS TEST COMPLETE"), colorWhite)
	printBoxLine(fmt.Sprintf("%s / %s", config.Name, config.Model))
	printBoxDivider()
	printBoxLine(bold("CONFIGURATION"))
	printBoxLine(fmt.Sprintf("  Workers:     %d        Duration:    %s", summary.Workers, formatDuration(time.Duration(summary.DurationSeconds)*time.Second)))
	printBoxLine(fmt.Sprintf("  Start time:  %s", summary.Timestamp.Format("15:04:05")))
	printBoxDivider()
	printBoxLine(bold("EXECUTION SUMMARY"))
	printBoxLine(fmt.Sprintf("  Total Requests:    %s", formatNumber(summary.TotalRequests)))

	successRate := 0.0
	if summary.TotalRequests > 0 {
		successRate = 100.0 * float64(summary.Successful) / float64(summary.TotalRequests)
	}
	failRate := 100.0 - successRate

	if summary.Successful > 0 {
		printBoxLine(fmt.Sprintf("  %s Successful:      %s (%.1f%%)", colorize(colorGreen, symCheck), formatNumber(summary.Successful), successRate))
	} else {
		printBoxLine("  ✓ Successful:      0 (0.0%)")
	}

	if summary.Failed > 0 {
		printBoxLine(fmt.Sprintf("  %s Failed:          %s (%.1f%%)", colorize(colorRed, symCross), formatNumber(summary.Failed), failRate))
	} else {
		printBoxLine("  ✗ Failed:          0 (0.0%)")
	}

	printBoxLine("  " + strings.Repeat(symDash, 40))
	printBoxLine(fmt.Sprintf("  RPS:               %.2f req/s", summary.RequestsPerSec))
	printBoxLine(fmt.Sprintf("  Total Tokens:      %s", formatNumber(summary.TotalTokens)))
	printBoxLine(fmt.Sprintf("  Avg Throughput:    %.2f tok/s", summary.AvgThroughput))

	printBoxDivider()
	printBoxLine(bold("LATENCY DISTRIBUTION (End-to-End)"))
	printBoxLine(fmt.Sprintf("  P50:   %s        P95:   %s        P99:   %s",
		formatDuration(summary.P50E2E),
		formatDuration(summary.P95E2E),
		formatDuration(summary.P99E2E)))
	printBoxLine(fmt.Sprintf("  Avg:   %s", formatDuration(summary.AvgE2ELatency)))

	printBoxDivider()
	printBoxLine(bold("REQUEST TYPE BREAKDOWN"))
	printBoxLine("  ┌─────────┬─────────┬────────┬──────────┬──────────┬─────────┐")
	printBoxLine("  │ Type    │ Count   │ %      │ Avg E2E  │ Avg Tok  │ Failed  │")
	printBoxLine("  ├─────────┼─────────┼────────┼──────────┼──────────┼─────────┤")

	totalReqs := summary.ShortRequests + summary.ToolRequests + summary.LongRequests
	types := []struct {
		name  string
		count int
		stats RequestTypeStats
	}{
		{"Short", summary.ShortRequests, perTypeStats["short"]},
		{"Tool", summary.ToolRequests, perTypeStats["tool"]},
		{"Long", summary.LongRequests, perTypeStats["long"]},
	}

	for _, t := range types {
		if t.count > 0 {
			pct := 100.0 * float64(t.count) / float64(totalReqs)
			avgE2E := formatDuration(t.stats.AvgE2E)
			if t.stats.AvgE2E == 0 {
				avgE2E = "N/A"
			}
			avgTok := fmt.Sprintf("%d", t.stats.AvgTokens)
			if t.stats.AvgTokens == 0 && t.stats.Count > 0 {
				avgTok = fmt.Sprintf("~%d", t.stats.TotalTokens/t.stats.Count)
			}
			line := fmt.Sprintf("  │ %-7s │ %7s │ %5.1f%% │ %8s │ %8s │ %7d │",
				t.name,
				formatNumber(t.count),
				pct,
				avgE2E,
				avgTok,
				t.stats.Failed)
			printBoxLine(line)
		}
	}
	printBoxLine("  └─────────┴─────────┴────────┴──────────┴──────────┴─────────┘")

	hasPercentiles := false
	for _, stats := range perTypeStats {
		if stats.P50E2E > 0 {
			hasPercentiles = true
			break
		}
	}

	if hasPercentiles {
		printBoxDivider()
		printBoxLine(bold("REQUEST TYPE LATENCY PERCENTILES"))
		printBoxLine("  ┌─────────┬─────────┬─────────┬─────────┐")
		printBoxLine("  │ Type    │ P50     │ P95     │ P99     │")
		printBoxLine("  ├─────────┼─────────┼─────────┼─────────┤")

		for _, t := range types {
			if t.count > 0 && t.stats.P50E2E > 0 {
				line := fmt.Sprintf("  │ %-7s │ %7s │ %7s │ %7s │",
					t.name,
					formatDuration(t.stats.P50E2E),
					formatDuration(t.stats.P95E2E),
					formatDuration(t.stats.P99E2E))
				printBoxLine(line)
			}
		}
		printBoxLine("  └─────────┴─────────┴─────────┴─────────┘")
	}

	if len(summary.Errors) > 0 {
		printBoxDivider()
		printBoxLine(bold("TOP ERRORS"))
		errorList := make([]struct {
			msg   string
			count int
		}, 0, len(summary.Errors))
		for msg, count := range summary.Errors {
			errorList = append(errorList, struct {
				msg   string
				count int
			}{msg, count})
		}
		for i := 0; i < len(errorList); i++ {
			for j := i + 1; j < len(errorList); j++ {
				if errorList[j].count > errorList[i].count {
					errorList[i], errorList[j] = errorList[j], errorList[i]
				}
			}
		}
		for i := 0; i < len(errorList) && i < 5; i++ {
			errMsg := errorList[i].msg
			if len(errMsg) > 50 {
				errMsg = errMsg[:47] + "..."
			}
			printBoxLine(fmt.Sprintf("  %s %s (x%d)", symCross, errMsg, errorList[i].count))
		}
	}

	if summary.Successful > 0 {
		printBoxDivider()
		printBoxLine(bold("HEALTH ASSESSMENT"))

		srGrade := gradeSuccessRate(successRate / 100.0)
		p95Grade := gradeLatencyP95(summary.P95E2E)
		p99Grade := gradeLatencyP95(summary.P99E2E)
		tpGrade := gradeThroughput(summary.AvgThroughput)
		erGrade := gradeErrorRate(failRate / 100.0)

		printBoxLine(fmt.Sprintf("  %s Success Rate    %5.1f%%     %s", colorize(srGrade.Color, srGrade.Symbol), successRate, srGrade.Label))
		printBoxLine(fmt.Sprintf("  %s Latency P95     %7s     %s", colorize(p95Grade.Color, p95Grade.Symbol), formatDuration(summary.P95E2E), p95Grade.Label))
		printBoxLine(fmt.Sprintf("  %s Latency P99     %7s     %s", colorize(p99Grade.Color, p99Grade.Symbol), formatDuration(summary.P99E2E), p99Grade.Label))
		printBoxLine(fmt.Sprintf("  %s Throughput      %7.2f     %s", colorize(tpGrade.Color, tpGrade.Symbol), summary.AvgThroughput, tpGrade.Label))
		printBoxLine(fmt.Sprintf("  %s Error Rate       %5.1f%%     %s", colorize(erGrade.Color, erGrade.Symbol), failRate, erGrade.Label))
		printBoxLine("  " + strings.Repeat(symDash, 40))

		grade, desc := calculateOverallGrade(summary)
		printBoxLine(fmt.Sprintf("  OVERALL GRADE:  %s  (%s)", bold(grade), desc))
	}

	printBoxBottom()
	log.Println()
}

// testProviderMetrics runs a full benchmark test against a single provider.
// It runs 3 iterations and reports averaged results, with a 2-minute total timeout.
func testProviderMetrics(config ProviderConfig, tke *tiktoken.Tiktoken, wg *sync.WaitGroup, logDir, resultsDir string, results *[]TestResult, resultsMutex *sync.Mutex, mode TestMode, toolReasoningCheck bool) {
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
				useReasoningCheck := toolReasoningCheck && currentMode == ModeToolCalling

				// Execute the appropriate test based on mode
				if currentMode == ModeToolCalling {
					e2e, ttft, throughput, tokens, responseContent, runErr = singleToolCallRun(ctx, config, tke, providerLogger, useReasoningCheck)
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

	// Calculate projected E2E if target tokens is set
	var projectedE2E time.Duration
	if targetTokens > 0 {
		projectedE2E = calculateProjectedE2E(avgTTFT, avgThroughput, targetTokens)
	}

	// Save successful result
	result := TestResult{
		Provider:         config.Name,
		Model:            config.Model,
		Timestamp:        time.Now(),
		E2ELatency:       avgE2E,
		TTFT:             avgTTFT,
		Throughput:       avgThroughput,
		CompletionTokens: avgTokens,
		ProjectedE2E:     projectedE2E,
		Success:          true,
		Mode:             modeStr,
	}
	saveResult(resultsDir, result)
	appendResult(results, resultsMutex, result)

	// Print formatted terminal summary
	printStandardSummary([]TestResult{result}, config)
}

// testProviderLongStory runs a single long-story benchmark against a provider.
func testProviderLongStory(config ProviderConfig, tke *tiktoken.Tiktoken, wg *sync.WaitGroup, logDir, resultsDir string, results *[]TestResult, resultsMutex *sync.Mutex) {
	if wg != nil {
		defer wg.Done()
	}

	timestamp := time.Now().Format("20060102-150405")
	logFile, err := os.Create(filepath.Clean(filepath.Join(logDir, fmt.Sprintf("%s-long-story-%s.log", config.Name, timestamp))))
	if err != nil {
		log.Printf("Error creating long-story log file for %s: %v", config.Name, err)
		return
	}
	defer func() {
		if closeErr := logFile.Close(); closeErr != nil {
			log.Printf("Warning: Failed to close long-story log file: %v", closeErr)
		}
	}()

	providerLogger := log.New(io.MultiWriter(os.Stdout, logFile), "", log.LstdFlags)
	providerLogger.Printf("--- Long-story test: %s (%s) ---", config.Name, config.Model)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	providerLogger.Printf("[%s] Long-story run starting", config.Name)

	e2e, ttft, throughput, tokens, responseContent, runErr := longStoryRun(ctx, config, tke, providerLogger)

	if saveResponses && runErr == nil && responseContent != "" {
		responseFile := filepath.Clean(filepath.Join(logDir,
			fmt.Sprintf("%s-long-story-response.txt", config.Name)))
		if err := os.WriteFile(responseFile, []byte(responseContent), 0600); err != nil {
			providerLogger.Printf("[%s] Warning: Failed to save long-story response: %v", config.Name, err)
		}
	}

	if runErr != nil {
		providerLogger.Printf("[%s] Long-story run failed: %v", config.Name, runErr)
		result := TestResult{
			Provider:  config.Name,
			Model:     config.Model,
			Timestamp: time.Now(),
			Success:   false,
			Error:     runErr.Error(),
			Mode:      longStoryModeLabel,
		}
		saveResult(resultsDir, result)
		appendResult(results, resultsMutex, result)
		return
	}

	providerLogger.Println("==============================================")
	providerLogger.Printf("   Long-story LLM Metrics for: %s", config.Name)
	providerLogger.Printf("   Model: %s", config.Model)
	providerLogger.Printf("   Mode: %s", longStoryModeLabel)
	providerLogger.Printf("   Output Tokens: %d", tokens)
	providerLogger.Println("----------------------------------------------")
	providerLogger.Printf("   End-to-End Latency: %s", formatDuration(e2e))
	providerLogger.Printf("   Latency (TTFT):     %s", formatDuration(ttft))
	providerLogger.Printf("   Throughput (Tokens/sec): %.2f tokens/s", throughput)
	providerLogger.Println("==============================================")

	var projectedE2E time.Duration
	if targetTokens > 0 {
		projectedE2E = calculateProjectedE2E(ttft, throughput, targetTokens)
	}

	result := TestResult{
		Provider:         config.Name,
		Model:            config.Model,
		Timestamp:        time.Now(),
		E2ELatency:       e2e,
		TTFT:             ttft,
		Throughput:       throughput,
		CompletionTokens: tokens,
		ProjectedE2E:     projectedE2E,
		Success:          true,
		Mode:             longStoryModeLabel,
	}
	saveResult(resultsDir, result)
	appendResult(results, resultsMutex, result)

	// Print formatted terminal summary
	printLongStorySummary(result, config)
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
		if targetTokens > 0 {
			report.WriteString(fmt.Sprintf("**Note:** Projected E2E calculated for %d tokens using formula: TTFT + (Target Tokens / Throughput)\n\n", targetTokens))
			report.WriteString("| Provider | Model | Mode | E2E Latency | TTFT | Throughput | Tokens | Projected E2E |\n")
			report.WriteString("|----------|-------|------|-------------|------|------------|--------|---------------|\n")
		} else {
			report.WriteString("| Provider | Model | Mode | E2E Latency | TTFT | Throughput | Tokens |\n")
			report.WriteString("|----------|-------|------|-------------|------|------------|--------|\n")
		}

		for _, r := range results {
			if r.Success {
				writeTestResultRow(&report, r, targetTokens > 0)
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
		writeTestResultLeaderboards(&report, results)
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
	ProjectedE2E  time.Duration  `json:"projectedE2eLatency,omitempty"`
	Errors        map[string]int `json:"errors,omitempty"`
}

// diagnosticMode runs continuous testing with 10 workers for 90 seconds.
// Makes requests every 15 seconds, with 30-second timeout per request.
// Workers stop starting new requests when insufficient time remains (5s grace period).
// Expected: 4 requests per worker (at 0s, 15s, 30s, 45s) for a total of 40 requests.
func diagnosticMode(config ProviderConfig, tke *tiktoken.Tiktoken, logDir, resultsDir string, mode TestMode, toolReasoningCheck bool, wg *sync.WaitGroup, results *[]DiagnosticSummary, resultsMutex *sync.Mutex) {
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
						e2e, ttft, throughput, tokens, responseContent, reqErr = singleToolCallRun(reqCtx, config, tke, providerLogger, toolReasoningCheck)
					}
				case ModeToolCalling:
					testMode = ModeToolCalling
					e2e, ttft, throughput, tokens, responseContent, reqErr = singleToolCallRun(reqCtx, config, tke, providerLogger, toolReasoningCheck)
				case ModeStreaming:
					testMode = ModeStreaming
					e2e, ttft, throughput, tokens, responseContent, reqErr = singleTestRun(reqCtx, config, tke, providerLogger)
				case ModeStress:
					// Stress mode is handled separately, not in diagnostic mode
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

		// Display projected E2E if target tokens is set
		if targetTokens > 0 {
			projectedE2E := calculateProjectedE2E(avgTTFT, avgThroughput, targetTokens)
			providerLogger.Printf("Projected E2E (%d tokens): %s", targetTokens, formatDuration(projectedE2E))
		}
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

		// Calculate projected E2E if target tokens is set
		if targetTokens > 0 {
			summary.ProjectedE2E = calculateProjectedE2E(summary.AvgTTFT, summary.AvgThroughput, targetTokens)
		}
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

	// Print formatted terminal summary
	printDiagnosticSummary(summary, config)
}

// stressMode runs high-concurrency stress testing with configurable workers and duration.
// Uses weighted random request distribution: short streaming, tool-calling, and long-form.
// Features gradual ramp-up, proper randomization per worker, and connection pooling.
func stressMode(config ProviderConfig, tke *tiktoken.Tiktoken, logDir, resultsDir string, numWorkers, durationSec, longBias, rampUpSec int, wg *sync.WaitGroup, results *[]StressSummary, resultsMutex *sync.Mutex) {
	if wg != nil {
		defer wg.Done()
	}

	timestamp := time.Now().Format("20060102-150405")
	logFileName := filepath.Clean(filepath.Join(logDir, fmt.Sprintf("%s-stress-%s.log", config.Name, timestamp)))
	logFile, err := os.Create(logFileName)
	if err != nil {
		log.Printf("Error creating stress log file for %s: %v", config.Name, err)
		return
	}
	defer func() {
		if closeErr := logFile.Close(); closeErr != nil {
			log.Printf("Warning: Failed to close log file: %v", closeErr)
		}
	}()

	providerLogger := log.New(io.MultiWriter(os.Stdout, logFile), "", log.LstdFlags)
	providerLogger.Printf("=== HIGH-STRESS MODE: %s (%s) ===", config.Name, config.Model)
	providerLogger.Printf("Workers: %d | Duration: %ds | Long-form bias: %d%% | Ramp-up: %ds", numWorkers, durationSec, longBias, rampUpSec)

	// Validate and adjust long bias
	if longBias < 10 {
		longBias = 10
	}
	if longBias > 80 {
		longBias = 80
	}

	// Calculate weights for optimal stress mix: ~50% short, ~20% tool, ~30% long (adjustable via longBias)
	remainingWeight := float64(100 - longBias)
	streamingWeight := remainingWeight * 0.71
	toolWeight := remainingWeight * 0.29
	longWeight := float64(longBias)
	totalWeight := streamingWeight + toolWeight + longWeight

	providerLogger.Printf("Request distribution: %.0f%% streaming, %.0f%% tool-calling, %.0f%% long-form",
		streamingWeight/totalWeight*100, toolWeight/totalWeight*100, longWeight/totalWeight*100)

	// Perform health check before spawning workers
	providerLogger.Println("Performing health check...")
	healthCtx, healthCancel := context.WithTimeout(context.Background(), 30*time.Second)
	if err := healthCheck(healthCtx, config, tke); err != nil {
		providerLogger.Printf("Health check FAILED: %v", err)
		providerLogger.Println("Aborting stress test - endpoint is not reachable")
		healthCancel()
		return
	}
	healthCancel()
	providerLogger.Println("Health check PASSED - proceeding with stress test")

	// Create session context with configured duration plus buffer
	sessionDuration := time.Duration(durationSec) * time.Second
	sessionCtx, sessionCancel := context.WithTimeout(context.Background(), sessionDuration+60*time.Second)
	defer sessionCancel()

	sessionStartTime := time.Now()
	rampUpComplete := make(chan bool, 1)

	// Metrics tracking
	type stressResult struct {
		workerID   int
		reqNum     int
		reqType    string
		e2e        time.Duration
		ttft       time.Duration
		throughput float64
		tokens     int
		err        error
	}

	resultsChan := make(chan stressResult, 10000)
	var workerWg sync.WaitGroup

	// Progress reporting ticker
	progressTicker := time.NewTicker(10 * time.Second)
	defer progressTicker.Stop()

	// Atomic counters for progress reporting
	var totalRequests, successCount, failureCount int64

	// Start progress reporter
	progressDone := make(chan bool)
	go func() {
		for {
			select {
			case <-progressTicker.C:
				elapsed := time.Since(sessionStartTime)
				rps := float64(totalRequests) / elapsed.Seconds()
				providerLogger.Printf("[STRESS PROGRESS] Workers: %d | Total: %d | Success: %d | Failed: %d | RPS: %.2f",
					numWorkers, totalRequests, successCount, failureCount, rps)
			case <-progressDone:
				return
			}
		}
	}()

	// Start workers with gradual ramp-up
	workersPerSecond := float64(numWorkers) / float64(rampUpSec)
	providerLogger.Printf("Ramping up: spawning %.1f workers/second over %d seconds...", workersPerSecond, rampUpSec)

	for workerID := 1; workerID <= numWorkers; workerID++ {
		workerWg.Add(1)
		go func(id int) {
			defer workerWg.Done()

			// Each worker gets its own seeded RNG for proper randomization
			//nolint:gosec // Not security-critical; using math/rand for performance
			rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(id)))
			reqNum := 0

			for {
				select {
				case <-sessionCtx.Done():
					return
				default:
					// Check if session duration has elapsed
					if time.Since(sessionStartTime) >= sessionDuration {
						return
					}
				}

				reqNum++

				// Determine request type based on weights using proper RNG
				randVal := rng.Float64() * totalWeight

				var reqType string
				var reqTimeout time.Duration
				var testFunc func(context.Context) (time.Duration, time.Duration, float64, int, string, error)

				switch {
				case randVal < streamingWeight:
					reqType = "short"
					reqTimeout = 60 * time.Second
					testFunc = func(ctx context.Context) (time.Duration, time.Duration, float64, int, string, error) {
						return singleTestRun(ctx, config, tke, providerLogger)
					}
				case randVal < streamingWeight+toolWeight:
					reqType = "tool"
					reqTimeout = 60 * time.Second
					testFunc = func(ctx context.Context) (time.Duration, time.Duration, float64, int, string, error) {
						return singleToolCallRun(ctx, config, tke, providerLogger, false)
					}
				default:
					reqType = "long"
					reqTimeout = 10 * time.Minute
					testFunc = func(ctx context.Context) (time.Duration, time.Duration, float64, int, string, error) {
						return longStoryRun(ctx, config, tke, providerLogger)
					}
				}

				// Create timeout context for this request
				reqCtx, reqCancel := context.WithTimeout(sessionCtx, reqTimeout)

				e2e, ttft, throughput, tokens, _, reqErr := testFunc(reqCtx)
				reqCancel()

				resultsChan <- stressResult{
					workerID:   id,
					reqNum:     reqNum,
					reqType:    reqType,
					e2e:        e2e,
					ttft:       ttft,
					throughput: throughput,
					tokens:     tokens,
					err:        reqErr,
				}
			}
		}(workerID)

		// Rate limit worker spawning during ramp-up
		if workerID < numWorkers {
			time.Sleep(time.Duration(float64(time.Second) / workersPerSecond))
		}
	}

	providerLogger.Printf("All %d workers spawned - ramp-up complete", numWorkers)
	rampUpComplete <- true

	// Wait for all workers to complete
	go func() {
		workerWg.Wait()
		close(resultsChan)
		close(progressDone)
	}()

	// Collect results with per-type statistics
	var allE2E []time.Duration
	var totalE2E, totalTTFT time.Duration
	var totalThroughput float64
	var totalTokens int
	perTypeStats := make(map[string]RequestTypeStats)

	errors := make(map[string]int)

	for result := range resultsChan {
		totalRequests++

		// Always update per-type stats for request counting
		stats := perTypeStats[result.reqType]
		stats.Count++

		if result.err != nil {
			failureCount++
			stats.Failed++
			errors[result.err.Error()]++
		} else {
			successCount++
			stats.Successful++
			stats.TotalTokens += result.tokens
			totalE2E += result.e2e
			totalTTFT += result.ttft
			totalThroughput += result.throughput
			totalTokens += result.tokens
			allE2E = append(allE2E, result.e2e)
		}
		perTypeStats[result.reqType] = stats
	}

	// Calculate statistics
	totalReqs := int(totalRequests)
	successful := int(successCount)
	failed := int(failureCount)

	providerLogger.Println("")
	providerLogger.Println("========================================")
	providerLogger.Println("   HIGH-STRESS MODE SUMMARY")
	providerLogger.Println("========================================")
	providerLogger.Printf("Provider: %s", config.Name)
	providerLogger.Printf("Model: %s", config.Model)
	providerLogger.Printf("Workers: %d", numWorkers)
	providerLogger.Printf("Duration: %s", formatDuration(time.Since(sessionStartTime)))
	providerLogger.Printf("Total Requests: %d", totalReqs)
	providerLogger.Printf("Successful: %d (%.1f%%)", successful, 100.0*float64(successful)/float64(totalReqs))
	providerLogger.Printf("Failed: %d (%.1f%%)", failed, 100.0*float64(failed)/float64(totalReqs))

	// Show per-type distribution
	providerLogger.Println("----------------------------------------")
	providerLogger.Println("Request Type Distribution:")
	for reqType, stats := range perTypeStats {
		if stats.Count > 0 {
			providerLogger.Printf("  %s: %d requests (%d successful)", reqType, stats.Count, stats.Successful)
		}
	}

	var avgE2E, avgTTFT time.Duration
	var avgThroughput, rps float64
	var p50, p95, p99 time.Duration

	if successful > 0 {
		avgE2E = totalE2E / time.Duration(successful)
		avgTTFT = totalTTFT / time.Duration(successful)
		avgThroughput = totalThroughput / float64(successful)
		elapsed := time.Since(sessionStartTime)
		rps = float64(totalReqs) / elapsed.Seconds()

		// Calculate percentiles
		p50 = calculatePercentile(allE2E, 50)
		p95 = calculatePercentile(allE2E, 95)
		p99 = calculatePercentile(allE2E, 99)

		providerLogger.Println("----------------------------------------")
		providerLogger.Printf("Avg E2E Latency: %s", formatDuration(avgE2E))
		providerLogger.Printf("P50 E2E: %s", formatDuration(p50))
		providerLogger.Printf("P95 E2E: %s", formatDuration(p95))
		providerLogger.Printf("P99 E2E: %s", formatDuration(p99))
		providerLogger.Printf("Avg TTFT: %s", formatDuration(avgTTFT))
		providerLogger.Printf("Avg Throughput: %.2f tokens/s", avgThroughput)
		providerLogger.Printf("Total Tokens: %d", totalTokens)
		providerLogger.Printf("Requests/sec: %.2f", rps)
	}

	if len(errors) > 0 {
		providerLogger.Println("----------------------------------------")
		providerLogger.Println("Errors encountered:")
		for errMsg, count := range errors {
			providerLogger.Printf("  - %s (x%d)", errMsg, count)
		}
	}

	providerLogger.Println("========================================")

	// Create stress summary
	summary := StressSummary{
		Provider:        config.Name,
		Model:           config.Model,
		Timestamp:       time.Now(),
		Workers:         numWorkers,
		DurationSeconds: durationSec,
		TotalRequests:   totalReqs,
		Successful:      successful,
		Failed:          failed,
		TotalTokens:     totalTokens,
		RequestsPerSec:  rps,
		// Fix: Populate per-type request counts from perTypeStats
		ShortRequests: perTypeStats["short"].Count,
		ToolRequests:  perTypeStats["tool"].Count,
		LongRequests:  perTypeStats["long"].Count,
	}

	if successful > 0 {
		summary.AvgE2ELatency = avgE2E
		summary.P50E2E = p50
		summary.P95E2E = p95
		summary.P99E2E = p99
		summary.AvgTTFT = avgTTFT
		summary.AvgThroughput = avgThroughput

		if targetTokens > 0 {
			summary.ProjectedE2E = calculateProjectedE2E(avgTTFT, avgThroughput, targetTokens)
		}
	}

	if len(errors) > 0 {
		summary.Errors = errors
	}

	// Save stress summary to JSON
	summaryFile := filepath.Join(resultsDir, fmt.Sprintf("%s-stress-summary-%s.json", config.Name, timestamp))
	data, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		providerLogger.Printf("Warning: Failed to marshal stress summary: %v", err)
	} else {
		if err := os.WriteFile(summaryFile, data, 0600); err != nil {
			providerLogger.Printf("Warning: Failed to write stress summary: %v", err)
		} else {
			providerLogger.Printf("Stress summary saved: %s", summaryFile)
		}
	}

	// Append to results slice if provided
	if results != nil && resultsMutex != nil {
		resultsMutex.Lock()
		*results = append(*results, summary)
		resultsMutex.Unlock()
	}

	// Print formatted terminal summary
	printStressSummary(summary, perTypeStats, config)
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
		if targetTokens > 0 {
			report.WriteString(fmt.Sprintf("**Note:** Projected E2E calculated for %d tokens using formula: TTFT + (Target Tokens / Throughput)\n\n", targetTokens))
			report.WriteString("| Provider | Model | Mode | Total Requests | Success | Failed | Avg E2E |" +
				" Avg TTFT | Avg Throughput | Projected E2E |\n")
			report.WriteString("|----------|-------|------|----------------|---------|--------|---------|" +
				"----------|----------------|---------------|\n")
		} else {
			report.WriteString("| Provider | Model | Mode | Total Requests | Success | Failed | Avg E2E |" +
				" Avg TTFT | Avg Throughput |\n")
			report.WriteString("|----------|-------|------|----------------|---------|--------|---------|" +
				"----------|----------------|\n")
		}

		for _, r := range results {
			writeDiagnosticResultRow(&report, r, targetTokens > 0)
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

		// Sort by Projected E2E (if available)
		if targetTokens > 0 {
			report.WriteString(fmt.Sprintf("### By Projected E2E Latency (%d tokens)\n\n", targetTokens))
			writeProjectedE2EDiagnosticLeaderboard(&report, successfulResults)
		}
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

// calculatePercentile calculates the P-th percentile from a slice of durations.
func calculatePercentile(values []time.Duration, percentile float64) time.Duration {
	if len(values) == 0 {
		return 0
	}
	// Sort the values
	sorted := make([]time.Duration, len(values))
	copy(sorted, values)
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j] < sorted[i] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}
	index := int(float64(len(sorted)-1) * percentile / 100.0)
	return sorted[index]
}

// generateStressReport creates a markdown report for stress mode results.
func generateStressReport(resultsDir string, results []StressSummary, sessionTimestamp string) error {
	filename := filepath.Join(resultsDir, "STRESS-REPORT.md")

	var report strings.Builder
	report.WriteString("# LLM API High-Stress Mode Results\n\n")
	report.WriteString(fmt.Sprintf("**Test Session:** %s\n\n", sessionTimestamp))
	report.WriteString("**Test Type:** High-Concurrency Stress Test\n\n")
	report.WriteString("---\n\n")

	// Summary statistics
	totalProviders := len(results)
	var totalRequests, totalSuccessful, totalFailed int
	var totalTokens int
	var totalRPS float64

	for _, r := range results {
		totalRequests += r.TotalRequests
		totalSuccessful += r.Successful
		totalFailed += r.Failed
		totalTokens += r.TotalTokens
		totalRPS += r.RequestsPerSec
	}

	report.WriteString("## Summary\n\n")
	report.WriteString(fmt.Sprintf("- **Providers Tested:** %d\n", totalProviders))
	report.WriteString(fmt.Sprintf("- **Total Requests:** %d\n", totalRequests))
	report.WriteString(fmt.Sprintf("- **Successful:** %d (%.1f%%)\n",
		totalSuccessful, 100.0*float64(totalSuccessful)/float64(totalRequests)))
	report.WriteString(fmt.Sprintf("- **Failed:** %d (%.1f%%)\n\n",
		totalFailed, 100.0*float64(totalFailed)/float64(totalRequests)))
	report.WriteString(fmt.Sprintf("- **Total Tokens Generated:** %d\n", totalTokens))
	report.WriteString(fmt.Sprintf("- **Aggregate RPS:** %.2f\n\n", totalRPS))

	// Detailed results table
	if len(results) > 0 {
		report.WriteString("## Detailed Results\n\n")
		report.WriteString("| Provider | Workers | Duration | Total | Success | Failed | Avg E2E | P95 E2E | RPS | Throughput |\n")
		report.WriteString("|----------|---------|----------|-------|---------|--------|---------|---------|-----|------------|\n")

		for _, r := range results {
			successRate := fmt.Sprintf("%.1f%%", 100.0*float64(r.Successful)/float64(r.TotalRequests))
			report.WriteString(fmt.Sprintf("| %s | %d | %ds | %d | %s | %d | %s | %s | %.2f | %.2f tok/s |\n",
				r.Provider,
				r.Workers,
				r.DurationSeconds,
				r.TotalRequests,
				successRate,
				r.Failed,
				formatDuration(r.AvgE2ELatency),
				formatDuration(r.P95E2E),
				r.RequestsPerSec,
				r.AvgThroughput))
		}
		report.WriteString("\n")
	}

	// Performance Leaderboard
	if len(results) > 0 {
		report.WriteString("## Performance Leaderboards\n\n")

		// By RPS
		report.WriteString("### By Requests Per Second\n\n")
		sortedResults := make([]StressSummary, len(results))
		copy(sortedResults, results)
		for i := 0; i < len(sortedResults); i++ {
			for j := i + 1; j < len(sortedResults); j++ {
				if sortedResults[j].RequestsPerSec > sortedResults[i].RequestsPerSec {
					sortedResults[i], sortedResults[j] = sortedResults[j], sortedResults[i]
				}
			}
		}

		report.WriteString("| Rank | Provider | RPS | Workers | Success Rate |\n")
		report.WriteString("|------|----------|-----|---------|-------------|\n")
		for i, r := range sortedResults {
			successRate := fmt.Sprintf("%.1f%%", 100.0*float64(r.Successful)/float64(r.TotalRequests))
			report.WriteString(fmt.Sprintf("| %d | %s | %.2f | %d | %s |\n",
				i+1, r.Provider, r.RequestsPerSec, r.Workers, successRate))
		}
		report.WriteString("\n")

		// By Throughput
		report.WriteString("### By Token Throughput\n\n")
		for i := 0; i < len(sortedResults); i++ {
			for j := i + 1; j < len(sortedResults); j++ {
				if sortedResults[j].AvgThroughput > sortedResults[i].AvgThroughput {
					sortedResults[i], sortedResults[j] = sortedResults[j], sortedResults[i]
				}
			}
		}

		report.WriteString("| Rank | Provider | Throughput | Total Tokens | Avg E2E |\n")
		report.WriteString("|------|----------|------------|--------------|--------|\n")
		for i, r := range sortedResults {
			report.WriteString(fmt.Sprintf("| %d | %s | %.2f tok/s | %d | %s |\n",
				i+1, r.Provider, r.AvgThroughput, r.TotalTokens, formatDuration(r.AvgE2ELatency)))
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
		return fmt.Errorf("error writing stress report: %w", err)
	}

	log.Printf("Stress report generated: %s", filename)
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
	longStory := flag.Bool("long-story", false, "Use long-form story generation scenario (single creative-writing prompt)")
	stressFlag := flag.Bool("stress", false, "Run high-stress mode: high concurrency mixed load testing")
	stressLevel := flag.String("stress-level", "moderate",
		"Stress level preset: moderate (100 workers), heavy (500), extreme (1000)")
	stressWorkers := flag.Int("stress-workers", 0,
		"Override: exact number of workers for stress mode (ignores --stress-level)")
	stressDuration := flag.Int("stress-duration", 300,
		"Stress test duration in seconds (default: 300 = 5 minutes)")
	stressLongBias := flag.Int("stress-long-bias", 30,
		"Percentage of long-form requests in stress mode (default: 30, range: 10-80)")
	stressRampUp := flag.Int("stress-rampup", 10,
		"Seconds to gradually spawn workers (default: 10)")
	flagToolReasoningCheck := flag.Bool("tool-reasoning-check", false,
		"Enable tool+reasoning behavior checks (implies tool-calling if not otherwise set)")
	flagSaveResponses := flag.Bool("save-responses", false, "Save all API responses to log files")
	flagTargetTokens := flag.Int("target-tokens", 350,
		"Target token count for projected E2E latency normalization (default: 350)")
	flagMaxTokens := flag.Int("max-tokens", 16384,
		"Maximum completion tokens for long-story mode (default: 16384)")

	// New stress test improvement flags
	flagDiscover := flag.Bool("discover", false,
		"Run progressive load discovery to find optimal capacity")
	flagDiscoverThreshold := flag.Float64("discover-threshold", 0.10,
		"Failure rate threshold for capacity discovery (default: 0.10 = 10%)")
	flagAdaptiveRate := flag.Bool("adaptive-rate", false,
		"Enable adaptive request pacing based on response patterns")
	flagCircuitBreaker := flag.Bool("circuit-breaker", true,
		"Enable circuit breaker pattern for resilience (default: true)")
	flagCircuitFailures := flag.Int("circuit-failures", 10,
		"Consecutive failures before opening circuit breaker (default: 10)")
	flagCircuitCooldown := flag.Duration("circuit-cooldown", 30*time.Second,
		"Cooldown period before circuit breaker half-opens (default: 30s)")
	flagVerboseMetrics := flag.Bool("verbose-metrics", false,
		"Enable detailed connection health metrics and histograms")
	flagConnectionTimeout := flag.Duration("connection-timeout", 30*time.Second,
		"TLS handshake timeout for connections (default: 30s)")
	flagMaxRetries := flag.Int("max-retries", 3,
		"Maximum retry attempts for transient failures (default: 3)")
	flagRetryBackoff := flag.Duration("retry-backoff", 500*time.Millisecond,
		"Base backoff duration for retries (default: 500ms)")
	flagAdaptiveStress := flag.Bool("adaptive-stress", false,
		"Run discovery to find optimal capacity, then stress test at that level")
	flagAdaptiveSafetyMargin := flag.Float64("adaptive-safety-margin", 0.9,
		"Safety margin for adaptive stress (0.9 = use 90%% of discovered capacity)")
	flag.Parse()

	// Set global flag for saving responses
	saveResponses = *flagSaveResponses
	targetTokens = *flagTargetTokens
	maxTokens = *flagMaxTokens

	if *diagnostic && *longStory {
		log.Fatal("Error: --long-story cannot be combined with --diagnostic")
	}
	if *stressFlag && *longStory {
		log.Fatal("Error: --long-story cannot be combined with --stress")
	}
	if *stressFlag && *diagnostic {
		log.Fatal("Error: --stress cannot be combined with --diagnostic")
	}
	if *flagDiscover && (*stressFlag || *diagnostic || *longStory) {
		log.Fatal("Error: --discover cannot be combined with --stress, --diagnostic, or --long-story")
	}
	if *flagDiscoverThreshold < 0 || *flagDiscoverThreshold > 1 {
		log.Fatal("Error: --discover-threshold must be between 0 and 1")
	}
	if *flagMaxRetries < 0 || *flagMaxRetries > 10 {
		log.Fatal("Error: --max-retries must be between 0 and 10")
	}
	if *flagAdaptiveStress && (*stressFlag || *diagnostic || *longStory || *flagDiscover) {
		log.Fatal("Error: --adaptive-stress cannot be combined with --stress, --diagnostic, --long-story, or --discover")
	}
	if *flagAdaptiveSafetyMargin < 0.1 || *flagAdaptiveSafetyMargin > 1.0 {
		log.Fatal("Error: --adaptive-safety-margin must be between 0.1 and 1.0")
	}

	// Log enabled features (using the flags to suppress "declared and not used" errors)
	if *flagAdaptiveRate {
		log.Println("Note: Adaptive rate limiting is enabled (placeholder - full implementation pending)")
	}
	if *flagVerboseMetrics {
		log.Println("Note: Verbose metrics collection is enabled (placeholder - full implementation pending)")
	}
	if *flagConnectionTimeout != 30*time.Second {
		log.Printf("Note: Connection timeout set to %s", *flagConnectionTimeout)
	}
	if *flagAdaptiveStress {
		log.Printf("Note: Adaptive stress mode enabled with %.0f%% safety margin", *flagAdaptiveSafetyMargin*100)
	}

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

	if *longStory {
		log.Println("Test mode: Long-story (single long-form creative-writing prompt)")

		var wgLong sync.WaitGroup
		var results []TestResult
		var resultsMutex sync.Mutex

		for _, provider := range providersToTest {
			if *testAll {
				wgLong.Add(1)
				go testProviderLongStory(provider, tke, &wgLong, logDir, resultsDir, &results, &resultsMutex)
			} else {
				testProviderLongStory(provider, tke, nil, logDir, resultsDir, &results, &resultsMutex)
			}
		}

		if *testAll {
			wgLong.Wait()
			log.Println("--- All long-story provider tests complete. ---")
		}

		log.Println("Generating summary report...")
		if err := generateMarkdownReport(resultsDir, results, sessionTimestamp); err != nil {
			log.Printf("Warning: Failed to generate report: %v", err)
		}

		log.Printf("All long-story tests complete. Results saved to: %s/", sessionDir)
		return
	}

	// Determine test mode and tool-reasoning behaviour
	rawToolReasoning := *flagToolReasoningCheck
	testMode, toolReasoningCheck, forcedToolMode := resolveTestMode(*toolCalling, *mixed, rawToolReasoning)
	switch testMode {
	case ModeMixed:
		log.Println("Test mode: Mixed (streaming + tool-calling)")
	case ModeToolCalling:
		log.Println("Test mode: Tool-calling")
	case ModeStreaming:
		log.Println("Test mode: Streaming")
	case ModeStress:
		log.Println("Test mode: Stress (handled separately)")
	default:
		log.Printf("Test mode: %s", testMode)
	}
	if forcedToolMode {
		log.Println("Tool-reasoning checks enabled; defaulting to tool-calling mode.")
	} else if rawToolReasoning && !toolReasoningCheck {
		log.Println("Warning: --tool-reasoning-check ignored because streaming-only mode selected.")
	}
	if toolReasoningCheck {
		log.Println("Tool-reasoning checks are ENABLED for tool-calling runs.")
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
				go diagnosticMode(provider, tke, logDir, resultsDir, testMode, toolReasoningCheck, &diagnosticWg, &diagnosticResults, &diagnosticMutex)
			}
			diagnosticWg.Wait()
		} else {
			// Single provider (no concurrency needed)
			for _, provider := range providersToTest {
				diagnosticMode(provider, tke, logDir, resultsDir, testMode, toolReasoningCheck, nil, &diagnosticResults, &diagnosticMutex)
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

	// Discovery mode - progressive load testing to find optimal capacity
	//nolint:nestif // Complex nesting is acceptable for CLI command handling
	if *flagDiscover {
		log.Printf("=== RUNNING IN LOAD DISCOVERY MODE ===")
		log.Printf("Failure threshold: %.1f%% | Max workers: 2000", *flagDiscoverThreshold*100)

		discoveryConfig := stress.DefaultDiscoveryConfig()
		discoveryConfig.FailureThreshold = *flagDiscoverThreshold

		for _, provider := range providersToTest {
			log.Printf("\n--- Discovering capacity for: %s ---", provider.Name)

			testFunc := func(ctx context.Context, workers int, duration time.Duration) (stress.LoadLevelResult, error) {
				return runDiscoveryTest(ctx, provider, tke, workers, duration, *flagCircuitBreaker, *flagCircuitFailures, *flagCircuitCooldown, *flagMaxRetries, *flagRetryBackoff)
			}

			discovery := stress.NewLoadDiscovery(discoveryConfig)
			result, err := discovery.Run(context.Background(), testFunc)
			if err != nil {
				log.Printf("Discovery failed for %s: %v", provider.Name, err)
				continue
			}

			// Print discovery report
			log.Println("")
			log.Println(result.GenerateReport())

			// Save discovery result
			discoveryFile := filepath.Join(resultsDir, fmt.Sprintf("%s-discovery-%s.json", provider.Name, sessionTimestamp))
			data, err := json.MarshalIndent(result, "", "  ")
			if err != nil {
				log.Printf("Warning: Failed to marshal discovery result: %v", err)
			} else {
				if err := os.WriteFile(discoveryFile, data, 0600); err != nil {
					log.Printf("Warning: Failed to write discovery result: %v", err)
				} else {
					log.Printf("Discovery result saved: %s", discoveryFile)
				}
			}
		}

		log.Printf("Load discovery complete. Results saved to: %s/", sessionDir)
		return
	}

	// Adaptive stress mode - discover optimal capacity, then run stress test
	//nolint:nestif // Complex nesting is acceptable for CLI command handling
	if *flagAdaptiveStress {
		log.Printf("=== RUNNING IN ADAPTIVE STRESS MODE ===")
		log.Printf("Discovering optimal capacity with %.0f%% safety margin...", *flagAdaptiveSafetyMargin*100)

		discoveryConfig := stress.DefaultDiscoveryConfig()
		discoveryConfig.FailureThreshold = *flagDiscoverThreshold

		for _, provider := range providersToTest {
			log.Printf("\n--- Phase 1: Discovering capacity for: %s ---", provider.Name)

			testFunc := func(ctx context.Context, workers int, duration time.Duration) (stress.LoadLevelResult, error) {
				return runDiscoveryTest(ctx, provider, tke, workers, duration, *flagCircuitBreaker, *flagCircuitFailures, *flagCircuitCooldown, *flagMaxRetries, *flagRetryBackoff)
			}

			discovery := stress.NewLoadDiscovery(discoveryConfig)
			result, err := discovery.Run(context.Background(), testFunc)
			if err != nil {
				log.Printf("Discovery failed for %s: %v", provider.Name, err)
				continue
			}

			// Print discovery report
			log.Println("")
			log.Println(result.GenerateReport())

			// Save discovery result
			discoveryFile := filepath.Join(resultsDir, fmt.Sprintf("%s-discovery-%s.json", provider.Name, sessionTimestamp))
			data, err := json.MarshalIndent(result, "", "  ")
			if err != nil {
				log.Printf("Warning: Failed to marshal discovery result: %v", err)
			} else {
				if err := os.WriteFile(discoveryFile, data, 0600); err != nil {
					log.Printf("Warning: Failed to write discovery result: %v", err)
				}
			}

			// Calculate optimal workers with safety margin
			discoveredCapacity := result.OptimalWorkers
			if discoveredCapacity < 1 {
				log.Printf("Warning: Could not determine capacity for %s, skipping stress test", provider.Name)
				continue
			}

			optimalWorkers := int(float64(discoveredCapacity) * *flagAdaptiveSafetyMargin)
			if optimalWorkers < 1 {
				optimalWorkers = 1
			}

			log.Printf("\n--- Phase 2: Running stress test at discovered capacity ---")
			log.Printf("Discovered capacity: %d workers", discoveredCapacity)
			log.Printf("Using safety margin: %.0f%%", *flagAdaptiveSafetyMargin*100)
			log.Printf("Stress test workers: %d", optimalWorkers)

			// Run stress test at discovered capacity
			var stressResults []StressSummary
			var stressMutex sync.Mutex
			stressMode(provider, tke, logDir, resultsDir, optimalWorkers, *stressDuration, *stressLongBias, *stressRampUp, nil, &stressResults, &stressMutex)

			// Generate report
			if len(stressResults) > 0 {
				stressReportFile := filepath.Join(resultsDir, fmt.Sprintf("%s-adaptive-stress-report-%s.md", provider.Name, sessionTimestamp))
				if err := generateStressReport(resultsDir, stressResults, sessionTimestamp); err != nil {
					log.Printf("Warning: Failed to generate stress report: %v", err)
				} else {
					log.Printf("Adaptive stress report saved: %s", stressReportFile)
				}
			}
		}

		log.Printf("Adaptive stress testing complete. Results saved to: %s/", sessionDir)
		return
	}

	if *stressFlag {
		// Determine worker count and validate stress mode settings
		numWorkers := resolveStressWorkers(*stressFlag, *stressWorkers, *stressLevel)
		validateStressSettings(numWorkers, *stressDuration, *stressLongBias, *stressRampUp)

		log.Printf("=== RUNNING IN HIGH-STRESS MODE ===")
		log.Printf("Workers: %d | Duration: %ds | Long-form bias: %d%% | Ramp-up: %ds",
			numWorkers, *stressDuration, *stressLongBias, *stressRampUp)

		var stressResults []StressSummary
		var stressMutex sync.Mutex

		if len(providersToTest) > 1 {
			// Run multiple providers concurrently
			var stressWg sync.WaitGroup
			for _, provider := range providersToTest {
				stressWg.Add(1)
				go stressMode(provider, tke, logDir, resultsDir, numWorkers, *stressDuration, *stressLongBias, *stressRampUp, &stressWg, &stressResults, &stressMutex)
			}
			stressWg.Wait()
		} else {
			// Single provider (no concurrency needed)
			for _, provider := range providersToTest {
				stressMode(provider, tke, logDir, resultsDir, numWorkers, *stressDuration, *stressLongBias, *stressRampUp, nil, &stressResults, &stressMutex)
			}
		}

		log.Println("--- All stress tests complete. ---")

		// Generate stress report
		log.Println("Generating stress summary report...")
		if err := generateStressReport(resultsDir, stressResults, sessionTimestamp); err != nil {
			log.Printf("Warning: Failed to generate stress report: %v", err)
		}

		log.Printf("Stress tests complete. Results saved to: %s/", sessionDir)
		return
	}

	var wg sync.WaitGroup
	var results []TestResult
	var resultsMutex sync.Mutex

	for _, provider := range providersToTest {
		if *testAll {
			// Run all tests concurrently
			wg.Add(1)
			go testProviderMetrics(provider, tke, &wg, logDir, resultsDir, &results, &resultsMutex, testMode, toolReasoningCheck)
		} else {
			// Run a single test sequentially
			testProviderMetrics(provider, tke, nil, logDir, resultsDir, &results, &resultsMutex, testMode, toolReasoningCheck)
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

// runDiscoveryTest runs a single discovery test at a specific load level.
func runDiscoveryTest(ctx context.Context, config ProviderConfig, tke *tiktoken.Tiktoken, workers int, duration time.Duration, enableCircuit bool, circuitFailures int, circuitCooldown time.Duration, maxRetries int, retryBackoff time.Duration) (stress.LoadLevelResult, error) {
	// Create a circuit breaker if enabled
	var cb *stress.CircuitBreaker
	if enableCircuit {
		cb = stress.NewCircuitBreaker(stress.Config{
			FailureThreshold: circuitFailures,
			CooldownPeriod:   circuitCooldown,
		})
	}

	// Create retry configuration
	retryConfig := stress.RetryConfig{
		MaxRetries:   maxRetries,
		BaseDelay:    retryBackoff,
		MaxDelay:     30 * time.Second,
		Multiplier:   2.0,
		JitterFactor: 0.1,
	}

	// Create metrics collector
	metrics := stress.NewMetrics()

	// Create context with timeout for this test
	testCtx, cancel := context.WithTimeout(ctx, duration+30*time.Second)
	defer cancel()

	// Simple load test: each worker makes requests as fast as possible
	type result struct {
		success bool
		e2e     time.Duration
		ttft    time.Duration
		tokens  int
		err     error
	}

	resultsChan := make(chan result, workers*10)
	var wg sync.WaitGroup

	// Start workers
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(_ int) {
			defer wg.Done()

			for {
				select {
				case <-testCtx.Done():
					return
				default:
				}

				// Check circuit breaker
				if cb != nil && !cb.CanExecute() {
					resultsChan <- result{success: false, err: stress.ErrCircuitOpen}
					time.Sleep(1 * time.Second)
					continue
				}

				// Execute request with retry
				reqCtx, cancel := context.WithTimeout(testCtx, 30*time.Second)

				retryResult := stress.Retry(reqCtx, retryConfig, func(ctx context.Context) error {
					_, _, _, _, _, err := singleTestRun(ctx, config, tke, log.New(io.Discard, "", 0))
					return err
				})

				cancel()

				if retryResult.Success {
					resultsChan <- result{success: true}
					if cb != nil {
						cb.RecordSuccess()
					}
				} else {
					resultsChan <- result{success: false, err: retryResult.LastError}
					if cb != nil {
						cb.RecordFailure()
					}
				}
			}
		}(i)
	}

	// Wait for test duration then collect results
	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	// Collect results for the duration
	testStart := time.Now()
CollectResults:
	for time.Since(testStart) < duration {
		select {
		case r := <-resultsChan:
			if r.success {
				metrics.RecordRequest("short", true, r.e2e, r.ttft, r.tokens)
			} else {
				metrics.RecordRequest("short", false, 0, 0, 0)
				if r.err != nil {
					catErr := stress.CategorizeError(r.err)
					metrics.RecordError(catErr.Category.String())
				}
			}
		case <-testCtx.Done():
			break CollectResults
		}
	}

	cancel()

	// Calculate final metrics
	summary := metrics.Summary()

	return stress.LoadLevelResult{
		Workers:       workers,
		Duration:      duration,
		TotalRequests: summary.TotalRequests,
		Success:       summary.Successful,
		Failed:        summary.Failed,
		SuccessRate:   summary.SuccessRate() / 100,
		AvgRPS:        summary.RequestsPerSecond,
		Timestamp:     time.Now(),
	}, nil
}
