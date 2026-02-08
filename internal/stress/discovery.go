package stress

import (
	"context"
	"fmt"
	"log"
	"time"
)

// DiscoveryConfig holds configuration for load discovery.
type DiscoveryConfig struct {
	// Starting worker count
	StartWorkers int
	// Maximum workers to test
	MaxWorkers int
	// Failure rate threshold (0.0 - 1.0) to consider capacity exceeded
	FailureThreshold float64
	// Duration for each load level test
	TestDuration time.Duration
	// Whether to use binary search after initial discovery
	UseBinarySearch bool
	// Ramp-up time for each test
	RampUpTime time.Duration
}

// DefaultDiscoveryConfig returns default discovery configuration.
func DefaultDiscoveryConfig() DiscoveryConfig {
	return DiscoveryConfig{
		StartWorkers:     10,
		MaxWorkers:       2000,
		FailureThreshold: 0.10, // 10% failure rate
		TestDuration:     60 * time.Second,
		UseBinarySearch:  true,
		RampUpTime:       5 * time.Second,
	}
}

// DiscoveryResult holds the results of a load discovery run.
type DiscoveryResult struct {
	// The maximum sustainable worker count
	OptimalWorkers int
	// The worker count that exceeded threshold
	MaxTestedWorkers int
	// Detailed results for each tested level
	LevelResults []LoadLevelResult
	// Final capacity estimate with confidence
	CapacityEstimate CapacityEstimate
	// Total time for discovery
	TotalDuration time.Duration
}

// LoadLevelResult holds results for a single load level test.
type LoadLevelResult struct {
	Workers       int
	Duration      time.Duration
	TotalRequests int64
	Success       int64
	Failed        int64
	SuccessRate   float64
	AvgRPS        float64
	AvgLatency    time.Duration
	P95Latency    time.Duration
	AvgThroughput float64
	Timestamp     time.Time
}

// CapacityEstimate provides a confidence-weighted capacity estimate.
type CapacityEstimate struct {
	Workers    int
	Confidence float64 // 0.0 - 1.0
	LowerBound int
	UpperBound int
}

// LoadDiscovery performs progressive load testing to find provider capacity.
type LoadDiscovery struct {
	config DiscoveryConfig
}

// NewLoadDiscovery creates a new load discovery instance.
func NewLoadDiscovery(config DiscoveryConfig) *LoadDiscovery {
	return &LoadDiscovery{config: config}
}

// Run executes the load discovery process.
func (ld *LoadDiscovery) Run(ctx context.Context, testFunc LoadTestFunc) (*DiscoveryResult, error) {
	startTime := time.Now()
	result := &DiscoveryResult{
		LevelResults: make([]LoadLevelResult, 0),
	}

	// Phase 1: Exponential discovery
	log.Println("=== Phase 1: Exponential Load Discovery ===")
	low, high, found := ld.exponentialDiscovery(ctx, testFunc, result)
	if !found {
		return nil, fmt.Errorf("could not determine capacity within tested range")
	}

	// Phase 2: Binary search refinement (optional)
	if ld.config.UseBinarySearch && high > low*2 {
		log.Println("\n=== Phase 2: Binary Search Refinement ===")
		ld.binarySearch(ctx, testFunc, low, high, result)
	}

	// Calculate final estimate
	result.OptimalWorkers = ld.calculateOptimalCapacity(result.LevelResults)
	result.CapacityEstimate = ld.calculateConfidenceEstimate(result.LevelResults)
	result.TotalDuration = time.Since(startTime)

	return result, nil
}

// LoadTestFunc is the function signature for running a load test.
type LoadTestFunc func(ctx context.Context, workers int, duration time.Duration) (LoadLevelResult, error)

// exponentialDiscovery finds the capacity range using exponential growth.
func (ld *LoadDiscovery) exponentialDiscovery(ctx context.Context, testFunc LoadTestFunc, result *DiscoveryResult) (low, high int, found bool) {
	workers := ld.config.StartWorkers
	low = workers

	for workers <= ld.config.MaxWorkers {
		select {
		case <-ctx.Done():
			return low, workers, true
		default:
		}

		log.Printf("\nTesting with %d workers...\n", workers)

		levelResult, err := testFunc(ctx, workers, ld.config.TestDuration)
		if err != nil {
			log.Printf("  Error: %v\n", err)
			// Treat errors as high failure rate
			levelResult.Workers = workers
			levelResult.SuccessRate = 0
		}

		levelResult.Timestamp = time.Now()
		result.LevelResults = append(result.LevelResults, levelResult)
		result.MaxTestedWorkers = workers

		log.Printf("  Success Rate: %.1f%%\n", levelResult.SuccessRate*100)
		log.Printf("  Avg RPS: %.2f\n", levelResult.AvgRPS)
		log.Printf("  P95 Latency: %s\n", levelResult.P95Latency)

		// Check if we've exceeded the threshold
		if levelResult.SuccessRate < (1 - ld.config.FailureThreshold) {
			log.Printf("\n  -> Failure threshold exceeded at %d workers\n", workers)
			high = workers
			return low, high, true
		}

		// This level passed, move to next
		low = workers

		// Double the workers for next test
		workers *= 2
		if workers > ld.config.MaxWorkers {
			workers = ld.config.MaxWorkers
		}

		// If we hit max, we're done
		if workers == ld.config.MaxWorkers && low == ld.config.MaxWorkers {
			log.Printf("\n  -> Reached maximum worker count (%d)\n", ld.config.MaxWorkers)
			return low, workers, true
		}
	}

	return low, workers, false
}

// binarySearch narrows down the optimal capacity using binary search.
func (ld *LoadDiscovery) binarySearch(ctx context.Context, testFunc LoadTestFunc, low, high int, result *DiscoveryResult) {
	// Keep track of tested values to avoid redundant tests
	tested := make(map[int]bool)
	for _, r := range result.LevelResults {
		tested[r.Workers] = true
	}

	for high-low > max(low/10, 5) { // Stop when range is within 10% or 5 workers
		mid := (low + high) / 2

		// Skip if already tested
		if tested[mid] {
			// Find nearest untested value
			for offset := 1; offset < (high-low)/2; offset++ {
				if !tested[mid+offset] {
					mid += offset
					break
				}
				if !tested[mid-offset] {
					mid -= offset
					break
				}
			}
		}

		select {
		case <-ctx.Done():
			return
		default:
		}

		log.Printf("\nBinary search: testing %d workers (range: %d-%d)...\n", mid, low, high)

		levelResult, err := testFunc(ctx, mid, ld.config.TestDuration)
		if err != nil {
			levelResult.Workers = mid
			levelResult.SuccessRate = 0
		}

		levelResult.Timestamp = time.Now()
		result.LevelResults = append(result.LevelResults, levelResult)
		tested[mid] = true

		log.Printf("  Success Rate: %.1f%%\n", levelResult.SuccessRate*100)

		if levelResult.SuccessRate >= (1 - ld.config.FailureThreshold) {
			// This level passed, move low up
			low = mid
		} else {
			// This level failed, move high down
			high = mid
		}
	}
}

// calculateOptimalCapacity determines the recommended worker count.
func (ld *LoadDiscovery) calculateOptimalCapacity(results []LoadLevelResult) int {
	if len(results) == 0 {
		return 0
	}

	// Find the highest passing level
	var bestResult *LoadLevelResult
	for i := range results {
		if results[i].SuccessRate >= (1 - ld.config.FailureThreshold) {
			if bestResult == nil || results[i].Workers > bestResult.Workers {
				bestResult = &results[i]
			}
		}
	}

	if bestResult == nil {
		// All failed, return the lowest tested
		return results[0].Workers
	}

	return bestResult.Workers
}

// calculateConfidenceEstimate provides a confidence-weighted estimate.
func (ld *LoadDiscovery) calculateConfidenceEstimate(results []LoadLevelResult) CapacityEstimate {
	if len(results) < 2 {
		return CapacityEstimate{
			Workers:    0,
			Confidence: 0,
			LowerBound: 0,
			UpperBound: 0,
		}
	}

	// Find the transition point (where success rate drops)
	var lower, upper int
	for i := len(results) - 1; i >= 0; i-- {
		if results[i].SuccessRate >= (1 - ld.config.FailureThreshold) {
			lower = results[i].Workers
			break
		}
	}

	for i := 0; i < len(results); i++ {
		if results[i].SuccessRate < (1 - ld.config.FailureThreshold) {
			upper = results[i].Workers
			break
		}
	}

	if upper == 0 {
		upper = results[len(results)-1].Workers * 2
	}

	// Calculate confidence based on gap size
	gap := float64(upper - lower)
	optimal := float64(lower)
	confidence := 1.0 - (gap / optimal)
	if confidence < 0 {
		confidence = 0
	}

	return CapacityEstimate{
		Workers:    lower,
		Confidence: confidence,
		LowerBound: int(float64(lower) * 0.9),
		UpperBound: upper,
	}
}

// GenerateReport creates a detailed capacity discovery report.
func (dr *DiscoveryResult) GenerateReport() string {
	report := "=== Load Discovery Report ===\n\n"

	report += "Test Configuration:\n"
	report += fmt.Sprintf("  Failure Threshold: %.0f%%\n", 10.0)
	report += fmt.Sprintf("  Test Duration: %s\n", "60s")
	report += fmt.Sprintf("  Total Discovery Time: %s\n\n", dr.TotalDuration)

	report += "Capacity Results:\n"
	report += fmt.Sprintf("  Optimal Workers: %d\n", dr.OptimalWorkers)
	report += fmt.Sprintf("  Max Tested: %d\n", dr.MaxTestedWorkers)
	report += fmt.Sprintf("  Confidence: %.0f%%\n", dr.CapacityEstimate.Confidence*100)
	report += fmt.Sprintf("  Recommended Range: %d - %d workers\n\n",
		dr.CapacityEstimate.LowerBound, dr.CapacityEstimate.UpperBound)

	report += "Detailed Results:\n"
	report += "  Workers | Success | Failed | Rate    | RPS    | P95 Latency\n"
	report += "  --------|---------|--------|---------|--------|------------\n"

	for _, r := range dr.LevelResults {
		report += fmt.Sprintf("  %7d | %7d | %6d | %6.1f%% | %6.2f | %s\n",
			r.Workers, r.Success, r.Failed, r.SuccessRate*100, r.AvgRPS, r.P95Latency)
	}

	report += "\nRecommendations:\n"
	report += fmt.Sprintf("  1. Use %d workers for reliable stress testing\n", dr.OptimalWorkers)
	report += fmt.Sprintf("  2. Scale up to %d workers during peak load\n", dr.CapacityEstimate.UpperBound)
	report += fmt.Sprintf("  3. Monitor for failure rates above %.0f%%\n", 10.0)

	return report
}

// QuickCapacityTest runs a single quick capacity test.
func QuickCapacityTest(ctx context.Context, testFunc LoadTestFunc, workers int, duration time.Duration) (*LoadLevelResult, error) {
	result, err := testFunc(ctx, workers, duration)
	if err != nil {
		return nil, err
	}
	return &result, nil
}
