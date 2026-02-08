package stress

import (
	"sync"
	"sync/atomic"
	"time"
)

// ConnectionMetrics tracks detailed connection and request metrics.
type ConnectionMetrics struct {
	// Connection phase timings
	DNSLookup    DurationHistogram
	TCPConnect   DurationHistogram
	TLSHandshake DurationHistogram
	TTFB         DurationHistogram // Time to First Byte

	// Connection pool stats
	ActiveConnections int64
	IdleConnections   int64
	TotalConnections  int64

	// Error categorization
	ErrorsByCategory map[string]int64

	// Retry stats
	RetryAttempts int64
	RetrySuccess  int64

	// Circuit breaker stats
	CircuitBreakerState    string
	CircuitBreakerChanges  int64
	CircuitBlockedRequests int64
}

// DurationHistogram tracks a distribution of durations.
type DurationHistogram struct {
	mu        sync.RWMutex
	values    []time.Duration
	min       time.Duration
	max       time.Duration
	sum       time.Duration
	count     int64
	sorted    bool
	maxValues int // Maximum number of values to store (for memory efficiency)
}

// NewDurationHistogram creates a new duration histogram.
func NewDurationHistogram(maxValues int) *DurationHistogram {
	if maxValues <= 0 {
		maxValues = 10000
	}
	return &DurationHistogram{
		values:    make([]time.Duration, 0, maxValues),
		min:       0,
		max:       0,
		maxValues: maxValues,
	}
}

// Record records a duration in the histogram.
func (h *DurationHistogram) Record(d time.Duration) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.count == 0 || d < h.min {
		h.min = d
	}
	if d > h.max {
		h.max = d
	}

	h.sum += d
	h.count++

	// Reservoir sampling for memory efficiency
	if int64(len(h.values)) < int64(h.maxValues) {
		h.values = append(h.values, d)
	} else {
		// Randomly replace existing values to maintain distribution
		// This is a simple form of reservoir sampling
		idx := int(time.Now().UnixNano() % int64(len(h.values)))
		h.values[idx] = d
	}

	h.sorted = false
}

// Count returns the total count of recorded values.
func (h *DurationHistogram) Count() int64 {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.count
}

// Min returns the minimum recorded value.
func (h *DurationHistogram) Min() time.Duration {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.min
}

// Max returns the maximum recorded value.
func (h *DurationHistogram) Max() time.Duration {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.max
}

// Avg returns the average of recorded values.
func (h *DurationHistogram) Avg() time.Duration {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if h.count == 0 {
		return 0
	}
	return h.sum / time.Duration(h.count)
}

// Percentile returns the p-th percentile (0-100).
func (h *DurationHistogram) Percentile(p float64) time.Duration {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.count == 0 || len(h.values) == 0 {
		return 0
	}

	if !h.sorted {
		// Simple insertion sort for small arrays
		for i := 1; i < len(h.values); i++ {
			key := h.values[i]
			j := i - 1
			for j >= 0 && h.values[j] > key {
				h.values[j+1] = h.values[j]
				j--
			}
			h.values[j+1] = key
		}
		h.sorted = true
	}

	idx := int(float64(len(h.values)-1) * p / 100.0)
	if idx < 0 {
		idx = 0
	}
	if idx >= len(h.values) {
		idx = len(h.values) - 1
	}
	return h.values[idx]
}

// P50 returns the 50th percentile (median).
func (h *DurationHistogram) P50() time.Duration {
	return h.Percentile(50)
}

// P95 returns the 95th percentile.
func (h *DurationHistogram) P95() time.Duration {
	return h.Percentile(95)
}

// P99 returns the 99th percentile.
func (h *DurationHistogram) P99() time.Duration {
	return h.Percentile(99)
}

// Metrics holds comprehensive metrics for a stress test run.
type Metrics struct {
	mu sync.RWMutex

	// Overall stats
	TotalRequests int64
	Successful    int64
	Failed        int64
	TotalTokens   int64
	TotalDuration time.Duration
	StartTime     time.Time

	// Request type breakdown
	RequestTypes map[string]*RequestTypeMetrics

	// Connection metrics
	ConnectionMetrics ConnectionMetrics

	// Latency stats
	Latency LatencyStats

	// Throughput
	ThroughputThroughput float64
	RequestsPerSecond    float64

	// Error categorization
	ErrorsByCategory map[string]int64

	// Per-worker stats
	WorkerStats map[int]*WorkerMetrics
}

// RequestTypeMetrics holds metrics for a specific request type.
type RequestTypeMetrics struct {
	Total   int64
	Success int64
	Failed  int64
	Tokens  int64
	Latency LatencyStats
}

// LatencyStats holds latency statistics.
type LatencyStats struct {
	E2E  DurationHistogram
	TTFT DurationHistogram
}

// WorkerMetrics holds per-worker metrics.
type WorkerMetrics struct {
	Requests   int64
	Success    int64
	Failed     int64
	LastActive time.Time
}

// NewMetrics creates a new stress metrics collector.
func NewMetrics() *Metrics {
	return &Metrics{
		StartTime:        time.Now(),
		RequestTypes:     make(map[string]*RequestTypeMetrics),
		ErrorsByCategory: make(map[string]int64),
		WorkerStats:      make(map[int]*WorkerMetrics),
		Latency: LatencyStats{
			E2E:  *NewDurationHistogram(10000),
			TTFT: *NewDurationHistogram(10000),
		},
		ConnectionMetrics: ConnectionMetrics{
			DNSLookup:        *NewDurationHistogram(5000),
			TCPConnect:       *NewDurationHistogram(5000),
			TLSHandshake:     *NewDurationHistogram(5000),
			TTFB:             *NewDurationHistogram(5000),
			ErrorsByCategory: make(map[string]int64),
		},
	}
}

// RecordRequest records a request result.
func (m *Metrics) RecordRequest(reqType string, success bool, e2e, ttft time.Duration, tokens int) {
	atomic.AddInt64(&m.TotalRequests, 1)

	if success {
		atomic.AddInt64(&m.Successful, 1)
		atomic.AddInt64(&m.TotalTokens, int64(tokens))
	} else {
		atomic.AddInt64(&m.Failed, 1)
	}

	// Record latency
	m.Latency.E2E.Record(e2e)
	m.Latency.TTFT.Record(ttft)

	// Record request type metrics
	m.mu.Lock()
	if m.RequestTypes[reqType] == nil {
		m.RequestTypes[reqType] = &RequestTypeMetrics{
			Latency: LatencyStats{
				E2E:  *NewDurationHistogram(5000),
				TTFT: *NewDurationHistogram(5000),
			},
		}
	}
	reqMetrics := m.RequestTypes[reqType]
	m.mu.Unlock()

	atomic.AddInt64(&reqMetrics.Total, 1)
	if success {
		atomic.AddInt64(&reqMetrics.Success, 1)
		atomic.AddInt64(&reqMetrics.Tokens, int64(tokens))
	} else {
		atomic.AddInt64(&reqMetrics.Failed, 1)
	}
	reqMetrics.Latency.E2E.Record(e2e)
	reqMetrics.Latency.TTFT.Record(ttft)
}

// RecordError records an error by category.
func (m *Metrics) RecordError(category string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.ErrorsByCategory[category]++
}

// RecordConnectionError records a connection error by category.
func (m *Metrics) RecordConnectionError(category string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.ConnectionMetrics.ErrorsByCategory[category]++
}

// RecordRetry records a retry attempt.
func (m *Metrics) RecordRetry(success bool) {
	atomic.AddInt64(&m.ConnectionMetrics.RetryAttempts, 1)
	if success {
		atomic.AddInt64(&m.ConnectionMetrics.RetrySuccess, 1)
	}
}

// UpdateWorkerStats updates per-worker statistics.
func (m *Metrics) UpdateWorkerStats(workerID int, success bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.WorkerStats[workerID] == nil {
		m.WorkerStats[workerID] = &WorkerMetrics{}
	}

	worker := m.WorkerStats[workerID]
	worker.Requests++
	worker.LastActive = time.Now()
	if success {
		worker.Success++
	} else {
		worker.Failed++
	}
}

// Finalize calculates final metrics.
func (m *Metrics) Finalize() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.TotalDuration = time.Since(m.StartTime)

	if m.TotalDuration.Seconds() > 0 {
		m.RequestsPerSecond = float64(m.TotalRequests) / m.TotalDuration.Seconds()
		m.ThroughputThroughput = float64(m.TotalTokens) / m.TotalDuration.Seconds()
	}
}

// Summary returns a summary of the stress test.
func (m *Metrics) Summary() Summary {
	m.Finalize()

	return Summary{
		TotalRequests:     atomic.LoadInt64(&m.TotalRequests),
		Successful:        atomic.LoadInt64(&m.Successful),
		Failed:            atomic.LoadInt64(&m.Failed),
		TotalTokens:       atomic.LoadInt64(&m.TotalTokens),
		TotalDuration:     m.TotalDuration,
		RequestsPerSecond: m.RequestsPerSecond,
		Throughput:        m.ThroughputThroughput,
		AvgE2E:            m.Latency.E2E.Avg(),
		P50E2E:            m.Latency.E2E.P50(),
		P95E2E:            m.Latency.E2E.P95(),
		P99E2E:            m.Latency.E2E.P99(),
		AvgTTFT:           m.Latency.TTFT.Avg(),
		RequestTypes:      m.getRequestTypeSummary(),
		ErrorsByCategory:  m.getErrorSummary(),
	}
}

// getRequestTypeSummary returns a summary of request type metrics.
func (m *Metrics) getRequestTypeSummary() map[string]RequestTypeSummary {
	m.mu.RLock()
	defer m.mu.RUnlock()

	summary := make(map[string]RequestTypeSummary)
	for reqType, metrics := range m.RequestTypes {
		avgTokens := 0
		if metrics.Success > 0 {
			avgTokens = int(atomic.LoadInt64(&metrics.Tokens) / metrics.Success)
		}
		summary[reqType] = RequestTypeSummary{
			Total:     atomic.LoadInt64(&metrics.Total),
			Success:   atomic.LoadInt64(&metrics.Success),
			Failed:    atomic.LoadInt64(&metrics.Failed),
			Tokens:    atomic.LoadInt64(&metrics.Tokens),
			AvgE2E:    metrics.Latency.E2E.Avg(),
			P95E2E:    metrics.Latency.E2E.P95(),
			AvgTTFT:   metrics.Latency.TTFT.Avg(),
			AvgTokens: avgTokens,
		}
	}
	return summary
}

// getErrorSummary returns a copy of error categorization.
func (m *Metrics) getErrorSummary() map[string]int64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	summary := make(map[string]int64)
	for k, v := range m.ErrorsByCategory {
		summary[k] = v
	}
	return summary
}

// Summary provides a summary of stress test results.
type Summary struct {
	TotalRequests     int64
	Successful        int64
	Failed            int64
	TotalTokens       int64
	TotalDuration     time.Duration
	RequestsPerSecond float64
	Throughput        float64
	AvgE2E            time.Duration
	P50E2E            time.Duration
	P95E2E            time.Duration
	P99E2E            time.Duration
	AvgTTFT           time.Duration
	RequestTypes      map[string]RequestTypeSummary
	ErrorsByCategory  map[string]int64
}

// RequestTypeSummary summarizes metrics for a request type.
type RequestTypeSummary struct {
	Total     int64
	Success   int64
	Failed    int64
	Tokens    int64
	AvgE2E    time.Duration
	P95E2E    time.Duration
	AvgTTFT   time.Duration
	AvgTokens int
}

// SuccessRate returns the success rate as a percentage.
func (s *Summary) SuccessRate() float64 {
	if s.TotalRequests == 0 {
		return 0
	}
	return float64(s.Successful) / float64(s.TotalRequests) * 100
}

// ErrorRate returns the error rate as a percentage.
func (s *Summary) ErrorRate() float64 {
	if s.TotalRequests == 0 {
		return 0
	}
	return float64(s.Failed) / float64(s.TotalRequests) * 100
}
