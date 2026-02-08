package stress

import (
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

// State represents the state of the circuit breaker.
type State int

const (
	// StateClosed means the circuit is closed and requests flow normally.
	StateClosed State = iota
	// StateOpen means the circuit is open and requests fail fast.
	StateOpen
	// StateHalfOpen means the circuit is testing if the service recovered.
	StateHalfOpen
)

func (s State) String() string {
	switch s {
	case StateClosed:
		return "closed"
	case StateOpen:
		return "open"
	case StateHalfOpen:
		return "half-open"
	default:
		return "unknown"
	}
}

// CircuitBreaker implements the circuit breaker pattern for fault tolerance.
type CircuitBreaker struct {
	mu sync.RWMutex

	// Configuration
	failureThreshold int
	successThreshold int
	cooldownPeriod   time.Duration
	timeout          time.Duration

	// State
	state           State
	failures        int
	successes       int
	lastFailure     time.Time
	lastStateChange time.Time

	// Metrics
	stats CircuitStats
}

// CircuitStats holds circuit breaker statistics.
type CircuitStats struct {
	StateChanges      int64
	RequestsAllowed   int64
	RequestsBlocked   int64
	FailuresRecorded  int64
	SuccessesRecorded int64
}

// Config holds configuration for the circuit breaker.
type Config struct {
	FailureThreshold int
	SuccessThreshold int
	CooldownPeriod   time.Duration
	Timeout          time.Duration
}

// DefaultConfig returns a default circuit breaker configuration.
func DefaultConfig() Config {
	return Config{
		FailureThreshold: 10,
		SuccessThreshold: 3,
		CooldownPeriod:   30 * time.Second,
		Timeout:          60 * time.Second,
	}
}

// NewCircuitBreaker creates a new circuit breaker with the given configuration.
func NewCircuitBreaker(config Config) *CircuitBreaker {
	return &CircuitBreaker{
		state:            StateClosed,
		failureThreshold: config.FailureThreshold,
		successThreshold: config.SuccessThreshold,
		cooldownPeriod:   config.CooldownPeriod,
		timeout:          config.Timeout,
		lastStateChange:  time.Now(),
	}
}

// State returns the current state of the circuit breaker.
func (cb *CircuitBreaker) State() State {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.state
}

// Stats returns circuit breaker statistics.
func (cb *CircuitBreaker) Stats() CircuitStats {
	return CircuitStats{
		StateChanges:      atomic.LoadInt64(&cb.stats.StateChanges),
		RequestsAllowed:   atomic.LoadInt64(&cb.stats.RequestsAllowed),
		RequestsBlocked:   atomic.LoadInt64(&cb.stats.RequestsBlocked),
		FailuresRecorded:  atomic.LoadInt64(&cb.stats.FailuresRecorded),
		SuccessesRecorded: atomic.LoadInt64(&cb.stats.SuccessesRecorded),
	}
}

// CanExecute checks if a request should be allowed to proceed.
func (cb *CircuitBreaker) CanExecute() bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.state {
	case StateClosed:
		atomic.AddInt64(&cb.stats.RequestsAllowed, 1)
		return true

	case StateOpen:
		// Check if cooldown period has elapsed
		if time.Since(cb.lastFailure) > cb.cooldownPeriod {
			cb.transitionTo(StateHalfOpen)
			atomic.AddInt64(&cb.stats.RequestsAllowed, 1)
			return true
		}
		atomic.AddInt64(&cb.stats.RequestsBlocked, 1)
		return false

	case StateHalfOpen:
		atomic.AddInt64(&cb.stats.RequestsAllowed, 1)
		return true

	default:
		return false
	}
}

// RecordSuccess records a successful request.
func (cb *CircuitBreaker) RecordSuccess() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	atomic.AddInt64(&cb.stats.SuccessesRecorded, 1)

	switch cb.state {
	case StateHalfOpen:
		cb.successes++
		if cb.successes >= cb.successThreshold {
			cb.transitionTo(StateClosed)
			cb.failures = 0
			cb.successes = 0
		}

	case StateClosed:
		// Reset failure count on success
		if cb.failures > 0 {
			cb.failures = 0
		}
	case StateOpen:
		// No-op when circuit is open
	}
}

// RecordFailure records a failed request.
func (cb *CircuitBreaker) RecordFailure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	atomic.AddInt64(&cb.stats.FailuresRecorded, 1)
	cb.lastFailure = time.Now()

	switch cb.state {
	case StateHalfOpen:
		// A single failure in half-open state moves back to open
		cb.transitionTo(StateOpen)
		cb.failures = 0
		cb.successes = 0

	case StateClosed:
		cb.failures++
		if cb.failures >= cb.failureThreshold {
			cb.transitionTo(StateOpen)
		}
	case StateOpen:
		// No-op when circuit is open
	}
}

// transitionTo transitions the circuit breaker to a new state.
func (cb *CircuitBreaker) transitionTo(state State) {
	if cb.state != state {
		cb.state = state
		cb.lastStateChange = time.Now()
		atomic.AddInt64(&cb.stats.StateChanges, 1)
	}
}

// Execute runs the given function if the circuit allows it.
// Returns ErrCircuitOpen if the circuit is open.
func (cb *CircuitBreaker) Execute(operation func() error) error {
	if !cb.CanExecute() {
		return ErrCircuitOpen
	}

	err := operation()
	if err != nil {
		cb.RecordFailure()
		return err
	}

	cb.RecordSuccess()
	return nil
}

// ErrCircuitOpen is returned when the circuit breaker is open.
var ErrCircuitOpen = errors.New("circuit breaker is open")

// Reset forces the circuit breaker back to closed state.
func (cb *CircuitBreaker) Reset() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.state = StateClosed
	cb.failures = 0
	cb.successes = 0
	cb.lastFailure = time.Time{}
	cb.lastStateChange = time.Now()
}

// TimeInCurrentState returns how long the circuit has been in its current state.
func (cb *CircuitBreaker) TimeInCurrentState() time.Duration {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return time.Since(cb.lastStateChange)
}

// AdaptiveCircuitBreaker extends CircuitBreaker with adaptive threshold adjustment.
type AdaptiveCircuitBreaker struct {
	*CircuitBreaker
	mu sync.RWMutex

	// Adaptive configuration
	minThreshold       int
	maxThreshold       int
	adjustmentInterval time.Duration

	// Window tracking
	windowStart   time.Time
	windowSuccess int
	windowTotal   int
}

// AdaptiveConfig extends Config with adaptive settings.
type AdaptiveConfig struct {
	Config
	MinThreshold       int
	MaxThreshold       int
	AdjustmentInterval time.Duration
}

// DefaultAdaptiveConfig returns default adaptive configuration.
func DefaultAdaptiveConfig() AdaptiveConfig {
	return AdaptiveConfig{
		Config:             DefaultConfig(),
		MinThreshold:       5,
		MaxThreshold:       50,
		AdjustmentInterval: 60 * time.Second,
	}
}

// NewAdaptiveCircuitBreaker creates an adaptive circuit breaker.
func NewAdaptiveCircuitBreaker(config AdaptiveConfig) *AdaptiveCircuitBreaker {
	return &AdaptiveCircuitBreaker{
		CircuitBreaker:     NewCircuitBreaker(config.Config),
		minThreshold:       config.MinThreshold,
		maxThreshold:       config.MaxThreshold,
		adjustmentInterval: config.AdjustmentInterval,
		windowStart:        time.Now(),
	}
}

// RecordResult records a request result and adjusts thresholds if needed.
func (acb *AdaptiveCircuitBreaker) RecordResult(success bool) {
	acb.mu.Lock()
	defer acb.mu.Unlock()

	// Update window stats
	acb.windowTotal++
	if success {
		acb.windowSuccess++
	}

	// Check if we should adjust thresholds
	if time.Since(acb.windowStart) >= acb.adjustmentInterval {
		acb.adjustThresholds()
	}

	acb.mu.Unlock()
	if success {
		acb.RecordSuccess()
	} else {
		acb.RecordFailure()
	}
	acb.mu.Lock()
}

// adjustThresholds adjusts the failure threshold based on recent success rate.
func (acb *AdaptiveCircuitBreaker) adjustThresholds() {
	if acb.windowTotal == 0 {
		return
	}

	successRate := float64(acb.windowSuccess) / float64(acb.windowTotal)

	// If success rate is very high, lower the threshold to fail faster
	// If success rate is moderate, raise the threshold to be more tolerant
	if successRate > 0.95 {
		acb.failureThreshold = maxInt(acb.failureThreshold-1, acb.minThreshold)
	} else if successRate < 0.80 {
		acb.failureThreshold = minInt(acb.failureThreshold+1, acb.maxThreshold)
	}

	// Reset window
	acb.windowStart = time.Now()
	acb.windowSuccess = 0
	acb.windowTotal = 0
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
