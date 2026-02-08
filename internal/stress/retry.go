package stress

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"net"
	"net/http"
	"strings"
	"time"
)

// RetryConfig holds configuration for retry behavior.
type RetryConfig struct {
	MaxRetries      int
	BaseDelay       time.Duration
	MaxDelay        time.Duration
	Multiplier      float64
	JitterFactor    float64
	RetryableErrors []string
}

// DefaultRetryConfig returns a default retry configuration.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:   3,
		BaseDelay:    500 * time.Millisecond,
		MaxDelay:     30 * time.Second,
		Multiplier:   2.0,
		JitterFactor: 0.1,
		RetryableErrors: []string{
			"TLS handshake timeout",
			"connection reset by peer",
			"i/o timeout",
			"connection timed out",
			"EOF",
			"no such host",
			"timeout exceeded",
			"context deadline exceeded",
		},
	}
}

// RetryResult holds the result of a retry operation.
type RetryResult struct {
	Attempts  int
	Success   bool
	LastError error
	TotalTime time.Duration
}

// IsRetryableError checks if an error is retryable.
func IsRetryableError(err error) bool {
	if err == nil {
		return false
	}

	errStr := err.Error()
	lowerErr := strings.ToLower(errStr)

	// Check for specific network errors
	var netErr net.Error
	if errors.As(err, &netErr) {
		// Retry on timeout
		if netErr.Timeout() {
			return true
		}
	}

	// Check for specific error patterns
	retryablePatterns := []string{
		"tls handshake timeout",
		"connection reset by peer",
		"connection refused",
		"i/o timeout",
		"connection timed out",
		"eof",
		"timeout exceeded",
		"context deadline exceeded",
		"temporary failure",
		"server closed",
	}

	for _, pattern := range retryablePatterns {
		if strings.Contains(lowerErr, pattern) {
			return true
		}
	}

	// Check for HTTP 5xx errors (server errors)
	var httpErr *HTTPStatusError
	if errors.As(err, &httpErr) {
		return httpErr.StatusCode >= 500 && httpErr.StatusCode < 600
	}

	// Check for HTTP 429 (rate limit) - these are retryable with backoff
	if errors.As(err, &httpErr) && httpErr.StatusCode == http.StatusTooManyRequests {
		return true
	}

	return false
}

// HTTPStatusError represents an HTTP error with status code.
type HTTPStatusError struct {
	StatusCode int
	Message    string
}

func (e *HTTPStatusError) Error() string {
	return fmt.Sprintf("HTTP %d: %s", e.StatusCode, e.Message)
}

// IsRateLimitError checks if the error is a rate limit error (429).
func IsRateLimitError(err error) bool {
	var httpErr *HTTPStatusError
	if errors.As(err, &httpErr) {
		return httpErr.StatusCode == http.StatusTooManyRequests
	}
	return false
}

// IsServerError checks if the error is a server error (5xx).
func IsServerError(err error) bool {
	var httpErr *HTTPStatusError
	if errors.As(err, &httpErr) {
		return httpErr.StatusCode >= 500 && httpErr.StatusCode < 600
	}
	return false
}

// calculateBackoff calculates the delay for the next retry attempt.
func calculateBackoff(attempt int, config RetryConfig) time.Duration {
	// Calculate exponential backoff
	delay := float64(config.BaseDelay) * math.Pow(config.Multiplier, float64(attempt))

	// Cap at max delay
	if delay > float64(config.MaxDelay) {
		delay = float64(config.MaxDelay)
	}

	// Add jitter to prevent thundering herd
	//nolint:gosec // Using math/rand for performance; not security-critical (just jitter)
	if config.JitterFactor > 0 {
		jitter := delay * config.JitterFactor * (rand.Float64()*2 - 1)
		delay += jitter
	}

	return time.Duration(delay)
}

// Retry executes the given function with retry logic.
func Retry(ctx context.Context, config RetryConfig, operation func(context.Context) error) RetryResult {
	start := time.Now()
	result := RetryResult{
		Attempts: 0,
		Success:  false,
	}

	for attempt := 0; attempt <= config.MaxRetries; attempt++ {
		result.Attempts = attempt + 1

		// Check context cancellation
		select {
		case <-ctx.Done():
			result.LastError = ctx.Err()
			result.TotalTime = time.Since(start)
			return result
		default:
		}

		// Execute the operation
		err := operation(ctx)
		if err == nil {
			result.Success = true
			result.TotalTime = time.Since(start)
			return result
		}

		result.LastError = err

		// Check if error is retryable
		if !IsRetryableError(err) {
			result.TotalTime = time.Since(start)
			return result
		}

		// Don't retry after the last attempt
		if attempt >= config.MaxRetries {
			break
		}

		// Calculate and apply backoff
		backoff := calculateBackoff(attempt, config)

		// For rate limit errors, use the Retry-After header if available, or longer backoff
		if IsRateLimitError(err) {
			minBackoff := 2 * time.Second
			if backoff < minBackoff {
				backoff = minBackoff
			}
		}

		// Wait before retrying, respecting context cancellation
		timer := time.NewTimer(backoff)
		select {
		case <-ctx.Done():
			timer.Stop()
			result.LastError = ctx.Err()
			result.TotalTime = time.Since(start)
			return result
		case <-timer.C:
		}
	}

	result.TotalTime = time.Since(start)
	return result
}

// RetryWithResult is like Retry but returns a result value.
func RetryWithResult[T any](ctx context.Context, config RetryConfig, operation func(context.Context) (T, error)) (T, RetryResult) {
	var result T
	retryResult := Retry(ctx, config, func(ctx context.Context) error {
		var err error
		result, err = operation(ctx)
		return err
	})
	return result, retryResult
}

// WrapHTTPResponse wraps an HTTP response to create an error if status is not 2xx.
func WrapHTTPResponse(resp *http.Response) error {
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}

	return &HTTPStatusError{
		StatusCode: resp.StatusCode,
		Message:    resp.Status,
	}
}

// CategorizedError provides categorization of errors.
type CategorizedError struct {
	Error     error
	Category  ErrorCategory
	Retryable bool
}

// ErrorCategory represents the type of error.
type ErrorCategory int

// Error category constants for categorizing errors.
const (
	CategoryUnknown ErrorCategory = iota // Unknown error category
	CategoryConnection
	CategoryTLS
	CategoryTimeout
	CategoryServer
	CategoryRateLimit
	CategoryClient
	CategoryInference
)

const categoryUnknownString = "unknown"

func (c ErrorCategory) String() string {
	switch c {
	case CategoryUnknown:
		return categoryUnknownString
	case CategoryConnection:
		return "connection"
	case CategoryTLS:
		return "tls"
	case CategoryTimeout:
		return "timeout"
	case CategoryServer:
		return "server"
	case CategoryRateLimit:
		return "rate_limit"
	case CategoryClient:
		return "client"
	case CategoryInference:
		return "inference"
	default:
		return categoryUnknownString
	}
}

// CategorizeError categorizes an error for better reporting.
func CategorizeError(err error) CategorizedError {
	if err == nil {
		return CategorizedError{Error: nil, Category: CategoryUnknown, Retryable: false}
	}

	errStr := strings.ToLower(err.Error())

	// Check for timeout errors
	if strings.Contains(errStr, "timeout") || strings.Contains(errStr, "context deadline exceeded") {
		return CategorizedError{
			Error:     err,
			Category:  CategoryTimeout,
			Retryable: true,
		}
	}

	// Check for TLS errors
	if strings.Contains(errStr, "tls") {
		return CategorizedError{
			Error:     err,
			Category:  CategoryTLS,
			Retryable: true,
		}
	}

	// Check for connection errors
	if strings.Contains(errStr, "connection") || strings.Contains(errStr, "dial") ||
		strings.Contains(errStr, "reset") || strings.Contains(errStr, "refused") ||
		strings.Contains(errStr, "eof") {
		return CategorizedError{
			Error:     err,
			Category:  CategoryConnection,
			Retryable: true,
		}
	}

	// Check for rate limit errors
	if IsRateLimitError(err) || strings.Contains(errStr, "rate limit") || strings.Contains(errStr, "too many requests") {
		return CategorizedError{
			Error:     err,
			Category:  CategoryRateLimit,
			Retryable: true,
		}
	}

	// Check for server errors
	if IsServerError(err) {
		return CategorizedError{
			Error:     err,
			Category:  CategoryServer,
			Retryable: true,
		}
	}

	// Check for inference/stream errors (mid-stream failures)
	if strings.Contains(errStr, "stream") || strings.Contains(errStr, "no content") ||
		strings.Contains(errStr, "no tool calls") {
		return CategorizedError{
			Error:     err,
			Category:  CategoryInference,
			Retryable: false,
		}
	}

	// Default to client error (likely not retryable)
	return CategorizedError{
		Error:     err,
		Category:  CategoryClient,
		Retryable: false,
	}
}
