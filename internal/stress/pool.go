// Package stress provides utilities for high-concurrency stress testing of LLM APIs.
package stress

import (
	"crypto/tls"
	"fmt"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// PoolStats holds connection pool statistics.
type PoolStats struct {
	ActiveConnections     int64
	IdleConnections       int64
	TotalConnections      int64
	TLSHandshakes         int64
	TLSHandshakeDurations time.Duration
}

// PoolConfig holds configuration for the connection pool.
type PoolConfig struct {
	Workers             int
	TLSHandshakeTimeout time.Duration
	DialTimeout         time.Duration
	KeepAlive           time.Duration
	IdleConnTimeout     time.Duration
}

// DefaultPoolConfig returns a default configuration scaled for the given worker count.
func DefaultPoolConfig(workers int) PoolConfig {
	return PoolConfig{
		Workers:             workers,
		TLSHandshakeTimeout: 30 * time.Second,
		DialTimeout:         10 * time.Second,
		KeepAlive:           30 * time.Second,
		IdleConnTimeout:     90 * time.Second,
	}
}

// Pool wraps http.Transport with dynamic connection pool sizing.
type Pool struct {
	config    PoolConfig
	transport *http.Transport
	stats     PoolStats
	mu        sync.RWMutex
}

// NewPool creates a new connection pool optimized for the given worker count.
func NewPool(config PoolConfig) *Pool {
	// Scale connection pool based on worker count
	// Use workers * 2 to account for potential connection churn
	maxConns := config.Workers * 2
	if maxConns < 100 {
		maxConns = 100
	}

	transport := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   config.DialTimeout,
			KeepAlive: config.KeepAlive,
		}).DialContext,
		MaxIdleConns:          maxConns,
		MaxIdleConnsPerHost:   maxConns,
		MaxConnsPerHost:       maxConns,
		IdleConnTimeout:       config.IdleConnTimeout,
		TLSHandshakeTimeout:   config.TLSHandshakeTimeout,
		ExpectContinueTimeout: 1 * time.Second,
		DisableKeepAlives:     false,
		ForceAttemptHTTP2:     true,
	}

	return &Pool{
		config:    config,
		transport: transport,
	}
}

// Transport returns the underlying HTTP transport.
func (p *Pool) Transport() *http.Transport {
	return p.transport
}

// Stats returns current connection pool statistics.
func (p *Pool) Stats() PoolStats {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return PoolStats{
		ActiveConnections:     atomic.LoadInt64(&p.stats.ActiveConnections),
		IdleConnections:       atomic.LoadInt64(&p.stats.IdleConnections),
		TotalConnections:      atomic.LoadInt64(&p.stats.TotalConnections),
		TLSHandshakes:         atomic.LoadInt64(&p.stats.TLSHandshakes),
		TLSHandshakeDurations: time.Duration(atomic.LoadInt64((*int64)(&p.stats.TLSHandshakeDurations))),
	}
}

// CloseIdleConnections closes all idle connections in the pool.
func (p *Pool) CloseIdleConnections() {
	p.transport.CloseIdleConnections()
}

// ConnectionState wraps connection state tracking.
type ConnectionState struct {
	StartTime    time.Time
	TLSTime      time.Duration
	ConnectTime  time.Duration
	DNSTime      time.Duration
	StateChanged chan<- ConnectionPhase
}

// ConnectionPhase represents different phases of connection establishment.
type ConnectionPhase int

// Connection phase constants.
const (
	PhaseDNS ConnectionPhase = iota // DNS lookup phase
	PhaseTCP
	PhaseTLS
	PhaseReady
)

// DialContext wraps the dialer with connection metrics tracking.
func (p *Pool) DialContext() func(network, addr string) (net.Conn, error) {
	dialer := &net.Dialer{
		Timeout:   p.config.DialTimeout,
		KeepAlive: p.config.KeepAlive,
	}

	return func(network, addr string) (net.Conn, error) {
		start := time.Now()
		conn, err := dialer.Dial(network, addr)
		if err != nil {
			return nil, fmt.Errorf("dial failed: %w", err)
		}

		atomic.AddInt64(&p.stats.TotalConnections, 1)

		// Wrap connection to track TLS handshake timing
		return &trackedConn{
			Conn:         conn,
			pool:         p,
			connectStart: start,
		}, nil
	}
}

// trackedConn wraps a net.Conn to track TLS handshake timing.
type trackedConn struct {
	net.Conn
	pool         *Pool
	connectStart time.Time
}

// ConnectionState is called when the TLS handshake starts/ends.
func (c *trackedConn) ConnectionState() tls.ConnectionState {
	if tlsConn, ok := c.Conn.(*tls.Conn); ok {
		return tlsConn.ConnectionState()
	}
	return tls.ConnectionState{}
}

// SetDeadline sets the read and write deadlines.
func (c *trackedConn) SetDeadline(t time.Time) error {
	if err := c.Conn.SetDeadline(t); err != nil {
		return fmt.Errorf("set deadline failed: %w", err)
	}
	return nil
}

// SetReadDeadline sets the read deadline.
func (c *trackedConn) SetReadDeadline(t time.Time) error {
	if err := c.Conn.SetReadDeadline(t); err != nil {
		return fmt.Errorf("set read deadline failed: %w", err)
	}
	return nil
}

// SetWriteDeadline sets the write deadline.
func (c *trackedConn) SetWriteDeadline(t time.Time) error {
	if err := c.Conn.SetWriteDeadline(t); err != nil {
		return fmt.Errorf("set write deadline failed: %w", err)
	}
	return nil
}
