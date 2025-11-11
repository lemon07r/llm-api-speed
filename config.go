package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/BurntSushi/toml"
)

// Config represents the entire TOML configuration
type Config struct {
	Global GlobalSettings     `toml:"global"`
	APIKeys map[string]string `toml:"api_keys"`
	Groups []TestGroup        `toml:"groups"`
}

// GlobalSettings contains global configuration
type GlobalSettings struct {
	SaveResponses  bool   `toml:"save_responses"`
	LogLevel       string `toml:"log_level"`
	TimeoutSeconds int    `toml:"timeout_seconds"`
	ResultsDir     string `toml:"results_dir"`
}

// TestGroup represents a group of tests to run
type TestGroup struct {
	Name             string                 `toml:"name"`
	Description      string                 `toml:"description"`
	Mode             string                 `toml:"mode"`
	Concurrent       bool                   `toml:"concurrent"`
	Providers        []GroupProviderConfig  `toml:"providers"`
	TestParams       *TestParameters        `toml:"test_params"`
	DiagnosticParams *DiagnosticParameters  `toml:"diagnostic_params"`
}

// GroupProviderConfig defines a provider within a group
type GroupProviderConfig struct {
	Provider string `toml:"provider"`
	Model    string `toml:"model"`
	BaseURL  string `toml:"base_url"`
}

// TestParameters for standard tests
type TestParameters struct {
	Iterations     int  `toml:"iterations"`
	TimeoutSeconds int  `toml:"timeout_seconds"`
	SaveResponses  bool `toml:"save_responses"`
}

// DiagnosticParameters for diagnostic mode
type DiagnosticParameters struct {
	DurationSeconds          int  `toml:"duration_seconds"`
	Workers                  int  `toml:"workers"`
	IntervalSeconds          int  `toml:"interval_seconds"`
	TimeoutPerRequestSeconds int  `toml:"timeout_per_request_seconds"`
	SaveResponses            bool `toml:"save_responses"`
}

// LoadConfig loads and parses a TOML configuration file
func LoadConfig(filename string) (*Config, error) {
	absPath, err := filepath.Abs(filename)
	if err != nil {
		return nil, fmt.Errorf("error resolving config path: %w", err)
	}

	data, err := os.ReadFile(absPath)
	if err != nil {
		return nil, fmt.Errorf("error reading config file: %w", err)
	}

	var cfg Config
	if err := toml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("error parsing TOML: %w", err)
	}

	// Resolve environment variables in API keys
	for key, value := range cfg.APIKeys {
		cfg.APIKeys[key] = ResolveEnvVars(value)
	}

	// Apply defaults
	cfg = *MergeDefaults(&cfg)

	// Validate configuration
	if err := ValidateConfig(&cfg); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	return &cfg, nil
}

// ResolveEnvVars replaces ${VAR} or $VAR with environment variables
func ResolveEnvVars(s string) string {
	// Match ${VAR} pattern
	re := regexp.MustCompile(`\$\{([^}]+)\}`)
	result := re.ReplaceAllStringFunc(s, func(match string) string {
		varName := match[2 : len(match)-1]
		return os.Getenv(varName)
	})

	// Match $VAR pattern (word characters only)
	re2 := regexp.MustCompile(`\$([A-Za-z_][A-Za-z0-9_]*)`)
	result = re2.ReplaceAllStringFunc(result, func(match string) string {
		varName := match[1:]
		return os.Getenv(varName)
	})

	return result
}

// ValidateConfig ensures the configuration is valid
func ValidateConfig(cfg *Config) error {
	if len(cfg.Groups) == 0 {
		return fmt.Errorf("no test groups defined")
	}

	// Validate each group
	for i, group := range cfg.Groups {
		if group.Name == "" {
			return fmt.Errorf("group %d: name is required", i)
		}

		if group.Mode == "" {
			return fmt.Errorf("group '%s': mode is required", group.Name)
		}

		validModes := map[string]bool{
			"streaming":    true,
			"tool-calling": true,
			"mixed":        true,
			"diagnostic":   true,
		}

		if !validModes[group.Mode] {
			return fmt.Errorf("group '%s': invalid mode '%s' (must be: streaming, tool-calling, mixed, or diagnostic)", group.Name, group.Mode)
		}

		if len(group.Providers) == 0 {
			return fmt.Errorf("group '%s': no providers defined", group.Name)
		}

		// Validate diagnostic mode has diagnostic params (or will use defaults)
		if group.Mode == "diagnostic" {
			// This is fine, defaults will be applied
		} else {
			// Standard modes shouldn't have diagnostic params
			if group.DiagnosticParams != nil {
				return fmt.Errorf("group '%s': diagnostic_params can only be used with mode='diagnostic'", group.Name)
			}
		}

		// Validate each provider in the group
		for j, provider := range group.Providers {
			if provider.Provider == "" {
				return fmt.Errorf("group '%s', provider %d: provider name is required", group.Name, j)
			}

			if provider.Model == "" {
				return fmt.Errorf("group '%s', provider %d (%s): model is required", group.Name, j, provider.Provider)
			}

			// Check if API key exists for this provider
			if _, ok := cfg.APIKeys[provider.Provider]; !ok {
				return fmt.Errorf("group '%s', provider %d (%s): no API key defined in [api_keys]", group.Name, j, provider.Provider)
			}
		}
	}

	return nil
}

// MergeDefaults applies default values to incomplete configs
func MergeDefaults(cfg *Config) *Config {
	// Global defaults
	if cfg.Global.TimeoutSeconds == 0 {
		cfg.Global.TimeoutSeconds = 120
	}
	if cfg.Global.ResultsDir == "" {
		cfg.Global.ResultsDir = "results"
	}
	if cfg.Global.LogLevel == "" {
		cfg.Global.LogLevel = "info"
	}

	// Group defaults
	for i := range cfg.Groups {
		group := &cfg.Groups[i]

		// Standard test defaults
		if group.Mode != "diagnostic" {
			if group.TestParams == nil {
				group.TestParams = &TestParameters{}
			}

			if group.TestParams.Iterations == 0 {
				group.TestParams.Iterations = 3
			}

			if group.TestParams.TimeoutSeconds == 0 {
				group.TestParams.TimeoutSeconds = cfg.Global.TimeoutSeconds
			}
		}

		// Diagnostic test defaults
		if group.Mode == "diagnostic" {
			if group.DiagnosticParams == nil {
				group.DiagnosticParams = &DiagnosticParameters{}
			}

			if group.DiagnosticParams.DurationSeconds == 0 {
				group.DiagnosticParams.DurationSeconds = 60
			}

			if group.DiagnosticParams.Workers == 0 {
				group.DiagnosticParams.Workers = 10
			}

			if group.DiagnosticParams.IntervalSeconds == 0 {
				group.DiagnosticParams.IntervalSeconds = 15
			}

			if group.DiagnosticParams.TimeoutPerRequestSeconds == 0 {
				group.DiagnosticParams.TimeoutPerRequestSeconds = 30
			}
		}

		// Provider defaults - resolve base URLs
		for j := range group.Providers {
			provider := &group.Providers[j]
			if provider.BaseURL == "" {
				provider.BaseURL = getDefaultBaseURL(provider.Provider)
			}
		}
	}

	return cfg
}

// getDefaultBaseURL returns the default base URL for a provider
func getDefaultBaseURL(provider string) string {
	defaults := map[string]string{
		"generic": "https://openrouter.ai/api/v1",
		"nim":     "https://integrate.api.nvidia.com/v1",
		"nahcrof": "https://ai.nahcrof.com/v2",
		"novita":  "https://api.novita.ai/openai",
		"nebius":  "https://api.tokenfactory.nebius.com/v1",
		"minimax": "https://api.minimax.io/v1",
	}

	if url, ok := defaults[provider]; ok {
		return url
	}

	// For unknown providers, default to OpenRouter
	return "https://openrouter.ai/api/v1"
}

// ConvertGroupToProviderConfig converts group provider to ProviderConfig
func ConvertGroupToProviderConfig(gpc GroupProviderConfig, apiKeys map[string]string) (ProviderConfig, error) {
	apiKey, ok := apiKeys[gpc.Provider]
	if !ok {
		return ProviderConfig{}, fmt.Errorf("no API key found for provider '%s'", gpc.Provider)
	}

	if apiKey == "" {
		return ProviderConfig{}, fmt.Errorf("API key for provider '%s' is empty (check environment variables)", gpc.Provider)
	}

	return ProviderConfig{
		Name:    fmt.Sprintf("%s-%s", gpc.Provider, sanitizeModelName(gpc.Model)),
		BaseURL: gpc.BaseURL,
		APIKey:  apiKey,
		Model:   gpc.Model,
	}, nil
}

// sanitizeModelName creates a safe name from model string
func sanitizeModelName(model string) string {
	// Replace slashes and special chars with dashes
	name := strings.ReplaceAll(model, "/", "-")
	name = strings.ReplaceAll(name, " ", "-")
	name = strings.ReplaceAll(name, ":", "-")

	// Remove consecutive dashes
	re := regexp.MustCompile(`-+`)
	name = re.ReplaceAllString(name, "-")

	// Trim dashes from ends
	name = strings.Trim(name, "-")

	return name
}

// ListGroups prints available groups in the config
func ListGroups(cfg *Config) {
	fmt.Println("\nAvailable Test Groups:")
	fmt.Println("======================")

	for _, group := range cfg.Groups {
		fmt.Printf("\n• %s\n", group.Name)
		if group.Description != "" {
			fmt.Printf("  Description: %s\n", group.Description)
		}
		fmt.Printf("  Mode: %s\n", group.Mode)
		fmt.Printf("  Providers: %d\n", len(group.Providers))
		fmt.Printf("  Concurrent: %v\n", group.Concurrent)

		if group.Mode == "diagnostic" && group.DiagnosticParams != nil {
			fmt.Printf("  Duration: %ds | Workers: %d | Interval: %ds\n",
				group.DiagnosticParams.DurationSeconds,
				group.DiagnosticParams.Workers,
				group.DiagnosticParams.IntervalSeconds)
		} else if group.TestParams != nil {
			fmt.Printf("  Iterations: %d | Timeout: %ds\n",
				group.TestParams.Iterations,
				group.TestParams.TimeoutSeconds)
		}
	}

	fmt.Println()
}

// GetGroupByName finds a group by name
func GetGroupByName(cfg *Config, name string) (*TestGroup, error) {
	for i := range cfg.Groups {
		if cfg.Groups[i].Name == name {
			return &cfg.Groups[i], nil
		}
	}
	return nil, fmt.Errorf("group '%s' not found", name)
}
