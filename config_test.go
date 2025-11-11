package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveEnvVars(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		envVars  map[string]string
		expected string
	}{
		{
			name:     "Simple ${VAR} syntax",
			input:    "${TEST_KEY}",
			envVars:  map[string]string{"TEST_KEY": "test-value"},
			expected: "test-value",
		},
		{
			name:     "Simple $VAR syntax",
			input:    "$TEST_KEY",
			envVars:  map[string]string{"TEST_KEY": "test-value"},
			expected: "test-value",
		},
		{
			name:     "Mixed text and variable",
			input:    "prefix-${API_KEY}-suffix",
			envVars:  map[string]string{"API_KEY": "abc123"},
			expected: "prefix-abc123-suffix",
		},
		{
			name:     "Multiple variables",
			input:    "${VAR1}:${VAR2}",
			envVars:  map[string]string{"VAR1": "value1", "VAR2": "value2"},
			expected: "value1:value2",
		},
		{
			name:     "No variable",
			input:    "plain-text",
			envVars:  map[string]string{},
			expected: "plain-text",
		},
		{
			name:     "Undefined variable returns empty",
			input:    "${UNDEFINED_VAR}",
			envVars:  map[string]string{},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set environment variables
			for k, v := range tt.envVars {
				os.Setenv(k, v)
				defer os.Unsetenv(k)
			}

			result := ResolveEnvVars(tt.input)
			if result != tt.expected {
				t.Errorf("ResolveEnvVars(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestMergeDefaults(t *testing.T) {
	t.Run("Global defaults", func(t *testing.T) {
		cfg := &Config{}
		cfg = MergeDefaults(cfg)

		if cfg.Global.TimeoutSeconds != 120 {
			t.Errorf("Expected default timeout 120, got %d", cfg.Global.TimeoutSeconds)
		}

		if cfg.Global.ResultsDir != "results" {
			t.Errorf("Expected default results dir 'results', got %s", cfg.Global.ResultsDir)
		}

		if cfg.Global.LogLevel != "info" {
			t.Errorf("Expected default log level 'info', got %s", cfg.Global.LogLevel)
		}
	})

	t.Run("Standard test defaults", func(t *testing.T) {
		cfg := &Config{
			Groups: []TestGroup{
				{
					Name: "test-group",
					Mode: "streaming",
				},
			},
		}
		cfg = MergeDefaults(cfg)

		if cfg.Groups[0].TestParams == nil {
			t.Fatal("TestParams should be initialized")
		}

		if cfg.Groups[0].TestParams.Iterations != 3 {
			t.Errorf("Expected default iterations 3, got %d", cfg.Groups[0].TestParams.Iterations)
		}

		if cfg.Groups[0].TestParams.TimeoutSeconds != 120 {
			t.Errorf("Expected default timeout 120, got %d", cfg.Groups[0].TestParams.TimeoutSeconds)
		}
	})

	t.Run("Diagnostic defaults", func(t *testing.T) {
		cfg := &Config{
			Groups: []TestGroup{
				{
					Name: "diag-group",
					Mode: "diagnostic",
				},
			},
		}
		cfg = MergeDefaults(cfg)

		if cfg.Groups[0].DiagnosticParams == nil {
			t.Fatal("DiagnosticParams should be initialized")
		}

		params := cfg.Groups[0].DiagnosticParams

		if params.DurationSeconds != 60 {
			t.Errorf("Expected default duration 60, got %d", params.DurationSeconds)
		}

		if params.Workers != 10 {
			t.Errorf("Expected default workers 10, got %d", params.Workers)
		}

		if params.IntervalSeconds != 15 {
			t.Errorf("Expected default interval 15, got %d", params.IntervalSeconds)
		}

		if params.TimeoutPerRequestSeconds != 30 {
			t.Errorf("Expected default per-request timeout 30, got %d", params.TimeoutPerRequestSeconds)
		}
	})

	t.Run("Provider base URL defaults", func(t *testing.T) {
		cfg := &Config{
			Groups: []TestGroup{
				{
					Name: "test-group",
					Mode: "streaming",
					Providers: []GroupProviderConfig{
						{Provider: "nim", Model: "test-model"},
						{Provider: "novita", Model: "test-model"},
					},
				},
			},
		}
		cfg = MergeDefaults(cfg)

		if cfg.Groups[0].Providers[0].BaseURL != "https://integrate.api.nvidia.com/v1" {
			t.Errorf("Expected NIM default URL, got %s", cfg.Groups[0].Providers[0].BaseURL)
		}

		if cfg.Groups[0].Providers[1].BaseURL != "https://api.novita.ai/openai" {
			t.Errorf("Expected Novita default URL, got %s", cfg.Groups[0].Providers[1].BaseURL)
		}
	})
}

func TestValidateConfig(t *testing.T) {
	t.Run("Valid config", func(t *testing.T) {
		cfg := &Config{
			APIKeys: map[string]string{"nim": "test-key"},
			Groups: []TestGroup{
				{
					Name: "test-group",
					Mode: "streaming",
					Providers: []GroupProviderConfig{
						{Provider: "nim", Model: "test-model"},
					},
				},
			},
		}

		if err := ValidateConfig(cfg); err != nil {
			t.Errorf("Expected valid config, got error: %v", err)
		}
	})

	t.Run("No groups", func(t *testing.T) {
		cfg := &Config{}
		err := ValidateConfig(cfg)

		if err == nil {
			t.Error("Expected error for no groups")
		}
	})

	t.Run("Group missing name", func(t *testing.T) {
		cfg := &Config{
			Groups: []TestGroup{
				{Mode: "streaming"},
			},
		}
		err := ValidateConfig(cfg)

		if err == nil {
			t.Error("Expected error for missing group name")
		}
	})

	t.Run("Group missing mode", func(t *testing.T) {
		cfg := &Config{
			Groups: []TestGroup{
				{Name: "test"},
			},
		}
		err := ValidateConfig(cfg)

		if err == nil {
			t.Error("Expected error for missing mode")
		}
	})

	t.Run("Invalid mode", func(t *testing.T) {
		cfg := &Config{
			APIKeys: map[string]string{"nim": "test-key"},
			Groups: []TestGroup{
				{
					Name: "test",
					Mode: "invalid-mode",
					Providers: []GroupProviderConfig{
						{Provider: "nim", Model: "test-model"},
					},
				},
			},
		}
		err := ValidateConfig(cfg)

		if err == nil {
			t.Error("Expected error for invalid mode")
		}
	})

	t.Run("No providers", func(t *testing.T) {
		cfg := &Config{
			Groups: []TestGroup{
				{
					Name:      "test",
					Mode:      "streaming",
					Providers: []GroupProviderConfig{},
				},
			},
		}
		err := ValidateConfig(cfg)

		if err == nil {
			t.Error("Expected error for no providers")
		}
	})

	t.Run("Provider missing model", func(t *testing.T) {
		cfg := &Config{
			APIKeys: map[string]string{"nim": "test-key"},
			Groups: []TestGroup{
				{
					Name: "test",
					Mode: "streaming",
					Providers: []GroupProviderConfig{
						{Provider: "nim"},
					},
				},
			},
		}
		err := ValidateConfig(cfg)

		if err == nil {
			t.Error("Expected error for missing model")
		}
	})

	t.Run("Missing API key", func(t *testing.T) {
		cfg := &Config{
			APIKeys: map[string]string{},
			Groups: []TestGroup{
				{
					Name: "test",
					Mode: "streaming",
					Providers: []GroupProviderConfig{
						{Provider: "nim", Model: "test-model"},
					},
				},
			},
		}
		err := ValidateConfig(cfg)

		if err == nil {
			t.Error("Expected error for missing API key")
		}
	})

	t.Run("Diagnostic params with non-diagnostic mode", func(t *testing.T) {
		cfg := &Config{
			APIKeys: map[string]string{"nim": "test-key"},
			Groups: []TestGroup{
				{
					Name: "test",
					Mode: "streaming",
					Providers: []GroupProviderConfig{
						{Provider: "nim", Model: "test-model"},
					},
					DiagnosticParams: &DiagnosticParameters{
						DurationSeconds: 60,
					},
				},
			},
		}
		err := ValidateConfig(cfg)

		if err == nil {
			t.Error("Expected error for diagnostic params with non-diagnostic mode")
		}
	})
}

func TestSanitizeModelName(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"meta-llama/llama-3.1-8b", "meta-llama-llama-3.1-8b"},
		{"minimaxai/minimax-m2", "minimaxai-minimax-m2"},
		{"model:with:colons", "model-with-colons"},
		{"model with spaces", "model-with-spaces"},
		{"model//double//slash", "model-double-slash"},
		{"--dashes--", "dashes"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := sanitizeModelName(tt.input)
			if result != tt.expected {
				t.Errorf("sanitizeModelName(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestConvertGroupToProviderConfig(t *testing.T) {
	t.Run("Successful conversion", func(t *testing.T) {
		gpc := GroupProviderConfig{
			Provider: "nim",
			Model:    "meta-llama/llama-3.1-8b",
			BaseURL:  "https://api.example.com/v1",
		}

		apiKeys := map[string]string{
			"nim": "test-api-key-123",
		}

		result, err := ConvertGroupToProviderConfig(gpc, apiKeys)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if result.Name != "nim-meta-llama-llama-3.1-8b" {
			t.Errorf("Expected name 'nim-meta-llama-llama-3.1-8b', got '%s'", result.Name)
		}

		if result.BaseURL != "https://api.example.com/v1" {
			t.Errorf("Expected BaseURL 'https://api.example.com/v1', got '%s'", result.BaseURL)
		}

		if result.APIKey != "test-api-key-123" {
			t.Errorf("Expected APIKey 'test-api-key-123', got '%s'", result.APIKey)
		}

		if result.Model != "meta-llama/llama-3.1-8b" {
			t.Errorf("Expected Model 'meta-llama/llama-3.1-8b', got '%s'", result.Model)
		}
	})

	t.Run("Missing API key", func(t *testing.T) {
		gpc := GroupProviderConfig{
			Provider: "nim",
			Model:    "test-model",
		}

		apiKeys := map[string]string{}

		_, err := ConvertGroupToProviderConfig(gpc, apiKeys)
		if err == nil {
			t.Error("Expected error for missing API key")
		}
	})

	t.Run("Empty API key", func(t *testing.T) {
		gpc := GroupProviderConfig{
			Provider: "nim",
			Model:    "test-model",
		}

		apiKeys := map[string]string{
			"nim": "",
		}

		_, err := ConvertGroupToProviderConfig(gpc, apiKeys)
		if err == nil {
			t.Error("Expected error for empty API key")
		}
	})
}

func TestLoadConfig(t *testing.T) {
	t.Run("Valid config file", func(t *testing.T) {
		// Create temporary config file
		tmpDir := t.TempDir()
		configFile := filepath.Join(tmpDir, "test-config.toml")

		content := `
[api_keys]
nim = "test-key-123"

[[groups]]
name = "test-group"
description = "Test group description"
mode = "streaming"
concurrent = true

  [[groups.providers]]
  provider = "nim"
  model = "test-model"
`

		if err := os.WriteFile(configFile, []byte(content), 0644); err != nil {
			t.Fatalf("Failed to create test config: %v", err)
		}

		cfg, err := LoadConfig(configFile)
		if err != nil {
			t.Fatalf("Failed to load config: %v", err)
		}

		if cfg.APIKeys["nim"] != "test-key-123" {
			t.Errorf("Expected API key 'test-key-123', got '%s'", cfg.APIKeys["nim"])
		}

		if len(cfg.Groups) != 1 {
			t.Fatalf("Expected 1 group, got %d", len(cfg.Groups))
		}

		if cfg.Groups[0].Name != "test-group" {
			t.Errorf("Expected group name 'test-group', got '%s'", cfg.Groups[0].Name)
		}
	})

	t.Run("Config with env var substitution", func(t *testing.T) {
		// Set test env var
		os.Setenv("TEST_API_KEY", "env-key-456")
		defer os.Unsetenv("TEST_API_KEY")

		tmpDir := t.TempDir()
		configFile := filepath.Join(tmpDir, "test-config.toml")

		content := `
[api_keys]
nim = "${TEST_API_KEY}"

[[groups]]
name = "test-group"
mode = "streaming"

  [[groups.providers]]
  provider = "nim"
  model = "test-model"
`

		if err := os.WriteFile(configFile, []byte(content), 0644); err != nil {
			t.Fatalf("Failed to create test config: %v", err)
		}

		cfg, err := LoadConfig(configFile)
		if err != nil {
			t.Fatalf("Failed to load config: %v", err)
		}

		if cfg.APIKeys["nim"] != "env-key-456" {
			t.Errorf("Expected resolved API key 'env-key-456', got '%s'", cfg.APIKeys["nim"])
		}
	})

	t.Run("Non-existent file", func(t *testing.T) {
		_, err := LoadConfig("/non/existent/file.toml")
		if err == nil {
			t.Error("Expected error for non-existent file")
		}
	})

	t.Run("Invalid TOML syntax", func(t *testing.T) {
		tmpDir := t.TempDir()
		configFile := filepath.Join(tmpDir, "invalid.toml")

		content := `
[api_keys
invalid toml syntax
`

		if err := os.WriteFile(configFile, []byte(content), 0644); err != nil {
			t.Fatalf("Failed to create test config: %v", err)
		}

		_, err := LoadConfig(configFile)
		if err == nil {
			t.Error("Expected error for invalid TOML")
		}
	})
}

func TestGetGroupByName(t *testing.T) {
	cfg := &Config{
		Groups: []TestGroup{
			{Name: "group1", Mode: "streaming"},
			{Name: "group2", Mode: "diagnostic"},
			{Name: "group3", Mode: "mixed"},
		},
	}

	t.Run("Find existing group", func(t *testing.T) {
		group, err := GetGroupByName(cfg, "group2")
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if group.Name != "group2" {
			t.Errorf("Expected group name 'group2', got '%s'", group.Name)
		}

		if group.Mode != "diagnostic" {
			t.Errorf("Expected mode 'diagnostic', got '%s'", group.Mode)
		}
	})

	t.Run("Group not found", func(t *testing.T) {
		_, err := GetGroupByName(cfg, "non-existent")
		if err == nil {
			t.Error("Expected error for non-existent group")
		}
	})
}
