package main

import (
	"os"
	"testing"
)

func TestProviderConfig(t *testing.T) {
	config := ProviderConfig{
		Name:    "test",
		BaseURL: "https://api.example.com/v1",
		APIKey:  "test-key",
		Model:   "test-model",
	}

	if config.Name != "test" {
		t.Errorf("Expected Name to be 'test', got '%s'", config.Name)
	}

	if config.BaseURL != "https://api.example.com/v1" {
		t.Errorf("Expected BaseURL to be 'https://api.example.com/v1', got '%s'", config.BaseURL)
	}

	if config.APIKey != "test-key" {
		t.Errorf("Expected APIKey to be 'test-key', got '%s'", config.APIKey)
	}

	if config.Model != "test-model" {
		t.Errorf("Expected Model to be 'test-model', got '%s'", config.Model)
	}
}

func TestEnvironmentVariables(t *testing.T) {
	// Test that environment variables can be set and read
	testKey := "TEST_API_KEY"
	testValue := "test-value-12345"

	if err := os.Setenv(testKey, testValue); err != nil {
		t.Fatalf("Failed to set environment variable: %v", err)
	}
	defer func() {
		if err := os.Unsetenv(testKey); err != nil {
			t.Logf("Warning: Failed to unset environment variable: %v", err)
		}
	}()

	value := os.Getenv(testKey)
	if value != testValue {
		t.Errorf("Expected environment variable to be '%s', got '%s'", testValue, value)
	}
}

func TestProviderConfigCreation(t *testing.T) {
	tests := []struct {
		name         string
		providerName string
		baseURL      string
		apiKey       string
		model        string
	}{
		{"Generic Provider", "generic", "https://openrouter.ai/api/v1", "key1", "model1"},
		{"NIM Provider", "nim", "https://integrate.api.nvidia.com/v1", "key2", "model2"},
		{"NovitaAI Provider", "novita", "https://api.novita.ai/openai", "key3", "model3"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := ProviderConfig{
				Name:    tt.providerName,
				BaseURL: tt.baseURL,
				APIKey:  tt.apiKey,
				Model:   tt.model,
			}

			if config.Name != tt.providerName {
				t.Errorf("Expected Name to be '%s', got '%s'", tt.providerName, config.Name)
			}

			if config.BaseURL != tt.baseURL {
				t.Errorf("Expected BaseURL to be '%s', got '%s'", tt.baseURL, config.BaseURL)
			}

			if config.APIKey != tt.apiKey {
				t.Errorf("Expected APIKey to be '%s', got '%s'", tt.apiKey, config.APIKey)
			}

			if config.Model != tt.model {
				t.Errorf("Expected Model to be '%s', got '%s'", tt.model, config.Model)
			}
		})
	}
}

func TestResolveTestMode(t *testing.T) {
	tests := []struct {
		name            string
		toolCalling     bool
		mixed           bool
		interleavedFlag bool
		wantMode        TestMode
		wantInterleaved bool
		wantForcedTool  bool
	}{
		{
			name:            "default streaming",
			wantMode:        ModeStreaming,
			wantInterleaved: false,
		},
		{
			name:            "explicit tool-calling",
			toolCalling:     true,
			wantMode:        ModeToolCalling,
			wantInterleaved: false,
		},
		{
			name:            "interleaved implies tool-calling",
			interleavedFlag: true,
			wantMode:        ModeToolCalling,
			wantInterleaved: true,
			wantForcedTool:  true,
		},
		{
			name:            "mixed keeps interleaved",
			mixed:           true,
			interleavedFlag: true,
			wantMode:        ModeMixed,
			wantInterleaved: true,
		},
		{
			name:            "tool-calling with interleaved",
			toolCalling:     true,
			interleavedFlag: true,
			wantMode:        ModeToolCalling,
			wantInterleaved: true,
		},
		{
			name:            "mixed without interleaved",
			mixed:           true,
			wantMode:        ModeMixed,
			wantInterleaved: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mode, interleaved, forced := resolveTestMode(tt.toolCalling, tt.mixed, tt.interleavedFlag)
			if mode != tt.wantMode {
				t.Fatalf("expected mode %s, got %s", tt.wantMode, mode)
			}
			if interleaved != tt.wantInterleaved {
				t.Fatalf("expected interleaved=%t, got %t", tt.wantInterleaved, interleaved)
			}
			if forced != tt.wantForcedTool {
				t.Fatalf("expected forced=%t, got %t", tt.wantForcedTool, forced)
			}
		})
	}
}
