# LLM API Speed

A fast, concurrent benchmarking tool for measuring LLM API performance metrics across multiple providers written in Go.

## Features

- **Simple Single Binary CLI Tool**: No dependencies, installation, or scripts required - just download and run
- **Multiple Provider Support**: Test OpenAI, NVIDIA NIM, NovitaAI, NebiusAI, MiniMax, and any OpenAI-compatible API
- **Concurrent Testing**: Benchmark all providers simultaneously with `--all` flag
- **Real Metrics**: Measures End-to-End Latency, Time to First Token (TTFT), and Throughput
- **Accurate Token Counting**: Uses tiktoken for precise token measurements
- **Multi-Run Averaging**: Runs 3 concurrent iterations per provider and averages results for more reliable metrics
- **Multiple Test Modes**: Streaming, tool-calling, and mixed modes for comprehensive testing
- **Diagnostic Mode**: 1-minute stress test with 10 concurrent workers for in-depth performance analysis
- **Session-Based Organization**: Each test run creates its own timestamped folder with logs and results
- **Markdown Reports**: Auto-generates performance summaries with leaderboards and failure analysis
- **Timeout Protection**: 2-minute timeout prevents indefinite hangs on stuck providers

## Quick Start

### Option 1: Using Pre-built Binaries (Recommended)

```bash
# Download the latest release for your platform from:
# https://github.com/lemon07r/llm-api-speed/releases

# Linux/macOS: Make executable
chmod +x llm-api-speed-linux-amd64

# Download example.env from the release
# Or create .env file with your API keys
cat > .env << 'EOF'
OAI_API_KEY=your_key_here
NIM_API_KEY=your_key_here
NIM_MODEL=minimaxai/minimax-m2
EOF

# Run
./llm-api-speed-linux-amd64 --provider nim
```

### Option 2: Build from Source

```bash
# Clone the repository
git clone https://github.com/lemon07r/llm-api-speed.git
cd llm-api-speed

# Build (downloads deps, formats, vets, tests, and builds)
make

# Or just build without checks
make build

# Setup environment
cp example.env .env
# Edit .env and add your API keys

# Test a provider
./llm-api-speed --provider nim

# Test all configured providers
./llm-api-speed --all
```

## Usage

### TOML Configuration (Recommended)

The most flexible way to run tests is using TOML configuration files. This allows you to:
- Test multiple models across multiple providers in one run
- Create reusable test configurations
- Fine-tune test parameters per group
- Run sequential or concurrent tests

```bash
# List available test groups
./llm-api-speed --config config.toml --list-groups

# Run all groups in config
./llm-api-speed --config example.config.toml

# Run specific group
./llm-api-speed --config example.config.toml --group quick-provider-comparison

# Use preset configurations
./llm-api-speed --config configs/quick-comparison.toml
./llm-api-speed --config configs/diagnostic-stress.toml
./llm-api-speed --config configs/model-comparison.toml
```

See `example.config.toml` for a comprehensive configuration example, or check the `configs/` directory for preset configurations.

### Legacy CLI Usage (Backward Compatible)

The original CLI flags still work for quick tests:

```bash
# Generic provider with custom API endpoint
./llm-api-speed --url https://api.openai.com/v1 --model gpt-4

# Generic provider with default OpenRouter
./llm-api-speed --model meta-llama/llama-3.1-8b-instruct

# Specific provider (requires .env configuration)
./llm-api-speed --provider novita
./llm-api-speed --provider nebius

# Test all configured providers concurrently
./llm-api-speed --all
```

### Test Modes

The tool supports three different test modes to measure different aspects of API performance:

#### Streaming Mode (Default)
Tests regular chat completion with streaming responses. This is the default mode.

```bash
# Explicit streaming mode (default behavior)
./llm-api-speed --provider nim
```

#### Tool-Calling Mode
Tests the API's tool/function calling capabilities with streaming. Measures performance when the model needs to generate tool calls.

```bash
# Test tool-calling performance
./llm-api-speed --provider nim --tool-calling
```

#### Mixed Mode
Runs 3 iterations of both streaming and tool-calling modes (6 total runs). Provides comprehensive performance metrics for both use cases.

```bash
# Test both streaming and tool-calling
./llm-api-speed --provider nim --mixed

# Test all providers with both modes
./llm-api-speed --all --mixed
```

### Diagnostic Mode

Diagnostic mode runs intensive stress testing with 10 concurrent workers for 1 minute, making requests every 15 seconds with a 30-second timeout per request. Perfect for:
- Load testing your API endpoints
- Identifying rate limits and throttling behavior
- Measuring performance under sustained concurrent load
- Debugging intermittent issues

```bash
# Run diagnostic mode with streaming
./llm-api-speed --provider nim --diagnostic

# Run diagnostic mode with tool-calling
./llm-api-speed --provider nim --diagnostic --tool-calling

# Run diagnostic mode with mixed workload (alternates between streaming and tool-calling)
./llm-api-speed --provider nim --diagnostic --mixed
```

Diagnostic mode produces:
- Detailed per-request logs for each worker
- Aggregated success/failure statistics
- Average metrics (E2E latency, TTFT, throughput, tokens) across all successful requests
- Error frequency analysis
- JSON summary file with all metrics
- Markdown report with leaderboards and error analysis (DIAGNOSTIC-REPORT.md)

**Multiple Providers:** Diagnostic mode supports testing multiple providers concurrently:

```bash
# Test all providers in diagnostic mode
./llm-api-speed --all --diagnostic --mixed
```

### Save Response Content

Use the `--save-responses` flag to save all API response content to files in the logs directory:

```bash
# Save responses in normal mode
./llm-api-speed --provider nim --save-responses

# Save responses in diagnostic mode
./llm-api-speed --provider nim --diagnostic --save-responses

# Save responses for all providers
./llm-api-speed --all --mixed --save-responses
```

Response files are saved with descriptive names:
- **Normal mode:** `{provider}-run{N}-{mode}-response.txt`
- **Diagnostic mode:** `{provider}-worker{N}-req{N}-{mode}-response.txt`

## Output

Each test run creates a session folder: `results/session-YYYYMMDD-HHMMSS/`

```
results/session-20251110-004615/
├── logs/
│   ├── nim-20251110-004615.log
│   ├── novita-20251110-004615.log
│   └── minimax-20251110-004615.log
├── nim-20251110-004637.json
├── novita-20251110-004640.json
├── minimax-20251110-004642.json
└── REPORT.md  # Performance summary with leaderboards
```

**REPORT.md** includes:
- Summary statistics (success/failure counts)
- Performance leaderboards (by throughput and TTFT)
- Detailed metrics for all providers
- Error details for failed tests

## Supported Providers

- **generic** - OpenRouter (default) or any OpenAI-compatible API (use `--url` to override)
- **nim** - NVIDIA NIM
- **nahcrof** - Nahcrof AI
- **novita** - NovitaAI
- **nebius** - NebiusAI  
- **minimax** - MiniMax

## Configuration

### TOML Configuration Files

Create a TOML file to define test groups with multiple providers and models:

```toml
# API Keys - can reference environment variables
[api_keys]
nim = "${NIM_API_KEY}"
novita = "${NOVITA_API_KEY}"
generic = "${OAI_API_KEY}"

# Test Group
[[groups]]
name = "provider-comparison"
description = "Compare multiple providers"
mode = "streaming"  # streaming | tool-calling | mixed | diagnostic
concurrent = true   # Run all providers in parallel

  [groups.test_params]
  iterations = 3
  timeout_seconds = 120
  save_responses = false

  [[groups.providers]]
  provider = "nim"
  model = "minimaxai/minimax-m2"

  [[groups.providers]]
  provider = "novita"
  model = "minimax/minimax-m2"

  [[groups.providers]]
  provider = "generic"
  model = "meta-llama/llama-3.1-8b-instruct"
```

See `example.config.toml` for all available options and `CONFIGURATION.md` for detailed documentation.

### Environment Variables (.env)

For legacy CLI usage or to store API keys referenced in TOML configs:

```env
# Generic provider (uses --url flag or defaults to OpenRouter)
OAI_API_KEY=your_key_here

# Provider-specific configuration
NIM_API_KEY=your_key_here
NIM_MODEL=minimaxai/minimax-m2

NOVITA_API_KEY=your_key_here
NOVITA_MODEL=minimaxai/minimax-m2
```

## Development

```bash
# Run all checks and build
make

# Run tests
make test

# Format code
make fmt

# Static analysis
make vet

# Clean build artifacts
make clean

# Build for all platforms (outputs to build/ directory)
make release-build

# Show all available targets
make help
```

## License

MIT
