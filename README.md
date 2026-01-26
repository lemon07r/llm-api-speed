# LLM API Speed

A fast, concurrent benchmarking tool for measuring LLM API performance across multiple providers. Written in Go as a single binary with no dependencies.

## Features

- **Single Binary** - No installation or dependencies, just download and run
- **Multi-Provider** - Test any OpenAI-compatible API (OpenAI, NVIDIA NIM, NovitaAI, NebiusAI, MiniMax, etc.)
- **Concurrent Testing** - Benchmark all providers simultaneously
- **Real Metrics** - E2E Latency, Time to First Token (TTFT), Throughput (tokens/sec)
- **Projected E2E Latency** - Normalized comparison across different output lengths
- **Multiple Test Modes** - Streaming, tool-calling, mixed, diagnostic stress-test, long-story generation
- **Markdown Reports** - Auto-generated performance summaries with leaderboards

## Quick Start

```bash
# Download the latest release for your platform
# https://github.com/lemon07r/llm-api-speed/releases

# Make executable (Linux/macOS)
chmod +x llm-api-speed

# Create .env with your API key
echo "OAI_API_KEY=your_key_here" > .env

# Run a quick test (uses OpenRouter by default)
./llm-api-speed --model meta-llama/llama-3.1-8b-instruct
```

### Build from Source

```bash
git clone https://github.com/lemon07r/llm-api-speed.git
cd llm-api-speed
make        # builds to ./llm-api-speed
```

## Usage

### Basic Commands

```bash
# Test with OpenRouter (default)
./llm-api-speed --model meta-llama/llama-3.1-8b-instruct

# Test with custom OpenAI-compatible endpoint
./llm-api-speed --url https://api.openai.com/v1 --model gpt-4

# Test a specific provider (requires .env config)
./llm-api-speed --provider nim

# Test all configured providers at once
./llm-api-speed --all
```

### Test Modes

| Mode | Flag | Description |
|------|------|-------------|
| Streaming | *(default)* | Standard chat completion with streaming |
| Tool-Calling | `--tool-calling` | Tests function calling capabilities |
| Mixed | `--mixed` | Runs both streaming and tool-calling |
| Diagnostic | `--diagnostic` | Stress test: 10 workers, 90 seconds |
| Long-Story | `--long-story` | Long-form generation (4000+ words) |

```bash
# Examples
./llm-api-speed --provider nim --tool-calling
./llm-api-speed --provider nim --mixed
./llm-api-speed --provider nim --diagnostic
./llm-api-speed --provider nim --long-story
./llm-api-speed --all --diagnostic --mixed
```

### Interleaved Tool-Call Testing

Test if a model supports parallel tool calls with reasoning:

```bash
./llm-api-speed --provider nim --tool-calling --interleaved-tools
```

### Projected E2E Latency

Normalize performance comparison across different output lengths:

```bash
# Compare providers normalized to 500 tokens
./llm-api-speed --all --diagnostic --target-tokens 500
```

**Formula:** `Projected E2E = TTFT + (Target Tokens / Throughput)`

### Save Responses

```bash
./llm-api-speed --provider nim --save-responses
```

## Configuration

Create a `.env` file (or copy from `example.env`):

```env
# Generic (OpenRouter by default, or use --url to override)
OAI_API_KEY=your_key_here

# Provider-specific
NIM_API_KEY=your_key_here
NIM_MODEL=deepseek-ai/deepseek-v3.1

NOVITA_API_KEY=your_key_here
NOVITA_MODEL=minimaxai/minimax-m2

NEBIUS_API_KEY=your_key_here
NEBIUS_MODEL=moonshotai/Kimi-K2-Instruct

MINIMAX_API_KEY=your_key_here
MINIMAX_MODEL=MiniMax-M2

NAHCROF_API_KEY=your_key_here
NAHCROF_MODEL=minimax-m2
```

### Supported Providers

| Provider | Base URL |
|----------|----------|
| generic | OpenRouter (default) or any `--url` |
| nim | NVIDIA NIM |
| novita | NovitaAI |
| nebius | NebiusAI |
| minimax | MiniMax |
| nahcrof | Nahcrof AI |

## Output

Each test creates a session folder with logs, JSON results, and a markdown report:

```
results/session-20251110-012642/
├── logs/
│   ├── nim-20251110-012646.log
│   └── novita-20251110-012650.log
├── nim-20251110-012646.json
├── novita-20251110-012650.json
└── REPORT.md
```

**REPORT.md** includes:
- Success/failure summary
- Performance leaderboards (by throughput, TTFT, projected E2E)
- Detailed metrics for all providers
- Error analysis for failed tests

## CLI Reference

| Flag | Description |
|------|-------------|
| `--provider` | Specific provider to test |
| `--all` | Test all configured providers |
| `--url` | Custom API base URL |
| `--model` | Model name |
| `--tool-calling` | Enable tool-calling mode |
| `--mixed` | Run both streaming and tool-calling |
| `--diagnostic` | Run stress test mode |
| `--long-story` | Long-form story generation |
| `--interleaved-tools` | Test parallel tool calls |
| `--target-tokens` | Target tokens for projected E2E (default: 350) |
| `--max-tokens` | Max tokens for long-story (default: 16384) |
| `--save-responses` | Save API responses to files |

## Development

```bash
make            # Run all (default)
make all        # Run deps, fmt, vet, lint, test, and build
make build      # Build for current platform only
make test       # Run tests
make fmt        # Format code
make vet        # Static analysis
make clean      # Clean build artifacts
make release-build  # Build for all platforms
make help       # Show all targets
```

## License

MIT
