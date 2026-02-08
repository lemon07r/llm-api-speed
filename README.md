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
# Download and run (https://github.com/lemon07r/llm-api-speed/releases)
chmod +x llm-api-speed
echo "OAI_API_KEY=your_key_here" > .env
./llm-api-speed --model meta-llama/llama-3.1-8b-instruct

# Or build from source
git clone https://github.com/lemon07r/llm-api-speed.git
cd llm-api-speed && make
```

---

## Choosing a Test Mode

| Goal | Command |
|------|---------|
| Quick health check | `./llm-api-speed --model gpt-4` |
| Function calling | `./llm-api-speed --provider X --tool-calling` |
| Compare providers | `./llm-api-speed --all` |
| Light load (10 users) | `./llm-api-speed --provider X --diagnostic` |
| Heavy stress (100-1000+ users) | `./llm-api-speed --provider X --stress` |
| Endurance test (30+ min) | `./llm-api-speed --provider X --stress --stress-duration 1800` |
| Max generation test | `./llm-api-speed --provider X --long-story` |

---

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
| Diagnostic | `--diagnostic` | Light stress test: 10 workers, 90 seconds |
| High-Stress | `--stress` | Heavy stress test: 100-1000+ workers, mixed load |
| Long-Story | `--long-story` | Long-form generation (4000+ words) |

```bash
# Examples
./llm-api-speed --provider nahcrof --tool-calling
./llm-api-speed --provider nahcrof --mixed
./llm-api-speed --provider nahcrof --diagnostic
./llm-api-speed --provider nahcrof --stress --stress-level heavy
./llm-api-speed --provider nahcrof --long-story
./llm-api-speed --all --diagnostic --mixed
```

### High-Stress Mode

Simulate extreme traffic with configurable concurrency:

```bash
./llm-api-speed --provider nahcrof --stress                    # 100 workers, 5min
./llm-api-speed --provider nahcrof --stress --stress-level heavy --stress-duration 600  # 500 workers, 10min
./llm-api-speed --provider nahcrof --stress --stress-level extreme                      # 1000 workers
./llm-api-speed --provider nahcrof --stress --stress-workers 250 --stress-long-bias 50  # Custom
./llm-api-speed --all --stress --stress-level heavy            # Test all providers
```

**Default Request Mix:** 50% short streaming (~512 tokens) / 20% tool-calling / 30% long-form (~16K tokens)

| Request Type | Prompt | Typical Response |
|--------------|--------|------------------|
| **Short** | 150-word robot story | ~100-200 tokens, 1-3s |
| **Tool** | Weather for 3 cities | ~150 tokens + tool calls |
| **Long** | 4,000+ word dragon story | ~4K-8K tokens, 60-300s |

Use `--stress-long-bias 10-80` to adjust the long-form percentage.

### Other Features

```bash
# Test parallel tool calls with reasoning
./llm-api-speed --provider nahcrof --tool-calling --interleaved-tools

# Compare providers normalized to 500 tokens
./llm-api-speed --all --diagnostic --target-tokens 500

# Save all API responses to files
./llm-api-speed --provider nahcrof --save-responses
```

### Supported Providers

| Provider | Base URL | Environment Variables |
|----------|----------|----------------------|
| generic | OpenRouter (default) or any `--url` | `OAI_API_KEY` |
| nim | NVIDIA NIM | `NIM_API_KEY`, `NIM_MODEL` |
| novita | NovitaAI | `NOVITA_API_KEY`, `NOVITA_MODEL` |
| nebius | NebiusAI | `NEBIUS_API_KEY`, `NEBIUS_MODEL` |
| minimax | MiniMax | `MINIMAX_API_KEY`, `MINIMAX_MODEL` |
| nahcrof | Nahcrof AI | `NAHCROF_API_KEY`, `NAHCROF_MODEL` |

## Output

Results are saved to `results/session-YYYYMMDD-HHMMSS/`:

```
results/session-20251110-012642/
├── logs/                          # Detailed execution logs
├── *.json                         # Structured results per provider
├── REPORT.md                      # Standard mode report
└── STRESS-REPORT.md               # Stress mode report (if applicable)
```

---

## Metrics Reference

| Metric | Description |
|--------|-------------|
| **E2E Latency** | Total time from request to final token |
| **TTFT** | Time to First Token - measures initial responsiveness |
| **Throughput** | Tokens per second after first token |
| **Projected E2E** | `TTFT + (Target Tokens / Throughput)` - normalized comparison |
| **RPS** | Requests per second (stress mode only) |
| **P50/P95/P99** | Latency percentiles - reveals tail latency issues |

## CLI Reference

### General Flags
| Flag | Description |
|------|-------------|
| `--provider` | Specific provider to test (nim, novita, nebius, minimax, nahcrof) |
| `--all` | Test all configured providers |
| `--url` | Custom API base URL |
| `--model` | Model name |
| `--save-responses` | Save API responses to files |

### Test Mode Flags
| Flag | Description |
|------|-------------|
| `--tool-calling` | Test function calling capabilities |
| `--mixed` | Run both streaming and tool-calling |
| `--diagnostic` | Light stress test (10 workers, 90s) |
| `--stress` | High-stress mode (100-1000+ workers) |
| `--stress-level` | Preset: moderate (100), heavy (500), extreme (1000) |
| `--stress-workers` | Exact worker count override |
| `--stress-duration` | Test duration in seconds (default: 300) |
| `--stress-long-bias` | % of long-form requests (default: 30, range: 10-80) |
| `--long-story` | Long-form generation (4000+ words) |
| `--interleaved-tools` | Test parallel tool calls |
| `--target-tokens` | Target tokens for projected E2E (default: 350) |
| `--max-tokens` | Max tokens for long-story (default: 16384) |

---

## Best Practices & Troubleshooting

### Before Stress Testing
1. **Check rate limits** - Most APIs have RPM/TPM limits
2. **Start small** - Use `--stress-duration 60` first
3. **Watch for 429 errors** - Reduce workers if rate limited
4. **Control costs** - Use `--stress-long-bias 10` for cheaper tests

### Recommended Progression
```bash
./llm-api-speed --provider X --stress --stress-duration 60   # Quick validation
./llm-api-speed --provider X --stress                         # Standard 5min test
./llm-api-speed --provider X --stress --stress-level heavy    # Heavy load
./llm-api-speed --provider X --stress --stress-duration 1800  # 30min endurance
```

### Common Issues
| Issue | Solution |
|-------|----------|
| `context deadline exceeded` | Reduce load or check timeouts |
| `429 Too Many Requests` | Reduce `--stress-workers` |
| `no tool calls observed` | Provider/model may not support tools |
| Empty responses | Try lower concurrency |
| Build errors | Run `make fmt && make all` |

---

## Configuration

Create `.env` file:

```env
OAI_API_KEY=your_key_here          # Generic/OpenRouter

NIM_API_KEY=your_key_here          # NVIDIA NIM
NIM_MODEL=deepseek-ai/deepseek-v3.1

NOVITA_API_KEY=your_key_here       # NovitaAI
NOVITA_MODEL=minimaxai/minimax-m2

NEBIUS_API_KEY=your_key_here       # NebiusAI
NEBIUS_MODEL=moonshotai/Kimi-K2-Instruct

MINIMAX_API_KEY=your_key_here      # MiniMax
MINIMAX_MODEL=MiniMax-M2

NAHCROF_API_KEY=your_key_here      # Nahcrof
NAHCROF_MODEL=kimi-k2-thinking
```

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
