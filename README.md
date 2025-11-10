# llm-api-speed

A fast, concurrent benchmarking tool for measuring LLM API performance metrics across multiple providers.

## Features

- **Multiple Provider Support**: Test OpenAI, NVIDIA NIM, NovitaAI, NebiusAI, MiniMax, and any OpenAI-compatible API
- **Concurrent Testing**: Benchmark all providers simultaneously with `--all` flag
- **Real Metrics**: Measures End-to-End Latency, Time to First Token (TTFT), and Throughput
- **Accurate Token Counting**: Uses tiktoken for precise token measurements
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

Copy `example.env` to `.env` and configure:

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
