# llm-api-speed

A fast, concurrent benchmarking tool for measuring LLM API performance metrics across multiple providers.

## Features

- **Multiple Provider Support**: Test OpenAI, NVIDIA NIM, NovitaAI, NebiusAI, MiniMax, and any OpenAI-compatible API
- **Concurrent Testing**: Benchmark all providers simultaneously with `--all` flag
- **Real Metrics**: Measures End-to-End Latency, Time to First Token (TTFT), and Throughput
- **Accurate Token Counting**: Uses tiktoken for precise token measurements

## Quick Start

```bash
# Download dependencies
go mod download

# Build
go build -o llm-api-speed main.go

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
# Run tests
go test -v

# Format code
go fmt ./...

# Static analysis
go vet ./...
```

## License

MIT
