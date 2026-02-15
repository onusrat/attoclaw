# AttoClaw

A single-file AI agent in Go with I2C/SPI hardware access. Zero dependencies. Compiles everywhere.

AttoClaw bridges language models and the physical world. It provides a ReAct agent loop that connects to any OpenAI-compatible API, with built-in tools for filesystem operations, shell execution, and direct hardware access via I2C and SPI on Linux.

## Features

- **Single file, zero dependencies** — stdlib only, compiles on any platform
- **Hardware access** — I2C bus scanning, register read/write; SPI transfer, loopback testing (Linux)
- **ReAct agent loop** — reasoning + acting with configurable iteration limits
- **Sliding-window context** — bounded memory usage with configurable window size
- **Safety filters** — blocks destructive commands (rm -rf /, dd, fork bombs, etc.)
- **REPL + one-shot modes** — interactive or scriptable

## Install

```bash
go install github.com/onusrat/attoclaw@latest
```

Or build from source:

```bash
git clone https://github.com/onusrat/attoclaw.git
cd attoclaw
make build
```

## Quick Start

```bash
export ATTOCLAW_API_KEY=sk-...
./attoclaw                              # interactive REPL
./attoclaw -m "list files in /tmp"      # one-shot mode
```

## Usage

### REPL Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/tools` | List available tools |
| `/status` | Show agent statistics |
| `/config` | Show current configuration |
| `/history` | Show recent messages |
| `/clear` | Clear conversation history |
| `/quit` | Exit |

### Keys

- `Ctrl+C` — cancel current operation
- `Ctrl+D` — exit REPL

## Tools

| Tool | Description |
|------|-------------|
| `exec` | Execute shell commands (with safety filter and 60s timeout) |
| `read_file` | Read file contents (up to 10MB, supports offset/limit) |
| `write_file` | Write or append to files |
| `list_dir` | List directory contents with types and sizes |
| `edit_file` | Replace exact string matches in files |
| `search_files` | Find files matching glob patterns |
| `system_info` | Get OS, arch, hostname, CPU count, env vars |
| `i2c` | I2C bus detect, device scan, read/write, register ops (Linux) |
| `spi` | SPI device list, transfer, read, info, loopback test (Linux) |

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ATTOCLAW_API_KEY` | API key (required) | — |
| `ATTOCLAW_API_BASE` | API endpoint | `https://api.openai.com/v1` |
| `ATTOCLAW_MODEL` | LLM model | `gpt-4o` |
| `ATTOCLAW_DEBUG` | Enable debug logging | `false` |

### Config File

`~/.attoclaw.json` — supports all options. Priority: env vars > config file > defaults.

```json
{
  "api_key": "sk-...",
  "model": "gpt-4o",
  "max_iterations": 20,
  "session_window": 50,
  "exec_timeout_seconds": 60
}
```

## Building

```bash
make build          # current platform
make build-all      # linux/amd64, linux/arm64, linux/riscv64, darwin/arm64
make install        # copy to ~/.local/bin
make vet            # run go vet
make fmt            # run go fmt
make clean          # remove artifacts
```

## Hardware Access

AttoClaw includes direct I2C and SPI tools that work on Linux via ioctl system calls:

**I2C** — detect buses, scan for devices (hybrid SMBus quick write + read byte), read/write raw bytes, register-level operations. Example: `"scan the I2C bus and read temperature from the sensor at 0x48"`

**SPI** — list devices, full-duplex transfer, read-only mode, device info, loopback testing with proper struct alignment. Example: `"do a loopback test on /dev/spidev0.0"`

Hardware tools are compiled on all platforms but runtime-check for Linux before executing.

## License

MIT
