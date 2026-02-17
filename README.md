# AttoClaw

The smallest AI agent that talks to hardware. One Go file, zero dependencies, direct I2C/SPI access on Linux.

## Quick Start

```bash
export ATTOCLAW_API_KEY=sk-...
./attoclaw -m "what's my system info?"
```

> **Note:** Output is illustrative; actual format depends on the LLM response.

```
[tool: system_info]
## OS Information
OS:         linux
Arch:       arm64
Hostname:   rpi4
NumCPU:     4
...

The system is running linux/arm64 with 4 CPUs on host "rpi4".
```

## Hardware Example (Raspberry Pi + TMP102)

On a Raspberry Pi with a TMP102 temperature sensor connected to the I2C bus:

```
$ ./attoclaw
AttoClaw v0.1.0 | gpt-4o | linux/arm64
Hardware: I2C: /dev/i2c-1 | SPI: /dev/spidev0.0
API: https://api.openai.com/v1 (key: sk-a...XXXX)
Tools: 9 registered

> scan the I2C bus and read temperature from 0x48
[tool: i2c]
I2C bus scan: /dev/i2c-1

     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:          -- -- -- -- -- -- -- -- -- -- -- -- -- --
40:  -- -- -- -- -- -- -- -- 48 -- -- -- -- -- -- --

Found device at 0x48. Reading register 0x00 (2 bytes)...
[tool: i2c]
Temperature: 0x01 0x94 → 25.25°C

The sensor at 0x48 is a TMP102. Current temperature is 25.25°C.
```

## Dry Run Mode

Preview hardware writes without touching any device:

```bash
./attoclaw --dry-run -m "write 0xFF to I2C address 0x48"
```

```
[tool: i2c]
[dry-run] would write 1 byte to bus 1 addr 0x48: ff
```

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

## Try It

```bash
./attoclaw                              # interactive REPL
./attoclaw -m "list files in /tmp"      # one-shot mode
./attoclaw -m "what's my system info?"  # quick check
./attoclaw --dry-run                    # hardware writes describe instead of execute
```

## Why This Exists

Debugging I2C sensors means writing throwaway Python scripts, memorizing `i2cdetect` flags, and converting hex in your head. SPI is worse. Every new board means starting from scratch.

AttoClaw lets you talk to hardware in plain English. It knows I2C and SPI at the ioctl level — bus scanning, register reads, full-duplex transfers — and it runs as a single static binary with zero dependencies. Drop it on a Raspberry Pi, a BeagleBone, or a RISC-V board and go.

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

## Hardware Safety

Hardware writes are **blocked by default**. You must explicitly allowlist which devices the agent can write to:

```json
{
  "allowed_i2c_write_addrs": [72, 76],
  "allowed_spi_devices": ["/dev/spidev0.0"],
  "hardware_dry_run": false
}
```

- **I2C reads** are always allowed (reads don't change device state on well-behaved hardware)
- **I2C writes** require the target address to be in `allowed_i2c_write_addrs` (0x03-0x77)
- **SPI transfers** require the device path to be in `allowed_spi_devices`
- **Dry-run mode** (`--dry-run` flag or `"hardware_dry_run": true`) makes writes return a description of what *would* happen without touching hardware

The exec tool also blocks destructive commands: `rm -rf /`, `dd`, fork bombs, `shutdown`, `mkfs`, and writes to block devices.

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
  "exec_timeout_seconds": 60,
  "allowed_i2c_write_addrs": [72, 76],
  "allowed_spi_devices": ["/dev/spidev0.0"]
}
```

## Building

```bash
make build          # current platform
make build-all      # linux/amd64, linux/arm64, linux/riscv64, darwin/arm64
make install        # copy to ~/.local/bin
make vet            # run go vet
make fmt            # run go fmt
make test           # run tests
make clean          # remove artifacts
```

## License

MIT
