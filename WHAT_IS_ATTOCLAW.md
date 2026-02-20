# What is AttoClaw?

AttoClaw is the smallest member of the Claw family of AI agents. OpenClaw (430K lines of TypeScript) started it, NanoClaw cut it to 500 lines, PicoClaw rewrote it in Go across 121 files — and AttoClaw compresses the whole thing into a single 2,912-line Go file with zero dependencies.

It's a complete ReAct agent: it connects to any OpenAI-compatible API, has 9 built-in tools (filesystem, shell, system info, I2C, SPI), manages sessions, retries with backoff, and runs as either an interactive REPL or a one-shot CLI. It compiles to a 5.6MB static binary, starts in under a second, and uses less than 10MB of RAM — matching PicoClaw's headline numbers with a fraction of the code.

The one thing it does that none of the others shipped: direct hardware access via raw Linux ioctl calls. You can type English and it scans an I2C bus or reads a temperature sensor. No Python scripts, no wrappers, no dependencies.
