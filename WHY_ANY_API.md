# Why "any OpenAI-compatible API" is a strength, not a weakness

"OpenAI-compatible API" doesn't mean "OpenAI only." It's become the de facto standard. Almost every LLM provider exposes this same REST interface:

- **Ollama** (local models) — `http://localhost:11434/v1`
- **LM Studio** — `http://localhost:1234/v1`
- **Groq**, **Together AI**, **Fireworks** — all OpenAI-compatible
- **Anthropic** has compatible proxies
- **vLLM**, **llama.cpp server** — self-hosted, same API

So that one HTTP client in AttoClaw isn't a lock-in to OpenAI — it's a universal connector. You set `ATTOCLAW_API_BASE` to point at whatever you want. Run Llama 3 locally on the same Pi that's reading the I2C bus, and you have a fully offline, air-gapped AI agent talking to hardware. No cloud needed.

That's actually a better story for the embedded/hardware audience. The people who care about ioctl-level I2C access on a Raspberry Pi are exactly the people who'd want to run a local model instead of sending their sensor data to OpenAI's servers.

If anything, it's an under-marketed feature. The README says "any OpenAI-compatible API" in passing, but it could say "runs fully offline with Ollama" — that's a different pitch entirely.
