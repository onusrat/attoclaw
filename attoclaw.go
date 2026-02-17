// AttoClaw — Single-file AI agent with I2C/SPI hardware access.
// Zero dependencies. Stdlib only. Compiles everywhere, hardware access on Linux.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

const (
	DefaultModel          = "gpt-4o"
	DefaultAPIBase        = "https://api.openai.com/v1"
	DefaultMaxIterations  = 20
	DefaultSessionWindow  = 50
	DefaultHTTPTimeout    = 120 * time.Second // generous for complex tool-use responses
	DefaultExecTimeout    = 60 * time.Second
	DefaultMaxRetries     = 3
	DefaultRetryBaseDelay = 1 * time.Second
	DefaultMaxOutputSize  = 100000

	EnvAPIKey  = "ATTOCLAW_API_KEY"
	EnvAPIBase = "ATTOCLAW_API_BASE"
	EnvModel   = "ATTOCLAW_MODEL"
	EnvDebug   = "ATTOCLAW_DEBUG"

	ConfigFileName    = ".attoclaw.json"
	MaxReadFileSize   = 10 * 1024 * 1024
	MaxExecOutputSize = 1024 * 1024
)

var (
	Version    = "0.1.0"
	CommitHash = "dev"
	BuildDate  = "unknown"
)

type LogLevel int

const (
	LogDebug LogLevel = iota
	LogInfo
	LogWarn
	LogError
)

type Logger struct {
	level  LogLevel
	prefix string
}

var globalLogger = &Logger{level: LogInfo, prefix: "attoclaw"}

func (l *Logger) SetDebug(enabled bool) {
	if enabled {
		l.level = LogDebug
	} else {
		l.level = LogInfo
	}
}

func (l *Logger) Debug(format string, args ...interface{}) {
	if l.level <= LogDebug {
		msg := fmt.Sprintf(format, args...)
		fmt.Fprintf(os.Stderr, "[%s][DEBUG] %s\n", l.prefix, msg)
	}
}

func (l *Logger) Info(format string, args ...interface{}) {
	if l.level <= LogInfo {
		msg := fmt.Sprintf(format, args...)
		fmt.Fprintf(os.Stderr, "[%s] %s\n", l.prefix, msg)
	}
}

func (l *Logger) Warn(format string, args ...interface{}) {
	if l.level <= LogWarn {
		msg := fmt.Sprintf(format, args...)
		fmt.Fprintf(os.Stderr, "[%s][WARN] %s\n", l.prefix, msg)
	}
}

func (l *Logger) Error(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	fmt.Fprintf(os.Stderr, "[%s][ERROR] %s\n", l.prefix, msg)
}

type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type ToolCall struct {
	ID       string        `json:"id"`
	Type     string        `json:"type"`
	Function *FunctionCall `json:"function,omitempty"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type LLMResponse struct {
	Content      string     `json:"content"`
	ToolCalls    []ToolCall `json:"tool_calls,omitempty"`
	FinishReason string     `json:"finish_reason"`
	Usage        *UsageInfo `json:"usage,omitempty"`
}

type UsageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type ToolDefinition struct {
	Type     string          `json:"type"`
	Function ToolDefFunction `json:"function"`
}

type ToolDefFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

type AgentStats struct {
	TotalRequests     int
	TotalToolCalls    int
	TotalTokensUsed   int
	TotalPromptTokens int
	TotalCompTokens   int
	TotalErrors       int
	SessionStartTime  time.Time
	LastRequestTime   time.Time
	LastRequestDur    time.Duration
	ToolCallCounts    map[string]int
}

type Config struct {
	APIKey             string   `json:"api_key,omitempty"`
	APIBase            string   `json:"api_base,omitempty"`
	Model              string   `json:"model,omitempty"`
	MaxIterations      int      `json:"max_iterations,omitempty"`
	SessionWindow      int      `json:"session_window,omitempty"`
	Debug              bool     `json:"debug,omitempty"`
	MaxRetries         int      `json:"max_retries,omitempty"`
	RetryBaseDelayMs   int      `json:"retry_base_delay_ms,omitempty"`
	ExecTimeoutSeconds int      `json:"exec_timeout_seconds,omitempty"`
	HTTPTimeoutSeconds int      `json:"http_timeout_seconds,omitempty"`
	MaxOutputSize      int      `json:"max_output_size,omitempty"`
	AllowedI2CAddrs    []int    `json:"allowed_i2c_write_addrs,omitempty"`
	AllowedSPIDevices  []string `json:"allowed_spi_devices,omitempty"`
	HardwareDryRun     bool     `json:"hardware_dry_run,omitempty"`
}

func LoadConfig() (*Config, error) {
	cfg := &Config{
		APIBase:            DefaultAPIBase,
		Model:              DefaultModel,
		MaxIterations:      DefaultMaxIterations,
		SessionWindow:      DefaultSessionWindow,
		MaxRetries:         DefaultMaxRetries,
		RetryBaseDelayMs:   int(DefaultRetryBaseDelay / time.Millisecond),
		ExecTimeoutSeconds: int(DefaultExecTimeout / time.Second),
		HTTPTimeoutSeconds: int(DefaultHTTPTimeout / time.Second),
		MaxOutputSize:      DefaultMaxOutputSize,
	}

	if home, err := os.UserHomeDir(); err == nil {
		cfgPath := filepath.Join(home, ConfigFileName)
		if data, err := os.ReadFile(cfgPath); err == nil {
			globalLogger.Debug("loading config from %s", cfgPath)
			var fileCfg Config
			if err := json.Unmarshal(data, &fileCfg); err == nil {
				mergeConfig(cfg, &fileCfg)
			} else {
				globalLogger.Warn("failed to parse config file %s: %v", cfgPath, err)
			}
		}
	}

	if v := os.Getenv(EnvAPIKey); v != "" {
		cfg.APIKey = v
	}
	if v := os.Getenv(EnvAPIBase); v != "" {
		cfg.APIBase = v
	}
	if v := os.Getenv(EnvModel); v != "" {
		cfg.Model = v
	}
	if v := os.Getenv(EnvDebug); v == "1" || strings.ToLower(v) == "true" {
		cfg.Debug = true
	}

	if cfg.Debug {
		globalLogger.SetDebug(true)
		globalLogger.Debug("debug logging enabled")
	}

	if cfg.APIKey == "" {
		return nil, fmt.Errorf("API key not set. Set %s env var or api_key in ~/%s", EnvAPIKey, ConfigFileName)
	}

	if cfg.MaxIterations < 1 {
		cfg.MaxIterations = DefaultMaxIterations
	}
	if cfg.MaxIterations > 100 {
		globalLogger.Warn("max_iterations=%d is very high, capping at 100", cfg.MaxIterations)
		cfg.MaxIterations = 100
	}
	if cfg.SessionWindow < 5 {
		cfg.SessionWindow = 5
	}
	if cfg.MaxRetries < 0 {
		cfg.MaxRetries = 0
	}
	if cfg.MaxRetries > 10 {
		cfg.MaxRetries = 10
	}
	if cfg.ExecTimeoutSeconds < 1 {
		cfg.ExecTimeoutSeconds = int(DefaultExecTimeout / time.Second)
	}
	if cfg.HTTPTimeoutSeconds < 10 {
		cfg.HTTPTimeoutSeconds = int(DefaultHTTPTimeout / time.Second)
	}
	if cfg.MaxOutputSize < 1000 {
		cfg.MaxOutputSize = DefaultMaxOutputSize
	}

	for _, addr := range cfg.AllowedI2CAddrs {
		if addr < 0x03 || addr > 0x77 {
			return nil, fmt.Errorf("invalid I2C address 0x%02x in allowed_i2c_write_addrs (must be 0x03-0x77)", addr)
		}
	}

	cfg.APIBase = strings.TrimRight(cfg.APIBase, "/")

	globalLogger.Debug("config loaded: model=%s api_base=%s max_iter=%d window=%d retries=%d",
		cfg.Model, cfg.APIBase, cfg.MaxIterations, cfg.SessionWindow, cfg.MaxRetries)

	return cfg, nil
}

func (c *Config) ExecTimeout() time.Duration {
	return time.Duration(c.ExecTimeoutSeconds) * time.Second
}

func (c *Config) HTTPTimeout() time.Duration {
	return time.Duration(c.HTTPTimeoutSeconds) * time.Second
}

func (c *Config) RetryBaseDelay() time.Duration {
	return time.Duration(c.RetryBaseDelayMs) * time.Millisecond
}

func (c *Config) MaskedAPIKey() string {
	if len(c.APIKey) <= 8 {
		return "***"
	}
	return c.APIKey[:4] + "..." + c.APIKey[len(c.APIKey)-4:]
}

func (c *Config) IsI2CWriteAllowed(addr int) bool {
	for _, a := range c.AllowedI2CAddrs {
		if a == addr {
			return true
		}
	}
	return false
}

func (c *Config) IsSPIDeviceAllowed(device string) bool {
	for _, d := range c.AllowedSPIDevices {
		if d == device {
			return true
		}
	}
	return false
}

func mergeConfig(dst, src *Config) {
	if src.APIKey != "" {
		dst.APIKey = src.APIKey
	}
	if src.APIBase != "" {
		dst.APIBase = src.APIBase
	}
	if src.Model != "" {
		dst.Model = src.Model
	}
	if src.MaxIterations > 0 {
		dst.MaxIterations = src.MaxIterations
	}
	if src.SessionWindow > 0 {
		dst.SessionWindow = src.SessionWindow
	}
	if src.Debug {
		dst.Debug = true
	}
	if src.MaxRetries > 0 {
		dst.MaxRetries = src.MaxRetries
	}
	if src.RetryBaseDelayMs > 0 {
		dst.RetryBaseDelayMs = src.RetryBaseDelayMs
	}
	if src.ExecTimeoutSeconds > 0 {
		dst.ExecTimeoutSeconds = src.ExecTimeoutSeconds
	}
	if src.HTTPTimeoutSeconds > 0 {
		dst.HTTPTimeoutSeconds = src.HTTPTimeoutSeconds
	}
	if src.MaxOutputSize > 0 {
		dst.MaxOutputSize = src.MaxOutputSize
	}
	if src.AllowedI2CAddrs != nil {
		dst.AllowedI2CAddrs = src.AllowedI2CAddrs
	}
	if src.AllowedSPIDevices != nil {
		dst.AllowedSPIDevices = src.AllowedSPIDevices
	}
	if src.HardwareDryRun {
		dst.HardwareDryRun = true
	}
}

type Tool interface {
	Name() string
	Description() string
	Parameters() map[string]interface{}
	Execute(ctx context.Context, args map[string]interface{}) *ToolResult
}

type ToolResult struct {
	ForLLM  string
	ForUser string
	IsError bool
}

func NewToolResult(forLLM string) *ToolResult {
	return &ToolResult{ForLLM: forLLM}
}

func ErrorResult(msg string) *ToolResult {
	return &ToolResult{ForLLM: msg, IsError: true}
}

func ErrorResultf(format string, args ...interface{}) *ToolResult {
	return &ToolResult{ForLLM: fmt.Sprintf(format, args...), IsError: true}
}

func getStringArg(args map[string]interface{}, key string) string {
	if args == nil {
		return ""
	}
	if v, ok := args[key]; ok {
		switch s := v.(type) {
		case string:
			return s
		case float64:
			// LLM sometimes produces numbers where strings are expected
			return strconv.FormatFloat(s, 'f', -1, 64)
		case bool:
			return strconv.FormatBool(s)
		}
	}
	return ""
}

func getBoolArg(args map[string]interface{}, key string) bool {
	if args == nil {
		return false
	}
	if v, ok := args[key]; ok {
		switch b := v.(type) {
		case bool:
			return b
		case string:
			lower := strings.ToLower(b)
			return lower == "true" || lower == "1" || lower == "yes"
		case float64:
			return b != 0
		}
	}
	return false
}

func getIntArg(args map[string]interface{}, key string, defaultVal int) int {
	if args == nil {
		return defaultVal
	}
	if v, ok := args[key]; ok {
		switch n := v.(type) {
		case float64:
			return int(n)
		case int:
			return n
		case int64:
			return int(n)
		case string:
			if i, err := strconv.Atoi(n); err == nil {
				return i
			}
			// Try parsing as float then converting (e.g. "1.0")
			if f, err := strconv.ParseFloat(n, 64); err == nil {
				return int(f)
			}
		}
	}
	return defaultVal
}

func requireStringArg(args map[string]interface{}, key string) (string, *ToolResult) {
	v := getStringArg(args, key)
	if v == "" {
		return "", ErrorResultf("%s argument is required", key)
	}
	return v, nil
}

func requireIntArg(args map[string]interface{}, key string) (int, *ToolResult) {
	v := getIntArg(args, key, -1)
	if v < 0 {
		return 0, ErrorResultf("%s argument is required", key)
	}
	return v, nil
}

type ToolRegistry struct {
	tools []Tool
}

func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{tools: make([]Tool, 0, 16)}
}

func (r *ToolRegistry) Register(t Tool) {
	r.tools = append(r.tools, t)
	globalLogger.Debug("registered tool: %s", t.Name())
}

func (r *ToolRegistry) Get(name string) Tool {
	for _, t := range r.tools {
		if t.Name() == name {
			return t
		}
	}
	return nil
}

func (r *ToolRegistry) Has(name string) bool {
	return r.Get(name) != nil
}

func (r *ToolRegistry) Count() int {
	return len(r.tools)
}

func (r *ToolRegistry) Execute(ctx context.Context, name string, args map[string]interface{}) *ToolResult {
	t := r.Get(name)
	if t == nil {
		available := strings.Join(r.Names(), ", ")
		return ErrorResult(fmt.Sprintf("unknown tool: %s (available: %s)", name, available))
	}
	return t.Execute(ctx, args)
}

func (r *ToolRegistry) Definitions() []ToolDefinition {
	defs := make([]ToolDefinition, len(r.tools))
	for i, t := range r.tools {
		defs[i] = ToolDefinition{
			Type: "function",
			Function: ToolDefFunction{
				Name:        t.Name(),
				Description: t.Description(),
				Parameters:  t.Parameters(),
			},
		}
	}
	return defs
}

func (r *ToolRegistry) Names() []string {
	names := make([]string, len(r.tools))
	for i, t := range r.tools {
		names[i] = t.Name()
	}
	return names
}

type HTTPProvider struct {
	apiBase        string
	apiKey         string
	model          string
	client         *http.Client
	maxRetries     int
	retryBaseDelay time.Duration
}

func NewHTTPProvider(cfg *Config) *HTTPProvider {
	return &HTTPProvider{
		apiBase: cfg.APIBase,
		apiKey:  cfg.APIKey,
		model:   cfg.Model,
		client: &http.Client{
			Timeout: cfg.HTTPTimeout(),
		},
		maxRetries:     cfg.MaxRetries,
		retryBaseDelay: cfg.RetryBaseDelay(),
	}
}

type chatCompletionRequest struct {
	Model    string           `json:"model"`
	Messages []Message        `json:"messages"`
	Tools    []ToolDefinition `json:"tools,omitempty"`
}

type chatCompletionResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int     `json:"index"`
		Message      Message `json:"message"`
		FinishReason string  `json:"finish_reason"`
	} `json:"choices"`
	Usage *UsageInfo `json:"usage,omitempty"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error,omitempty"`
}

func isRetryableStatusCode(code int) bool {
	return code == 429 || (code >= 500 && code < 600)
}

type apiError struct {
	StatusCode int
	Retryable  bool
	Body       string
}

func (e *apiError) Error() string {
	return fmt.Sprintf("API returned status %d: %s", e.StatusCode, e.Body)
}

func (p *HTTPProvider) Call(ctx context.Context, messages []Message, tools []ToolDefinition) (*LLMResponse, error) {
	reqBody := chatCompletionRequest{
		Model:    p.model,
		Messages: messages,
	}
	if len(tools) > 0 {
		reqBody.Tools = tools
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	globalLogger.Debug("request: %d messages, %d tools, %d bytes",
		len(messages), len(tools), len(bodyBytes))

	url := p.apiBase + "/chat/completions"
	var lastErr error
	for attempt := 0; attempt <= p.maxRetries; attempt++ {
		if attempt > 0 {
			// Exponential backoff: baseDelay * 2^(attempt-1)
			delay := p.retryBaseDelay * time.Duration(1<<uint(attempt-1))
			globalLogger.Info("retrying request (attempt %d/%d) after %v...",
				attempt+1, p.maxRetries+1, delay)

			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}

		resp, err := p.doRequest(ctx, url, bodyBytes)
		if err != nil {
			lastErr = err
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			// Only retry errors explicitly marked as retryable (429, 5xx)
			var ae *apiError
			if errors.As(err, &ae) && !ae.Retryable {
				return nil, err
			}
			globalLogger.Debug("request attempt %d failed: %v", attempt+1, err)
			continue
		}

		return resp, nil
	}

	return nil, fmt.Errorf("all %d attempts failed, last error: %w", p.maxRetries+1, lastErr)
}

func (p *HTTPProvider) doRequest(ctx context.Context, url string, bodyBytes []byte) (*LLMResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.apiKey)
	req.Header.Set("User-Agent", "AttoClaw/"+Version)

	startTime := time.Now()
	resp, err := p.client.Do(req)
	elapsed := time.Since(startTime)

	if err != nil {
		return nil, fmt.Errorf("HTTP request failed after %v: %w", elapsed, err)
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	globalLogger.Debug("response: status=%d size=%d bytes time=%v",
		resp.StatusCode, len(respBytes), elapsed)

	if resp.StatusCode != http.StatusOK {
		return nil, &apiError{
			StatusCode: resp.StatusCode,
			Retryable:  isRetryableStatusCode(resp.StatusCode),
			Body:       truncate(string(respBytes), 500),
		}
	}

	var chatResp chatCompletionResponse
	if err := json.Unmarshal(respBytes, &chatResp); err != nil {
		return nil, fmt.Errorf("failed to parse response JSON (%d bytes): %w", len(respBytes), err)
	}

	if chatResp.Error != nil {
		return nil, fmt.Errorf("API error (type=%s, code=%s): %s",
			chatResp.Error.Type, chatResp.Error.Code, chatResp.Error.Message)
	}

	if len(chatResp.Choices) == 0 {
		return nil, fmt.Errorf("API returned no choices (model=%s)", chatResp.Model)
	}

	choice := chatResp.Choices[0]

	globalLogger.Debug("finish_reason=%s tool_calls=%d content_len=%d",
		choice.FinishReason, len(choice.Message.ToolCalls), len(choice.Message.Content))

	llmResp := &LLMResponse{
		Content:      choice.Message.Content,
		ToolCalls:    choice.Message.ToolCalls,
		FinishReason: choice.FinishReason,
		Usage:        chatResp.Usage,
	}

	return llmResp, nil
}

type Session struct {
	messages     []Message
	window       int
	totalAdded   int
	totalDropped int
}

func NewSession(window int) *Session {
	return &Session{
		messages: make([]Message, 0, window+10),
		window:   window,
	}
}

func (s *Session) Add(msg Message) {
	s.messages = append(s.messages, msg)
	s.totalAdded++
	s.trim()
}

func (s *Session) AddAll(msgs []Message) {
	s.messages = append(s.messages, msgs...)
	s.totalAdded += len(msgs)
	s.trim()
}

func (s *Session) trim() {
	if len(s.messages) <= s.window {
		return
	}
	excess := len(s.messages) - s.window
	// Try to find a clean break point (start of a user message)
	cutPoint := excess
	for i := excess; i < len(s.messages) && i < excess+5; i++ {
		if s.messages[i].Role == "user" {
			cutPoint = i
			break
		}
	}
	s.totalDropped += cutPoint
	s.messages = s.messages[cutPoint:]
}

func (s *Session) Get() []Message {
	return s.messages
}

func (s *Session) Clear() {
	s.messages = s.messages[:0]
	globalLogger.Debug("session cleared")
}

func (s *Session) Len() int {
	return len(s.messages)
}

func (s *Session) Stats() string {
	roleCounts := make(map[string]int)
	for _, m := range s.messages {
		roleCounts[m.Role]++
	}
	return fmt.Sprintf("messages=%d (user=%d assistant=%d tool=%d) total_added=%d dropped=%d window=%d",
		len(s.messages), roleCounts["user"], roleCounts["assistant"], roleCounts["tool"],
		s.totalAdded, s.totalDropped, s.window)
}

func BuildSystemPrompt(registry *ToolRegistry) string {
	var b strings.Builder

	b.WriteString("You are AttoClaw v")
	b.WriteString(Version)
	b.WriteString(", a lightweight AI agent that can interact with the local system ")
	b.WriteString("and physical hardware. You run as a single Go binary with zero external dependencies.\n\n")

	b.WriteString("## Capabilities\n" +
		"- Execute shell commands with safety filtering\n" +
		"- Read, write, edit, and list files and directories\n" +
		"- Search for files by name pattern\n" +
		"- Access I2C/SPI devices for hardware communication (Linux only)\n" +
		"- Gather system information\n\n")

	hostname, _ := os.Hostname()
	cwd, _ := os.Getwd()
	b.WriteString("## Environment\n")
	b.WriteString(fmt.Sprintf("- OS: %s\n", runtime.GOOS))
	b.WriteString(fmt.Sprintf("- Arch: %s\n", runtime.GOARCH))
	b.WriteString(fmt.Sprintf("- CPUs: %d\n", runtime.NumCPU()))
	b.WriteString(fmt.Sprintf("- Hostname: %s\n", hostname))
	b.WriteString(fmt.Sprintf("- Working directory: %s\n", cwd))
	b.WriteString(fmt.Sprintf("- Go version: %s\n", runtime.Version()))
	b.WriteString(fmt.Sprintf("- Time: %s\n", time.Now().Format(time.RFC3339)))

	if user := os.Getenv("USER"); user != "" {
		b.WriteString(fmt.Sprintf("- User: %s\n", user))
	} else if user := os.Getenv("USERNAME"); user != "" {
		b.WriteString(fmt.Sprintf("- User: %s\n", user))
	}

	if shell := os.Getenv("SHELL"); shell != "" {
		b.WriteString(fmt.Sprintf("- Shell: %s\n", shell))
	}

	b.WriteString("\n")

	b.WriteString("## Available Tools\n")
	for _, name := range registry.Names() {
		t := registry.Get(name)
		b.WriteString(fmt.Sprintf("- **%s**: %s\n", name, t.Description()))
	}
	b.WriteString("\n")

	b.WriteString("## Safety Guidelines\n" +
		"- Always explain what you are about to do before calling a tool.\n" +
		"- The exec tool blocks dangerous commands (rm -rf /, dd, fork bombs, shutdown, mkfs).\n" +
		"- I2C writes restricted by allowed_i2c_write_addrs config (empty = all blocked).\n" +
		"- SPI transfers restricted by allowed_spi_devices config (empty = all blocked).\n" +
		"- When hardware_dry_run is enabled, writes describe what would happen without executing.\n" +
		"- Never reveal API keys, passwords, or sensitive data in responses.\n\n")
	b.WriteString("## Behavioral Guidelines\n" +
		"- Keep responses concise and actionable.\n" +
		"- If a tool returns an error, explain it clearly and suggest next steps.\n" +
		"- For multi-step tasks, explain your plan before starting.\n" +
		"- If unsure about something, ask the user rather than guessing.\n" +
		"- When working with hardware (I2C/SPI), explain what each operation does.\n" +
		"- Proceed step by step and report intermediate results.\n")

	if runtime.GOOS == "linux" {
		b.WriteString("\n## Hardware Notes (Linux)\n")
		b.WriteString("- I2C buses are available at /dev/i2c-N (requires i2c-dev module or permissions)\n")
		b.WriteString("- SPI devices are available at /dev/spidevB.C (bus B, chip select C)\n")
		b.WriteString("- You may need root permissions or appropriate group membership for hardware access\n")
		b.WriteString("- Use 'i2c detect' and 'spi list' first to discover available buses/devices\n")
		b.WriteString("- Always scan an I2C bus before reading/writing to discover connected devices\n")
		b.WriteString("- Common I2C devices: sensors (0x76/0x77 BME280), EEPROMs (0x50-0x57), ")
		b.WriteString("displays (0x3C/0x3D SSD1306), RTC (0x68 DS3231)\n")
	}

	return b.String()
}

type Agent struct {
	config   *Config
	provider *HTTPProvider
	registry *ToolRegistry
	session  *Session
	stats    *AgentStats
	output   io.Writer
}

func NewAgent(cfg *Config, provider *HTTPProvider, registry *ToolRegistry) *Agent {
	return &Agent{
		config:   cfg,
		provider: provider,
		registry: registry,
		session:  NewSession(cfg.SessionWindow),
		stats: &AgentStats{
			SessionStartTime: time.Now(),
			ToolCallCounts:   make(map[string]int),
		},
		output: os.Stderr,
	}
}

func (a *Agent) Run(ctx context.Context, userInput string) (string, error) {
	a.session.Add(Message{Role: "user", Content: userInput})

	globalLogger.Debug("starting agent loop for input: %s", truncate(userInput, 100))

	systemPrompt := BuildSystemPrompt(a.registry)
	toolDefs := a.registry.Definitions()

	for iteration := 0; iteration < a.config.MaxIterations; iteration++ {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}

		globalLogger.Debug("iteration %d/%d", iteration+1, a.config.MaxIterations)

		sessionMsgs := a.session.Get()
		messages := make([]Message, 0, len(sessionMsgs)+1)
		messages = append(messages, Message{Role: "system", Content: systemPrompt})
		messages = append(messages, sessionMsgs...)
		startTime := time.Now()
		resp, err := a.provider.Call(ctx, messages, toolDefs)
		elapsed := time.Since(startTime)

		a.stats.TotalRequests++
		a.stats.LastRequestTime = startTime
		a.stats.LastRequestDur = elapsed

		if err != nil {
			a.stats.TotalErrors++
			return "", fmt.Errorf("LLM call failed (iteration %d): %w", iteration+1, err)
		}

		if resp.Usage != nil {
			a.stats.TotalTokensUsed += resp.Usage.TotalTokens
			a.stats.TotalPromptTokens += resp.Usage.PromptTokens
			a.stats.TotalCompTokens += resp.Usage.CompletionTokens
			fmt.Fprintf(a.output, "[tokens: prompt=%d completion=%d total=%d | time=%v]\n",
				resp.Usage.PromptTokens, resp.Usage.CompletionTokens,
				resp.Usage.TotalTokens, elapsed.Round(time.Millisecond))
		}

		if len(resp.ToolCalls) == 0 {
			assistantMsg := Message{Role: "assistant", Content: resp.Content}
			a.session.Add(assistantMsg)
			globalLogger.Debug("agent loop complete after %d iterations", iteration+1)
			return resp.Content, nil
		}

		assistantMsg := Message{
			Role:      "assistant",
			Content:   resp.Content,
			ToolCalls: resp.ToolCalls,
		}
		a.session.Add(assistantMsg)

		if resp.Content != "" {
			fmt.Println(resp.Content)
		}

		toolMessages, err := a.executeToolCalls(ctx, resp.ToolCalls)
		if err != nil {
			return "", err
		}

		a.session.AddAll(toolMessages)
	}

	return "", fmt.Errorf("agent exceeded maximum iterations (%d); use /clear to reset or increase max_iterations",
		a.config.MaxIterations)
}

func (a *Agent) executeToolCalls(ctx context.Context, toolCalls []ToolCall) ([]Message, error) {
	var toolMessages []Message

	for _, tc := range toolCalls {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		if tc.Function == nil {
			globalLogger.Warn("tool call %s has no function", tc.ID)
			toolMessages = append(toolMessages, Message{
				Role:       "tool",
				Content:    "error: tool call has no function definition",
				ToolCallID: tc.ID,
			})
			continue
		}

		toolName := tc.Function.Name
		fmt.Fprintf(a.output, "[tool: %s]\n", toolName)
		globalLogger.Debug("executing tool %s with args: %s", toolName,
			truncate(tc.Function.Arguments, 200))

		args, err := parseToolArgs(tc.Function.Arguments)
		if err != nil {
			a.stats.TotalErrors++
			toolMessages = append(toolMessages, Message{
				Role:       "tool",
				Content:    fmt.Sprintf("error parsing tool arguments: %v", err),
				ToolCallID: tc.ID,
			})
			continue
		}

		startTime := time.Now()
		result := a.registry.Execute(ctx, toolName, args)
		elapsed := time.Since(startTime)

		a.stats.TotalToolCalls++
		a.stats.ToolCallCounts[toolName]++
		if result.IsError {
			a.stats.TotalErrors++
		}

		toolOutput := result.ForLLM
		if len(toolOutput) > a.config.MaxOutputSize {
			truncatedNote := fmt.Sprintf("\n\n[output truncated: %d bytes total, showing first %d bytes]",
				len(toolOutput), a.config.MaxOutputSize)
			toolOutput = toolOutput[:a.config.MaxOutputSize] + truncatedNote
		}

		display := toolOutput
		if result.ForUser != "" {
			display = result.ForUser
		}
		if result.IsError {
			fmt.Fprintf(a.output, "[error (%v): %s]\n", elapsed.Round(time.Millisecond),
				truncate(display, 200))
		} else {
			preview := truncate(display, 500)
			if preview != "" {
				fmt.Fprintf(a.output, "[result (%v): %s]\n", elapsed.Round(time.Millisecond), preview)
			}
		}

		toolMessages = append(toolMessages, Message{
			Role:       "tool",
			Content:    toolOutput,
			ToolCallID: tc.ID,
		})
	}

	return toolMessages, nil
}

func parseToolArgs(argsJSON string) (map[string]interface{}, error) {
	if argsJSON == "" || argsJSON == "{}" {
		return make(map[string]interface{}), nil
	}

	var args map[string]interface{}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return nil, fmt.Errorf("invalid JSON arguments: %w (raw: %s)", err, truncate(argsJSON, 200))
	}
	if args == nil {
		args = make(map[string]interface{})
	}
	return args, nil
}

func (a *Agent) ClearSession() {
	a.session.Clear()
}

func (a *Agent) GetStats() *AgentStats {
	return a.stats
}

func (a *Agent) GetSession() *Session {
	return a.session
}

func truncate(s string, maxLen int) string {
	s = strings.TrimSpace(s)
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

type ExecTool struct {
	timeout time.Duration
}

func NewExecTool(timeout time.Duration) *ExecTool {
	return &ExecTool{timeout: timeout}
}

func (t *ExecTool) Name() string { return "exec" }

func (t *ExecTool) Description() string {
	return "Execute a shell command and return its output. Dangerous commands " +
		"(rm -rf /, dd, fork bombs, shutdown, mkfs, etc.) are blocked by a safety filter. " +
		"Commands have a timeout of " + t.timeout.String() + "."
}

func (t *ExecTool) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"command": map[string]interface{}{
				"type":        "string",
				"description": "The shell command to execute via /bin/sh -c",
			},
			"working_dir": map[string]interface{}{
				"type":        "string",
				"description": "Working directory for the command (optional, defaults to agent cwd)",
			},
			"timeout_seconds": map[string]interface{}{
				"type":        "integer",
				"description": "Override timeout in seconds (optional, max 300)",
			},
		},
		"required": []string{"command"},
	}
}

var denyPatterns = []*regexp.Regexp{
	regexp.MustCompile(`\brm\s+(-[^\s]*)?-r[^\s]*f[^\s]*\s+/\s*$`),
	regexp.MustCompile(`\brm\s+(-[^\s]*)?-f[^\s]*r[^\s]*\s+/\s*$`),
	regexp.MustCompile(`\brm\s+-rf\s+/\b`),
	regexp.MustCompile(`\brm\s+-fr\s+/\b`),
	regexp.MustCompile(`\brm\s+-rf\s+\*`),
	regexp.MustCompile(`\bdd\s+if=`),
	regexp.MustCompile(`:\(\)\s*\{\s*:\|:\s*&\s*\}\s*;`),
	regexp.MustCompile(`\.\(\)\s*\{\s*\.\|\.\s*&\s*\}\s*;`),
	regexp.MustCompile(`\b(shutdown|reboot|poweroff|halt|init\s+[06])\b`),
	regexp.MustCompile(`\b(mkfs|mkfs\.\w+|format)\b`),
	regexp.MustCompile(`>\s*/dev/sd`),
	regexp.MustCompile(`>\s*/dev/nvme`),
	regexp.MustCompile(`>\s*/dev/vd`),
	regexp.MustCompile(`>\s*/dev/hd`),
	regexp.MustCompile(`>\s*/dev/mmcblk`),
	regexp.MustCompile(`\bchmod\s+-R\s+\d+\s+/\s*$`),
	regexp.MustCompile(`\bchown\s+-R\s+\S+\s+/\s*$`),
	regexp.MustCompile(`>\s*/dev/sda\b`),
	regexp.MustCompile(`\bwipefs\b.*-a\b`),
	regexp.MustCompile(`\bsgdisk\b.*--zap-all\b`),
	regexp.MustCompile(`\byes\b.*>\s*/dev/`),
	regexp.MustCompile(`\bcat\s+/dev/(zero|urandom)\s*>\s*/dev/`),
}

func isDeniedCommand(command string) (bool, string) {
	for _, pat := range denyPatterns {
		if pat.MatchString(command) {
			return true, pat.String()
		}
	}
	return false, ""
}

func (t *ExecTool) Execute(ctx context.Context, args map[string]interface{}) *ToolResult {
	command := getStringArg(args, "command")
	if command == "" {
		return ErrorResult("command argument is required")
	}

	if denied, pattern := isDeniedCommand(command); denied {
		globalLogger.Warn("blocked dangerous command: %s (matched: %s)", command, pattern)
		return ErrorResult(fmt.Sprintf("command blocked by safety filter: %s", command))
	}

	workDir := getStringArg(args, "working_dir")

	timeout := t.timeout
	if overrideSec := getIntArg(args, "timeout_seconds", 0); overrideSec > 0 {
		if overrideSec > 300 {
			overrideSec = 300
		}
		timeout = time.Duration(overrideSec) * time.Second
	}

	execCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	shell := "/bin/sh"
	shellFlag := "-c"
	if runtime.GOOS == "windows" {
		shell = "cmd"
		shellFlag = "/C"
	}

	globalLogger.Debug("exec: shell=%s command=%s workdir=%s timeout=%v",
		shell, truncate(command, 100), workDir, timeout)

	cmd := exec.CommandContext(execCtx, shell, shellFlag, command)
	if workDir != "" {
		if info, err := os.Stat(workDir); err != nil {
			return ErrorResultf("working directory does not exist: %s", workDir)
		} else if !info.IsDir() {
			return ErrorResultf("working_dir is not a directory: %s", workDir)
		}
		cmd.Dir = workDir
	}

	var output bytes.Buffer
	output.Grow(4096)
	cmd.Stdout = &output
	cmd.Stderr = &output

	startTime := time.Now()
	err := cmd.Run()
	elapsed := time.Since(startTime)

	result := output.String()

	if len(result) > MaxExecOutputSize {
		result = result[:MaxExecOutputSize] +
			fmt.Sprintf("\n\n[output truncated at %d bytes, total was %d bytes]",
				MaxExecOutputSize, output.Len())
	}

	if err != nil {
		if execCtx.Err() == context.DeadlineExceeded {
			return ErrorResult(fmt.Sprintf("command timed out after %v\nPartial output:\n%s",
				timeout, truncate(result, 10000)))
		}
		if ctx.Err() == context.Canceled {
			return ErrorResult("command cancelled by user")
		}
		exitCode := -1
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		}
		return NewToolResult(fmt.Sprintf("Exit code: %d (%.2fs)\n%s", exitCode, elapsed.Seconds(), result))
	}

	if result == "" {
		result = "(no output)"
	}

	return NewToolResult(fmt.Sprintf("%s\n(%.2fs)", result, elapsed.Seconds()))
}

type ReadFileTool struct{}

func (t *ReadFileTool) Name() string { return "read_file" }

func (t *ReadFileTool) Description() string {
	return "Read the contents of a file. For large files, use offset and limit to read specific line ranges. " +
		"Maximum file size: 10MB."
}

func (t *ReadFileTool) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "Absolute or relative path to the file to read",
			},
			"offset": map[string]interface{}{
				"type":        "integer",
				"description": "Line number to start reading from (1-based, optional)",
			},
			"limit": map[string]interface{}{
				"type":        "integer",
				"description": "Maximum number of lines to read (optional, default: all)",
			},
		},
		"required": []string{"path"},
	}
}

func (t *ReadFileTool) Execute(ctx context.Context, args map[string]interface{}) *ToolResult {
	path := getStringArg(args, "path")
	if path == "" {
		return ErrorResult("path argument is required")
	}

	info, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return ErrorResult(fmt.Sprintf("file not found: %s", path))
		}
		return ErrorResult(fmt.Sprintf("failed to stat file: %v", err))
	}

	if info.IsDir() {
		return ErrorResult(fmt.Sprintf("%s is a directory, not a file (use list_dir instead)", path))
	}

	if info.Size() > MaxReadFileSize {
		return ErrorResult(fmt.Sprintf("file too large: %d bytes (max %d bytes). Use exec with head/tail to read portions.",
			info.Size(), MaxReadFileSize))
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to read file: %v", err))
	}

	content := string(data)
	if content == "" {
		return NewToolResult(fmt.Sprintf("(empty file: %s, size=0 bytes)", path))
	}

	totalLines := strings.Count(content, "\n")
	if len(content) > 0 && content[len(content)-1] != '\n' {
		totalLines++ // Count last line without trailing newline
	}

	offset := getIntArg(args, "offset", 0)
	limit := getIntArg(args, "limit", 0)

	if offset > 0 || limit > 0 {
		lines := strings.Split(content, "\n")

		startIdx := 0
		if offset > 0 {
			startIdx = offset - 1
		}
		if startIdx >= len(lines) {
			return ErrorResult(fmt.Sprintf("offset %d exceeds file line count (%d)", offset, totalLines))
		}

		endIdx := len(lines)
		if limit > 0 {
			endIdx = startIdx + limit
			if endIdx > len(lines) {
				endIdx = len(lines)
			}
		}

		selectedLines := lines[startIdx:endIdx]
		content = strings.Join(selectedLines, "\n")

		return NewToolResult(fmt.Sprintf("File: %s (%d bytes, %d total lines, showing lines %d-%d)\n\n%s",
			path, info.Size(), totalLines, startIdx+1, startIdx+len(selectedLines), content))
	}

	return NewToolResult(fmt.Sprintf("File: %s (%d bytes, %d lines)\n\n%s",
		path, info.Size(), totalLines, content))
}

type WriteFileTool struct{}

func (t *WriteFileTool) Name() string { return "write_file" }

func (t *WriteFileTool) Description() string {
	return "Write content to a file. Creates parent directories if they don't exist. " +
		"Set append=true to append instead of overwriting."
}

func (t *WriteFileTool) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "Absolute or relative path to the file to write",
			},
			"content": map[string]interface{}{
				"type":        "string",
				"description": "The content to write to the file",
			},
			"append": map[string]interface{}{
				"type":        "boolean",
				"description": "If true, append to existing file instead of overwriting (default: false)",
			},
			"mode": map[string]interface{}{
				"type":        "string",
				"description": "File permissions in octal (default: '0644')",
			},
		},
		"required": []string{"path", "content"},
	}
}

func (t *WriteFileTool) Execute(ctx context.Context, args map[string]interface{}) *ToolResult {
	path := getStringArg(args, "path")
	if path == "" {
		return ErrorResult("path argument is required")
	}
	content := getStringArg(args, "content")
	appendMode := getBoolArg(args, "append")

	fileMode := os.FileMode(0644)
	if modeStr := getStringArg(args, "mode"); modeStr != "" {
		parsed, err := strconv.ParseUint(modeStr, 8, 32)
		if err != nil {
			return ErrorResult(fmt.Sprintf("invalid file mode %q: %v", modeStr, err))
		}
		fileMode = os.FileMode(parsed)
	}

	existed := false
	if _, err := os.Stat(path); err == nil {
		existed = true
	}

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return ErrorResult(fmt.Sprintf("failed to create directory %s: %v", dir, err))
	}

	if appendMode {
		f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, fileMode)
		if err != nil {
			return ErrorResult(fmt.Sprintf("failed to open file for append: %v", err))
		}
		defer f.Close()
		n, err := f.WriteString(content)
		if err != nil {
			return ErrorResult(fmt.Sprintf("failed to append to file: %v", err))
		}
		return NewToolResult(fmt.Sprintf("appended %d bytes to %s", n, path))
	}

	if err := os.WriteFile(path, []byte(content), fileMode); err != nil {
		return ErrorResult(fmt.Sprintf("failed to write file: %v", err))
	}

	lineCount := strings.Count(content, "\n")
	if len(content) > 0 && content[len(content)-1] != '\n' {
		lineCount++
	}

	action := "created"
	if existed {
		action = "wrote"
	}

	return NewToolResult(fmt.Sprintf("%s %s (%d bytes, %d lines)", action, path, len(content), lineCount))
}

type ListDirTool struct{}

func (t *ListDirTool) Name() string { return "list_dir" }

func (t *ListDirTool) Description() string {
	return "List files and directories in the given path. Shows type (DIR/FILE/LINK), " +
		"file sizes, and entry count. Use show_hidden=true to include dotfiles."
}

func (t *ListDirTool) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "Path to the directory to list",
			},
			"show_hidden": map[string]interface{}{
				"type":        "boolean",
				"description": "Include hidden files/directories starting with . (default: false)",
			},
			"show_sizes": map[string]interface{}{
				"type":        "boolean",
				"description": "Show file sizes (default: true)",
			},
		},
		"required": []string{"path"},
	}
}

func formatSize(bytes int64) string {
	const (
		KB = 1024
		MB = KB * 1024
		GB = MB * 1024
	)
	switch {
	case bytes >= GB:
		return fmt.Sprintf("%.1fG", float64(bytes)/float64(GB))
	case bytes >= MB:
		return fmt.Sprintf("%.1fM", float64(bytes)/float64(MB))
	case bytes >= KB:
		return fmt.Sprintf("%.1fK", float64(bytes)/float64(KB))
	default:
		return fmt.Sprintf("%dB", bytes)
	}
}

func (t *ListDirTool) Execute(ctx context.Context, args map[string]interface{}) *ToolResult {
	path := getStringArg(args, "path")
	if path == "" {
		return ErrorResult("path argument is required")
	}

	info, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return ErrorResult(fmt.Sprintf("directory not found: %s", path))
		}
		return ErrorResult(fmt.Sprintf("failed to stat path: %v", err))
	}
	if !info.IsDir() {
		return ErrorResult(fmt.Sprintf("%s is a file, not a directory (use read_file instead)", path))
	}

	entries, err := os.ReadDir(path)
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to list directory: %v", err))
	}

	showHidden := getBoolArg(args, "show_hidden")
	showSizes := true // default to showing sizes
	if v, ok := args["show_sizes"]; ok {
		if b, ok := v.(bool); ok {
			showSizes = b
		}
	}

	var b strings.Builder
	dirCount := 0
	fileCount := 0
	linkCount := 0
	totalSize := int64(0)

	for _, entry := range entries {
		name := entry.Name()

		if !showHidden && strings.HasPrefix(name, ".") {
			continue
		}

		entryInfo, err := entry.Info()
		if err != nil {
			b.WriteString(fmt.Sprintf("ERR:  %s (cannot stat)\n", name))
			continue
		}

		mode := entryInfo.Mode()
		switch {
		case mode&os.ModeSymlink != 0:
			linkCount++
			target, _ := os.Readlink(filepath.Join(path, name))
			if target != "" {
				b.WriteString(fmt.Sprintf("LINK: %s -> %s\n", name, target))
			} else {
				b.WriteString(fmt.Sprintf("LINK: %s\n", name))
			}
		case entry.IsDir():
			dirCount++
			b.WriteString(fmt.Sprintf("DIR:  %s/\n", name))
		default:
			fileCount++
			size := entryInfo.Size()
			totalSize += size
			if showSizes {
				b.WriteString(fmt.Sprintf("FILE: %-40s %s\n", name, formatSize(size)))
			} else {
				b.WriteString(fmt.Sprintf("FILE: %s\n", name))
			}
		}
	}

	result := b.String()
	if result == "" {
		if showHidden {
			return NewToolResult(fmt.Sprintf("(empty directory: %s)", path))
		}
		return NewToolResult(fmt.Sprintf("(no visible entries in %s, try show_hidden=true)", path))
	}

	summary := fmt.Sprintf("\n[%s: %d dirs, %d files", path, dirCount, fileCount)
	if linkCount > 0 {
		summary += fmt.Sprintf(", %d links", linkCount)
	}
	if totalSize > 0 {
		summary += fmt.Sprintf(", total %s", formatSize(totalSize))
	}
	summary += "]"

	return NewToolResult(result + summary)
}

type EditFileTool struct{}

func (t *EditFileTool) Name() string { return "edit_file" }

func (t *EditFileTool) Description() string {
	return "Edit a file by replacing an exact string match with a new string. " +
		"By default, old_string must appear exactly once (set replace_all=true for multiple). " +
		"Reports the line number of the replacement."
}

func (t *EditFileTool) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "Path to the file to edit",
			},
			"old_string": map[string]interface{}{
				"type":        "string",
				"description": "The exact string to find in the file",
			},
			"new_string": map[string]interface{}{
				"type":        "string",
				"description": "The string to replace it with",
			},
			"replace_all": map[string]interface{}{
				"type":        "boolean",
				"description": "Replace all occurrences instead of requiring exactly one (default: false)",
			},
		},
		"required": []string{"path", "old_string", "new_string"},
	}
}

func (t *EditFileTool) Execute(ctx context.Context, args map[string]interface{}) *ToolResult {
	path := getStringArg(args, "path")
	if path == "" {
		return ErrorResult("path argument is required")
	}
	oldStr := getStringArg(args, "old_string")
	if oldStr == "" {
		return ErrorResult("old_string argument is required")
	}
	newStr := getStringArg(args, "new_string")
	replaceAll := getBoolArg(args, "replace_all")

	info, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return ErrorResult(fmt.Sprintf("file not found: %s", path))
		}
		return ErrorResult(fmt.Sprintf("failed to stat file: %v", err))
	}
	origMode := info.Mode()

	data, err := os.ReadFile(path)
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to read file: %v", err))
	}

	content := string(data)
	count := strings.Count(content, oldStr)

	if count == 0 {
		lines := strings.Split(content, "\n")
		preview := strings.Join(lines[:min(len(lines), 5)], "\n")
		return ErrorResult(fmt.Sprintf("old_string not found in file. File starts with:\n%s",
			truncate(preview, 300)))
	}

	if !replaceAll && count > 1 {
		lineNums := findOccurrenceLines(content, oldStr)
		return ErrorResult(fmt.Sprintf("old_string found %d times (at lines %s). "+
			"Provide more context to make it unique, or set replace_all=true.",
			count, formatLineNums(lineNums)))
	}

	firstIdx := strings.Index(content, oldStr)
	firstLine := strings.Count(content[:firstIdx], "\n") + 1

	var newContent string
	if replaceAll {
		newContent = strings.ReplaceAll(content, oldStr, newStr)
	} else {
		newContent = strings.Replace(content, oldStr, newStr, 1)
	}

	if err := os.WriteFile(path, []byte(newContent), origMode); err != nil {
		return ErrorResult(fmt.Sprintf("failed to write file: %v", err))
	}

	oldLines := strings.Count(oldStr, "\n") + 1
	newLines := strings.Count(newStr, "\n") + 1
	lineDelta := (newLines - oldLines) * count

	if replaceAll {
		return NewToolResult(fmt.Sprintf("edited %s: replaced %d occurrences (first at line %d, line delta: %+d)",
			path, count, firstLine, lineDelta))
	}

	return NewToolResult(fmt.Sprintf("edited %s: replaced 1 occurrence at line %d (line delta: %+d)",
		path, firstLine, newLines-oldLines))
}

func findOccurrenceLines(content, substr string) []int {
	var lines []int
	offset := 0
	for {
		idx := strings.Index(content[offset:], substr)
		if idx == -1 {
			break
		}
		lineNum := strings.Count(content[:offset+idx], "\n") + 1
		lines = append(lines, lineNum)
		offset += idx + len(substr)
	}
	return lines
}

func formatLineNums(nums []int) string {
	strs := make([]string, len(nums))
	for i, n := range nums {
		strs[i] = strconv.Itoa(n)
	}
	if len(strs) > 10 {
		return strings.Join(strs[:10], ", ") + ", ..."
	}
	return strings.Join(strs, ", ")
}

var errMaxResults = errors.New("max results reached")

type SearchFilesTool struct{}

func (t *SearchFilesTool) Name() string { return "search_files" }

func (t *SearchFilesTool) Description() string {
	return "Search for files matching a glob pattern (e.g. '*.go', '**/*.json'). " +
		"Searches recursively from the given path."
}

func (t *SearchFilesTool) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "Directory to search in (default: current directory)",
			},
			"pattern": map[string]interface{}{
				"type":        "string",
				"description": "Glob pattern to match file names (e.g. '*.go', 'test_*.py', '*.{js,ts}')",
			},
			"max_depth": map[string]interface{}{
				"type":        "integer",
				"description": "Maximum directory depth to search (default: 10)",
			},
			"max_results": map[string]interface{}{
				"type":        "integer",
				"description": "Maximum number of results to return (default: 100)",
			},
		},
		"required": []string{"pattern"},
	}
}

func (t *SearchFilesTool) Execute(ctx context.Context, args map[string]interface{}) *ToolResult {
	pattern := getStringArg(args, "pattern")
	if pattern == "" {
		return ErrorResult("pattern argument is required")
	}

	searchPath := getStringArg(args, "path")
	if searchPath == "" {
		searchPath = "."
	}

	maxDepth := getIntArg(args, "max_depth", 10)
	maxResults := getIntArg(args, "max_results", 100)

	absPath, err := filepath.Abs(searchPath)
	if err != nil {
		return ErrorResult(fmt.Sprintf("invalid search path: %v", err))
	}
	baseDepth := strings.Count(absPath, string(os.PathSeparator))

	var matches []string
	err = filepath.WalkDir(absPath, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil // Skip entries we cannot access
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		depth := strings.Count(path, string(os.PathSeparator)) - baseDepth
		if depth > maxDepth {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		if d.IsDir() {
			name := d.Name()
			if name == ".git" || name == "node_modules" || name == "__pycache__" ||
				name == ".venv" || name == "vendor" || name == ".cache" {
				return filepath.SkipDir
			}
			return nil
		}

		matched, err := filepath.Match(pattern, d.Name())
		if err != nil {
			return nil // Invalid pattern character, skip
		}
		if matched {
			relPath, err := filepath.Rel(absPath, path)
			if err != nil {
				relPath = path
			}
			matches = append(matches, relPath)
			if len(matches) >= maxResults {
				return errMaxResults
			}
		}

		return nil
	})

	if err != nil && !errors.Is(err, errMaxResults) && !errors.Is(err, context.Canceled) {
		return ErrorResult(fmt.Sprintf("search error: %v", err))
	}

	if len(matches) == 0 {
		return NewToolResult(fmt.Sprintf("no files matching '%s' found in %s", pattern, absPath))
	}

	var b strings.Builder
	for _, m := range matches {
		b.WriteString(m)
		b.WriteString("\n")
	}

	truncated := ""
	if len(matches) >= maxResults {
		truncated = fmt.Sprintf(" (truncated at %d results)", maxResults)
	}

	return NewToolResult(fmt.Sprintf("Found %d files matching '%s' in %s%s:\n\n%s",
		len(matches), pattern, absPath, truncated, b.String()))
}

type SystemInfoTool struct{}

func (t *SystemInfoTool) Name() string { return "system_info" }

func (t *SystemInfoTool) Description() string {
	return "Get system information: OS, architecture, hostname, CPU count, " +
		"environment variables, and working directory."
}

func (t *SystemInfoTool) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"category": map[string]interface{}{
				"type":        "string",
				"description": "Category of info: 'all', 'os', 'env', 'runtime' (default: 'all')",
				"enum":        []string{"all", "os", "env", "runtime"},
			},
		},
		"required": []string{},
	}
}

func (t *SystemInfoTool) Execute(ctx context.Context, args map[string]interface{}) *ToolResult {
	category := getStringArg(args, "category")
	if category == "" {
		category = "all"
	}

	var b strings.Builder

	if category == "all" || category == "os" {
		hostname, _ := os.Hostname()
		cwd, _ := os.Getwd()
		execPath, _ := os.Executable()

		b.WriteString("## OS Information\n")
		b.WriteString(fmt.Sprintf("OS:         %s\n", runtime.GOOS))
		b.WriteString(fmt.Sprintf("Arch:       %s\n", runtime.GOARCH))
		b.WriteString(fmt.Sprintf("Hostname:   %s\n", hostname))
		b.WriteString(fmt.Sprintf("PID:        %d\n", os.Getpid()))
		b.WriteString(fmt.Sprintf("UID:        %d\n", os.Getuid()))
		b.WriteString(fmt.Sprintf("GID:        %d\n", os.Getgid()))
		b.WriteString(fmt.Sprintf("CWD:        %s\n", cwd))
		b.WriteString(fmt.Sprintf("Executable: %s\n", execPath))
		b.WriteString(fmt.Sprintf("Temp dir:   %s\n", os.TempDir()))
		b.WriteString("\n")
	}

	if category == "all" || category == "runtime" {
		b.WriteString("## Runtime Information\n")
		b.WriteString(fmt.Sprintf("Go version: %s\n", runtime.Version()))
		b.WriteString(fmt.Sprintf("Compiler:   %s\n", runtime.Compiler))
		b.WriteString(fmt.Sprintf("NumCPU:     %d\n", runtime.NumCPU()))
		b.WriteString(fmt.Sprintf("GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0)))
		b.WriteString(fmt.Sprintf("Goroutines: %d\n", runtime.NumGoroutine()))

		var memStats runtime.MemStats
		runtime.ReadMemStats(&memStats)
		b.WriteString(fmt.Sprintf("Alloc:      %s\n", formatSize(int64(memStats.Alloc))))
		b.WriteString(fmt.Sprintf("Sys:        %s\n", formatSize(int64(memStats.Sys))))
		b.WriteString(fmt.Sprintf("GC cycles:  %d\n", memStats.NumGC))
		b.WriteString("\n")
	}

	if category == "all" || category == "env" {
		b.WriteString("## Key Environment Variables\n")
		envKeys := []string{
			"HOME", "USER", "USERNAME", "SHELL", "PATH", "LANG",
			"TERM", "EDITOR", "GOPATH", "GOROOT", "VIRTUAL_ENV",
			"CONDA_DEFAULT_ENV", "SSH_CONNECTION", "DISPLAY", "XDG_SESSION_TYPE",
		}
		for _, key := range envKeys {
			if val := os.Getenv(key); val != "" {
				if key == "PATH" && len(val) > 200 {
					val = val[:200] + "..."
				}
				b.WriteString(fmt.Sprintf("%-20s = %s\n", key, val))
			}
		}
		b.WriteString("\n")
	}

	return NewToolResult(b.String())
}

// I2C ioctl constants from Linux kernel headers:
//
//	/usr/include/linux/i2c.h, /usr/include/linux/i2c-dev.h
const (
	I2C_SLAVE = 0x0703 // ioctl(fd, I2C_SLAVE, addr)
	I2C_SMBUS = 0x0720 // ioctl(fd, I2C_SMBUS, &smbusIoctlData)
	I2C_RDWR  = 0x0707 // ioctl(fd, I2C_RDWR, &i2cRdwrData)

	I2C_SMBUS_READ  = 1
	I2C_SMBUS_WRITE = 0

	I2C_SMBUS_QUICK     = 0
	I2C_SMBUS_BYTE      = 1
	I2C_SMBUS_BYTE_DATA = 2
	I2C_SMBUS_WORD_DATA = 3
)

// smbusIoctlData mirrors kernel's i2c_smbus_ioctl_data (include/uapi/linux/i2c-dev.h).
type smbusIoctlData struct {
	readWrite uint8
	command   uint8
	size      uint32
	data      uintptr
}

func i2cOpenSlave(bus, addr int) (int, func(), *ToolResult) {
	devPath := fmt.Sprintf("/dev/i2c-%d", bus)
	fd, err := syscall.Open(devPath, syscall.O_RDWR, 0)
	if err != nil {
		return 0, nil, ErrorResultf("failed to open %s: %v", devPath, err)
	}
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), I2C_SLAVE, uintptr(addr)); errno != 0 {
		syscall.Close(fd)
		return 0, nil, ErrorResultf("failed to set I2C slave address 0x%02x: %v", addr, errno)
	}
	return fd, func() { syscall.Close(fd) }, nil
}

type I2CTool struct {
	config *Config
}

func (t *I2CTool) Name() string { return "i2c" }

func (t *I2CTool) Description() string {
	return "Interact with I2C devices on Linux. Actions: detect (find I2C buses), " +
		"scan (probe addresses 0x03-0x77), read (read N bytes), read_register (read from register), " +
		"write (write bytes; restricted by allowed_i2c_write_addrs config)."
}

func (t *I2CTool) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"action": map[string]interface{}{
				"type":        "string",
				"description": "Action to perform: detect, scan, read, read_register, write",
				"enum":        []string{"detect", "scan", "read", "read_register", "write"},
			},
			"bus": map[string]interface{}{
				"type":        "integer",
				"description": "I2C bus number (e.g. 1 for /dev/i2c-1)",
			},
			"address": map[string]interface{}{
				"type":        "integer",
				"description": "I2C device address (0x03-0x77, decimal or hex)",
			},
			"register": map[string]interface{}{
				"type":        "integer",
				"description": "Register address to read from (0-255, for read_register action)",
			},
			"count": map[string]interface{}{
				"type":        "integer",
				"description": "Number of bytes to read (for read/read_register, default: 1, max: 256)",
			},
			"data": map[string]interface{}{
				"type":        "string",
				"description": "Hex string of bytes to write (for write action, e.g. 'ff01ab', max 32 bytes)",
			},
		},
		"required": []string{"action"},
	}
}

func (t *I2CTool) Execute(ctx context.Context, args map[string]interface{}) *ToolResult {
	if runtime.GOOS != "linux" {
		return ErrorResult("I2C is only supported on Linux (current OS: " + runtime.GOOS + "). " +
			"I2C requires direct hardware access via /dev/i2c-* device nodes.")
	}

	action := getStringArg(args, "action")
	switch action {
	case "detect":
		return t.detect()
	case "scan":
		return t.scan(args)
	case "read":
		return t.read(args)
	case "read_register":
		return t.readRegister(args)
	case "write":
		return t.write(args)
	default:
		return ErrorResult("unknown action: " + action +
			" (valid: detect, scan, read, read_register, write)")
	}
}

func (t *I2CTool) detect() *ToolResult {
	matches, err := filepath.Glob("/dev/i2c-*")
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to glob /dev/i2c-*: %v", err))
	}
	if len(matches) == 0 {
		return NewToolResult("no I2C buses found (no /dev/i2c-* devices)")
	}

	var b strings.Builder
	b.WriteString("I2C buses found:\n")
	for _, m := range matches {
		b.WriteString(fmt.Sprintf("  %s\n", m))
	}
	return NewToolResult(b.String())
}

func (t *I2CTool) scan(args map[string]interface{}) *ToolResult {
	bus := getIntArg(args, "bus", -1)
	if bus < 0 {
		return ErrorResult("bus argument is required for scan")
	}

	devPath := fmt.Sprintf("/dev/i2c-%d", bus)
	fd, err := syscall.Open(devPath, syscall.O_RDWR, 0)
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to open %s: %v", devPath, err))
	}
	defer syscall.Close(fd)

	foundSet := make(map[int]bool)
	var found []int
	for addr := 0x03; addr <= 0x77; addr++ {
		if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), I2C_SLAVE, uintptr(addr)); errno != 0 {
			continue
		}

		// Hybrid detection approach:
		// 1. Try SMBus Quick Write first (most compatible)
		// 2. If that fails, try SMBus Read Byte as fallback
		// This matches the approach used by i2cdetect in i2c-tools
		if t.smbusQuickWrite(fd) || t.smbusReadByte(fd) {
			found = append(found, addr)
			foundSet[addr] = true
		}
	}

	var b strings.Builder
	b.WriteString(fmt.Sprintf("I2C bus scan: /dev/i2c-%d\n\n", bus))

	b.WriteString("     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f\n")

	for row := 0; row < 8; row++ {
		b.WriteString(fmt.Sprintf("%02x: ", row*16))
		for col := 0; col < 16; col++ {
			addr := row*16 + col
			if addr < 0x03 || addr > 0x77 {
				b.WriteString("   ") // Reserved address
			} else if foundSet[addr] {
				b.WriteString(fmt.Sprintf("%02x ", addr)) // Device found
			} else {
				b.WriteString("-- ") // No device
			}
		}
		b.WriteString("\n")
	}

	if len(found) == 0 {
		b.WriteString(fmt.Sprintf("\nNo devices found (scanned 0x03-0x77)"))
	} else {
		b.WriteString(fmt.Sprintf("\n%d device(s) found:", len(found)))
		for _, addr := range found {
			desc := describeI2CAddress(addr)
			if desc != "" {
				b.WriteString(fmt.Sprintf("\n  0x%02x - %s", addr, desc))
			} else {
				b.WriteString(fmt.Sprintf("\n  0x%02x", addr))
			}
		}
	}

	return NewToolResult(b.String())
}

var knownI2CDevices = map[int]string{
	0x1D: "ADXL345/MMA8451 accelerometer",
	0x1E: "HMC5883L magnetometer",
	0x20: "PCF8574/MCP23008 GPIO expander",
	0x21: "PCF8574/MCP23008 GPIO expander",
	0x22: "PCF8574/MCP23008 GPIO expander",
	0x23: "BH1750 light sensor / PCF8574 GPIO expander",
	0x24: "PCF8574 GPIO expander",
	0x25: "PCF8574 GPIO expander",
	0x26: "PCF8574 GPIO expander",
	0x27: "PCF8574/LCD backpack / PCF8574 GPIO expander",
	0x29: "VL53L0X ToF distance / TSL2561 light sensor",
	0x38: "AHT10/AHT20 temperature/humidity",
	0x39: "TSL2561 light sensor / APDS-9960 gesture",
	0x3C: "SSD1306 OLED display",
	0x3D: "SSD1306 OLED display (alt)",
	0x40: "INA219/INA260 current sensor / HTU21D/SHT31 temp/humidity / PCA9685 PWM",
	0x41: "INA219 current sensor (alt)",
	0x48: "ADS1015/ADS1115 ADC / TMP102 temperature / PCF8591 ADC/DAC",
	0x49: "ADS1015/ADS1115 ADC (alt) / TMP102 (alt)",
	0x4A: "ADS1015/ADS1115 ADC (alt)",
	0x4B: "ADS1015/ADS1115 ADC (alt)",
	0x50: "AT24C32/AT24C256 EEPROM",
	0x51: "AT24C32 EEPROM (alt)",
	0x52: "AT24C32 EEPROM (alt)",
	0x53: "ADXL345 accelerometer (alt) / AT24C32 EEPROM (alt)",
	0x57: "MAX30102 pulse oximeter / AT24C32 EEPROM (alt)",
	0x5A: "MLX90614 IR thermometer / MPR121 capacitive touch",
	0x5B: "CCS811 air quality / MPR121 (alt)",
	0x60: "MCP4725 DAC / SI5351 clock generator",
	0x68: "DS3231/DS1307 RTC / MPU6050/MPU9250 IMU",
	0x69: "MPU6050/MPU9250 IMU (alt)",
	0x6A: "LSM6DS3 IMU",
	0x6B: "LSM6DS3 IMU (alt)",
	0x76: "BME280/BMP280/MS5611 pressure/temp/humidity",
	0x77: "BME280/BMP280 (alt) / BMP180 pressure/temp",
}

func describeI2CAddress(addr int) string {
	return knownI2CDevices[addr]
}

func (t *I2CTool) smbusQuickWrite(fd int) bool {
	data := smbusIoctlData{
		readWrite: I2C_SMBUS_WRITE,
		command:   0,
		size:      I2C_SMBUS_QUICK,
		data:      0,
	}
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), I2C_SMBUS, uintptr(unsafe.Pointer(&data)))
	return errno == 0
}

func (t *I2CTool) smbusReadByte(fd int) bool {
	var result [1]byte
	data := smbusIoctlData{
		readWrite: I2C_SMBUS_READ,
		command:   0,
		size:      I2C_SMBUS_BYTE,
		data:      uintptr(unsafe.Pointer(&result[0])),
	}
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), I2C_SMBUS, uintptr(unsafe.Pointer(&data)))
	return errno == 0
}

func (t *I2CTool) read(args map[string]interface{}) *ToolResult {
	bus, errResult := requireIntArg(args, "bus")
	if errResult != nil {
		return errResult
	}
	addr, errResult := requireIntArg(args, "address")
	if errResult != nil {
		return errResult
	}
	count := getIntArg(args, "count", 1)
	if count <= 0 || count > 256 {
		return ErrorResult("count must be between 1 and 256")
	}

	fd, closer, errResult := i2cOpenSlave(bus, addr)
	if errResult != nil {
		return errResult
	}
	defer closer()

	buf := make([]byte, count)
	n, err := syscall.Read(fd, buf)
	if err != nil {
		return ErrorResultf("failed to read from I2C device: %v", err)
	}

	return NewToolResult(fmt.Sprintf("read %d bytes from bus %d addr 0x%02x: %s",
		n, bus, addr, hex.EncodeToString(buf[:n])))
}

func (t *I2CTool) write(args map[string]interface{}) *ToolResult {
	bus, errResult := requireIntArg(args, "bus")
	if errResult != nil {
		return errResult
	}
	addr, errResult := requireIntArg(args, "address")
	if errResult != nil {
		return errResult
	}
	dataHex, errResult := requireStringArg(args, "data")
	if errResult != nil {
		return errResult
	}

	if !t.config.IsI2CWriteAllowed(addr) {
		return ErrorResultf("I2C write to address 0x%02x blocked: configure allowed_i2c_write_addrs in ~/.attoclaw.json", addr)
	}

	dataHex = strings.ReplaceAll(dataHex, " ", "")
	dataHex = strings.ReplaceAll(dataHex, "0x", "")
	dataHex = strings.ReplaceAll(dataHex, "0X", "")

	writeBytes, err := hex.DecodeString(dataHex)
	if err != nil {
		return ErrorResultf("invalid hex data '%s': %v", dataHex, err)
	}
	if len(writeBytes) == 0 {
		return ErrorResult("data must not be empty")
	}
	if len(writeBytes) > 32 {
		return ErrorResultf("data too long: %d bytes (max 32)", len(writeBytes))
	}

	if t.config.HardwareDryRun {
		return NewToolResult(fmt.Sprintf("[dry-run] would write %d bytes to bus %d addr 0x%02x: %s",
			len(writeBytes), bus, addr, hex.EncodeToString(writeBytes)))
	}

	fd, closer, errResult := i2cOpenSlave(bus, addr)
	if errResult != nil {
		return errResult
	}
	defer closer()

	n, err := syscall.Write(fd, writeBytes)
	if err != nil {
		return ErrorResultf("failed to write to I2C device: %v", err)
	}

	return NewToolResult(fmt.Sprintf("wrote %d bytes to bus %d addr 0x%02x: %s",
		n, bus, addr, hex.EncodeToString(writeBytes)))
}

func (t *I2CTool) readRegister(args map[string]interface{}) *ToolResult {
	bus, errResult := requireIntArg(args, "bus")
	if errResult != nil {
		return errResult
	}
	addr, errResult := requireIntArg(args, "address")
	if errResult != nil {
		return errResult
	}
	register := getIntArg(args, "register", -1)
	if register < 0 || register > 255 {
		return ErrorResult("register argument is required (0-255)")
	}
	count := getIntArg(args, "count", 1)
	if count <= 0 || count > 256 {
		return ErrorResult("count must be between 1 and 256")
	}

	fd, closer, errResult := i2cOpenSlave(bus, addr)
	if errResult != nil {
		return errResult
	}
	defer closer()

	regBuf := []byte{byte(register)}
	if _, err := syscall.Write(fd, regBuf); err != nil {
		return ErrorResultf("failed to write register address 0x%02x: %v", register, err)
	}

	buf := make([]byte, count)
	n, err := syscall.Read(fd, buf)
	if err != nil {
		return ErrorResultf("failed to read from register 0x%02x: %v", register, err)
	}

	return NewToolResult(fmt.Sprintf("read %d bytes from bus %d addr 0x%02x register 0x%02x: %s",
		n, bus, addr, register, hex.EncodeToString(buf[:n])))
}

// SPI ioctl constants from include/uapi/linux/spi/spidev.h.
const (
	SPI_IOC_MAGIC            = 'k'
	SPI_IOC_WR_MODE          = 0x40016B01
	SPI_IOC_RD_MODE          = 0x80016B01
	SPI_IOC_WR_BITS_PER_WORD = 0x40016B03
	SPI_IOC_RD_BITS_PER_WORD = 0x80016B03
	SPI_IOC_WR_MAX_SPEED_HZ  = 0x40046B04
	SPI_IOC_RD_MAX_SPEED_HZ  = 0x80046B04
	SPI_IOC_MESSAGE_1        = 0x40206B00 // _IOW(SPI_IOC_MAGIC, 0, spi_ioc_transfer)
)

// spiIocTransfer must be exactly 32 bytes, matching kernel's spi_ioc_transfer
// (include/uapi/linux/spi/spidev.h).
type spiIocTransfer struct {
	txBuf       uint64 // offset 0
	rxBuf       uint64 // offset 8
	length      uint32 // offset 16
	speedHz     uint32 // offset 20
	delayUsecs  uint16 // offset 24
	bitsPerWord uint8  // offset 26
	csChange    uint8  // offset 27
	txNbits     uint8  // offset 28
	rxNbits     uint8  // offset 29
	wordDelay   uint8  // offset 30
	pad         uint8  // offset 31
}

func spiOpenDevice(device string, args map[string]interface{}) (int, func(), *ToolResult) {
	fd, err := syscall.Open(device, syscall.O_RDWR, 0)
	if err != nil {
		return 0, nil, ErrorResultf("failed to open %s: %v", device, err)
	}
	if err := configureSPI(fd, args); err != nil {
		syscall.Close(fd)
		return 0, nil, ErrorResult(err.Error())
	}
	return fd, func() { syscall.Close(fd) }, nil
}

func spiDoTransfer(fd int, tx, rx []byte, speed, bpw int) error {
	xfer := spiIocTransfer{
		txBuf:       uint64(uintptr(unsafe.Pointer(&tx[0]))),
		rxBuf:       uint64(uintptr(unsafe.Pointer(&rx[0]))),
		length:      uint32(len(tx)),
		speedHz:     uint32(speed),
		bitsPerWord: uint8(bpw),
	}
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd),
		SPI_IOC_MESSAGE_1, uintptr(unsafe.Pointer(&xfer))); errno != 0 {
		return fmt.Errorf("SPI transfer failed: %v", errno)
	}
	runtime.KeepAlive(tx)
	runtime.KeepAlive(rx)
	return nil
}

type SPITool struct {
	config *Config
}

func (t *SPITool) Name() string { return "spi" }

func (t *SPITool) Description() string {
	return "Interact with SPI devices on Linux. Actions: list (find SPI devices), " +
		"info (read configuration), transfer (full-duplex send/receive; restricted by allowed_spi_devices config), " +
		"read (receive data by clocking zeros), loopback_test (verify MOSI->MISO wiring)."
}

func (t *SPITool) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"action": map[string]interface{}{
				"type":        "string",
				"description": "Action to perform: list, info, transfer, read, loopback_test",
				"enum":        []string{"list", "info", "transfer", "read", "loopback_test"},
			},
			"device": map[string]interface{}{
				"type":        "string",
				"description": "SPI device path (e.g. /dev/spidev0.0)",
			},
			"data": map[string]interface{}{
				"type":        "string",
				"description": "Hex string of bytes to send (for transfer action)",
			},
			"count": map[string]interface{}{
				"type":        "integer",
				"description": "Number of bytes to read (for read action)",
			},
			"mode": map[string]interface{}{
				"type":        "integer",
				"description": "SPI mode (0-3, default 0)",
			},
			"speed": map[string]interface{}{
				"type":        "integer",
				"description": "SPI speed in Hz (default 1000000)",
			},
			"bits_per_word": map[string]interface{}{
				"type":        "integer",
				"description": "Bits per word (default 8)",
			},
		},
		"required": []string{"action"},
	}
}

func (t *SPITool) Execute(ctx context.Context, args map[string]interface{}) *ToolResult {
	if runtime.GOOS != "linux" {
		return ErrorResult("SPI is only supported on Linux (current OS: " + runtime.GOOS + "). " +
			"SPI requires direct hardware access via /dev/spidev* device nodes.")
	}

	action := getStringArg(args, "action")
	switch action {
	case "list":
		return t.list()
	case "info":
		return t.info(args)
	case "transfer":
		return t.transfer(args)
	case "read":
		return t.readData(args)
	case "loopback_test":
		return t.loopbackTest(args)
	default:
		return ErrorResult("unknown action: " + action +
			" (valid: list, info, transfer, read, loopback_test)")
	}
}

func (t *SPITool) list() *ToolResult {
	matches, err := filepath.Glob("/dev/spidev*")
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to glob /dev/spidev*: %v", err))
	}
	if len(matches) == 0 {
		return NewToolResult("no SPI devices found (no /dev/spidev* devices)")
	}

	var b strings.Builder
	b.WriteString("SPI devices found:\n")
	for _, m := range matches {
		b.WriteString(fmt.Sprintf("  %s\n", m))
	}
	return NewToolResult(b.String())
}

func configureSPI(fd int, args map[string]interface{}) error {
	mode := getIntArg(args, "mode", 0)
	speed := getIntArg(args, "speed", 1000000)
	bpw := getIntArg(args, "bits_per_word", 8)

	modeVal := uint8(mode)
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd),
		SPI_IOC_WR_MODE, uintptr(unsafe.Pointer(&modeVal))); errno != 0 {
		return fmt.Errorf("failed to set SPI mode: %v", errno)
	}

	bpwVal := uint8(bpw)
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd),
		SPI_IOC_WR_BITS_PER_WORD, uintptr(unsafe.Pointer(&bpwVal))); errno != 0 {
		return fmt.Errorf("failed to set bits per word: %v", errno)
	}

	speedVal := uint32(speed)
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd),
		SPI_IOC_WR_MAX_SPEED_HZ, uintptr(unsafe.Pointer(&speedVal))); errno != 0 {
		return fmt.Errorf("failed to set SPI speed: %v", errno)
	}

	return nil
}

func (t *SPITool) transfer(args map[string]interface{}) *ToolResult {
	device, errResult := requireStringArg(args, "device")
	if errResult != nil {
		return errResult
	}
	dataHex, errResult := requireStringArg(args, "data")
	if errResult != nil {
		return errResult
	}

	if !t.config.IsSPIDeviceAllowed(device) {
		return ErrorResultf("SPI transfer to %s blocked: configure allowed_spi_devices in ~/.attoclaw.json", device)
	}

	txData, err := hex.DecodeString(dataHex)
	if err != nil {
		return ErrorResultf("invalid hex data: %v", err)
	}
	if len(txData) == 0 {
		return ErrorResult("data must not be empty")
	}

	if t.config.HardwareDryRun {
		return NewToolResult(fmt.Sprintf("[dry-run] would transfer %d bytes on %s: %s",
			len(txData), device, hex.EncodeToString(txData)))
	}

	fd, closer, errResult := spiOpenDevice(device, args)
	if errResult != nil {
		return errResult
	}
	defer closer()

	rxData := make([]byte, len(txData))
	speed := getIntArg(args, "speed", 1000000)
	bpw := getIntArg(args, "bits_per_word", 8)

	if err := spiDoTransfer(fd, txData, rxData, speed, bpw); err != nil {
		return ErrorResult(err.Error())
	}

	return NewToolResult(fmt.Sprintf("SPI transfer on %s:\n  TX: %s\n  RX: %s",
		device, hex.EncodeToString(txData), hex.EncodeToString(rxData)))
}

func (t *SPITool) readData(args map[string]interface{}) *ToolResult {
	device, errResult := requireStringArg(args, "device")
	if errResult != nil {
		return errResult
	}
	count := getIntArg(args, "count", 1)
	if count <= 0 || count > 4096 {
		return ErrorResult("count must be between 1 and 4096")
	}

	fd, closer, errResult := spiOpenDevice(device, args)
	if errResult != nil {
		return errResult
	}
	defer closer()

	txData := make([]byte, count)
	rxData := make([]byte, count)
	speed := getIntArg(args, "speed", 1000000)
	bpw := getIntArg(args, "bits_per_word", 8)

	if err := spiDoTransfer(fd, txData, rxData, speed, bpw); err != nil {
		return ErrorResult(err.Error())
	}

	return NewToolResult(fmt.Sprintf("SPI read %d bytes from %s: %s",
		count, device, hex.EncodeToString(rxData)))
}

func (t *SPITool) info(args map[string]interface{}) *ToolResult {
	device, errResult := requireStringArg(args, "device")
	if errResult != nil {
		return errResult
	}

	fd, err := syscall.Open(device, syscall.O_RDWR, 0)
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to open %s: %v", device, err))
	}
	defer syscall.Close(fd)

	var mode uint8
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd),
		SPI_IOC_RD_MODE, uintptr(unsafe.Pointer(&mode))); errno != 0 {
		return ErrorResult(fmt.Sprintf("failed to read SPI mode: %v", errno))
	}

	var bpw uint8
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd),
		SPI_IOC_RD_BITS_PER_WORD, uintptr(unsafe.Pointer(&bpw))); errno != 0 {
		return ErrorResult(fmt.Sprintf("failed to read bits per word: %v", errno))
	}

	var speed uint32
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd),
		SPI_IOC_RD_MAX_SPEED_HZ, uintptr(unsafe.Pointer(&speed))); errno != 0 {
		return ErrorResult(fmt.Sprintf("failed to read SPI speed: %v", errno))
	}

	cpol := (mode >> 1) & 1
	cpha := mode & 1
	modeDesc := fmt.Sprintf("Mode %d (CPOL=%d, CPHA=%d)", mode&3, cpol, cpha)

	var b strings.Builder
	b.WriteString(fmt.Sprintf("SPI device: %s\n", device))
	b.WriteString(fmt.Sprintf("  Mode:          %s\n", modeDesc))
	b.WriteString(fmt.Sprintf("  Bits per word: %d\n", bpw))
	b.WriteString(fmt.Sprintf("  Max speed:     %d Hz", speed))
	if speed >= 1000000 {
		b.WriteString(fmt.Sprintf(" (%.1f MHz)", float64(speed)/1000000.0))
	} else if speed >= 1000 {
		b.WriteString(fmt.Sprintf(" (%.1f kHz)", float64(speed)/1000.0))
	}
	b.WriteString("\n")

	return NewToolResult(b.String())
}

func (t *SPITool) loopbackTest(args map[string]interface{}) *ToolResult {
	device, errResult := requireStringArg(args, "device")
	if errResult != nil {
		return errResult
	}

	fd, closer, errResult := spiOpenDevice(device, args)
	if errResult != nil {
		return errResult
	}
	defer closer()

	testLen := 32
	txData := make([]byte, testLen)
	for i := range txData {
		txData[i] = byte(i)
	}
	rxData := make([]byte, testLen)
	speed := getIntArg(args, "speed", 1000000)
	bpw := getIntArg(args, "bits_per_word", 8)

	if err := spiDoTransfer(fd, txData, rxData, speed, bpw); err != nil {
		return ErrorResultf("SPI loopback test failed: %v", err)
	}

	mismatches := 0
	for i := 0; i < testLen; i++ {
		if txData[i] != rxData[i] {
			mismatches++
		}
	}

	var b strings.Builder
	b.WriteString(fmt.Sprintf("SPI loopback test on %s (%d bytes):\n", device, testLen))
	b.WriteString(fmt.Sprintf("  TX: %s\n", hex.EncodeToString(txData)))
	b.WriteString(fmt.Sprintf("  RX: %s\n", hex.EncodeToString(rxData)))

	if mismatches == 0 {
		b.WriteString("\n  Result: PASS - all bytes match (MOSI->MISO loopback verified)")
	} else {
		b.WriteString(fmt.Sprintf("\n  Result: FAIL - %d/%d bytes mismatched", mismatches, testLen))
		b.WriteString("\n  Check: Is MOSI physically connected to MISO?")
	}

	return NewToolResult(b.String())
}

type CLI struct {
	agent *Agent
}

func NewCLI(agent *Agent) *CLI {
	return &CLI{agent: agent}
}

func (c *CLI) RunREPL() {
	fmt.Printf("AttoClaw v%s - AI Agent with Hardware Access\n", Version)
	fmt.Printf("Model: %s | Max iterations: %d | Session window: %d\n",
		c.agent.config.Model, c.agent.config.MaxIterations, c.agent.config.SessionWindow)
	fmt.Println("Type /help for commands, /quit to exit.")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB input buffer

	var cancelFunc context.CancelFunc
	var cancelMu sync.Mutex

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)

	go func() {
		for range sigCh {
			cancelMu.Lock()
			if cancelFunc != nil {
				fmt.Fprintln(os.Stderr, "\n[interrupted]")
				cancelFunc()
			}
			cancelMu.Unlock()
		}
	}()

	for {
		fmt.Print("attoclaw> ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		if strings.HasPrefix(input, "/") {
			c.handleCommand(input)
			continue
		}

		ctx, cancel := context.WithCancel(context.Background())
		cancelMu.Lock()
		cancelFunc = cancel
		cancelMu.Unlock()

		response, err := c.agent.Run(ctx, input)

		cancelMu.Lock()
		cancelFunc = nil
		cancelMu.Unlock()
		cancel()

		if err != nil {
			if ctx.Err() == context.Canceled {
				fmt.Println("[operation cancelled]")
			} else {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			}
			continue
		}

		fmt.Println()
		fmt.Println(response)
		fmt.Println()
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "input error: %v\n", err)
	}

	fmt.Println("\nGoodbye.")
}

func (c *CLI) handleCommand(input string) {
	fields := strings.Fields(input)
	cmd := strings.ToLower(fields[0])

	switch cmd {
	case "/quit", "/exit", "/q":
		fmt.Println("Goodbye.")
		os.Exit(0)

	case "/help", "/h", "/?":
		fmt.Print(`AttoClaw REPL Commands:

  /help, /h       Show this help
  /quit, /q       Exit
  /clear          Clear conversation history
  /tools          List available tools
  /status         Show agent statistics
  /config         Show current configuration
  /history        Show recent messages
  /session        Show session statistics
  /version        Show version

Tips: Ctrl+C cancels current operation, Ctrl+D exits.

`)

	case "/clear":
		c.agent.ClearSession()
		fmt.Println("[session cleared]")

	case "/tools":
		fmt.Println("\nAvailable tools:")
		for _, name := range c.agent.registry.Names() {
			t := c.agent.registry.Get(name)
			fmt.Printf("  %-16s %s\n", name, t.Description())
		}
		fmt.Printf("\n(%d tools registered)\n\n", c.agent.registry.Count())

	case "/status", "/stats":
		s := c.agent.GetStats()
		uptime := time.Since(s.SessionStartTime).Round(time.Second)
		fmt.Printf("Agent Statistics:\n\n"+
			"  Uptime:     %v\n  Requests:   %d\n  Tool calls: %d\n  Errors:     %d\n"+
			"  Tokens:     %d (prompt: %d, completion: %d)\n",
			uptime, s.TotalRequests, s.TotalToolCalls, s.TotalErrors,
			s.TotalTokensUsed, s.TotalPromptTokens, s.TotalCompTokens)
		if s.LastRequestDur > 0 {
			fmt.Printf("  Last req:   %v\n", s.LastRequestDur.Round(time.Millisecond))
		}
		if len(s.ToolCallCounts) > 0 {
			fmt.Println("\n  Tool call counts:")
			for name, count := range s.ToolCallCounts {
				fmt.Printf("    %-16s %d\n", name, count)
			}
		}
		fmt.Println()

	case "/config":
		cfg := c.agent.config
		fmt.Printf("Configuration:\n\n"+
			"  API key:     %s\n  API base:    %s\n  Model:       %s\n"+
			"  Iterations:  %d\n  Window:      %d\n  Retries:     %d\n"+
			"  HTTP timeout: %ds\n  Exec timeout: %ds\n  Max output:  %d bytes\n"+
			"  Debug:       %v\n\n",
			cfg.MaskedAPIKey(), cfg.APIBase, cfg.Model, cfg.MaxIterations, cfg.SessionWindow,
			cfg.MaxRetries, cfg.HTTPTimeoutSeconds, cfg.ExecTimeoutSeconds, cfg.MaxOutputSize, cfg.Debug)

	case "/history":
		msgs := c.agent.GetSession().Get()
		if len(msgs) == 0 {
			fmt.Println("(no messages in session)")
			return
		}
		startIdx := 0
		if len(msgs) > 20 {
			startIdx = len(msgs) - 20
			fmt.Printf("(showing last 20 of %d messages)\n\n", len(msgs))
		}
		for i := startIdx; i < len(msgs); i++ {
			m := msgs[i]
			preview := truncate(m.Content, 200)
			if m.Role == "assistant" && len(m.ToolCalls) > 0 {
				names := make([]string, len(m.ToolCalls))
				for j, tc := range m.ToolCalls {
					if tc.Function != nil {
						names[j] = tc.Function.Name
					}
				}
				fmt.Printf("[assistant] %s [tools: %s]\n", preview, strings.Join(names, ", "))
			} else if m.Role == "tool" {
				fmt.Printf("[tool %s] %s\n", m.ToolCallID, preview)
			} else {
				fmt.Printf("[%s] %s\n", m.Role, preview)
			}
		}
		fmt.Println()

	case "/session":
		fmt.Printf("Session: %s\n\n", c.agent.GetSession().Stats())

	case "/version":
		fmt.Printf("AttoClaw v%s (%s/%s) [%s/%s, Go %s]\n\n",
			Version, CommitHash, BuildDate, runtime.GOOS, runtime.GOARCH, runtime.Version())

	default:
		fmt.Printf("Unknown command: %s (type /help for commands)\n\n", cmd)
	}
}

func (c *CLI) RunOneShot(message string) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Fprintln(os.Stderr, "\n[interrupted]")
		cancel()
	}()

	response, err := c.agent.Run(ctx, message)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(response)
}

func printDiagnostics(cfg *Config, registry *ToolRegistry) {
	fmt.Fprintf(os.Stderr, "AttoClaw v%s | %s | %s/%s\n", Version, cfg.Model, runtime.GOOS, runtime.GOARCH)
	var hwParts []string
	if runtime.GOOS == "linux" {
		if buses, err := filepath.Glob("/dev/i2c-*"); err == nil && len(buses) > 0 {
			hwParts = append(hwParts, fmt.Sprintf("I2C: %s", strings.Join(buses, ", ")))
		} else {
			hwParts = append(hwParts, "no I2C buses found")
		}
		if devs, err := filepath.Glob("/dev/spidev*"); err == nil && len(devs) > 0 {
			hwParts = append(hwParts, fmt.Sprintf("SPI: %s", strings.Join(devs, ", ")))
		} else {
			hwParts = append(hwParts, "no SPI devices found")
		}
	} else {
		hwParts = append(hwParts, "no I2C buses (non-Linux)", "no SPI devices (non-Linux)")
	}
	fmt.Fprintf(os.Stderr, "Hardware: %s\n", strings.Join(hwParts, " | "))

	fmt.Fprintf(os.Stderr, "API: %s (key: %s)\n", cfg.APIBase, cfg.MaskedAPIKey())
	safetyNote := ""
	if cfg.HardwareDryRun {
		safetyNote = " | dry-run enabled"
	}
	fmt.Fprintf(os.Stderr, "Tools: %d registered%s\n", registry.Count(), safetyNote)
}

func main() {
	message := flag.String("m", "", "Run a single message and exit (one-shot mode)")
	version := flag.Bool("version", false, "Print version and exit")
	debug := flag.Bool("debug", false, "Enable debug logging")
	model := flag.String("model", "", "Override LLM model (e.g. gpt-4o-mini)")
	apiBase := flag.String("api-base", "", "Override API base URL")
	maxIter := flag.Int("max-iterations", 0, "Override maximum agent loop iterations")
	dryRun := flag.Bool("dry-run", false, "Hardware dry-run mode: describe writes without executing")
	flag.Parse()

	if *version {
		fmt.Printf("AttoClaw v%s (%s/%s) [%s/%s, Go %s]\n",
			Version, CommitHash, BuildDate, runtime.GOOS, runtime.GOARCH, runtime.Version())
		os.Exit(0)
	}

	if *debug {
		globalLogger.SetDebug(true)
	}

	cfg, err := LoadConfig()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Configuration error: %v\n", err)
		fmt.Fprintf(os.Stderr, "\nTo get started:\n")
		fmt.Fprintf(os.Stderr, "  export %s=your-api-key\n", EnvAPIKey)
		fmt.Fprintf(os.Stderr, "  ./attoclaw\n\n")
		fmt.Fprintf(os.Stderr, "Or create ~/.attoclaw.json with {\"api_key\": \"your-key\"}\n")
		os.Exit(1)
	}

	if *debug {
		cfg.Debug = true
		globalLogger.SetDebug(true)
	}
	if *model != "" {
		cfg.Model = *model
	}
	if *apiBase != "" {
		cfg.APIBase = strings.TrimRight(*apiBase, "/")
	}
	if *maxIter > 0 {
		cfg.MaxIterations = *maxIter
	}
	if *dryRun {
		cfg.HardwareDryRun = true
	}

	globalLogger.Debug("starting AttoClaw v%s on %s/%s", Version, runtime.GOOS, runtime.GOARCH)

	provider := NewHTTPProvider(cfg)

	registry := NewToolRegistry()
	registry.Register(NewExecTool(cfg.ExecTimeout()))
	registry.Register(&ReadFileTool{})
	registry.Register(&WriteFileTool{})
	registry.Register(&ListDirTool{})
	registry.Register(&EditFileTool{})
	registry.Register(&SearchFilesTool{})
	registry.Register(&SystemInfoTool{})
	registry.Register(&I2CTool{config: cfg})
	registry.Register(&SPITool{config: cfg})

	globalLogger.Debug("registered %d tools: %s", registry.Count(),
		strings.Join(registry.Names(), ", "))

	if *message == "" {
		printDiagnostics(cfg, registry)
	}

	agent := NewAgent(cfg, provider, registry)
	cli := NewCLI(agent)
	if *message != "" {
		globalLogger.Debug("one-shot mode: %s", truncate(*message, 100))
		cli.RunOneShot(*message)
	} else {
		cli.RunREPL()
	}
}
