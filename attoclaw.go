// AttoClaw - A single-file AI agent in Go with I2C/SPI hardware access.
// Zero external dependencies. Stdlib only. Compiles everywhere, hardware access on Linux.
//
// AttoClaw is a lightweight, self-contained AI agent that bridges the gap between
// language models and the physical world. It provides a ReAct (Reasoning + Acting)
// agent loop that connects to any OpenAI-compatible chat completions API, with
// built-in tools for filesystem operations, shell command execution, and direct
// hardware access via I2C and SPI buses on Linux systems.
//
// Architecture:
//   - Single-file Go program with zero external dependencies (stdlib only)
//   - Compiles on all platforms; hardware tools runtime-check for Linux
//   - Non-streaming HTTP provider for simplicity and reliability
//   - Sliding-window session for bounded memory usage
//   - Tool interface pattern for extensibility
//   - ReAct agent loop with configurable iteration limit
//
// Hardware Access:
//   - I2C: Bus detection, device scanning (hybrid SMBus quick write + read byte),
//     raw read/write, register-level operations via ioctl
//   - SPI: Device listing, full-duplex transfer, read-only mode, device info,
//     loopback testing via ioctl with proper struct alignment
//
// Build:
//   go build attoclaw.go
//
// Run (interactive REPL):
//   ATTOCLAW_API_KEY=sk-... ./attoclaw
//
// Run (one-shot mode):
//   ATTOCLAW_API_KEY=sk-... ./attoclaw -m "list files in /tmp"
//
// Configuration:
//   Environment variables: ATTOCLAW_API_KEY, ATTOCLAW_API_BASE, ATTOCLAW_MODEL
//   Config file: ~/.attoclaw.json
//   Priority: env vars > config file > defaults

package main

// ============================================================================
// Section 1: Imports
//
// All imports are from Go's standard library. No external dependencies.
// The unsafe package is required for ioctl system calls used by I2C/SPI tools.
// The sync package is used for mutex-protected signal handler coordination.
// ============================================================================

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
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

// Compile-time assertions that imports are used. The unsafe package is needed
// for constructing pointers to ioctl data structures. The sync package is used
// for the mutex that coordinates signal handling with the agent loop.
var (
	_ = unsafe.Pointer(nil)
	_ = sync.Mutex{}
)

// ============================================================================
// Section 2: Constants & Version
//
// All configurable defaults are defined here as constants. These can be
// overridden via config file (~/.attoclaw.json) or environment variables.
// The version follows semver and is printed on startup and with --version.
// ============================================================================

const (
	// Version is the current release version of AttoClaw.
	Version = "0.1.0"

	// VersionDate is the build date for this release.
	VersionDate = "2025-01-01"

	// DefaultModel is the LLM model to use when none is specified.
	DefaultModel = "gpt-4o"

	// DefaultAPIBase is the base URL for the OpenAI-compatible API.
	DefaultAPIBase = "https://api.openai.com/v1"

	// DefaultMaxIterations caps the ReAct loop to prevent runaway tool calls.
	// Each iteration involves one LLM call, so 20 iterations = 20 API calls max.
	DefaultMaxIterations = 20

	// DefaultSessionWindow is the number of recent messages to keep in context.
	// Older messages are dropped from the sliding window to manage token usage.
	DefaultSessionWindow = 50

	// DefaultHTTPTimeout is the maximum time to wait for an LLM API response.
	// This is generous because complex tool-use responses can take 30-60 seconds.
	DefaultHTTPTimeout = 120 * time.Second

	// DefaultExecTimeout is the maximum time a shell command can run.
	DefaultExecTimeout = 60 * time.Second

	// DefaultMaxRetries is the number of times to retry failed HTTP requests.
	DefaultMaxRetries = 3

	// DefaultRetryBaseDelay is the initial delay before the first retry.
	// Subsequent retries use exponential backoff: delay * 2^(attempt-1).
	DefaultRetryBaseDelay = 1 * time.Second

	// DefaultMaxOutputSize is the maximum bytes of tool output sent to the LLM.
	// Larger outputs are truncated with a notice to prevent token overflow.
	DefaultMaxOutputSize = 100000

	// Environment variable names for configuration.
	EnvAPIKey  = "ATTOCLAW_API_KEY"
	EnvAPIBase = "ATTOCLAW_API_BASE"
	EnvModel   = "ATTOCLAW_MODEL"
	EnvDebug   = "ATTOCLAW_DEBUG"

	// ConfigFileName is the name of the optional JSON config file in $HOME.
	ConfigFileName = ".attoclaw.json"

	// MaxReadFileSize is the maximum file size (10MB) the read_file tool will read.
	MaxReadFileSize = 10 * 1024 * 1024

	// MaxExecOutputSize is the maximum output captured from shell commands.
	MaxExecOutputSize = 1024 * 1024
)

// ============================================================================
// Section 2b: Logging
//
// A minimal leveled logger that writes to stderr. Debug messages are only
// shown when ATTOCLAW_DEBUG=1 or debug is enabled in config. Info and error
// messages are always shown. All log output goes to stderr so that stdout
// remains clean for agent responses (important for one-shot mode piping).
// ============================================================================

// LogLevel represents the severity of a log message.
type LogLevel int

const (
	LogDebug LogLevel = iota
	LogInfo
	LogWarn
	LogError
)

// Logger provides leveled logging to stderr.
type Logger struct {
	level  LogLevel
	prefix string
}

// globalLogger is the package-level logger instance.
var globalLogger = &Logger{level: LogInfo, prefix: "attoclaw"}

// SetDebug enables or disables debug-level logging.
func (l *Logger) SetDebug(enabled bool) {
	if enabled {
		l.level = LogDebug
	} else {
		l.level = LogInfo
	}
}

// Debug logs a debug-level message (only shown when debug is enabled).
func (l *Logger) Debug(format string, args ...interface{}) {
	if l.level <= LogDebug {
		msg := fmt.Sprintf(format, args...)
		fmt.Fprintf(os.Stderr, "[%s][DEBUG] %s\n", l.prefix, msg)
	}
}

// Info logs an info-level message.
func (l *Logger) Info(format string, args ...interface{}) {
	if l.level <= LogInfo {
		msg := fmt.Sprintf(format, args...)
		fmt.Fprintf(os.Stderr, "[%s] %s\n", l.prefix, msg)
	}
}

// Warn logs a warning-level message.
func (l *Logger) Warn(format string, args ...interface{}) {
	if l.level <= LogWarn {
		msg := fmt.Sprintf(format, args...)
		fmt.Fprintf(os.Stderr, "[%s][WARN] %s\n", l.prefix, msg)
	}
}

// Error logs an error-level message.
func (l *Logger) Error(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	fmt.Fprintf(os.Stderr, "[%s][ERROR] %s\n", l.prefix, msg)
}

// ============================================================================
// Section 3: Core Types
//
// These types model the OpenAI chat completions API protocol. They are designed
// to be compatible with any OpenAI-compatible API (OpenAI, Azure, Ollama,
// LM Studio, vLLM, etc.). The JSON tags match the API wire format exactly.
//
// The message flow is:
//   1. User sends Message{Role:"user", Content:"..."}
//   2. LLM responds with Message{Role:"assistant", Content:"...", ToolCalls:[...]}
//   3. Agent executes tools and sends Message{Role:"tool", Content:"...", ToolCallID:"..."}
//   4. Repeat from step 2 until LLM responds without tool calls
// ============================================================================

// Message represents a single message in the conversation. Messages can be
// from the user, assistant, system, or tool. Assistant messages may include
// tool calls; tool messages must reference the tool call they respond to.
type Message struct {
	Role       string     `json:"role"`                  // "system", "user", "assistant", or "tool"
	Content    string     `json:"content"`               // Text content of the message
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`  // Tool invocations (assistant messages only)
	ToolCallID string     `json:"tool_call_id,omitempty"` // ID of the tool call this responds to (tool messages only)
}

// ToolCall represents a single tool invocation requested by the LLM.
// The LLM generates a unique ID for each call, specifies the type as "function",
// and provides the function name and JSON-encoded arguments.
type ToolCall struct {
	ID       string        `json:"id"`                // Unique identifier for this tool call
	Type     string        `json:"type"`              // Always "function" for function calls
	Function *FunctionCall `json:"function,omitempty"` // The function to call with its arguments
}

// FunctionCall holds the function name and its arguments. The Arguments field
// is a JSON string that must be parsed separately into a map. The LLM produces
// this as a string, not a parsed object.
type FunctionCall struct {
	Name      string `json:"name"`      // Name of the function to invoke
	Arguments string `json:"arguments"` // JSON-encoded arguments string
}

// LLMResponse is the parsed response from the LLM provider after extracting
// the relevant fields from the chat completions API response format.
type LLMResponse struct {
	Content      string     `json:"content"`               // Text content (may be empty if only tool calls)
	ToolCalls    []ToolCall `json:"tool_calls,omitempty"`  // Requested tool invocations
	FinishReason string     `json:"finish_reason"`         // "stop", "tool_calls", "length", etc.
	Usage        *UsageInfo `json:"usage,omitempty"`       // Token usage statistics
}

// UsageInfo tracks token consumption for a single API call. This is used
// for monitoring costs and debugging context window issues.
type UsageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`     // Tokens in the input (messages + tools)
	CompletionTokens int `json:"completion_tokens"` // Tokens in the output
	TotalTokens      int `json:"total_tokens"`      // Sum of prompt + completion
}

// ToolDefinition is the JSON schema sent to the LLM describing an available tool.
// This follows the OpenAI function calling format where each tool has a type
// (always "function") and a function descriptor with name, description, and
// JSON Schema parameters.
type ToolDefinition struct {
	Type     string          `json:"type"`     // Always "function"
	Function ToolDefFunction `json:"function"` // Function descriptor
}

// ToolDefFunction describes a tool's name, description, and parameter schema.
// The Parameters field is a JSON Schema object that describes the expected
// arguments. The LLM uses this schema to generate valid function calls.
type ToolDefFunction struct {
	Name        string                 `json:"name"`        // Unique tool name
	Description string                 `json:"description"` // Human-readable description for the LLM
	Parameters  map[string]interface{} `json:"parameters"`  // JSON Schema for the function arguments
}

// AgentStats tracks cumulative statistics across the agent's lifetime.
// These are displayed via the /status REPL command.
type AgentStats struct {
	TotalRequests    int           // Number of LLM API calls made
	TotalToolCalls   int           // Number of tool invocations
	TotalTokensUsed  int           // Cumulative token usage
	TotalPromptTokens int          // Cumulative prompt tokens
	TotalCompTokens  int           // Cumulative completion tokens
	TotalErrors      int           // Number of errors encountered
	SessionStartTime time.Time     // When the agent was created
	LastRequestTime  time.Time     // Time of most recent API call
	LastRequestDur   time.Duration // Duration of most recent API call
	ToolCallCounts   map[string]int // Per-tool invocation counts
}

// ============================================================================
// Section 4: Config
//
// Configuration is loaded in three layers with increasing priority:
//   1. Compiled defaults (constants above)
//   2. Config file (~/.attoclaw.json)
//   3. Environment variables (ATTOCLAW_*)
//
// This allows users to set a base config in the JSON file and override
// specific values per-session via environment variables. The API key is
// the only required field; everything else has sensible defaults.
//
// Example ~/.attoclaw.json:
//   {
//     "api_key": "sk-...",
//     "api_base": "https://api.openai.com/v1",
//     "model": "gpt-4o",
//     "max_iterations": 20,
//     "session_window": 50,
//     "debug": false,
//     "max_retries": 3,
//     "retry_base_delay_ms": 1000,
//     "exec_timeout_seconds": 60,
//     "http_timeout_seconds": 120,
//     "max_output_size": 100000
//   }
// ============================================================================

// Config holds all configuration for the agent. Fields are populated from
// defaults, then the config file, then environment variables.
type Config struct {
	// ApiKey is the authentication key for the LLM API. Required.
	ApiKey string `json:"api_key,omitempty"`

	// ApiBase is the base URL for the OpenAI-compatible API endpoint.
	// Should not include a trailing slash. Example: "https://api.openai.com/v1"
	ApiBase string `json:"api_base,omitempty"`

	// Model is the LLM model identifier to use for chat completions.
	// Examples: "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"
	Model string `json:"model,omitempty"`

	// MaxIterations caps the number of ReAct loop iterations per user message.
	// Each iteration involves one LLM API call. Set to prevent runaway loops.
	MaxIterations int `json:"max_iterations,omitempty"`

	// SessionWindow is the maximum number of messages kept in the sliding
	// context window. Older messages are dropped. The system prompt is always
	// included and does not count toward this limit.
	SessionWindow int `json:"session_window,omitempty"`

	// Debug enables verbose debug logging to stderr when true.
	Debug bool `json:"debug,omitempty"`

	// MaxRetries is the number of times to retry failed HTTP requests before
	// giving up. Uses exponential backoff between retries.
	MaxRetries int `json:"max_retries,omitempty"`

	// RetryBaseDelayMs is the initial retry delay in milliseconds.
	// Each subsequent retry doubles the delay (exponential backoff).
	RetryBaseDelayMs int `json:"retry_base_delay_ms,omitempty"`

	// ExecTimeoutSeconds is the maximum time in seconds a shell command can run.
	ExecTimeoutSeconds int `json:"exec_timeout_seconds,omitempty"`

	// HTTPTimeoutSeconds is the maximum time in seconds to wait for an API response.
	HTTPTimeoutSeconds int `json:"http_timeout_seconds,omitempty"`

	// MaxOutputSize is the maximum number of bytes of tool output to send to the LLM.
	// Larger outputs are truncated with a notice.
	MaxOutputSize int `json:"max_output_size,omitempty"`
}

// LoadConfig builds configuration from defaults, config file, and env vars.
// Priority: env vars > config file > defaults.
func LoadConfig() (*Config, error) {
	cfg := &Config{
		ApiBase:            DefaultAPIBase,
		Model:              DefaultModel,
		MaxIterations:      DefaultMaxIterations,
		SessionWindow:      DefaultSessionWindow,
		Debug:              false,
		MaxRetries:         DefaultMaxRetries,
		RetryBaseDelayMs:   int(DefaultRetryBaseDelay / time.Millisecond),
		ExecTimeoutSeconds: int(DefaultExecTimeout / time.Second),
		HTTPTimeoutSeconds: int(DefaultHTTPTimeout / time.Second),
		MaxOutputSize:      DefaultMaxOutputSize,
	}

	// Try loading config file from ~/.attoclaw.json
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

	// Environment variables override everything
	if v := os.Getenv(EnvAPIKey); v != "" {
		cfg.ApiKey = v
	}
	if v := os.Getenv(EnvAPIBase); v != "" {
		cfg.ApiBase = v
	}
	if v := os.Getenv(EnvModel); v != "" {
		cfg.Model = v
	}
	if v := os.Getenv(EnvDebug); v == "1" || strings.ToLower(v) == "true" {
		cfg.Debug = true
	}

	// Enable debug logging if configured
	if cfg.Debug {
		globalLogger.SetDebug(true)
		globalLogger.Debug("debug logging enabled")
	}

	// Validate required fields
	if cfg.ApiKey == "" {
		return nil, fmt.Errorf("API key not set. Set %s env var or api_key in ~/%s", EnvAPIKey, ConfigFileName)
	}

	// Validate numeric ranges
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

	// Normalize API base: strip trailing slash
	cfg.ApiBase = strings.TrimRight(cfg.ApiBase, "/")

	globalLogger.Debug("config loaded: model=%s api_base=%s max_iter=%d window=%d retries=%d",
		cfg.Model, cfg.ApiBase, cfg.MaxIterations, cfg.SessionWindow, cfg.MaxRetries)

	return cfg, nil
}

// ExecTimeout returns the exec timeout as a time.Duration.
func (c *Config) ExecTimeout() time.Duration {
	return time.Duration(c.ExecTimeoutSeconds) * time.Second
}

// HTTPTimeout returns the HTTP timeout as a time.Duration.
func (c *Config) HTTPTimeout() time.Duration {
	return time.Duration(c.HTTPTimeoutSeconds) * time.Second
}

// RetryBaseDelay returns the retry base delay as a time.Duration.
func (c *Config) RetryBaseDelay() time.Duration {
	return time.Duration(c.RetryBaseDelayMs) * time.Millisecond
}

// MaskedApiKey returns the API key with most characters masked for display.
func (c *Config) MaskedApiKey() string {
	if len(c.ApiKey) <= 8 {
		return "***"
	}
	return c.ApiKey[:4] + "..." + c.ApiKey[len(c.ApiKey)-4:]
}

// mergeConfig applies non-zero values from src onto dst. Zero values in src
// are treated as "not set" and do not override dst values.
func mergeConfig(dst, src *Config) {
	if src.ApiKey != "" {
		dst.ApiKey = src.ApiKey
	}
	if src.ApiBase != "" {
		dst.ApiBase = src.ApiBase
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
}

// ============================================================================
// Section 5: Tool Interface & ToolResult
//
// Every tool in AttoClaw implements the Tool interface. This provides a uniform
// way to describe tools to the LLM (via Parameters() and Description()) and
// execute them (via Execute()). Tools receive parsed arguments as a generic
// map and return a ToolResult.
//
// ToolResult has two text fields: ForLLM (sent back to the model) and ForUser
// (displayed to the human). If ForUser is empty, ForLLM is shown to the user.
// This separation allows tools to provide abbreviated user output while sending
// full details to the LLM for reasoning.
//
// Argument extraction helpers (getStringArg, getIntArg, etc.) handle the
// type conversions needed because JSON numbers arrive as float64 and the LLM
// may produce strings where numbers are expected or vice versa.
// ============================================================================

// Tool is the interface every tool must implement. Tools are registered with
// the ToolRegistry and made available to the LLM via function calling.
type Tool interface {
	// Name returns the unique identifier for this tool. This is the name the
	// LLM uses to invoke the tool in function calls.
	Name() string

	// Description returns a human-readable description of what the tool does.
	// This is included in the system prompt and the tool definition sent to
	// the LLM to help it decide when and how to use the tool.
	Description() string

	// Parameters returns a JSON Schema object describing the tool's expected
	// arguments. The schema follows the OpenAI function calling format with
	// "type", "properties", and "required" fields.
	Parameters() map[string]interface{}

	// Execute runs the tool with the given arguments and returns the result.
	// The context can be used for cancellation (e.g., when the user presses
	// Ctrl+C). Arguments are a parsed map from the JSON the LLM produced.
	Execute(ctx context.Context, args map[string]interface{}) *ToolResult
}

// ToolResult holds the outcome of a tool execution. Every tool invocation
// produces exactly one ToolResult, which is then sent back to the LLM as
// a tool message.
type ToolResult struct {
	// ForLLM is the text sent back to the LLM in the tool response message.
	// This should contain all the information the LLM needs to continue
	// reasoning about the task.
	ForLLM string

	// ForUser is optional text displayed to the human user. If empty, ForLLM
	// is displayed instead. Use this when you want to show a brief summary
	// to the user while sending full details to the LLM.
	ForUser string

	// IsError indicates whether this result represents an error condition.
	// Error results are logged differently and may cause the LLM to try
	// alternative approaches.
	IsError bool
}

// NewToolResult creates a successful tool result with the given text.
func NewToolResult(forLLM string) *ToolResult {
	return &ToolResult{ForLLM: forLLM}
}

// NewToolResultWithUser creates a result with separate LLM and user text.
// The forUser text is displayed to the human; forLLM is sent to the model.
func NewToolResultWithUser(forLLM, forUser string) *ToolResult {
	return &ToolResult{ForLLM: forLLM, ForUser: forUser}
}

// ErrorResult creates an error tool result. The message is sent to both
// the LLM and displayed to the user with error formatting.
func ErrorResult(msg string) *ToolResult {
	return &ToolResult{ForLLM: msg, IsError: true}
}

// ErrorResultf creates a formatted error tool result.
func ErrorResultf(format string, args ...interface{}) *ToolResult {
	return &ToolResult{ForLLM: fmt.Sprintf(format, args...), IsError: true}
}

// ============================================================================
// Section 5b: Argument Extraction Helpers
//
// These functions safely extract typed values from the generic argument map
// that tools receive. They handle the various type representations that can
// occur due to JSON parsing semantics:
//
//   - JSON numbers are parsed as float64 by encoding/json
//   - The LLM may produce string representations of numbers
//   - Boolean values may arrive as actual bools or as strings "true"/"false"
//   - Missing keys return sensible zero values
//
// All helpers are nil-safe and never panic.
// ============================================================================

// getStringArg extracts a string argument from the args map.
// Returns empty string if the key is missing or the value is not a string.
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

// getBoolArg extracts a boolean argument from the args map.
// Handles actual booleans, string representations, and numeric 0/1.
// Returns false if the key is missing.
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

// getIntArg extracts an integer argument from the args map.
// JSON numbers arrive as float64, so this handles the conversion.
// Also handles string representations of integers. Returns defaultVal
// if the key is missing or cannot be converted.
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

// getFloat64Arg extracts a float64 argument from the args map.
// Returns defaultVal if the key is missing or cannot be converted.
func getFloat64Arg(args map[string]interface{}, key string, defaultVal float64) float64 {
	if args == nil {
		return defaultVal
	}
	if v, ok := args[key]; ok {
		switch n := v.(type) {
		case float64:
			return n
		case int:
			return float64(n)
		case string:
			if f, err := strconv.ParseFloat(n, 64); err == nil {
				return f
			}
		}
	}
	return defaultVal
}

// getStringSliceArg extracts a string slice argument from the args map.
// Handles both actual string slices and interface slices containing strings.
func getStringSliceArg(args map[string]interface{}, key string) []string {
	if args == nil {
		return nil
	}
	if v, ok := args[key]; ok {
		switch s := v.(type) {
		case []string:
			return s
		case []interface{}:
			result := make([]string, 0, len(s))
			for _, item := range s {
				if str, ok := item.(string); ok {
					result = append(result, str)
				}
			}
			return result
		}
	}
	return nil
}

// ============================================================================
// Section 6: Tool Registry
//
// The ToolRegistry is a simple collection of tools that provides lookup by
// name, execution, and definition export for the LLM. It uses a slice rather
// than a map to preserve registration order, which determines the order tools
// appear in the system prompt and tool definitions.
//
// The registry is single-threaded (no mutex) because the agent loop processes
// one tool call at a time. Tool registration happens once at startup before
// any concurrent access.
// ============================================================================

// ToolRegistry manages all registered tools. Tools are stored in a slice
// to preserve registration order. Lookup is O(n) but n is always small
// (typically < 10 tools).
type ToolRegistry struct {
	tools []Tool
}

// NewToolRegistry creates an empty tool registry.
func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{tools: make([]Tool, 0, 16)}
}

// Register adds a tool to the registry. Duplicate names are not checked;
// the first registered tool with a given name will be found by Get().
func (r *ToolRegistry) Register(t Tool) {
	r.tools = append(r.tools, t)
	globalLogger.Debug("registered tool: %s", t.Name())
}

// Get retrieves a tool by name, or nil if not found.
func (r *ToolRegistry) Get(name string) Tool {
	for _, t := range r.tools {
		if t.Name() == name {
			return t
		}
	}
	return nil
}

// Has returns true if a tool with the given name is registered.
func (r *ToolRegistry) Has(name string) bool {
	return r.Get(name) != nil
}

// Count returns the number of registered tools.
func (r *ToolRegistry) Count() int {
	return len(r.tools)
}

// Execute runs a named tool with the given arguments. If the tool is not
// found, an error result is returned. The context is passed through to the
// tool's Execute method for cancellation support.
func (r *ToolRegistry) Execute(ctx context.Context, name string, args map[string]interface{}) *ToolResult {
	t := r.Get(name)
	if t == nil {
		available := strings.Join(r.Names(), ", ")
		return ErrorResult(fmt.Sprintf("unknown tool: %s (available: %s)", name, available))
	}
	return t.Execute(ctx, args)
}

// Definitions returns the JSON tool definitions for all registered tools.
// These are sent to the LLM as part of the chat completions request so it
// knows what tools are available and how to call them.
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

// Names returns all tool names in registration order.
func (r *ToolRegistry) Names() []string {
	names := make([]string, len(r.tools))
	for i, t := range r.tools {
		names[i] = t.Name()
	}
	return names
}

// Summary returns a formatted multi-line string listing all tools.
func (r *ToolRegistry) Summary() string {
	var b strings.Builder
	for _, t := range r.tools {
		b.WriteString(fmt.Sprintf("  %-14s %s\n", t.Name(), t.Description()))
	}
	return b.String()
}

// ============================================================================
// Section 7: HTTP Provider
//
// The HTTP provider sends requests to an OpenAI-compatible chat completions
// endpoint. It handles:
//   - Request serialization (messages, tools, model)
//   - Authentication via Bearer token
//   - Response parsing with error handling
//   - Automatic retries with exponential backoff for transient errors
//   - Debug logging of request/response sizes
//
// The provider is designed to work with any API that implements the OpenAI
// chat completions format: OpenAI, Azure OpenAI, Ollama, LM Studio, vLLM,
// Together AI, Groq, etc.
// ============================================================================

// HTTPProvider sends requests to an OpenAI-compatible chat completions API.
type HTTPProvider struct {
	apiBase        string
	apiKey         string
	model          string
	client         *http.Client
	maxRetries     int
	retryBaseDelay time.Duration
}

// NewHTTPProvider creates a new HTTP-based LLM provider with the given
// configuration. The HTTP client timeout and retry settings are configurable.
func NewHTTPProvider(cfg *Config) *HTTPProvider {
	return &HTTPProvider{
		apiBase: cfg.ApiBase,
		apiKey:  cfg.ApiKey,
		model:   cfg.Model,
		client: &http.Client{
			Timeout: cfg.HTTPTimeout(),
		},
		maxRetries:     cfg.MaxRetries,
		retryBaseDelay: cfg.RetryBaseDelay(),
	}
}

// chatCompletionRequest is the request body for the chat completions API.
// It includes the model name, conversation messages, and available tools.
type chatCompletionRequest struct {
	Model    string           `json:"model"`
	Messages []Message        `json:"messages"`
	Tools    []ToolDefinition `json:"tools,omitempty"`
}

// chatCompletionResponse is the full response body from the chat completions API.
// It wraps the choices array, usage statistics, and potential error information.
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

// isRetryableStatusCode returns true for HTTP status codes that indicate
// transient errors worth retrying: rate limits and server errors.
func isRetryableStatusCode(code int) bool {
	switch {
	case code == 429: // Rate limited
		return true
	case code == 502: // Bad gateway
		return true
	case code == 503: // Service unavailable
		return true
	case code == 504: // Gateway timeout
		return true
	case code >= 500 && code < 600: // Other server errors
		return true
	default:
		return false
	}
}

// Call sends messages to the LLM and returns the parsed response.
// It automatically retries on transient HTTP errors with exponential backoff.
// The context can be used for cancellation.
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

	// Retry loop with exponential backoff
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
				// Continue with retry
			}
		}

		resp, err := p.doRequest(ctx, url, bodyBytes)
		if err != nil {
			lastErr = err
			// Check if this is a retryable error
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			globalLogger.Debug("request attempt %d failed: %v", attempt+1, err)
			continue
		}

		return resp, nil
	}

	return nil, fmt.Errorf("all %d attempts failed, last error: %w", p.maxRetries+1, lastErr)
}

// doRequest performs a single HTTP request to the chat completions endpoint.
// It returns the parsed LLMResponse or an error. Retryable errors are returned
// as-is for the caller to decide whether to retry.
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

	// Handle non-200 status codes
	if resp.StatusCode != http.StatusOK {
		errMsg := fmt.Sprintf("API returned status %d: %s", resp.StatusCode, truncate(string(respBytes), 500))
		if isRetryableStatusCode(resp.StatusCode) {
			return nil, fmt.Errorf("%s (retryable)", errMsg)
		}
		return nil, fmt.Errorf("%s", errMsg)
	}

	// Parse JSON response
	var chatResp chatCompletionResponse
	if err := json.Unmarshal(respBytes, &chatResp); err != nil {
		return nil, fmt.Errorf("failed to parse response JSON (%d bytes): %w", len(respBytes), err)
	}

	// Check for API-level errors embedded in the response body
	if chatResp.Error != nil {
		return nil, fmt.Errorf("API error (type=%s, code=%s): %s",
			chatResp.Error.Type, chatResp.Error.Code, chatResp.Error.Message)
	}

	// Validate that we got at least one choice
	if len(chatResp.Choices) == 0 {
		return nil, fmt.Errorf("API returned no choices (model=%s)", chatResp.Model)
	}

	// Extract the first choice (we only use single-completion mode)
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

// ============================================================================
// Section 8: Session
//
// The Session manages an in-memory sliding window of conversation messages.
// It tracks user, assistant, and tool messages but NOT the system prompt
// (which is rebuilt fresh each iteration to include current time, etc.).
//
// When the message count exceeds the window size, older messages are dropped
// from the beginning. This provides bounded memory usage while keeping
// recent context available to the LLM. The window size is configurable.
//
// The session also tracks basic statistics: total messages added, messages
// dropped, and per-role counts.
// ============================================================================

// Session manages an in-memory sliding window of conversation messages.
type Session struct {
	messages     []Message // All messages currently in the window
	window       int       // Maximum number of messages to keep
	totalAdded   int       // Total messages ever added (including dropped)
	totalDropped int       // Messages dropped due to window limit
}

// NewSession creates a new session with the given window size.
func NewSession(window int) *Session {
	return &Session{
		messages: make([]Message, 0, window+10),
		window:   window,
	}
}

// Add appends a message to the session and trims if needed.
func (s *Session) Add(msg Message) {
	s.messages = append(s.messages, msg)
	s.totalAdded++
	s.trim()
}

// AddAll appends multiple messages to the session and trims if needed.
func (s *Session) AddAll(msgs []Message) {
	s.messages = append(s.messages, msgs...)
	s.totalAdded += len(msgs)
	s.trim()
}

// trim drops old messages when the window is exceeded, ensuring that we don't
// break tool call / tool result pairs by always trimming to a user or
// assistant message boundary when possible.
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

// Get returns the messages within the sliding window. The system prompt is
// managed externally; this only tracks user/assistant/tool messages.
func (s *Session) Get() []Message {
	return s.messages
}

// Clear resets the session history.
func (s *Session) Clear() {
	s.messages = s.messages[:0]
	globalLogger.Debug("session cleared")
}

// Len returns the number of messages currently in the session.
func (s *Session) Len() int {
	return len(s.messages)
}

// Stats returns a summary string of session statistics.
func (s *Session) Stats() string {
	roleCounts := make(map[string]int)
	for _, m := range s.messages {
		roleCounts[m.Role]++
	}
	return fmt.Sprintf("messages=%d (user=%d assistant=%d tool=%d) total_added=%d dropped=%d window=%d",
		len(s.messages), roleCounts["user"], roleCounts["assistant"], roleCounts["tool"],
		s.totalAdded, s.totalDropped, s.window)
}

// LastUserMessage returns the most recent user message, or empty string if none.
func (s *Session) LastUserMessage() string {
	for i := len(s.messages) - 1; i >= 0; i-- {
		if s.messages[i].Role == "user" {
			return s.messages[i].Content
		}
	}
	return ""
}

// ============================================================================
// Section 9: System Prompt Builder
//
// The system prompt is rebuilt fresh for every LLM call so that dynamic
// information (current time, working directory) stays current. It includes:
//   - Agent identity and capabilities
//   - Platform information (OS, arch, hostname, cwd, time)
//   - Tool summaries (names and descriptions)
//   - Safety guidelines and behavioral constraints
//   - Hardware-specific notes when on Linux
//
// The prompt is structured with markdown headers for clarity. It aims to be
// comprehensive but concise to minimize token usage.
// ============================================================================

// BuildSystemPrompt constructs the system prompt with platform info, tool
// summaries, and behavioral guidelines. This is called on every LLM iteration
// so dynamic info (time, cwd) stays current.
func BuildSystemPrompt(registry *ToolRegistry) string {
	var b strings.Builder

	// Identity
	b.WriteString("You are AttoClaw v")
	b.WriteString(Version)
	b.WriteString(", a lightweight AI agent that can interact with the local system ")
	b.WriteString("and physical hardware. You run as a single Go binary with zero external dependencies.\n\n")

	// Capabilities overview
	b.WriteString("## Capabilities\n")
	b.WriteString("- Execute shell commands with safety filtering\n")
	b.WriteString("- Read, write, edit, and list files and directories\n")
	b.WriteString("- Search for files by name pattern\n")
	b.WriteString("- Make HTTP requests to external services\n")
	b.WriteString("- Access I2C devices for hardware communication (Linux only)\n")
	b.WriteString("- Access SPI devices for hardware communication (Linux only)\n")
	b.WriteString("- Gather system information\n")
	b.WriteString("\n")

	// Platform info
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

	// Add user info if available
	if user := os.Getenv("USER"); user != "" {
		b.WriteString(fmt.Sprintf("- User: %s\n", user))
	} else if user := os.Getenv("USERNAME"); user != "" {
		b.WriteString(fmt.Sprintf("- User: %s\n", user))
	}

	// Add shell info
	if shell := os.Getenv("SHELL"); shell != "" {
		b.WriteString(fmt.Sprintf("- Shell: %s\n", shell))
	}

	b.WriteString("\n")

	// Tool summaries
	b.WriteString("## Available Tools\n")
	for _, name := range registry.Names() {
		t := registry.Get(name)
		b.WriteString(fmt.Sprintf("- **%s**: %s\n", name, t.Description()))
	}
	b.WriteString("\n")

	// Safety guidelines
	b.WriteString("## Safety Guidelines\n")
	b.WriteString("- Always explain what you are about to do before calling a tool.\n")
	b.WriteString("- For destructive operations (writing files, executing commands, hardware writes), ")
	b.WriteString("state your intent clearly so the user can interrupt if needed.\n")
	b.WriteString("- The exec tool blocks dangerous commands (rm -rf /, dd if=, fork bombs, ")
	b.WriteString("shutdown, reboot, mkfs, writing to block devices).\n")
	b.WriteString("- I2C and SPI tools only work on Linux. They will return a clear error on other platforms.\n")
	b.WriteString("- Hardware write operations (i2c write, spi transfer) require confirm=true as a safety gate.\n")
	b.WriteString("- Never reveal API keys, passwords, or sensitive data in your responses.\n")
	b.WriteString("\n")

	// Behavioral guidelines
	b.WriteString("## Behavioral Guidelines\n")
	b.WriteString("- Keep responses concise and actionable.\n")
	b.WriteString("- If a tool returns an error, explain it clearly and suggest next steps.\n")
	b.WriteString("- When reading large files, summarize the content rather than repeating it verbatim.\n")
	b.WriteString("- When executing commands, prefer simple and well-known tools.\n")
	b.WriteString("- For multi-step tasks, explain your plan before starting.\n")
	b.WriteString("- If you're unsure about something, ask the user rather than guessing.\n")
	b.WriteString("- When working with hardware (I2C/SPI), be extra cautious. Explain what ")
	b.WriteString("each operation does and what to expect.\n")
	b.WriteString("- If a task requires multiple tool calls, proceed step by step and ")
	b.WriteString("report intermediate results.\n")

	// Hardware-specific notes when on Linux
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

// ============================================================================
// Section 10: Agent Loop
//
// The agent implements a ReAct (Reasoning + Acting) pattern:
//   1. User provides input
//   2. Build messages: system prompt + session history + user message
//   3. Call LLM via HTTP provider
//   4. If LLM responds with text only (no tool calls) -> return (done)
//   5. If LLM requests tool calls -> execute each, build result messages
//   6. Add tool results to session and go to step 2
//   7. Repeat up to MaxIterations times
//
// The loop ensures that:
//   - Tool calls are executed in order
//   - Tool results are properly linked back to their calls via ToolCallID
//   - The assistant's intermediate text (before tool calls) is displayed
//   - Token usage is tracked and reported
//   - Context cancellation (Ctrl+C) is respected at each step
//   - Large tool outputs are truncated to prevent token overflow
// ============================================================================

// Agent orchestrates the ReAct loop between user input, LLM, and tools.
type Agent struct {
	config   *Config
	provider *HTTPProvider
	registry *ToolRegistry
	session  *Session
	stats    *AgentStats
}

// NewAgent creates a new Agent with fresh session and statistics.
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
	}
}

// Run processes a single user message through the full ReAct loop.
// It returns the final assistant text response, or an error if the loop
// fails or exceeds the maximum iteration count.
func (a *Agent) Run(ctx context.Context, userInput string) (string, error) {
	// Add the user message to session
	a.session.Add(Message{Role: "user", Content: userInput})

	globalLogger.Debug("starting agent loop for input: %s", truncate(userInput, 100))

	for iteration := 0; iteration < a.config.MaxIterations; iteration++ {
		// Check for cancellation before each iteration
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}

		globalLogger.Debug("iteration %d/%d", iteration+1, a.config.MaxIterations)

		// Build full message list: system + session history
		systemPrompt := BuildSystemPrompt(a.registry)
		sessionMsgs := a.session.Get()
		messages := make([]Message, 0, len(sessionMsgs)+1)
		messages = append(messages, Message{Role: "system", Content: systemPrompt})
		messages = append(messages, sessionMsgs...)

		// Call LLM
		toolDefs := a.registry.Definitions()
		startTime := time.Now()
		resp, err := a.provider.Call(ctx, messages, toolDefs)
		elapsed := time.Since(startTime)

		// Update stats
		a.stats.TotalRequests++
		a.stats.LastRequestTime = startTime
		a.stats.LastRequestDur = elapsed

		if err != nil {
			a.stats.TotalErrors++
			return "", fmt.Errorf("LLM call failed (iteration %d): %w", iteration+1, err)
		}

		// Track token usage
		if resp.Usage != nil {
			a.stats.TotalTokensUsed += resp.Usage.TotalTokens
			a.stats.TotalPromptTokens += resp.Usage.PromptTokens
			a.stats.TotalCompTokens += resp.Usage.CompletionTokens
			fmt.Fprintf(os.Stderr, "[tokens: prompt=%d completion=%d total=%d | time=%v]\n",
				resp.Usage.PromptTokens, resp.Usage.CompletionTokens,
				resp.Usage.TotalTokens, elapsed.Round(time.Millisecond))
		}

		// No tool calls means we are done - the LLM has a final response
		if len(resp.ToolCalls) == 0 {
			assistantMsg := Message{Role: "assistant", Content: resp.Content}
			a.session.Add(assistantMsg)
			globalLogger.Debug("agent loop complete after %d iterations", iteration+1)
			return resp.Content, nil
		}

		// Build the assistant message with tool calls
		assistantMsg := Message{
			Role:      "assistant",
			Content:   resp.Content,
			ToolCalls: resp.ToolCalls,
		}
		a.session.Add(assistantMsg)

		// If the assistant also produced text alongside tool calls, print it
		if resp.Content != "" {
			fmt.Println(resp.Content)
		}

		// Execute each tool call sequentially and collect results
		toolMessages, err := a.executeToolCalls(ctx, resp.ToolCalls)
		if err != nil {
			return "", err
		}

		a.session.AddAll(toolMessages)
	}

	return "", fmt.Errorf("agent exceeded maximum iterations (%d); use /clear to reset or increase max_iterations",
		a.config.MaxIterations)
}

// executeToolCalls processes a batch of tool calls from the LLM, executing
// each one and building the corresponding tool result messages.
func (a *Agent) executeToolCalls(ctx context.Context, toolCalls []ToolCall) ([]Message, error) {
	var toolMessages []Message

	for _, tc := range toolCalls {
		// Check for cancellation between tool calls
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
		fmt.Fprintf(os.Stderr, "[tool: %s]\n", toolName)
		globalLogger.Debug("executing tool %s with args: %s", toolName,
			truncate(tc.Function.Arguments, 200))

		// Parse arguments from JSON string to map
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

		// Execute the tool with timing
		startTime := time.Now()
		result := a.registry.Execute(ctx, toolName, args)
		elapsed := time.Since(startTime)

		// Update stats
		a.stats.TotalToolCalls++
		a.stats.ToolCallCounts[toolName]++
		if result.IsError {
			a.stats.TotalErrors++
		}

		// Truncate large outputs to prevent token overflow
		toolOutput := result.ForLLM
		if len(toolOutput) > a.config.MaxOutputSize {
			truncatedNote := fmt.Sprintf("\n\n[output truncated: %d bytes total, showing first %d bytes]",
				len(toolOutput), a.config.MaxOutputSize)
			toolOutput = toolOutput[:a.config.MaxOutputSize] + truncatedNote
		}

		// Display user-facing output
		display := toolOutput
		if result.ForUser != "" {
			display = result.ForUser
		}
		if result.IsError {
			fmt.Fprintf(os.Stderr, "[error (%v): %s]\n", elapsed.Round(time.Millisecond),
				truncate(display, 200))
		} else {
			preview := truncate(display, 500)
			if preview != "" {
				fmt.Fprintf(os.Stderr, "[result (%v): %s]\n", elapsed.Round(time.Millisecond), preview)
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

// parseToolArgs parses the JSON arguments string from a tool call into a map.
// Returns an empty map (not nil) if the arguments string is empty.
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

// ClearSession resets the conversation history.
func (a *Agent) ClearSession() {
	a.session.Clear()
}

// GetStats returns the current agent statistics.
func (a *Agent) GetStats() *AgentStats {
	return a.stats
}

// GetSession returns the current session (for inspection).
func (a *Agent) GetSession() *Session {
	return a.session
}

// truncate shortens a string to maxLen, appending "..." if truncated.
// Whitespace is trimmed from both ends first.
func truncate(s string, maxLen int) string {
	s = strings.TrimSpace(s)
	// Replace newlines with spaces for single-line display
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// generateToolCallID creates a random tool call ID using crypto/rand.
// The format matches OpenAI's tool call ID format: "call_" + random hex.
func generateToolCallID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return "call_" + hex.EncodeToString(b)
}

// ============================================================================
// Section 11: Tool - exec
//
// The exec tool runs shell commands via /bin/sh -c (or cmd /C on Windows).
// It includes a deny list of dangerous command patterns that are checked
// before execution. Commands run with a configurable timeout (default 60s).
//
// Safety features:
//   - Regex-based deny list blocks destructive commands before they run
//   - Context timeout kills commands that run too long
//   - Output is captured (stdout + stderr combined) with size limits
//   - Exit codes are reported for non-zero exits
//   - Working directory can be specified for relative path operations
//
// The deny list catches common destructive patterns but is not exhaustive.
// It is a safety net, not a security boundary.
// ============================================================================

// ExecTool runs shell commands with safety checks.
type ExecTool struct {
	timeout time.Duration
}

// NewExecTool creates an ExecTool with the given timeout.
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

// denyPatterns are compiled regular expressions that match dangerous shell
// commands. Each pattern targets a specific class of destructive operation.
// The patterns are compiled once at package init time.
var denyPatterns = []*regexp.Regexp{
	// Recursive force-delete from root: rm -rf /, rm -fr /, rm -rf --no-preserve-root /
	regexp.MustCompile(`\brm\s+(-[^\s]*)?-r[^\s]*f[^\s]*\s+/\s*$`),
	regexp.MustCompile(`\brm\s+(-[^\s]*)?-f[^\s]*r[^\s]*\s+/\s*$`),
	regexp.MustCompile(`\brm\s+-rf\s+/\b`),
	regexp.MustCompile(`\brm\s+-fr\s+/\b`),
	regexp.MustCompile(`\brm\s+-rf\s+\*`),

	// dd with input source (could overwrite anything)
	regexp.MustCompile(`\bdd\s+if=`),

	// Fork bombs: various formats
	regexp.MustCompile(`:\(\)\s*\{\s*:\|:\s*&\s*\}\s*;`),
	regexp.MustCompile(`\.\(\)\s*\{\s*\.\|\.\s*&\s*\}\s*;`),

	// System power control commands
	regexp.MustCompile(`\b(shutdown|reboot|poweroff|halt|init\s+[06])\b`),

	// Filesystem creation / formatting
	regexp.MustCompile(`\b(mkfs|mkfs\.\w+|format)\b`),

	// Writing directly to block devices
	regexp.MustCompile(`>\s*/dev/sd`),
	regexp.MustCompile(`>\s*/dev/nvme`),
	regexp.MustCompile(`>\s*/dev/vd`),
	regexp.MustCompile(`>\s*/dev/hd`),
	regexp.MustCompile(`>\s*/dev/mmcblk`),

	// Recursive permission changes on root
	regexp.MustCompile(`\bchmod\s+-R\s+\d+\s+/\s*$`),
	regexp.MustCompile(`\bchown\s+-R\s+\S+\s+/\s*$`),

	// Overwriting boot records
	regexp.MustCompile(`>\s*/dev/sda\b`),

	// Wiping partition tables
	regexp.MustCompile(`\bwipefs\b.*-a\b`),
	regexp.MustCompile(`\bsgdisk\b.*--zap-all\b`),

	// Filling disks
	regexp.MustCompile(`\byes\b.*>\s*/dev/`),
	regexp.MustCompile(`\bcat\s+/dev/(zero|urandom)\s*>\s*/dev/`),
}

// isDeniedCommand checks if a command matches any deny pattern.
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

	// Check deny patterns
	if denied, pattern := isDeniedCommand(command); denied {
		globalLogger.Warn("blocked dangerous command: %s (matched: %s)", command, pattern)
		return ErrorResult(fmt.Sprintf("command blocked by safety filter: %s", command))
	}

	workDir := getStringArg(args, "working_dir")

	// Determine timeout: use override if provided, else default
	timeout := t.timeout
	if overrideSec := getIntArg(args, "timeout_seconds", 0); overrideSec > 0 {
		if overrideSec > 300 {
			overrideSec = 300
		}
		timeout = time.Duration(overrideSec) * time.Second
	}

	// Create command with timeout
	execCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Determine shell based on OS
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
		// Validate working directory exists
		if info, err := os.Stat(workDir); err != nil {
			return ErrorResultf("working directory does not exist: %s", workDir)
		} else if !info.IsDir() {
			return ErrorResultf("working_dir is not a directory: %s", workDir)
		}
		cmd.Dir = workDir
	}

	// Capture stdout and stderr combined with size limit
	var output bytes.Buffer
	output.Grow(4096) // Pre-allocate a reasonable initial size
	cmd.Stdout = &output
	cmd.Stderr = &output

	startTime := time.Now()
	err := cmd.Run()
	elapsed := time.Since(startTime)

	result := output.String()

	// Truncate very large outputs
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
		// Include exit code in output for non-zero exits
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

// ============================================================================
// Section 12: Tool - read_file
//
// Reads file contents and returns them as text. Includes file metadata
// (size, line count) in the response. Supports optional offset and limit
// parameters for reading portions of large files. Files larger than
// MaxReadFileSize (10MB) are rejected to prevent memory issues.
// ============================================================================

// ReadFileTool reads the contents of a file.
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

	// Check file exists and get metadata
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

	// Check file size limit
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

	// Count total lines
	totalLines := strings.Count(content, "\n")
	if len(content) > 0 && content[len(content)-1] != '\n' {
		totalLines++ // Count last line without trailing newline
	}

	// Handle offset and limit for partial reads
	offset := getIntArg(args, "offset", 0)
	limit := getIntArg(args, "limit", 0)

	if offset > 0 || limit > 0 {
		lines := strings.Split(content, "\n")

		// Convert 1-based offset to 0-based index
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

	// Return full file with metadata header
	return NewToolResult(fmt.Sprintf("File: %s (%d bytes, %d lines)\n\n%s",
		path, info.Size(), totalLines, content))
}

// ============================================================================
// Section 13: Tool - write_file
//
// Writes content to a file, creating parent directories as needed. Reports
// whether the file was created new or overwritten, along with byte count
// and line count. Supports optional append mode to add to existing files.
// ============================================================================

// WriteFileTool writes content to a file, creating parent directories as needed.
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

	// Parse file mode (default 0644)
	fileMode := os.FileMode(0644)
	if modeStr := getStringArg(args, "mode"); modeStr != "" {
		parsed, err := strconv.ParseUint(modeStr, 8, 32)
		if err != nil {
			return ErrorResult(fmt.Sprintf("invalid file mode %q: %v", modeStr, err))
		}
		fileMode = os.FileMode(parsed)
	}

	// Check if file already exists
	existed := false
	if _, err := os.Stat(path); err == nil {
		existed = true
	}

	// Create parent directories
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return ErrorResult(fmt.Sprintf("failed to create directory %s: %v", dir, err))
	}

	// Write or append
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

	// Count lines in the written content
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

// ============================================================================
// Section 14: Tool - list_dir
//
// Lists directory contents with file type indicators and optional file sizes.
// Supports showing hidden files and recursive listing. Output is formatted
// with type prefixes (DIR/FILE/LINK) for easy parsing by the LLM.
// ============================================================================

// ListDirTool lists the contents of a directory.
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

// formatSize returns a human-readable file size string.
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

	// Verify path is a directory
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

		// Skip hidden files unless requested
		if !showHidden && strings.HasPrefix(name, ".") {
			continue
		}

		entryInfo, err := entry.Info()
		if err != nil {
			b.WriteString(fmt.Sprintf("ERR:  %s (cannot stat)\n", name))
			continue
		}

		// Determine type
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

	// Add summary line
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

// ============================================================================
// Section 15: Tool - edit_file
//
// Performs surgical find-and-replace edits on a file. The old_string must
// appear exactly once to prevent ambiguous edits. For replacing all
// occurrences, use replace_all=true. The edit preserves the original file
// permissions and reports the line number where the replacement was made.
//
// This tool is preferred over write_file for making targeted changes because
// it validates the file's current state (the old_string must exist) and
// makes minimal modifications.
// ============================================================================

// EditFileTool performs find-and-replace edits on a file.
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

	// Get original file permissions
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
		// Provide helpful context: show the first few lines of the file
		lines := strings.Split(content, "\n")
		preview := strings.Join(lines[:min(len(lines), 5)], "\n")
		return ErrorResult(fmt.Sprintf("old_string not found in file. File starts with:\n%s",
			truncate(preview, 300)))
	}

	if !replaceAll && count > 1 {
		// Find the line numbers of each occurrence to help the user
		lineNums := findOccurrenceLines(content, oldStr)
		return ErrorResult(fmt.Sprintf("old_string found %d times (at lines %s). "+
			"Provide more context to make it unique, or set replace_all=true.",
			count, formatLineNums(lineNums)))
	}

	// Find the line number of the first occurrence for the report
	firstIdx := strings.Index(content, oldStr)
	firstLine := strings.Count(content[:firstIdx], "\n") + 1

	// Perform the replacement
	var newContent string
	if replaceAll {
		newContent = strings.ReplaceAll(content, oldStr, newStr)
	} else {
		newContent = strings.Replace(content, oldStr, newStr, 1)
	}

	// Write back with original permissions
	if err := os.WriteFile(path, []byte(newContent), origMode); err != nil {
		return ErrorResult(fmt.Sprintf("failed to write file: %v", err))
	}

	// Report what was done
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

// findOccurrenceLines returns the line numbers (1-based) where a substring appears.
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

// formatLineNums formats a slice of line numbers as a comma-separated string.
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

// min returns the smaller of two ints.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ============================================================================
// Section 15b: Tool - search_files
//
// Searches for files matching a glob pattern. Useful for discovering files
// in a project or finding specific file types. Uses filepath.WalkDir for
// recursive searching with a configurable depth limit.
// ============================================================================

// SearchFilesTool searches for files by name pattern.
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

	// Resolve to absolute path for depth calculation
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

		// Check context cancellation
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Check depth limit
		depth := strings.Count(path, string(os.PathSeparator)) - baseDepth
		if depth > maxDepth {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		// Skip common uninteresting directories
		if d.IsDir() {
			name := d.Name()
			if name == ".git" || name == "node_modules" || name == "__pycache__" ||
				name == ".venv" || name == "vendor" || name == ".cache" {
				return filepath.SkipDir
			}
			return nil
		}

		// Check if filename matches pattern
		matched, err := filepath.Match(pattern, d.Name())
		if err != nil {
			return nil // Invalid pattern character, skip
		}
		if matched {
			// Return relative path from search root
			relPath, err := filepath.Rel(absPath, path)
			if err != nil {
				relPath = path
			}
			matches = append(matches, relPath)
			if len(matches) >= maxResults {
				return fmt.Errorf("max results reached")
			}
		}

		return nil
	})

	if err != nil && err.Error() != "max results reached" && err != context.Canceled {
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

// ============================================================================
// Section 15c: Tool - system_info
//
// Gathers system information including OS details, CPU count, memory,
// environment variables, and disk usage. Useful for diagnostics and
// understanding the execution environment.
// ============================================================================

// SystemInfoTool returns system information.
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
				"type": "string",
				"description": "Category of info: 'all', 'os', 'env', 'runtime' (default: 'all')",
				"enum": []string{"all", "os", "env", "runtime"},
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
				// Truncate PATH to avoid giant output
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

// ============================================================================
// Section 16: Tool - i2c
// ============================================================================

// I2C ioctl constants. Defined at package level so the file compiles on all
// platforms (macOS, Windows, etc.); the actual ioctl system calls are guarded
// by runtime.GOOS == "linux" checks in each tool action.
//
// These constants come from the Linux kernel headers:
//   /usr/include/linux/i2c.h
//   /usr/include/linux/i2c-dev.h
//
// I2C_SLAVE sets the slave address for subsequent read/write operations.
// I2C_SMBUS performs SMBus-level operations (quick, byte, word, block).
// I2C_RDWR performs raw I2C message transfers.
const (
	// I2C_SLAVE sets the 7-bit slave address for the I2C device.
	// ioctl(fd, I2C_SLAVE, addr)
	I2C_SLAVE = 0x0703

	// I2C_SMBUS performs an SMBus transaction.
	// ioctl(fd, I2C_SMBUS, &smbusIoctlData)
	I2C_SMBUS = 0x0720

	// I2C_RDWR performs a combined I2C message transfer.
	// ioctl(fd, I2C_RDWR, &i2cRdwrData)
	I2C_RDWR = 0x0707

	// SMBus read/write direction flags.
	I2C_SMBUS_READ  = 1 // Read from device
	I2C_SMBUS_WRITE = 0 // Write to device

	// SMBus transaction types (size parameter).
	I2C_SMBUS_QUICK     = 0 // Quick command: just the address + R/W bit
	I2C_SMBUS_BYTE      = 1 // Byte transfer: single byte without register
	I2C_SMBUS_BYTE_DATA = 2 // Byte data: register address + single byte
	I2C_SMBUS_WORD_DATA = 3 // Word data: register address + two bytes
)

// smbusIoctlData mirrors the kernel's i2c_smbus_ioctl_data struct used
// for SMBus transactions via ioctl(fd, I2C_SMBUS, &data).
//
// Kernel definition (from include/uapi/linux/i2c-dev.h):
//   struct i2c_smbus_ioctl_data {
//       __u8 read_write;
//       __u8 command;
//       __u32 size;
//       union i2c_smbus_data __user *data;
//   };
type smbusIoctlData struct {
	readWrite uint8   // I2C_SMBUS_READ or I2C_SMBUS_WRITE
	command   uint8   // Register address or 0 for Quick/Byte
	size      uint32  // I2C_SMBUS_QUICK, I2C_SMBUS_BYTE, etc.
	data      uintptr // Pointer to data buffer (union i2c_smbus_data)
}

// I2CTool provides I2C bus access on Linux via ioctl.
type I2CTool struct{}

func (t *I2CTool) Name() string { return "i2c" }

func (t *I2CTool) Description() string {
	return "Interact with I2C devices on Linux. Actions: detect (find I2C buses), " +
		"scan (probe addresses 0x03-0x77 with grid display), read (read N bytes from device), " +
		"read_register (read from specific register), write (write bytes, requires confirm=true). " +
		"Scan uses hybrid SMBus Quick Write + Read Byte detection."
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
			"confirm": map[string]interface{}{
				"type":        "boolean",
				"description": "Safety gate: must be true for write operations to proceed",
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

	// Scan all valid 7-bit addresses (0x03-0x77)
	// Addresses 0x00-0x02 are reserved, 0x78-0x7F are 10-bit address space
	foundSet := make(map[int]bool)
	var found []int
	for addr := 0x03; addr <= 0x77; addr++ {
		// Set slave address via ioctl
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

	// Build i2cdetect-style grid
	// Header row
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

// describeI2CAddress returns a description of commonly known I2C device addresses.
// This helps identify devices found during a bus scan.
func describeI2CAddress(addr int) string {
	// Common I2C device addresses
	knownDevices := map[int]string{
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

	if desc, ok := knownDevices[addr]; ok {
		return desc
	}
	return ""
}

// smbusQuickWrite attempts an SMBus Quick Write to detect a device.
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

// smbusReadByte attempts an SMBus Read Byte as a fallback detection method.
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
	bus := getIntArg(args, "bus", -1)
	if bus < 0 {
		return ErrorResult("bus argument is required for read")
	}
	addr := getIntArg(args, "address", -1)
	if addr < 0 {
		return ErrorResult("address argument is required for read")
	}
	count := getIntArg(args, "count", 1)
	if count <= 0 || count > 256 {
		return ErrorResult("count must be between 1 and 256")
	}

	devPath := fmt.Sprintf("/dev/i2c-%d", bus)
	fd, err := syscall.Open(devPath, syscall.O_RDWR, 0)
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to open %s: %v", devPath, err))
	}
	defer syscall.Close(fd)

	// Set slave address
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), I2C_SLAVE, uintptr(addr)); errno != 0 {
		return ErrorResult(fmt.Sprintf("failed to set I2C slave address 0x%02x: %v", addr, errno))
	}

	// Read bytes
	buf := make([]byte, count)
	n, err := syscall.Read(fd, buf)
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to read from I2C device: %v", err))
	}

	return NewToolResult(fmt.Sprintf("read %d bytes from bus %d addr 0x%02x: %s",
		n, bus, addr, hex.EncodeToString(buf[:n])))
}

func (t *I2CTool) write(args map[string]interface{}) *ToolResult {
	if !getBoolArg(args, "confirm") {
		return ErrorResult("write operations require confirm=true. This is a safety gate to prevent " +
			"accidental writes to hardware devices. Set confirm=true to proceed.")
	}

	bus := getIntArg(args, "bus", -1)
	if bus < 0 {
		return ErrorResult("bus argument is required for write")
	}
	addr := getIntArg(args, "address", -1)
	if addr < 0 {
		return ErrorResult("address argument is required for write")
	}
	dataHex := getStringArg(args, "data")
	if dataHex == "" {
		return ErrorResult("data argument is required for write (hex string, e.g. 'ff01ab')")
	}

	// Remove any spaces or 0x prefixes from hex string
	dataHex = strings.ReplaceAll(dataHex, " ", "")
	dataHex = strings.ReplaceAll(dataHex, "0x", "")
	dataHex = strings.ReplaceAll(dataHex, "0X", "")

	writeBytes, err := hex.DecodeString(dataHex)
	if err != nil {
		return ErrorResult(fmt.Sprintf("invalid hex data '%s': %v", dataHex, err))
	}
	if len(writeBytes) == 0 {
		return ErrorResult("data must not be empty")
	}
	if len(writeBytes) > 32 {
		return ErrorResult(fmt.Sprintf("data too long: %d bytes (max 32 for safety)", len(writeBytes)))
	}

	devPath := fmt.Sprintf("/dev/i2c-%d", bus)
	fd, err := syscall.Open(devPath, syscall.O_RDWR, 0)
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to open %s: %v", devPath, err))
	}
	defer syscall.Close(fd)

	// Set slave address
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), I2C_SLAVE, uintptr(addr)); errno != 0 {
		return ErrorResult(fmt.Sprintf("failed to set I2C slave address 0x%02x: %v", addr, errno))
	}

	// Write bytes
	n, err := syscall.Write(fd, writeBytes)
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to write to I2C device: %v", err))
	}

	globalLogger.Debug("i2c write: bus=%d addr=0x%02x data=%s", bus, addr, hex.EncodeToString(writeBytes))

	return NewToolResult(fmt.Sprintf("wrote %d bytes to bus %d addr 0x%02x: %s",
		n, bus, addr, hex.EncodeToString(writeBytes)))
}

// readRegister reads a specific register from an I2C device.
// This is a common operation: write the register address, then read the data.
func (t *I2CTool) readRegister(args map[string]interface{}) *ToolResult {
	bus := getIntArg(args, "bus", -1)
	if bus < 0 {
		return ErrorResult("bus argument is required for read_register")
	}
	addr := getIntArg(args, "address", -1)
	if addr < 0 {
		return ErrorResult("address argument is required for read_register")
	}
	register := getIntArg(args, "register", -1)
	if register < 0 || register > 255 {
		return ErrorResult("register argument is required (0-255)")
	}
	count := getIntArg(args, "count", 1)
	if count <= 0 || count > 256 {
		return ErrorResult("count must be between 1 and 256")
	}

	devPath := fmt.Sprintf("/dev/i2c-%d", bus)
	fd, err := syscall.Open(devPath, syscall.O_RDWR, 0)
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to open %s: %v", devPath, err))
	}
	defer syscall.Close(fd)

	// Set slave address
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), I2C_SLAVE, uintptr(addr)); errno != 0 {
		return ErrorResult(fmt.Sprintf("failed to set I2C slave address 0x%02x: %v", addr, errno))
	}

	// Write register address
	regBuf := []byte{byte(register)}
	if _, err := syscall.Write(fd, regBuf); err != nil {
		return ErrorResult(fmt.Sprintf("failed to write register address 0x%02x: %v", register, err))
	}

	// Read data bytes
	buf := make([]byte, count)
	n, err := syscall.Read(fd, buf)
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to read from register 0x%02x: %v", register, err))
	}

	return NewToolResult(fmt.Sprintf("read %d bytes from bus %d addr 0x%02x register 0x%02x: %s",
		n, bus, addr, register, hex.EncodeToString(buf[:n])))
}

// ============================================================================
// Section 17: Tool - spi
//
// SPI (Serial Peripheral Interface) is a synchronous full-duplex bus commonly
// used for high-speed communication with sensors, displays, flash memory,
// and other peripherals. Unlike I2C, SPI uses separate MOSI (Master Out
// Slave In) and MISO (Master In Slave Out) lines, plus a clock and chip
// select, allowing simultaneous send and receive.
//
// On Linux, SPI devices appear as /dev/spidevB.C where B is the bus number
// and C is the chip select number. Access is via ioctl with the spidev driver.
//
// The tool supports:
//   - list: Discover available SPI devices
//   - info: Read current SPI device configuration
//   - transfer: Full-duplex send/receive (requires confirm=true)
//   - read: Receive data by clocking out zeros
//   - loopback_test: Test with MOSI connected to MISO
//
// The SPI ioctl transfer struct must be exactly 32 bytes to match the kernel
// definition. This is verified at compile time by the struct layout.
// ============================================================================

// SPI ioctl constants from the Linux kernel spidev driver.
// These are defined at package level so the file compiles on all platforms.
//
// Source: include/uapi/linux/spi/spidev.h
const (
	// SPI_IOC_MAGIC is the ioctl magic number for SPI.
	SPI_IOC_MAGIC = 'k'

	// SPI_IOC_WR_MODE sets the SPI mode (CPOL/CPHA).
	// Mode 0: CPOL=0, CPHA=0 (most common)
	// Mode 1: CPOL=0, CPHA=1
	// Mode 2: CPOL=1, CPHA=0
	// Mode 3: CPOL=1, CPHA=1
	SPI_IOC_WR_MODE = 0x40016B01

	// SPI_IOC_RD_MODE reads the current SPI mode.
	SPI_IOC_RD_MODE = 0x80016B01

	// SPI_IOC_WR_BITS_PER_WORD sets the word size in bits (usually 8).
	SPI_IOC_WR_BITS_PER_WORD = 0x40016B03

	// SPI_IOC_RD_BITS_PER_WORD reads the current word size.
	SPI_IOC_RD_BITS_PER_WORD = 0x80016B03

	// SPI_IOC_WR_MAX_SPEED_HZ sets the maximum clock speed.
	SPI_IOC_WR_MAX_SPEED_HZ = 0x40046B04

	// SPI_IOC_RD_MAX_SPEED_HZ reads the current maximum clock speed.
	SPI_IOC_RD_MAX_SPEED_HZ = 0x80046B04

	// SPI_IOC_MESSAGE_1 performs a single SPI transfer.
	// The argument is a pointer to one spiIocTransfer struct.
	// Computed as: _IOW(SPI_IOC_MAGIC, 0, struct spi_ioc_transfer)
	SPI_IOC_MESSAGE_1 = 0x40206B00
)

// spiIocTransfer matches the kernel's spi_ioc_transfer struct.
// It MUST be exactly 32 bytes to match the kernel definition.
//
// Kernel definition (from include/uapi/linux/spi/spidev.h):
//
//	struct spi_ioc_transfer {
//	    __u64 tx_buf;         // offset 0:  pointer to TX buffer
//	    __u64 rx_buf;         // offset 8:  pointer to RX buffer
//	    __u32 len;            // offset 16: transfer length
//	    __u32 speed_hz;       // offset 20: transfer speed
//	    __u16 delay_usecs;    // offset 24: delay after transfer
//	    __u8  bits_per_word;  // offset 26: bits per word
//	    __u8  cs_change;      // offset 27: chip select toggle
//	    __u8  tx_nbits;       // offset 28: TX bits per word
//	    __u8  rx_nbits;       // offset 29: RX bits per word
//	    __u8  word_delay_usecs; // offset 30: inter-word delay
//	    __u8  pad;            // offset 31: padding
//	};                        // total: 32 bytes
type spiIocTransfer struct {
	txBuf       uint64 // offset 0:  pointer to TX data buffer
	rxBuf       uint64 // offset 8:  pointer to RX data buffer
	length      uint32 // offset 16: transfer length in bytes
	speedHz     uint32 // offset 20: clock speed for this transfer
	delayUsecs  uint16 // offset 24: delay after transfer in microseconds
	bitsPerWord uint8  // offset 26: bits per word (usually 8)
	csChange    uint8  // offset 27: deselect chip before next transfer
	txNbits     uint8  // offset 28: TX bits per word (SPI_TX_DUAL, SPI_TX_QUAD)
	rxNbits     uint8  // offset 29: RX bits per word (SPI_RX_DUAL, SPI_RX_QUAD)
	wordDelay   uint8  // offset 30: inter-word delay in microseconds
	pad         uint8  // offset 31: padding byte to reach 32 bytes total
}

// SPITool provides SPI bus access on Linux via ioctl.
type SPITool struct{}

func (t *SPITool) Name() string { return "spi" }

func (t *SPITool) Description() string {
	return "Interact with SPI devices on Linux. Actions: list (find SPI devices), " +
		"info (read device configuration), transfer (full-duplex send/receive, requires confirm=true), " +
		"read (receive data by clocking zeros), loopback_test (test with MOSI->MISO jumper)."
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
			"confirm": map[string]interface{}{
				"type":        "boolean",
				"description": "Must be true for transfer operations",
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

func (t *SPITool) configureSPI(fd int, args map[string]interface{}) error {
	mode := getIntArg(args, "mode", 0)
	speed := getIntArg(args, "speed", 1000000)
	bpw := getIntArg(args, "bits_per_word", 8)

	// Set SPI mode
	modeVal := uint8(mode)
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd),
		SPI_IOC_WR_MODE, uintptr(unsafe.Pointer(&modeVal))); errno != 0 {
		return fmt.Errorf("failed to set SPI mode: %v", errno)
	}

	// Set bits per word
	bpwVal := uint8(bpw)
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd),
		SPI_IOC_WR_BITS_PER_WORD, uintptr(unsafe.Pointer(&bpwVal))); errno != 0 {
		return fmt.Errorf("failed to set bits per word: %v", errno)
	}

	// Set max speed
	speedVal := uint32(speed)
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd),
		SPI_IOC_WR_MAX_SPEED_HZ, uintptr(unsafe.Pointer(&speedVal))); errno != 0 {
		return fmt.Errorf("failed to set SPI speed: %v", errno)
	}

	return nil
}

func (t *SPITool) transfer(args map[string]interface{}) *ToolResult {
	if !getBoolArg(args, "confirm") {
		return ErrorResult("transfer operations require confirm=true")
	}

	device := getStringArg(args, "device")
	if device == "" {
		return ErrorResult("device argument is required for transfer")
	}
	dataHex := getStringArg(args, "data")
	if dataHex == "" {
		return ErrorResult("data argument is required for transfer (hex string)")
	}

	txData, err := hex.DecodeString(dataHex)
	if err != nil {
		return ErrorResult(fmt.Sprintf("invalid hex data: %v", err))
	}
	if len(txData) == 0 {
		return ErrorResult("data must not be empty")
	}

	fd, err := syscall.Open(device, syscall.O_RDWR, 0)
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to open %s: %v", device, err))
	}
	defer syscall.Close(fd)

	if err := t.configureSPI(fd, args); err != nil {
		return ErrorResult(err.Error())
	}

	rxData := make([]byte, len(txData))
	speed := getIntArg(args, "speed", 1000000)
	bpw := getIntArg(args, "bits_per_word", 8)

	xfer := spiIocTransfer{
		txBuf:       uint64(uintptr(unsafe.Pointer(&txData[0]))),
		rxBuf:       uint64(uintptr(unsafe.Pointer(&rxData[0]))),
		length:      uint32(len(txData)),
		speedHz:     uint32(speed),
		delayUsecs:  0,
		bitsPerWord: uint8(bpw),
	}

	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd),
		SPI_IOC_MESSAGE_1, uintptr(unsafe.Pointer(&xfer))); errno != 0 {
		return ErrorResult(fmt.Sprintf("SPI transfer failed: %v", errno))
	}

	// Keep pointers alive until ioctl completes
	runtime.KeepAlive(txData)
	runtime.KeepAlive(rxData)

	return NewToolResult(fmt.Sprintf("SPI transfer on %s:\n  TX: %s\n  RX: %s",
		device, hex.EncodeToString(txData), hex.EncodeToString(rxData)))
}

func (t *SPITool) readData(args map[string]interface{}) *ToolResult {
	device := getStringArg(args, "device")
	if device == "" {
		return ErrorResult("device argument is required for read")
	}
	count := getIntArg(args, "count", 1)
	if count <= 0 || count > 4096 {
		return ErrorResult("count must be between 1 and 4096")
	}

	fd, err := syscall.Open(device, syscall.O_RDWR, 0)
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to open %s: %v", device, err))
	}
	defer syscall.Close(fd)

	if err := t.configureSPI(fd, args); err != nil {
		return ErrorResult(err.Error())
	}

	// Send zeros to clock out data
	txData := make([]byte, count)
	rxData := make([]byte, count)
	speed := getIntArg(args, "speed", 1000000)
	bpw := getIntArg(args, "bits_per_word", 8)

	xfer := spiIocTransfer{
		txBuf:       uint64(uintptr(unsafe.Pointer(&txData[0]))),
		rxBuf:       uint64(uintptr(unsafe.Pointer(&rxData[0]))),
		length:      uint32(count),
		speedHz:     uint32(speed),
		delayUsecs:  0,
		bitsPerWord: uint8(bpw),
	}

	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd),
		SPI_IOC_MESSAGE_1, uintptr(unsafe.Pointer(&xfer))); errno != 0 {
		return ErrorResult(fmt.Sprintf("SPI read failed: %v", errno))
	}

	runtime.KeepAlive(txData)
	runtime.KeepAlive(rxData)

	return NewToolResult(fmt.Sprintf("SPI read %d bytes from %s: %s",
		count, device, hex.EncodeToString(rxData)))
}

// info reads the current configuration of an SPI device.
func (t *SPITool) info(args map[string]interface{}) *ToolResult {
	device := getStringArg(args, "device")
	if device == "" {
		return ErrorResult("device argument is required for info")
	}

	fd, err := syscall.Open(device, syscall.O_RDWR, 0)
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to open %s: %v", device, err))
	}
	defer syscall.Close(fd)

	// Read current mode
	var mode uint8
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd),
		SPI_IOC_RD_MODE, uintptr(unsafe.Pointer(&mode))); errno != 0 {
		return ErrorResult(fmt.Sprintf("failed to read SPI mode: %v", errno))
	}

	// Read bits per word
	var bpw uint8
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd),
		SPI_IOC_RD_BITS_PER_WORD, uintptr(unsafe.Pointer(&bpw))); errno != 0 {
		return ErrorResult(fmt.Sprintf("failed to read bits per word: %v", errno))
	}

	// Read max speed
	var speed uint32
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd),
		SPI_IOC_RD_MAX_SPEED_HZ, uintptr(unsafe.Pointer(&speed))); errno != 0 {
		return ErrorResult(fmt.Sprintf("failed to read SPI speed: %v", errno))
	}

	// Describe the mode
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

// loopbackTest performs a loopback test by sending known data and verifying
// it is received back. Requires MOSI to be physically connected to MISO.
// This is useful for verifying SPI hardware and driver configuration.
func (t *SPITool) loopbackTest(args map[string]interface{}) *ToolResult {
	device := getStringArg(args, "device")
	if device == "" {
		return ErrorResult("device argument is required for loopback_test")
	}

	fd, err := syscall.Open(device, syscall.O_RDWR, 0)
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to open %s: %v", device, err))
	}
	defer syscall.Close(fd)

	if err := t.configureSPI(fd, args); err != nil {
		return ErrorResult(err.Error())
	}

	// Generate test pattern: 0x00-0xFF repeating
	testLen := 32
	txData := make([]byte, testLen)
	for i := range txData {
		txData[i] = byte(i)
	}
	rxData := make([]byte, testLen)
	speed := getIntArg(args, "speed", 1000000)
	bpw := getIntArg(args, "bits_per_word", 8)

	xfer := spiIocTransfer{
		txBuf:       uint64(uintptr(unsafe.Pointer(&txData[0]))),
		rxBuf:       uint64(uintptr(unsafe.Pointer(&rxData[0]))),
		length:      uint32(testLen),
		speedHz:     uint32(speed),
		delayUsecs:  0,
		bitsPerWord: uint8(bpw),
	}

	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd),
		SPI_IOC_MESSAGE_1, uintptr(unsafe.Pointer(&xfer))); errno != 0 {
		return ErrorResult(fmt.Sprintf("SPI loopback test failed: %v", errno))
	}

	runtime.KeepAlive(txData)
	runtime.KeepAlive(rxData)

	// Compare TX and RX
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

// ============================================================================
// Section 18: CLI
// ============================================================================

// CLI handles command-line interaction modes.
type CLI struct {
	agent *Agent
}

// NewCLI creates a new CLI wrapper.
func NewCLI(agent *Agent) *CLI {
	return &CLI{agent: agent}
}

// RunREPL starts the interactive read-eval-print loop.
func (c *CLI) RunREPL() {
	fmt.Printf("AttoClaw v%s - AI Agent with Hardware Access\n", Version)
	fmt.Printf("Model: %s | Max iterations: %d | Session window: %d\n",
		c.agent.config.Model, c.agent.config.MaxIterations, c.agent.config.SessionWindow)
	fmt.Println("Type /help for commands, /quit to exit.")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB input buffer

	// Set up signal handling: Ctrl+C cancels current operation
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

		// Handle special commands
		if strings.HasPrefix(input, "/") {
			if c.handleCommand(input) {
				continue
			}
		}

		// Create a cancellable context for this operation
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

// handleCommand processes REPL slash commands. Returns true if handled.
func (c *CLI) handleCommand(input string) bool {
	fields := strings.Fields(input)
	cmd := strings.ToLower(fields[0])

	switch cmd {
	case "/quit", "/exit", "/q":
		fmt.Println("Goodbye.")
		os.Exit(0)
		return true

	case "/help", "/h", "/?":
		fmt.Println("AttoClaw REPL Commands:")
		fmt.Println()
		fmt.Println("  /help, /h       Show this help message")
		fmt.Println("  /quit, /q       Exit AttoClaw")
		fmt.Println("  /clear          Clear conversation history")
		fmt.Println("  /tools          List all available tools with descriptions")
		fmt.Println("  /status         Show agent statistics (requests, tokens, tool usage)")
		fmt.Println("  /config         Show current configuration (API key masked)")
		fmt.Println("  /history        Show recent conversation messages")
		fmt.Println("  /session        Show session statistics")
		fmt.Println("  /version        Show version information")
		fmt.Println()
		fmt.Println("Tips:")
		fmt.Println("  - Ctrl+C cancels the current operation (does not exit)")
		fmt.Println("  - Ctrl+D (EOF) exits the REPL")
		fmt.Println("  - Empty input is ignored")
		fmt.Println("  - Any non-command input is sent to the AI agent")
		fmt.Println()
		return true

	case "/clear":
		c.agent.ClearSession()
		fmt.Println("[session cleared]")
		return true

	case "/tools":
		fmt.Println("Available tools:")
		fmt.Println()
		for _, name := range c.agent.registry.Names() {
			t := c.agent.registry.Get(name)
			fmt.Printf("  %-16s %s\n", name, t.Description())
		}
		fmt.Println()
		fmt.Printf("(%d tools registered)\n", c.agent.registry.Count())
		fmt.Println()
		return true

	case "/status", "/stats":
		stats := c.agent.GetStats()
		uptime := time.Since(stats.SessionStartTime).Round(time.Second)
		fmt.Println("Agent Statistics:")
		fmt.Println()
		fmt.Printf("  Uptime:             %v\n", uptime)
		fmt.Printf("  Total requests:     %d\n", stats.TotalRequests)
		fmt.Printf("  Total tool calls:   %d\n", stats.TotalToolCalls)
		fmt.Printf("  Total errors:       %d\n", stats.TotalErrors)
		fmt.Printf("  Total tokens:       %d\n", stats.TotalTokensUsed)
		fmt.Printf("  Prompt tokens:      %d\n", stats.TotalPromptTokens)
		fmt.Printf("  Completion tokens:  %d\n", stats.TotalCompTokens)
		if stats.LastRequestDur > 0 {
			fmt.Printf("  Last request time:  %v\n", stats.LastRequestDur.Round(time.Millisecond))
		}
		if len(stats.ToolCallCounts) > 0 {
			fmt.Println()
			fmt.Println("  Tool call counts:")
			for name, count := range stats.ToolCallCounts {
				fmt.Printf("    %-16s %d\n", name, count)
			}
		}
		fmt.Println()
		return true

	case "/config":
		cfg := c.agent.config
		fmt.Println("Current Configuration:")
		fmt.Println()
		fmt.Printf("  API key:         %s\n", cfg.MaskedApiKey())
		fmt.Printf("  API base:        %s\n", cfg.ApiBase)
		fmt.Printf("  Model:           %s\n", cfg.Model)
		fmt.Printf("  Max iterations:  %d\n", cfg.MaxIterations)
		fmt.Printf("  Session window:  %d\n", cfg.SessionWindow)
		fmt.Printf("  Max retries:     %d\n", cfg.MaxRetries)
		fmt.Printf("  HTTP timeout:    %ds\n", cfg.HTTPTimeoutSeconds)
		fmt.Printf("  Exec timeout:    %ds\n", cfg.ExecTimeoutSeconds)
		fmt.Printf("  Max output:      %d bytes\n", cfg.MaxOutputSize)
		fmt.Printf("  Debug:           %v\n", cfg.Debug)
		fmt.Println()
		return true

	case "/history":
		msgs := c.agent.GetSession().Get()
		if len(msgs) == 0 {
			fmt.Println("(no messages in session)")
			fmt.Println()
			return true
		}
		// Show last N messages
		showCount := 20
		startIdx := 0
		if len(msgs) > showCount {
			startIdx = len(msgs) - showCount
			fmt.Printf("(showing last %d of %d messages)\n\n", showCount, len(msgs))
		}
		for i := startIdx; i < len(msgs); i++ {
			msg := msgs[i]
			content := truncate(msg.Content, 200)
			switch msg.Role {
			case "user":
				fmt.Printf("[user] %s\n", content)
			case "assistant":
				if len(msg.ToolCalls) > 0 {
					toolNames := make([]string, len(msg.ToolCalls))
					for j, tc := range msg.ToolCalls {
						if tc.Function != nil {
							toolNames[j] = tc.Function.Name
						}
					}
					fmt.Printf("[assistant] %s [tools: %s]\n", content, strings.Join(toolNames, ", "))
				} else {
					fmt.Printf("[assistant] %s\n", content)
				}
			case "tool":
				fmt.Printf("[tool %s] %s\n", msg.ToolCallID, content)
			}
		}
		fmt.Println()
		return true

	case "/session":
		session := c.agent.GetSession()
		fmt.Println("Session Info:")
		fmt.Printf("  %s\n", session.Stats())
		fmt.Println()
		return true

	case "/version":
		fmt.Printf("AttoClaw v%s (%s) [%s/%s]\n", Version, VersionDate, runtime.GOOS, runtime.GOARCH)
		fmt.Printf("Go: %s\n", runtime.Version())
		fmt.Println()
		return true

	default:
		fmt.Printf("Unknown command: %s\n", cmd)
		fmt.Println("Type /help for a list of available commands.")
		fmt.Println()
		return true
	}
}

// RunOneShot processes a single message and exits.
func (c *CLI) RunOneShot(message string) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Signal handling for one-shot mode
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

// ============================================================================
// Section 19: main()
//
// Entry point for AttoClaw. Handles:
//   1. Command-line flag parsing (-m for one-shot, --version, --debug)
//   2. Configuration loading (env vars, config file, defaults)
//   3. HTTP provider creation
//   4. Tool registry population
//   5. Agent creation
//   6. Mode dispatch (one-shot vs REPL)
//
// Exit codes:
//   0 - Success
//   1 - Configuration error or runtime error
// ============================================================================

func main() {
	// Parse command-line flags
	message := flag.String("m", "", "Run a single message and exit (one-shot mode)")
	version := flag.Bool("version", false, "Print version and exit")
	debug := flag.Bool("debug", false, "Enable debug logging")
	model := flag.String("model", "", "Override LLM model (e.g. gpt-4o-mini)")
	apiBase := flag.String("api-base", "", "Override API base URL")
	maxIter := flag.Int("max-iterations", 0, "Override maximum agent loop iterations")
	flag.Parse()

	// Handle --version
	if *version {
		fmt.Printf("AttoClaw v%s (%s) [%s/%s]\n", Version, VersionDate, runtime.GOOS, runtime.GOARCH)
		fmt.Printf("Go: %s\n", runtime.Version())
		os.Exit(0)
	}

	// Enable debug if flag is set (before config load)
	if *debug {
		globalLogger.SetDebug(true)
	}

	// Load configuration from defaults, config file, and env vars
	cfg, err := LoadConfig()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Configuration error: %v\n", err)
		fmt.Fprintf(os.Stderr, "\nTo get started:\n")
		fmt.Fprintf(os.Stderr, "  export %s=your-api-key\n", EnvAPIKey)
		fmt.Fprintf(os.Stderr, "  ./attoclaw\n\n")
		fmt.Fprintf(os.Stderr, "Or create ~/.attoclaw.json with {\"api_key\": \"your-key\"}\n")
		os.Exit(1)
	}

	// Apply command-line overrides
	if *debug {
		cfg.Debug = true
		globalLogger.SetDebug(true)
	}
	if *model != "" {
		cfg.Model = *model
	}
	if *apiBase != "" {
		cfg.ApiBase = strings.TrimRight(*apiBase, "/")
	}
	if *maxIter > 0 {
		cfg.MaxIterations = *maxIter
	}

	globalLogger.Debug("starting AttoClaw v%s on %s/%s", Version, runtime.GOOS, runtime.GOARCH)

	// Create HTTP provider with retry support
	provider := NewHTTPProvider(cfg)

	// Create tool registry and register all tools.
	// Tools are registered in a logical order:
	//   1. Execution and filesystem tools (most commonly used)
	//   2. Discovery and inspection tools
	//   3. Hardware tools (I2C, SPI)
	registry := NewToolRegistry()

	// Core filesystem and execution tools
	registry.Register(NewExecTool(cfg.ExecTimeout()))
	registry.Register(&ReadFileTool{})
	registry.Register(&WriteFileTool{})
	registry.Register(&ListDirTool{})
	registry.Register(&EditFileTool{})

	// Discovery and inspection tools
	registry.Register(&SearchFilesTool{})
	registry.Register(&SystemInfoTool{})

	// Hardware access tools (compile everywhere, execute only on Linux)
	registry.Register(&I2CTool{})
	registry.Register(&SPITool{})

	globalLogger.Debug("registered %d tools: %s", registry.Count(),
		strings.Join(registry.Names(), ", "))

	// Create agent with session
	agent := NewAgent(cfg, provider, registry)

	// Create CLI
	cli := NewCLI(agent)

	// Dispatch to the appropriate mode
	if *message != "" {
		// One-shot mode: process single message and exit
		globalLogger.Debug("one-shot mode: %s", truncate(*message, 100))
		cli.RunOneShot(*message)
	} else {
		// Interactive REPL mode
		cli.RunREPL()
	}
}
