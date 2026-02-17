package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

type mockTool struct {
	name   string
	result *ToolResult
}

func (m *mockTool) Name() string        { return m.name }
func (m *mockTool) Description() string { return "mock tool" }
func (m *mockTool) Parameters() map[string]interface{} {
	return map[string]interface{}{"type": "object", "properties": map[string]interface{}{}}
}
func (m *mockTool) Execute(ctx context.Context, args map[string]interface{}) *ToolResult {
	return m.result
}

func TestConfigDefaults(t *testing.T) {
	cfg := &Config{}
	if cfg.IsI2CWriteAllowed(0x48) {
		t.Error("expected IsI2CWriteAllowed to return false on zero-value Config")
	}
	if cfg.IsSPIDeviceAllowed("/dev/spidev0.0") {
		t.Error("expected IsSPIDeviceAllowed to return false on zero-value Config")
	}
}

func TestConfigIsI2CWriteAllowed(t *testing.T) {
	cfg := &Config{AllowedI2CAddrs: []int{0x48, 0x4C}}

	tests := []struct {
		addr int
		want bool
	}{
		{0x48, true},
		{0x4C, true},
		{0x50, false},
		{0x00, false},
	}
	for _, tc := range tests {
		got := cfg.IsI2CWriteAllowed(tc.addr)
		if got != tc.want {
			t.Errorf("IsI2CWriteAllowed(0x%02X) = %v, want %v", tc.addr, got, tc.want)
		}
	}
}

func TestConfigIsSPIDeviceAllowed(t *testing.T) {
	cfg := &Config{AllowedSPIDevices: []string{"/dev/spidev0.0"}}

	tests := []struct {
		device string
		want   bool
	}{
		{"/dev/spidev0.0", true},
		{"/dev/spidev0.1", false},
		{"/dev/spidev1.0", false},
		{"", false},
	}
	for _, tc := range tests {
		got := cfg.IsSPIDeviceAllowed(tc.device)
		if got != tc.want {
			t.Errorf("IsSPIDeviceAllowed(%q) = %v, want %v", tc.device, got, tc.want)
		}
	}
}

func TestConfigMaskedAPIKey(t *testing.T) {
	tests := []struct {
		key  string
		want string
	}{
		{"sk-abcdefghijklmnop", "sk-a...mnop"},
		{"abc", "***"},
		{"12345678", "***"},
		{"123456789", "1234...6789"},
	}
	for _, tc := range tests {
		cfg := &Config{APIKey: tc.key}
		got := cfg.MaskedAPIKey()
		if got != tc.want {
			t.Errorf("MaskedAPIKey() for key len %d = %q, want %q", len(tc.key), got, tc.want)
		}
	}
}

func TestMergeConfig(t *testing.T) {
	dst := &Config{
		Model:         "gpt-4o",
		MaxIterations: 20,
		APIBase:       "https://api.openai.com/v1",
	}
	src := &Config{
		Model: "gpt-4o-mini",
	}

	mergeConfig(dst, src)

	if dst.Model != "gpt-4o-mini" {
		t.Errorf("expected Model to be updated to %q, got %q", "gpt-4o-mini", dst.Model)
	}
	if dst.MaxIterations != 20 {
		t.Errorf("expected MaxIterations to remain 20, got %d", dst.MaxIterations)
	}
	if dst.APIBase != "https://api.openai.com/v1" {
		t.Errorf("expected APIBase to remain unchanged, got %q", dst.APIBase)
	}
}

func TestRequireStringArg(t *testing.T) {
	args := map[string]interface{}{"name": "hello"}

	val, err := requireStringArg(args, "name")
	if err != nil {
		t.Fatalf("expected no error, got %v", err.ForLLM)
	}
	if val != "hello" {
		t.Errorf("expected %q, got %q", "hello", val)
	}

	_, err = requireStringArg(args, "missing")
	if err == nil {
		t.Error("expected error for missing key")
	}
	if err != nil && !err.IsError {
		t.Error("expected IsError to be true")
	}

	_, err = requireStringArg(nil, "key")
	if err == nil {
		t.Error("expected error for nil args")
	}
}

func TestRequireIntArg(t *testing.T) {
	args := map[string]interface{}{
		"port":     float64(8080),
		"negative": float64(-1),
	}

	val, err := requireIntArg(args, "port")
	if err != nil {
		t.Fatalf("expected no error, got %v", err.ForLLM)
	}
	if val != 8080 {
		t.Errorf("expected 8080, got %d", val)
	}

	_, err = requireIntArg(args, "missing")
	if err == nil {
		t.Error("expected error for missing key")
	}

	// Negative values trigger the error path (default is -1, returned when missing;
	// negative values are treated as "not provided").
	_, err = requireIntArg(args, "negative")
	if err == nil {
		t.Error("expected error for negative value")
	}
}

func TestGetStringArg(t *testing.T) {
	args := map[string]interface{}{
		"str":  "hello",
		"num":  float64(42),
		"flag": true,
	}

	tests := []struct {
		key  string
		want string
	}{
		{"str", "hello"},
		{"num", "42"},
		{"flag", "true"},
		{"missing", ""},
	}
	for _, tc := range tests {
		got := getStringArg(args, tc.key)
		if got != tc.want {
			t.Errorf("getStringArg(args, %q) = %q, want %q", tc.key, got, tc.want)
		}
	}
}

func TestGetIntArg(t *testing.T) {
	args := map[string]interface{}{
		"float_val":  float64(99),
		"int_val":    int(42),
		"string_val": "7",
	}

	tests := []struct {
		key        string
		defaultVal int
		want       int
	}{
		{"float_val", 0, 99},
		{"int_val", 0, 42},
		{"string_val", 0, 7},
		{"missing", 55, 55},
	}
	for _, tc := range tests {
		got := getIntArg(args, tc.key, tc.defaultVal)
		if got != tc.want {
			t.Errorf("getIntArg(args, %q, %d) = %d, want %d", tc.key, tc.defaultVal, got, tc.want)
		}
	}
}

func TestToolRegistry(t *testing.T) {
	reg := NewToolRegistry()

	result := NewToolResult("mock output")
	mock := &mockTool{name: "test_tool", result: result}
	reg.Register(mock)

	if reg.Count() != 1 {
		t.Errorf("expected Count() = 1, got %d", reg.Count())
	}

	if !reg.Has("test_tool") {
		t.Error("expected Has(test_tool) to be true")
	}
	if reg.Has("nonexistent") {
		t.Error("expected Has(nonexistent) to be false")
	}

	got := reg.Get("test_tool")
	if got == nil {
		t.Fatal("expected Get(test_tool) to return a tool")
	}
	if got.Name() != "test_tool" {
		t.Errorf("expected tool name %q, got %q", "test_tool", got.Name())
	}

	if reg.Get("unknown") != nil {
		t.Error("expected Get(unknown) to return nil")
	}

	names := reg.Names()
	if len(names) != 1 || names[0] != "test_tool" {
		t.Errorf("expected Names() = [test_tool], got %v", names)
	}

	ctx := context.Background()
	execResult := reg.Execute(ctx, "test_tool", nil)
	if execResult.ForLLM != "mock output" {
		t.Errorf("expected Execute result %q, got %q", "mock output", execResult.ForLLM)
	}

	unknownResult := reg.Execute(ctx, "no_such_tool", nil)
	if !unknownResult.IsError {
		t.Error("expected Execute for unknown tool to return an error result")
	}
}

func TestSessionAddAndGet(t *testing.T) {
	s := NewSession(50)

	s.Add(Message{Role: "user", Content: "hello"})
	s.Add(Message{Role: "assistant", Content: "hi"})

	msgs := s.Get()
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}
	if msgs[0].Content != "hello" {
		t.Errorf("expected first message content %q, got %q", "hello", msgs[0].Content)
	}
	if msgs[1].Content != "hi" {
		t.Errorf("expected second message content %q, got %q", "hi", msgs[1].Content)
	}
	if s.Len() != 2 {
		t.Errorf("expected Len() = 2, got %d", s.Len())
	}
}

func TestSessionSlidingWindow(t *testing.T) {
	s := NewSession(5)

	for i := 0; i < 8; i++ {
		s.Add(Message{Role: "user", Content: "msg"})
	}

	if s.Len() > 5 {
		t.Errorf("expected session length <= 5 after sliding window, got %d", s.Len())
	}
}

func TestSessionClear(t *testing.T) {
	s := NewSession(50)
	s.Add(Message{Role: "user", Content: "one"})
	s.Add(Message{Role: "user", Content: "two"})

	s.Clear()

	if s.Len() != 0 {
		t.Errorf("expected Len() = 0 after Clear(), got %d", s.Len())
	}
	if len(s.Get()) != 0 {
		t.Errorf("expected Get() to return empty slice after Clear(), got %d items", len(s.Get()))
	}
}

func TestI2CWriteBlockedByDefault(t *testing.T) {
	cfg := &Config{}
	if cfg.IsI2CWriteAllowed(0x48) {
		t.Error("expected I2C write to be blocked by default (empty AllowedI2CAddrs)")
	}
}

func TestI2CWriteAllowedWithAllowlist(t *testing.T) {
	cfg := &Config{AllowedI2CAddrs: []int{0x48}}
	if !cfg.IsI2CWriteAllowed(0x48) {
		t.Error("expected I2C write to 0x48 to be allowed when in allowlist")
	}
}

func TestSPITransferBlockedByDefault(t *testing.T) {
	cfg := &Config{}
	if cfg.IsSPIDeviceAllowed("/dev/spidev0.0") {
		t.Error("expected SPI transfer to be blocked by default (empty AllowedSPIDevices)")
	}
}

func TestSPITransferAllowedWithAllowlist(t *testing.T) {
	cfg := &Config{AllowedSPIDevices: []string{"/dev/spidev0.0"}}
	if !cfg.IsSPIDeviceAllowed("/dev/spidev0.0") {
		t.Error("expected SPI device /dev/spidev0.0 to be allowed when in allowlist")
	}
}

func TestDenyPatterns(t *testing.T) {
	tests := []struct {
		command string
		blocked bool
	}{
		{"rm -rf /", true},
		{"rm -rf *", true},
		{"dd if=/dev/zero of=/dev/sda", true},
		{"shutdown now", true},
		{"mkfs.ext4 /dev/sda1", true},
		{"ls -la", false},
		{"cat /etc/hosts", false},
		{"echo hello", false},
	}
	for _, tc := range tests {
		denied, _ := isDeniedCommand(tc.command)
		if denied != tc.blocked {
			t.Errorf("isDeniedCommand(%q) = %v, want %v", tc.command, denied, tc.blocked)
		}
	}
}

func TestTruncate(t *testing.T) {
	short := "hello"
	if got := truncate(short, 100); got != short {
		t.Errorf("expected short string unchanged, got %q", got)
	}

	long := "abcdefghijklmnopqrstuvwxyz"
	got := truncate(long, 10)
	if len(got) > 13 { // 10 + len("...")
		t.Errorf("expected truncated string length <= 13, got %d (%q)", len(got), got)
	}
	if got != "abcdefghij..." {
		t.Errorf("expected %q, got %q", "abcdefghij...", got)
	}
}

func TestAgentLoopWithToolCall(t *testing.T) {
	var mu sync.Mutex
	var callCount int
	var requests []chatCompletionRequest

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req chatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("failed to decode request: %v", err)
			http.Error(w, "bad request", 400)
			return
		}

		mu.Lock()
		callCount++
		n := callCount
		requests = append(requests, req)
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")

		if n == 1 {
			// First call: return a tool_calls response
			json.NewEncoder(w).Encode(chatCompletionResponse{
				ID:    "chatcmpl-1",
				Model: "test-model",
				Choices: []struct {
					Index        int     `json:"index"`
					Message      Message `json:"message"`
					FinishReason string  `json:"finish_reason"`
				}{
					{
						Index: 0,
						Message: Message{
							Role: "assistant",
							ToolCalls: []ToolCall{
								{
									ID:   "call_abc123",
									Type: "function",
									Function: &FunctionCall{
										Name:      "test_tool",
										Arguments: "{}",
									},
								},
							},
						},
						FinishReason: "tool_calls",
					},
				},
				Usage: &UsageInfo{
					PromptTokens:     15,
					CompletionTokens: 10,
					TotalTokens:      25,
				},
			})
		} else {
			// Second call: return a plain text response (agent loop terminates)
			json.NewEncoder(w).Encode(chatCompletionResponse{
				ID:    "chatcmpl-2",
				Model: "test-model",
				Choices: []struct {
					Index        int     `json:"index"`
					Message      Message `json:"message"`
					FinishReason string  `json:"finish_reason"`
				}{
					{
						Index: 0,
						Message: Message{
							Role:    "assistant",
							Content: "The tool returned: mock output",
						},
						FinishReason: "stop",
					},
				},
				Usage: &UsageInfo{
					PromptTokens:     10,
					CompletionTokens: 10,
					TotalTokens:      20,
				},
			})
		}
	}))
	defer server.Close()

	cfg := &Config{
		APIKey:             "test-key",
		APIBase:            server.URL,
		Model:              "test-model",
		MaxIterations:      10,
		SessionWindow:      50,
		MaxRetries:         0,
		HTTPTimeoutSeconds: 10,
		MaxOutputSize:      100000,
	}
	provider := &HTTPProvider{
		apiBase:    cfg.APIBase,
		apiKey:     cfg.APIKey,
		model:      cfg.Model,
		client:     &http.Client{Timeout: 10 * time.Second},
		maxRetries: 0,
	}
	registry := NewToolRegistry()
	registry.Register(&mockTool{
		name:   "test_tool",
		result: NewToolResult("mock output"),
	})

	agent := NewAgent(cfg, provider, registry)
	agent.output = io.Discard

	ctx := context.Background()
	result, err := agent.Run(ctx, "please use the test tool")
	if err != nil {
		t.Fatalf("Agent.Run() returned error: %v", err)
	}

	// 1. Final text returned
	if result != "The tool returned: mock output" {
		t.Errorf("expected final response %q, got %q", "The tool returned: mock output", result)
	}

	// 2. Exactly 2 HTTP calls
	mu.Lock()
	finalCount := callCount
	capturedRequests := make([]chatCompletionRequest, len(requests))
	copy(capturedRequests, requests)
	mu.Unlock()

	if finalCount != 2 {
		t.Fatalf("expected 2 HTTP calls, got %d", finalCount)
	}

	// 3. First request contains system + user messages + tool definitions
	firstReq := capturedRequests[0]
	if len(firstReq.Messages) < 2 {
		t.Fatalf("expected at least 2 messages in first request, got %d", len(firstReq.Messages))
	}
	if firstReq.Messages[0].Role != "system" {
		t.Errorf("expected first message role 'system', got %q", firstReq.Messages[0].Role)
	}
	if firstReq.Messages[1].Role != "user" {
		t.Errorf("expected second message role 'user', got %q", firstReq.Messages[1].Role)
	}
	if !strings.Contains(firstReq.Messages[1].Content, "please use the test tool") {
		t.Errorf("expected user message in first request, got %q", firstReq.Messages[1].Content)
	}
	if len(firstReq.Tools) == 0 {
		t.Error("expected tool definitions in first request")
	}

	// 4. Second request contains the tool result with correct tool_call_id
	secondReq := capturedRequests[1]
	foundToolResult := false
	for _, msg := range secondReq.Messages {
		if msg.Role == "tool" && msg.ToolCallID == "call_abc123" {
			foundToolResult = true
			if msg.Content != "mock output" {
				t.Errorf("expected tool result content %q, got %q", "mock output", msg.Content)
			}
		}
	}
	if !foundToolResult {
		t.Error("expected tool result message with tool_call_id 'call_abc123' in second request")
	}

	// 5. Agent stats reflect 2 requests, 1 tool call, 45 total tokens
	stats := agent.GetStats()
	if stats.TotalRequests != 2 {
		t.Errorf("expected 2 total requests, got %d", stats.TotalRequests)
	}
	if stats.TotalToolCalls != 1 {
		t.Errorf("expected 1 total tool call, got %d", stats.TotalToolCalls)
	}
	if stats.TotalTokensUsed != 45 {
		t.Errorf("expected 45 total tokens, got %d", stats.TotalTokensUsed)
	}

	// 6. Session contains 4 messages: user -> assistant(tool_calls) -> tool -> assistant(final)
	msgs := agent.GetSession().Get()
	if len(msgs) != 4 {
		t.Fatalf("expected 4 session messages, got %d", len(msgs))
	}
	if msgs[0].Role != "user" {
		t.Errorf("expected msg[0] role 'user', got %q", msgs[0].Role)
	}
	if msgs[1].Role != "assistant" || len(msgs[1].ToolCalls) == 0 {
		t.Errorf("expected msg[1] to be assistant with tool_calls, got role=%q tool_calls=%d",
			msgs[1].Role, len(msgs[1].ToolCalls))
	}
	if msgs[2].Role != "tool" {
		t.Errorf("expected msg[2] role 'tool', got %q", msgs[2].Role)
	}
	if msgs[3].Role != "assistant" || msgs[3].Content != "The tool returned: mock output" {
		t.Errorf("expected msg[3] to be final assistant response, got role=%q content=%q",
			msgs[3].Role, msgs[3].Content)
	}
}

func TestAgentLoopMaxIterations(t *testing.T) {
	// Mock server that always returns a tool call, forcing the agent to loop forever
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(chatCompletionResponse{
			ID:    "chatcmpl-loop",
			Model: "test-model",
			Choices: []struct {
				Index        int     `json:"index"`
				Message      Message `json:"message"`
				FinishReason string  `json:"finish_reason"`
			}{
				{
					Index: 0,
					Message: Message{
						Role: "assistant",
						ToolCalls: []ToolCall{
							{
								ID:   "call_loop",
								Type: "function",
								Function: &FunctionCall{
									Name:      "test_tool",
									Arguments: "{}",
								},
							},
						},
					},
					FinishReason: "tool_calls",
				},
			},
			Usage: &UsageInfo{
				PromptTokens:     10,
				CompletionTokens: 5,
				TotalTokens:      15,
			},
		})
	}))
	defer server.Close()

	cfg := &Config{
		APIKey:             "test-key",
		APIBase:            server.URL,
		Model:              "test-model",
		MaxIterations:      3,
		SessionWindow:      50,
		MaxRetries:         0,
		HTTPTimeoutSeconds: 10,
		MaxOutputSize:      100000,
	}
	provider := &HTTPProvider{
		apiBase:    cfg.APIBase,
		apiKey:     cfg.APIKey,
		model:      cfg.Model,
		client:     &http.Client{Timeout: 10 * time.Second},
		maxRetries: 0,
	}
	registry := NewToolRegistry()
	registry.Register(&mockTool{
		name:   "test_tool",
		result: NewToolResult("looping"),
	})

	agent := NewAgent(cfg, provider, registry)
	agent.output = io.Discard

	ctx := context.Background()
	_, err := agent.Run(ctx, "loop forever")
	if err == nil {
		t.Fatal("expected error when max iterations exceeded, got nil")
	}
	if !strings.Contains(err.Error(), "exceeded maximum iterations") {
		t.Errorf("expected 'exceeded maximum iterations' error, got: %v", err)
	}
	if !strings.Contains(err.Error(), "(3)") {
		t.Errorf("expected error to mention max iterations count (3), got: %v", err)
	}

	stats := agent.GetStats()
	if stats.TotalRequests != 3 {
		t.Errorf("expected 3 total requests, got %d", stats.TotalRequests)
	}
	if stats.TotalToolCalls != 3 {
		t.Errorf("expected 3 total tool calls, got %d", stats.TotalToolCalls)
	}
}

func TestHTTPProviderRetry(t *testing.T) {
	var mu sync.Mutex
	var attempts int

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		attempts++
		n := attempts
		mu.Unlock()

		switch n {
		case 1:
			// First attempt: 429 Too Many Requests
			w.WriteHeader(429)
			w.Write([]byte(`{"error": {"message": "rate limited"}}`))
		case 2:
			// Second attempt: 500 Internal Server Error
			w.WriteHeader(500)
			w.Write([]byte(`{"error": {"message": "server error"}}`))
		case 3:
			// Third attempt: success
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(chatCompletionResponse{
				ID:    "chatcmpl-retry",
				Model: "test-model",
				Choices: []struct {
					Index        int     `json:"index"`
					Message      Message `json:"message"`
					FinishReason string  `json:"finish_reason"`
				}{
					{
						Index: 0,
						Message: Message{
							Role:    "assistant",
							Content: "recovered after retries",
						},
						FinishReason: "stop",
					},
				},
				Usage: &UsageInfo{
					PromptTokens:     10,
					CompletionTokens: 5,
					TotalTokens:      15,
				},
			})
		}
	}))
	defer server.Close()

	provider := &HTTPProvider{
		apiBase:        server.URL,
		apiKey:         "test-key",
		model:          "test-model",
		client:         &http.Client{Timeout: 10 * time.Second},
		maxRetries:     2,
		retryBaseDelay: 1 * time.Millisecond, // fast retries for testing
	}

	ctx := context.Background()
	resp, err := provider.Call(ctx, []Message{{Role: "user", Content: "hello"}}, nil)
	if err != nil {
		t.Fatalf("expected successful response after retries, got error: %v", err)
	}
	if resp.Content != "recovered after retries" {
		t.Errorf("expected content %q, got %q", "recovered after retries", resp.Content)
	}

	mu.Lock()
	finalAttempts := attempts
	mu.Unlock()
	if finalAttempts != 3 {
		t.Errorf("expected 3 attempts (1 initial + 2 retries), got %d", finalAttempts)
	}
}

func TestHTTPProviderRetryExhausted(t *testing.T) {
	var mu sync.Mutex
	var attempts int

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		attempts++
		mu.Unlock()

		// Always return 429
		w.WriteHeader(429)
		w.Write([]byte(`{"error": {"message": "rate limited"}}`))
	}))
	defer server.Close()

	provider := &HTTPProvider{
		apiBase:        server.URL,
		apiKey:         "test-key",
		model:          "test-model",
		client:         &http.Client{Timeout: 10 * time.Second},
		maxRetries:     2,
		retryBaseDelay: 1 * time.Millisecond,
	}

	ctx := context.Background()
	_, err := provider.Call(ctx, []Message{{Role: "user", Content: "hello"}}, nil)
	if err == nil {
		t.Fatal("expected error when all retries exhausted, got nil")
	}
	if !strings.Contains(err.Error(), "all 3 attempts failed") {
		t.Errorf("expected 'all 3 attempts failed' error, got: %v", err)
	}

	// The wrapped error should be an apiError with retryable=true
	var ae *apiError
	if !errors.As(err, &ae) {
		t.Fatalf("expected wrapped *apiError, got %T: %v", err, err)
	}
	if !ae.Retryable {
		t.Error("expected Retryable=true for 429")
	}

	mu.Lock()
	finalAttempts := attempts
	mu.Unlock()
	if finalAttempts != 3 {
		t.Errorf("expected 3 attempts (1 initial + 2 retries), got %d", finalAttempts)
	}
}

func TestHTTPProviderNoRetryOn4xx(t *testing.T) {
	var mu sync.Mutex
	var attempts int

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		attempts++
		mu.Unlock()

		// 401 Unauthorized — should NOT be retried
		w.WriteHeader(401)
		w.Write([]byte(`{"error": {"message": "invalid api key"}}`))
	}))
	defer server.Close()

	provider := &HTTPProvider{
		apiBase:        server.URL,
		apiKey:         "bad-key",
		model:          "test-model",
		client:         &http.Client{Timeout: 10 * time.Second},
		maxRetries:     2,
		retryBaseDelay: 1 * time.Millisecond,
	}

	ctx := context.Background()
	_, err := provider.Call(ctx, []Message{{Role: "user", Content: "hello"}}, nil)
	if err == nil {
		t.Fatal("expected error for 401 response, got nil")
	}

	var ae *apiError
	if !errors.As(err, &ae) {
		t.Fatalf("expected *apiError, got %T: %v", err, err)
	}
	if ae.StatusCode != 401 {
		t.Errorf("expected status code 401, got %d", ae.StatusCode)
	}
	if ae.Retryable {
		t.Error("expected Retryable=false for 401")
	}

	mu.Lock()
	finalAttempts := attempts
	mu.Unlock()
	if finalAttempts != 1 {
		t.Errorf("expected exactly 1 attempt (no retries for 401), got %d", finalAttempts)
	}
}
