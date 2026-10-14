import { describe, it, expect } from 'vitest';
import {
  createChatModel,
  OpenAIOfficialModel,
  GeminiOfficialModel,
  AnthropicOfficialModel,
  OpenAIProtocolModel,
  AnthropicProtocolModel,
} from '../src/index.js';
import type { UnifiedMessage, UnifiedTool } from '../src/index.js';

describe('Providers Bidirectional Translation & Factory', () => {
  const sampleMessages: UnifiedMessage[] = [
    { role: 'system', content: 'You are a helpful coding assistant.' },
    { role: 'user', content: 'What is the weather in Tokyo?' },
    {
      role: 'assistant',
      content: 'Let me check Tokyo weather.',
      toolCalls: [
        {
          id: 'call_weather_1',
          name: 'get_weather',
          arguments: { city: 'Tokyo', unit: 'celsius' },
        },
      ],
    },
    {
      role: 'tool',
      content: '{"temp": 18, "condition": "Sunny"}',
      toolCallId: 'call_weather_1',
    },
  ];

  const sampleTools: UnifiedTool[] = [
    {
      name: 'get_weather',
      description: 'Get current weather for a city',
      parameters: {
        type: 'object',
        properties: {
          city: { type: 'string' },
          unit: { type: 'string', enum: ['celsius', 'fahrenheit'] },
        },
        required: ['city'],
      },
    },
  ];

  describe('Factory Routing (createChatModel)', () => {
    it('should route official OpenAI config to OpenAIOfficialModel', () => {
      const model = createChatModel({
        category: 'official',
        vendor: 'openai',
        modelName: 'gpt-4o',
        apiKey: 'sk-test',
      });
      expect(model).toBeInstanceOf(OpenAIOfficialModel);
      expect(model.providerName).toBe('openai.official');
    });

    it('should route official Gemini config to GeminiOfficialModel', () => {
      const model = createChatModel({
        category: 'official',
        vendor: 'gemini',
        modelName: 'gemini-2.5-flash',
        apiKey: 'gemini-key',
      });
      expect(model).toBeInstanceOf(GeminiOfficialModel);
      expect(model.providerName).toBe('gemini.official');
    });

    it('should route official Anthropic config to AnthropicOfficialModel', () => {
      const model = createChatModel({
        category: 'official',
        vendor: 'anthropic',
        modelName: 'claude-3-7-sonnet',
        apiKey: 'anthropic-key',
      });
      expect(model).toBeInstanceOf(AnthropicOfficialModel);
      expect(model.providerName).toBe('anthropic.official');
    });

    it('should route protocol openai-compatible to OpenAIProtocolModel', () => {
      const model = createChatModel({
        category: 'protocol',
        protocol: 'openai-compatible',
        modelName: 'deepseek-r1-70b',
        baseURL: 'http://localhost:8000/v1',
      });
      expect(model).toBeInstanceOf(OpenAIProtocolModel);
      expect(model.providerName).toBe('openai.protocol');
    });

    it('should route protocol anthropic-compatible to AnthropicProtocolModel', () => {
      const model = createChatModel({
        category: 'protocol',
        protocol: 'anthropic-compatible',
        modelName: 'claude-3-5-sonnet',
        baseURL: 'https://anthropic-proxy.internal.net/v1',
      });
      expect(model).toBeInstanceOf(AnthropicProtocolModel);
      expect(model.providerName).toBe('anthropic.protocol');
    });
  });

  describe('OpenAIOfficialModel Translation', () => {
    const model = new OpenAIOfficialModel({
      category: 'official',
      vendor: 'openai',
      modelName: 'gpt-4o',
      apiKey: 'sk-test',
      temperature: 0.7,
      timeoutMs: 60000,
      maxRetries: 3,
    });

    it('should format input messages into OpenAI ChatCompletionMessageParam array', () => {
      const formatted = model.formatMessages(sampleMessages);
      expect(formatted).toHaveLength(4);
      expect(formatted[0].role).toBe('system');
      expect(formatted[1].role).toBe('user');
      expect(formatted[2].role).toBe('assistant');
      expect(formatted[3].role).toBe('tool');

      // 验证 toolCalls 序列化
      const assistant = formatted[2] as { tool_calls?: Array<{ function: { name: string; arguments: string } }> };
      expect(assistant.tool_calls?.[0]?.function.name).toBe('get_weather');
      expect(JSON.parse(assistant.tool_calls![0]!.function.arguments)).toEqual({ city: 'Tokyo', unit: 'celsius' });
    });

    it('should format tools into OpenAI function structure', () => {
      const formatted = model.formatTools(sampleTools);
      expect(formatted).toHaveLength(1);
      expect(formatted![0].type).toBe('function');
      expect(formatted![0].function.name).toBe('get_weather');
    });

    it('should parse raw OpenAI response into UnifiedChatResponse', () => {
      const raw = {
        id: 'chatcmpl-123',
        object: 'chat.completion',
        created: 1700000000,
        model: 'gpt-4o',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: 'Tokyo is currently sunny at 18°C.',
              reasoning_content: 'User asked for Tokyo weather.',
            },
            finish_reason: 'stop',
          },
        ],
        usage: {
          prompt_tokens: 50,
          completion_tokens: 25,
          total_tokens: 75,
          prompt_tokens_details: { cached_tokens: 10 },
        },
      };

      const parsed = model.parseResponse(raw as never);
      expect(parsed.content).toBe('Tokyo is currently sunny at 18°C.');
      expect(parsed.reasoning).toBe('User asked for Tokyo weather.');
      expect(parsed.stopReason).toBe('stop');
      expect(parsed.usage.inputTokens).toBe(50);
      expect(parsed.usage.outputTokens).toBe(25);
      expect(parsed.usage.cachedTokens).toBe(10);
      expect(parsed.usage.totalTokens).toBe(75);
    });
  });

  describe('AnthropicOfficialModel Translation', () => {
    const model = new AnthropicOfficialModel({
      category: 'official',
      vendor: 'anthropic',
      modelName: 'claude-3-7-sonnet',
      apiKey: 'test-key',
      temperature: 0.7,
      timeoutMs: 60000,
      maxRetries: 3,
    });

    it('should extract system message and map tool calls to Anthropic format', () => {
      const formatted = model.formatMessages(sampleMessages);
      expect(formatted.system).toBe('You are a helpful coding assistant.');
      expect(formatted.messages).toHaveLength(3); // system 已被剥离到顶层

      // Assistant 携带 tool_use block
      const assistant = formatted.messages[1] as { content: Array<{ type: string; id?: string }> };
      expect(assistant.content[0].type).toBe('text');
      expect(assistant.content[1].type).toBe('tool_use');

      // Tool 结果映射为 user 轮次的 tool_result
      const toolTurn = formatted.messages[2] as { content: Array<{ type: string; tool_use_id?: string }> };
      expect(toolTurn.content[0].type).toBe('tool_result');
      expect(toolTurn.content[0].tool_use_id).toBe('call_weather_1');
    });

    it('should parse Anthropic response with extended thinking blocks', () => {
      const raw = {
        id: 'msg_123',
        type: 'message',
        role: 'assistant',
        model: 'claude-3-7-sonnet-20250219',
        content: [
          {
            type: 'thinking',
            thinking: 'Analyzing user request about Tokyo.',
          },
          {
            type: 'text',
            text: 'Here is the weather information.',
          },
        ],
        stop_reason: 'end_turn',
        usage: {
          input_tokens: 120,
          output_tokens: 60,
          cache_read_input_tokens: 40,
        },
      };

      const parsed = model.parseResponse(raw as never);
      expect(parsed.content).toBe('Here is the weather information.');
      expect(parsed.reasoning).toBe('Analyzing user request about Tokyo.');
      expect(parsed.stopReason).toBe('stop');
      expect(parsed.usage.cachedTokens).toBe(40);
    });
  });

  describe('GeminiOfficialModel Translation', () => {
    const model = new GeminiOfficialModel({
      category: 'official',
      vendor: 'gemini',
      modelName: 'gemini-2.5-flash',
      apiKey: 'test-key',
      temperature: 0.7,
      timeoutMs: 60000,
      maxRetries: 3,
    });

    it('should format messages into Gemini contents and systemInstruction', () => {
      const formatted = model.formatMessages(sampleMessages);
      expect(formatted.systemInstruction).toBe('You are a helpful coding assistant.');
      expect(formatted.contents).toHaveLength(3);
      expect(formatted.contents[0].role).toBe('user');
      expect(formatted.contents[1].role).toBe('model'); // assistant -> model
      expect(formatted.contents[2].role).toBe('user');  // tool -> user with functionResponse
    });

    it('should parse Gemini response with thought part and functionCall', () => {
      const raw = {
        model: 'gemini-2.5-flash',
        candidates: [
          {
            finishReason: 'STOP',
            content: {
              parts: [
                {
                  thought: true,
                  text: 'Calculating weather dynamics...',
                },
                {
                  text: 'Tokyo is clear today.',
                },
                {
                  functionCall: {
                    id: 'fc_1',
                    name: 'get_weather',
                    args: { city: 'Tokyo' },
                  },
                },
              ],
            },
          },
        ],
        usageMetadata: {
          promptTokenCount: 80,
          candidatesTokenCount: 45,
          totalTokenCount: 125,
        },
      };

      const parsed = model.parseResponse(raw);
      expect(parsed.content).toBe('Tokyo is clear today.');
      expect(parsed.reasoning).toBe('Calculating weather dynamics...');
      expect(parsed.stopReason).toBe('tool_call');
      expect(parsed.toolCalls).toHaveLength(1);
      expect(parsed.toolCalls![0].name).toBe('get_weather');
      expect(parsed.toolCalls![0].arguments).toEqual({ city: 'Tokyo' });
      expect(parsed.usage.totalTokens).toBe(125);
    });
  });

  describe('OpenAIProtocolModel (Self-hosted DeepSeek / Ollama / vLLM)', () => {
    const model = new OpenAIProtocolModel({
      category: 'protocol',
      protocol: 'openai-compatible',
      modelName: 'deepseek-r1-70b',
      baseURL: 'http://localhost:8000/v1',
      apiKey: 'EMPTY',
      temperature: 0.6,
      timeoutMs: 60000,
      maxRetries: 3,
    });

    it('should automatically strip <think>...</think> from content when reasoning_content is absent', () => {
      const raw = {
        id: 'vllm-123',
        model: 'deepseek-r1-70b',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: '<think>\nThinking through algorithm complexity\nO(N log N)\n</think>\nThe optimal approach is merge sort.',
            },
            finish_reason: 'stop',
          },
        ],
        usage: {
          prompt_tokens: 30,
          completion_tokens: 40,
          total_tokens: 70,
        },
      };

      const parsed = model.parseResponse(raw as never);
      expect(parsed.content).toBe('The optimal approach is merge sort.');
      expect(parsed.reasoning).toBe('Thinking through algorithm complexity\nO(N log N)');
      expect(parsed.stopReason).toBe('stop');
    });

    it('should respect reasoning_content when natively provided by inference gateway', () => {
      const raw = {
        id: 'sglang-456',
        model: 'deepseek-r1-distill',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: 'Result is 42.',
              reasoning_content: 'Direct calculation gives 42.',
            },
            finish_reason: 'stop',
          },
        ],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 15,
          total_tokens: 25,
        },
      };

      const parsed = model.parseResponse(raw as never);
      expect(parsed.content).toBe('Result is 42.');
      expect(parsed.reasoning).toBe('Direct calculation gives 42.');
    });
  });
});
