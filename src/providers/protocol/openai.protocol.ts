import OpenAI from 'openai';
import { BaseChatModel } from '../../interfaces/base-model.js';
import type { ProtocolModelConfig } from '../../config/index.js';
import type {
  UnifiedMessage,
  UnifiedTool,
  UnifiedChatResponse,
  UnifiedStreamChunk,
  UnifiedStopReason,
  UnifiedToolCall,
} from '../../schemas/index.js';
import {
  safeParseToolArguments,
  stripThinkTags,
  StreamThinkExtractor,
} from '../../utils/normalizers.js';

export class OpenAIProtocolModel extends BaseChatModel<ProtocolModelConfig> {
  public readonly providerName = 'openai.protocol';
  protected client: OpenAI;

  constructor(config: ProtocolModelConfig) {
    super(config);
    this.client = new OpenAI({
      baseURL: config.baseURL,
      apiKey: config.apiKey || 'EMPTY',
      defaultHeaders: config.customHeaders,
    });
  }

  public formatMessages(messages: UnifiedMessage[]): OpenAI.Chat.Completions.ChatCompletionMessageParam[] {
    return messages.map((msg): OpenAI.Chat.Completions.ChatCompletionMessageParam => {
      switch (msg.role) {
        case 'system':
          return {
            role: 'system',
            content: msg.content,
            name: msg.name,
          };
        case 'user':
          return {
            role: 'user',
            content: msg.content,
            name: msg.name,
          };
        case 'assistant': {
          const assistantMsg: OpenAI.Chat.Completions.ChatCompletionAssistantMessageParam = {
            role: 'assistant',
            content: msg.content || null,
            name: msg.name,
          };
          if (msg.toolCalls && msg.toolCalls.length > 0) {
            assistantMsg.tool_calls = msg.toolCalls.map((tc) => ({
              id: tc.id,
              type: 'function',
              function: {
                name: tc.name,
                arguments: JSON.stringify(tc.arguments),
              },
            }));
          }
          return assistantMsg;
        }
        case 'tool':
          return {
            role: 'tool',
            content: msg.content,
            tool_call_id: msg.toolCallId || '',
          };
        default: {
          const exhaustiveCheck: never = msg.role;
          throw new Error(`Unsupported message role: ${exhaustiveCheck}`);
        }
      }
    });
  }

  public formatTools(tools?: UnifiedTool[]): OpenAI.Chat.Completions.ChatCompletionTool[] | undefined {
    if (!tools || tools.length === 0) {
      return undefined;
    }

    return tools.map((tool) => ({
      type: 'function',
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters,
      },
    }));
  }

  public parseResponse(rawResponse: OpenAI.Chat.Completions.ChatCompletion): UnifiedChatResponse {
    const choice = rawResponse.choices?.[0];
    const message = choice?.message;

    const rawContent = message?.content || '';
    let reasoning = (message as unknown as { reasoning_content?: string })?.reasoning_content;
    let content = rawContent;

    // 若未显式返回 reasoning_content，但正文中包含 <think>...</think>（自部署 DeepSeek/QwQ 常见输出），自动清洗剥离
    if (!reasoning && (rawContent.includes('<think>') || rawContent.includes('</think>'))) {
      const stripped = stripThinkTags(rawContent);
      content = stripped.content;
      reasoning = stripped.reasoning;
    } else if (rawContent.includes('<think>')) {
      // 即使已有 reasoning_content，若正文仍残留 <think>，进行内容清洗
      const stripped = stripThinkTags(rawContent);
      content = stripped.content;
      if (!reasoning) {
        reasoning = stripped.reasoning;
      }
    }

    let toolCalls: UnifiedToolCall[] | undefined;
    if (message?.tool_calls && message.tool_calls.length > 0) {
      toolCalls = message.tool_calls.map((tc) => {
        const parsedArgs = safeParseToolArguments(tc.function.arguments, {
          provider: this.providerName,
          modelName: this.config.modelName,
          toolName: tc.function.name,
        });
        return {
          id: tc.id,
          name: tc.function.name,
          arguments: parsedArgs,
        };
      });
    }

    let stopReason: UnifiedStopReason = 'stop';
    if (choice?.finish_reason === 'tool_calls') {
      stopReason = 'tool_call';
    } else if (choice?.finish_reason === 'length') {
      stopReason = 'max_tokens';
    }

    const usage = {
      inputTokens: rawResponse.usage?.prompt_tokens ?? 0,
      outputTokens: rawResponse.usage?.completion_tokens ?? 0,
      cachedTokens: rawResponse.usage?.prompt_tokens_details?.cached_tokens,
      totalTokens: rawResponse.usage?.total_tokens ?? 0,
    };

    return {
      content,
      reasoning,
      role: 'assistant',
      model: rawResponse.model || this.config.modelName,
      stopReason,
      toolCalls,
      usage,
      raw: rawResponse,
    };
  }

  public parseStreamChunk(rawChunk: OpenAI.Chat.Completions.ChatCompletionChunk): UnifiedStreamChunk | null {
    const choice = rawChunk.choices?.[0];
    if (!choice && !rawChunk.usage) {
      return null;
    }

    const delta = choice?.delta;
    let deltaContent = delta?.content || '';
    let deltaReasoning = (delta as unknown as { reasoning_content?: string })?.reasoning_content;

    let deltaToolCalls: Partial<UnifiedToolCall>[] | undefined;
    if (delta?.tool_calls) {
      deltaToolCalls = delta.tool_calls.map((tc) => {
        let args: Record<string, unknown> | undefined;
        if (tc.function?.arguments) {
          try {
            args = JSON.parse(tc.function.arguments);
          } catch {
            // 流式中间片段 arguments
          }
        }
        return {
          id: tc.id,
          name: tc.function?.name,
          arguments: args,
        };
      });
    }

    let stopReason: UnifiedStopReason | null = null;
    if (choice?.finish_reason === 'stop') {
      stopReason = 'stop';
    } else if (choice?.finish_reason === 'tool_calls') {
      stopReason = 'tool_call';
    } else if (choice?.finish_reason === 'length') {
      stopReason = 'max_tokens';
    }

    const usage = rawChunk.usage
      ? {
          inputTokens: rawChunk.usage.prompt_tokens,
          outputTokens: rawChunk.usage.completion_tokens,
          cachedTokens: rawChunk.usage.prompt_tokens_details?.cached_tokens,
          totalTokens: rawChunk.usage.total_tokens,
        }
      : undefined;

    return {
      deltaContent,
      deltaReasoning,
      deltaToolCalls,
      stopReason,
      usage,
    };
  }

  protected async executeNativeInvoke(params: {
    formattedMessages: unknown;
    formattedTools: unknown;
    signal?: AbortSignal;
  }): Promise<unknown> {
    const messages = params.formattedMessages as OpenAI.Chat.Completions.ChatCompletionMessageParam[];
    const tools = params.formattedTools as OpenAI.Chat.Completions.ChatCompletionTool[] | undefined;

    const requestParams: OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming = {
      model: this.config.modelName,
      messages,
      temperature: this.config.temperature,
      max_tokens: this.config.maxTokens,
      tools,
      // 透传自部署引擎特定参数 (top_k, repetition_penalty, min_p 等)
      ...(this.config.extraBody as Record<string, unknown>),
    };

    return await this.client.chat.completions.create(requestParams, {
      signal: params.signal,
    });
  }

  protected async *executeNativeStream(params: {
    formattedMessages: unknown;
    formattedTools: unknown;
    signal?: AbortSignal;
  }): AsyncGenerator<unknown, void, unknown> {
    const messages = params.formattedMessages as OpenAI.Chat.Completions.ChatCompletionMessageParam[];
    const tools = params.formattedTools as OpenAI.Chat.Completions.ChatCompletionTool[] | undefined;

    const requestParams: OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming = {
      model: this.config.modelName,
      messages,
      temperature: this.config.temperature,
      max_tokens: this.config.maxTokens,
      tools,
      stream: true,
      stream_options: {
        include_usage: true,
      },
      ...(this.config.extraBody as Record<string, unknown>),
    };

    const stream = await this.client.chat.completions.create(requestParams, {
      signal: params.signal,
    });

    const thinkExtractor = new StreamThinkExtractor();

    for await (const chunk of stream) {
      const parsed = this.parseStreamChunk(chunk);
      if (parsed) {
        // 如果没有原生的 reasoning_content 且正文中可能含有 <think>，使用流式切分状态机二次过滤
        if (!parsed.deltaReasoning && parsed.deltaContent) {
          const extracted = thinkExtractor.feed(parsed.deltaContent);
          yield {
            ...chunk,
            choices: [
              {
                ...chunk.choices[0],
                delta: {
                  ...chunk.choices[0]?.delta,
                  content: extracted.deltaContent,
                  reasoning_content: extracted.deltaReasoning,
                },
              },
            ],
          };
        } else {
          yield chunk;
        }
      } else {
        yield chunk;
      }
    }

    const flushed = thinkExtractor.flush();
    if (flushed.deltaContent || flushed.deltaReasoning) {
      yield {
        id: 'flush',
        object: 'chat.completion.chunk',
        created: Date.now(),
        model: this.config.modelName,
        choices: [
          {
            index: 0,
            delta: {
              content: flushed.deltaContent || undefined,
              reasoning_content: flushed.deltaReasoning,
            },
            finish_reason: null,
          },
        ],
      } as unknown as OpenAI.Chat.Completions.ChatCompletionChunk;
    }
  }
}
