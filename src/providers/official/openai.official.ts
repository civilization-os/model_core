import OpenAI from 'openai';
import { BaseChatModel, type ChatModelInvocationOptions } from '../../interfaces/base-model.js';
import type { OfficialModelConfig } from '../../config/index.js';
import type {
  UnifiedMessage,
  UnifiedTool,
  UnifiedChatResponse,
  UnifiedStreamChunk,
  UnifiedStopReason,
  UnifiedToolCall,
} from '../../schemas/index.js';
import { safeParseToolArguments } from '../../utils/normalizers.js';

export class OpenAIOfficialModel extends BaseChatModel<OfficialModelConfig> {
  public readonly providerName = 'openai.official';
  protected client: OpenAI;

  constructor(config: OfficialModelConfig) {
    super(config);
    this.client = new OpenAI({
      apiKey: config.apiKey,
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

    let content = message?.content || '';
    // 兼容原厂 thinking/reasoning 字段 (如 o1/o3-mini 的 reasoning_content)
    const reasoning = (message as unknown as { reasoning_content?: string })?.reasoning_content;

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
    const deltaContent = delta?.content || '';
    const deltaReasoning = (delta as unknown as { reasoning_content?: string })?.reasoning_content;

    let deltaToolCalls: Partial<UnifiedToolCall>[] | undefined;
    if (delta?.tool_calls) {
      deltaToolCalls = delta.tool_calls.map((tc) => {
        let args: Record<string, unknown> | undefined;
        if (tc.function?.arguments) {
          try {
            args = JSON.parse(tc.function.arguments);
          } catch {
            // 流式中间片段 arguments 往往是不完整 JSON，暂留未解析
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
    };

    if (this.config.reasoningEffort) {
      (requestParams as unknown as { reasoning_effort?: string }).reasoning_effort = this.config.reasoningEffort;
    }

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
    };

    if (this.config.reasoningEffort) {
      (requestParams as unknown as { reasoning_effort?: string }).reasoning_effort = this.config.reasoningEffort;
    }

    const stream = await this.client.chat.completions.create(requestParams, {
      signal: params.signal,
    });

    for await (const chunk of stream) {
      yield chunk;
    }
  }
}
