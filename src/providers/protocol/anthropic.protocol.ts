import Anthropic from '@anthropic-ai/sdk';
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
import { safeParseToolArguments } from '../../utils/normalizers.js';
import type { AnthropicFormattedMessages } from '../official/anthropic.official.js';

export class AnthropicProtocolModel extends BaseChatModel<ProtocolModelConfig> {
  public readonly providerName = 'anthropic.protocol';
  protected client: Anthropic;

  constructor(config: ProtocolModelConfig) {
    super(config);
    this.client = new Anthropic({
      baseURL: config.baseURL,
      apiKey: config.apiKey || 'EMPTY',
      defaultHeaders: config.customHeaders,
    });
  }

  public formatMessages(messages: UnifiedMessage[]): AnthropicFormattedMessages {
    const systemParts: string[] = [];
    const formattedMessages: Anthropic.Messages.MessageParam[] = [];

    for (const msg of messages) {
      if (msg.role === 'system') {
        systemParts.push(msg.content);
        continue;
      }

      if (msg.role === 'user') {
        formattedMessages.push({
          role: 'user',
          content: msg.content,
        });
      } else if (msg.role === 'assistant') {
        if (msg.toolCalls && msg.toolCalls.length > 0) {
          const contentBlocks: Anthropic.Messages.ContentBlockParam[] = [];
          if (msg.content) {
            contentBlocks.push({
              type: 'text',
              text: msg.content,
            });
          }
          for (const tc of msg.toolCalls) {
            contentBlocks.push({
              type: 'tool_use',
              id: tc.id,
              name: tc.name,
              input: tc.arguments,
            });
          }
          formattedMessages.push({
            role: 'assistant',
            content: contentBlocks,
          });
        } else {
          formattedMessages.push({
            role: 'assistant',
            content: msg.content,
          });
        }
      } else if (msg.role === 'tool') {
        formattedMessages.push({
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: msg.toolCallId || '',
              content: msg.content,
            },
          ],
        });
      }
    }

    return {
      system: systemParts.length > 0 ? systemParts.join('\n\n') : undefined,
      messages: formattedMessages,
    };
  }

  public formatTools(tools?: UnifiedTool[]): Anthropic.Messages.Tool[] | undefined {
    if (!tools || tools.length === 0) {
      return undefined;
    }

    return tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      input_schema: tool.parameters as Anthropic.Messages.Tool.InputSchema,
    }));
  }

  public parseResponse(rawResponse: Anthropic.Messages.Message): UnifiedChatResponse {
    let content = '';
    const reasoningBlocks: string[] = [];
    const toolCalls: UnifiedToolCall[] = [];

    for (const block of rawResponse.content) {
      if (block.type === 'text') {
        content += block.text;
      } else if (block.type === 'thinking') {
        reasoningBlocks.push((block as unknown as { thinking: string }).thinking);
      } else if (block.type === 'tool_use') {
        const parsedArgs = safeParseToolArguments(block.input, {
          provider: this.providerName,
          modelName: this.config.modelName,
          toolName: block.name,
        });
        toolCalls.push({
          id: block.id,
          name: block.name,
          arguments: parsedArgs,
        });
      }
    }

    let stopReason: UnifiedStopReason = 'stop';
    if (rawResponse.stop_reason === 'tool_use') {
      stopReason = 'tool_call';
    } else if (rawResponse.stop_reason === 'max_tokens') {
      stopReason = 'max_tokens';
    }

    const inputTokens = rawResponse.usage.input_tokens;
    const outputTokens = rawResponse.usage.output_tokens;
    const cachedTokens = (rawResponse.usage as unknown as { cache_read_input_tokens?: number })
      ?.cache_read_input_tokens;

    return {
      content,
      reasoning: reasoningBlocks.length > 0 ? reasoningBlocks.join('\n\n') : undefined,
      role: 'assistant',
      model: rawResponse.model || this.config.modelName,
      stopReason,
      toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
      usage: {
        inputTokens,
        outputTokens,
        cachedTokens,
        totalTokens: inputTokens + outputTokens,
      },
      raw: rawResponse,
    };
  }

  public parseStreamChunk(rawChunk: Anthropic.Messages.MessageStreamEvent): UnifiedStreamChunk | null {
    if (rawChunk.type === 'content_block_delta') {
      const delta = rawChunk.delta;
      if (delta.type === 'text_delta') {
        return { deltaContent: delta.text };
      }
      if ((delta as unknown as { type: string; thinking?: string }).type === 'thinking_delta') {
        return {
          deltaContent: '',
          deltaReasoning: (delta as unknown as { thinking: string }).thinking,
        };
      }
      if (delta.type === 'input_json_delta') {
        return {
          deltaContent: '',
          deltaToolCalls: [
            {
              arguments: { partial: delta.partial_json },
            },
          ],
        };
      }
    } else if (rawChunk.type === 'message_delta') {
      let stopReason: UnifiedStopReason | null = null;
      if (rawChunk.delta.stop_reason === 'end_turn') {
        stopReason = 'stop';
      } else if (rawChunk.delta.stop_reason === 'tool_use') {
        stopReason = 'tool_call';
      } else if (rawChunk.delta.stop_reason === 'max_tokens') {
        stopReason = 'max_tokens';
      }

      return {
        deltaContent: '',
        stopReason,
        usage: rawChunk.usage
          ? {
              inputTokens: 0,
              outputTokens: rawChunk.usage.output_tokens,
              totalTokens: rawChunk.usage.output_tokens,
            }
          : undefined,
      };
    }

    return null;
  }

  protected async executeNativeInvoke(params: {
    formattedMessages: unknown;
    formattedTools: unknown;
    signal?: AbortSignal;
  }): Promise<unknown> {
    const { system, messages } = params.formattedMessages as AnthropicFormattedMessages;
    const tools = params.formattedTools as Anthropic.Messages.Tool[] | undefined;

    const requestParams: Anthropic.Messages.MessageCreateParamsNonStreaming = {
      model: this.config.modelName,
      messages,
      system,
      max_tokens: this.config.maxTokens ?? 4096,
      tools,
      temperature: this.config.temperature,
      ...(this.config.extraBody as Record<string, unknown>),
    };

    return await this.client.messages.create(requestParams, {
      signal: params.signal,
    });
  }

  protected async *executeNativeStream(params: {
    formattedMessages: unknown;
    formattedTools: unknown;
    signal?: AbortSignal;
  }): AsyncGenerator<unknown, void, unknown> {
    const { system, messages } = params.formattedMessages as AnthropicFormattedMessages;
    const tools = params.formattedTools as Anthropic.Messages.Tool[] | undefined;

    const requestParams: Anthropic.Messages.MessageCreateParamsStreaming = {
      model: this.config.modelName,
      messages,
      system,
      max_tokens: this.config.maxTokens ?? 4096,
      tools,
      temperature: this.config.temperature,
      stream: true,
      ...(this.config.extraBody as Record<string, unknown>),
    };

    const stream = await this.client.messages.create(requestParams, {
      signal: params.signal,
    });

    for await (const chunk of stream) {
      yield chunk;
    }
  }
}
