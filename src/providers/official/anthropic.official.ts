import Anthropic from '@anthropic-ai/sdk';
import { BaseChatModel } from '../../interfaces/base-model.js';
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

export interface AnthropicFormattedMessages {
  system?: string;
  messages: Anthropic.Messages.MessageParam[];
}

export class AnthropicOfficialModel extends BaseChatModel<OfficialModelConfig> {
  public readonly providerName = 'anthropic.official';
  protected client: Anthropic;

  constructor(config: OfficialModelConfig) {
    super(config);
    this.client = new Anthropic({
      apiKey: config.apiKey,
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
        // Anthropic 将工具返回结果作为用户轮次的 tool_result 内容块
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
        // Claude 3.7 原生思考扩展块
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
        // partial tool input delta
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

    const maxTokens = this.config.maxTokens ?? 4096;

    const requestParams: Anthropic.Messages.MessageCreateParamsNonStreaming = {
      model: this.config.modelName,
      messages,
      system,
      max_tokens: maxTokens,
      tools,
      temperature: this.config.temperature,
    };

    if (this.config.thinkingBudget && this.config.thinkingBudget > 0) {
      // 开启 Claude 原生 Thinking 时，temperature 须为 1.0，max_tokens 必须大于 budget_tokens
      requestParams.temperature = 1.0;
      requestParams.max_tokens = Math.max(maxTokens, this.config.thinkingBudget + 1024);
      (requestParams as unknown as { thinking: { type: string; budget_tokens: number } }).thinking = {
        type: 'enabled',
        budget_tokens: this.config.thinkingBudget,
      };
    }

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

    const maxTokens = this.config.maxTokens ?? 4096;

    const requestParams: Anthropic.Messages.MessageCreateParamsStreaming = {
      model: this.config.modelName,
      messages,
      system,
      max_tokens: maxTokens,
      tools,
      temperature: this.config.temperature,
      stream: true,
    };

    if (this.config.thinkingBudget && this.config.thinkingBudget > 0) {
      requestParams.temperature = 1.0;
      requestParams.max_tokens = Math.max(maxTokens, this.config.thinkingBudget + 1024);
      (requestParams as unknown as { thinking: { type: string; budget_tokens: number } }).thinking = {
        type: 'enabled',
        budget_tokens: this.config.thinkingBudget,
      };
    }

    const stream = await this.client.messages.create(requestParams, {
      signal: params.signal,
    });

    for await (const chunk of stream) {
      yield chunk;
    }
  }
}
