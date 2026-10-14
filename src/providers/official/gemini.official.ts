import { GoogleGenAI } from '@google/genai';
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

export interface GeminiPart {
  text?: string;
  thought?: boolean;
  functionCall?: {
    id?: string;
    name: string;
    args?: Record<string, unknown>;
  };
  functionResponse?: {
    name: string;
    response: Record<string, unknown>;
  };
}

export interface GeminiContent {
  role: 'user' | 'model';
  parts: GeminiPart[];
}

export interface GeminiFormattedMessages {
  systemInstruction?: string;
  contents: GeminiContent[];
}

export class GeminiOfficialModel extends BaseChatModel<OfficialModelConfig> {
  public readonly providerName = 'gemini.official';
  protected client: GoogleGenAI;

  constructor(config: OfficialModelConfig) {
    super(config);
    this.client = new GoogleGenAI({
      apiKey: config.apiKey,
    });
  }

  public formatMessages(messages: UnifiedMessage[]): GeminiFormattedMessages {
    const systemParts: string[] = [];
    const contents: GeminiContent[] = [];

    for (const msg of messages) {
      if (msg.role === 'system') {
        systemParts.push(msg.content);
        continue;
      }

      if (msg.role === 'user') {
        contents.push({
          role: 'user',
          parts: [{ text: msg.content }],
        });
      } else if (msg.role === 'assistant') {
        const parts: GeminiPart[] = [];
        if (msg.content) {
          parts.push({ text: msg.content });
        }
        if (msg.toolCalls && msg.toolCalls.length > 0) {
          for (const tc of msg.toolCalls) {
            parts.push({
              functionCall: {
                id: tc.id,
                name: tc.name,
                args: tc.arguments,
              },
            });
          }
        }
        contents.push({
          role: 'model',
          parts: parts.length > 0 ? parts : [{ text: '' }],
        });
      } else if (msg.role === 'tool') {
        // Gemini 将外部工具执行结果封装为 functionResponse
        contents.push({
          role: 'user',
          parts: [
            {
              functionResponse: {
                name: msg.name || 'tool',
                response: { output: msg.content },
              },
            },
          ],
        });
      }
    }

    return {
      systemInstruction: systemParts.length > 0 ? systemParts.join('\n\n') : undefined,
      contents,
    };
  }

  public formatTools(tools?: UnifiedTool[]): Array<{ functionDeclarations: unknown[] }> | undefined {
    if (!tools || tools.length === 0) {
      return undefined;
    }

    const functionDeclarations = tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    }));

    return [{ functionDeclarations }];
  }

  public parseResponse(rawResponse: unknown): UnifiedChatResponse {
    const res = rawResponse as {
      model?: string;
      candidates?: Array<{
        finishReason?: string;
        content?: {
          parts?: GeminiPart[];
        };
      }>;
      usageMetadata?: {
        promptTokenCount?: number;
        candidatesTokenCount?: number;
        cachedContentTokenCount?: number;
        totalTokenCount?: number;
      };
    };

    const candidate = res.candidates?.[0];
    const parts = candidate?.content?.parts || [];

    const contentBlocks: string[] = [];
    const reasoningBlocks: string[] = [];
    const toolCalls: UnifiedToolCall[] = [];

    for (const part of parts) {
      if (part.thought) {
        if (part.text) {
          reasoningBlocks.push(part.text);
        }
      } else if (part.text) {
        contentBlocks.push(part.text);
      }

      if (part.functionCall) {
        toolCalls.push({
          id: part.functionCall.id || `call_${part.functionCall.name}_${Date.now()}`,
          name: part.functionCall.name,
          arguments: (part.functionCall.args as Record<string, unknown>) || {},
        });
      }
    }

    let stopReason: UnifiedStopReason = 'stop';
    if (toolCalls.length > 0) {
      stopReason = 'tool_call';
    } else if (candidate?.finishReason === 'MAX_TOKENS') {
      stopReason = 'max_tokens';
    }

    const inputTokens = res.usageMetadata?.promptTokenCount ?? 0;
    const outputTokens = res.usageMetadata?.candidatesTokenCount ?? 0;
    const cachedTokens = res.usageMetadata?.cachedContentTokenCount;
    const totalTokens = res.usageMetadata?.totalTokenCount ?? inputTokens + outputTokens;

    return {
      content: contentBlocks.join(''),
      reasoning: reasoningBlocks.length > 0 ? reasoningBlocks.join('\n\n') : undefined,
      role: 'assistant',
      model: res.model || this.config.modelName,
      stopReason,
      toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
      usage: {
        inputTokens,
        outputTokens,
        cachedTokens,
        totalTokens,
      },
      raw: rawResponse,
    };
  }

  public parseStreamChunk(rawChunk: unknown): UnifiedStreamChunk | null {
    const chunk = rawChunk as {
      candidates?: Array<{
        finishReason?: string;
        content?: {
          parts?: GeminiPart[];
        };
      }>;
      usageMetadata?: {
        promptTokenCount?: number;
        candidatesTokenCount?: number;
        totalTokenCount?: number;
      };
    };

    const candidate = chunk.candidates?.[0];
    const parts = candidate?.content?.parts;

    let deltaContent = '';
    let deltaReasoning: string | undefined;
    const deltaToolCalls: Partial<UnifiedToolCall>[] = [];

    if (parts) {
      for (const part of parts) {
        if (part.thought && part.text) {
          deltaReasoning = (deltaReasoning || '') + part.text;
        } else if (part.text) {
          deltaContent += part.text;
        }
        if (part.functionCall) {
          deltaToolCalls.push({
            id: part.functionCall.id,
            name: part.functionCall.name,
            arguments: part.functionCall.args,
          });
        }
      }
    }

    let stopReason: UnifiedStopReason | null = null;
    if (candidate?.finishReason === 'STOP') {
      stopReason = 'stop';
    } else if (candidate?.finishReason === 'MAX_TOKENS') {
      stopReason = 'max_tokens';
    }

    return {
      deltaContent,
      deltaReasoning,
      deltaToolCalls: deltaToolCalls.length > 0 ? deltaToolCalls : undefined,
      stopReason,
      usage: chunk.usageMetadata
        ? {
            inputTokens: chunk.usageMetadata.promptTokenCount ?? 0,
            outputTokens: chunk.usageMetadata.candidatesTokenCount ?? 0,
            totalTokens: chunk.usageMetadata.totalTokenCount ?? 0,
          }
        : undefined,
    };
  }

  protected async executeNativeInvoke(params: {
    formattedMessages: unknown;
    formattedTools: unknown;
    signal?: AbortSignal;
  }): Promise<unknown> {
    const { systemInstruction, contents } = params.formattedMessages as GeminiFormattedMessages;
    const tools = params.formattedTools;

    const requestConfig: Record<string, unknown> = {
      temperature: this.config.temperature,
      maxOutputTokens: this.config.maxTokens,
    };

    if (systemInstruction) {
      requestConfig.systemInstruction = systemInstruction;
    }

    if (tools) {
      requestConfig.tools = tools;
    }

    if (this.config.thinkingBudget && this.config.thinkingBudget > 0) {
      requestConfig.thinkingConfig = {
        thinkingBudget: this.config.thinkingBudget,
      };
    }

    return await this.client.models.generateContent({
      model: this.config.modelName,
      contents: contents as unknown as never,
      config: requestConfig,
    });
  }

  protected async *executeNativeStream(params: {
    formattedMessages: unknown;
    formattedTools: unknown;
    signal?: AbortSignal;
  }): AsyncGenerator<unknown, void, unknown> {
    const { systemInstruction, contents } = params.formattedMessages as GeminiFormattedMessages;
    const tools = params.formattedTools;

    const requestConfig: Record<string, unknown> = {
      temperature: this.config.temperature,
      maxOutputTokens: this.config.maxTokens,
    };

    if (systemInstruction) {
      requestConfig.systemInstruction = systemInstruction;
    }

    if (tools) {
      requestConfig.tools = tools;
    }

    if (this.config.thinkingBudget && this.config.thinkingBudget > 0) {
      requestConfig.thinkingConfig = {
        thinkingBudget: this.config.thinkingBudget,
      };
    }

    const responseStream = await this.client.models.generateContentStream({
      model: this.config.modelName,
      contents: contents as unknown as never,
      config: requestConfig,
    });

    for await (const chunk of responseStream) {
      yield chunk;
    }
  }
}
