import type { ModelConfig } from '../config/index.js';
import type {
  UnifiedMessage,
  UnifiedTool,
  UnifiedChatResponse,
  UnifiedStreamChunk,
} from '../schemas/index.js';
import { wrapProviderError, ModelTimeoutError } from '../utils/errors.js';
import { withExponentialBackoff } from '../utils/retry.js';

export interface ChatModelInvocationOptions {
  tools?: UnifiedTool[];
  signal?: AbortSignal;
}

/**
 * 所有模型提供商的顶层抽象基类 BaseChatModel
 * 固化模型调用生命周期，封装统一重试、超时与异常捕获，定义“两进两出”双向翻译契约
 */
export abstract class BaseChatModel<TConfig extends ModelConfig = ModelConfig> {
  public readonly config: TConfig;

  constructor(config: TConfig) {
    this.config = config;
  }

  /**
   * 提供商唯一标识名称 (如 'openai.official', 'gemini.official', 'openai.protocol' 等)
   */
  public abstract readonly providerName: string;

  // ==========================================
  // 【核心部分 3：两进两出受保护的双向翻译方法】
  // ==========================================

  /**
   * 去程翻译 1：将统一消息数组转换为厂商特定方言消息结构
   */
  public abstract formatMessages(messages: UnifiedMessage[]): unknown;

  /**
   * 去程翻译 2：将统一工具定义转换为厂商特定方言的工具/函数声明
   */
  public abstract formatTools(tools?: UnifiedTool[]): unknown;

  /**
   * 回程翻译 1：将厂商非流式原生返回结构转换为标准 UnifiedChatResponse
   */
  public abstract parseResponse(rawResponse: unknown): UnifiedChatResponse;

  /**
   * 回程翻译 2：将厂商流式原生切片转换为标准 UnifiedStreamChunk (若为无需上报的空切片则返回 null)
   */
  public abstract parseStreamChunk(rawChunk: unknown): UnifiedStreamChunk | null;

  // ==========================================
  // 【底层原生调用执行钩子】
  // ==========================================

  /**
   * 底层单次非流式调用实现
   */
  protected abstract executeNativeInvoke(params: {
    formattedMessages: unknown;
    formattedTools: unknown;
    signal?: AbortSignal;
  }): Promise<unknown>;

  /**
   * 底层流式调用实现
   */
  protected abstract executeNativeStream(params: {
    formattedMessages: unknown;
    formattedTools: unknown;
    signal?: AbortSignal;
  }): AsyncGenerator<unknown, void, unknown>;

  // ==========================================
  // 【对外统一暴露的标准方法】
  // ==========================================

  /**
   * 统一非流式调用入口
   * 封装：参数校验 -> 去程翻译 -> 指数退避重试与超时控制 -> 回程翻译 -> 统一异常包装
   */
  public async invoke(
    messages: UnifiedMessage[],
    options: ChatModelInvocationOptions = {}
  ): Promise<UnifiedChatResponse> {
    const formattedMessages = this.formatMessages(messages);
    const formattedTools = this.formatTools(options.tools);

    const executeWithTimeout = async (): Promise<unknown> => {
      const controller = new AbortController();
      let timeoutId: NodeJS.Timeout | undefined;

      if (this.config.timeoutMs && this.config.timeoutMs > 0) {
        timeoutId = setTimeout(() => {
          controller.abort(
            new ModelTimeoutError(`Request timed out after ${this.config.timeoutMs}ms`, {
              provider: this.providerName,
              modelName: this.config.modelName,
            })
          );
        }, this.config.timeoutMs);
      }

      // 如果外部传入了 signal，监听其触发
      if (options.signal) {
        options.signal.addEventListener('abort', () => {
          controller.abort(options.signal?.reason);
        });
      }

      try {
        const raw = await this.executeNativeInvoke({
          formattedMessages,
          formattedTools,
          signal: controller.signal,
        });
        return raw;
      } finally {
        if (timeoutId) {
          clearTimeout(timeoutId);
        }
      }
    };

    try {
      const rawResponse = await withExponentialBackoff(
        () => executeWithTimeout(),
        {
          maxRetries: this.config.maxRetries,
        }
      );
      return this.parseResponse(rawResponse);
    } catch (err) {
      throw wrapProviderError(err, this.providerName, this.config.modelName);
    }
  }

  /**
   * 统一流式调用入口
   * 封装：去程翻译 -> 底层流消费 -> 逐切片回程翻译
   */
  public async *stream(
    messages: UnifiedMessage[],
    options: ChatModelInvocationOptions = {}
  ): AsyncGenerator<UnifiedStreamChunk, void, unknown> {
    const formattedMessages = this.formatMessages(messages);
    const formattedTools = this.formatTools(options.tools);

    try {
      const rawStream = this.executeNativeStream({
        formattedMessages,
        formattedTools,
        signal: options.signal,
      });

      for await (const rawChunk of rawStream) {
        const parsed = this.parseStreamChunk(rawChunk);
        if (parsed !== null) {
          yield parsed;
        }
      }
    } catch (err) {
      throw wrapProviderError(err, this.providerName, this.config.modelName);
    }
  }
}
