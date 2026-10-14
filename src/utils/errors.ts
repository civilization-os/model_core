/**
 * 统一模型异常基类
 */
export class ModelCoreError extends Error {
  readonly provider: string;
  readonly modelName: string;
  readonly statusCode?: number;
  readonly rawError?: unknown;

  constructor(
    message: string,
    options: {
      provider: string;
      modelName: string;
      statusCode?: number;
      rawError?: unknown;
    }
  ) {
    super(message);
    this.name = 'ModelCoreError';
    this.provider = options.provider;
    this.modelName = options.modelName;
    this.statusCode = options.statusCode;
    this.rawError = options.rawError;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * 鉴权异常 (401 / 403 API Key 错误等)
 */
export class AuthenticationError extends ModelCoreError {
  constructor(
    message: string,
    options: { provider: string; modelName: string; statusCode?: number; rawError?: unknown }
  ) {
    super(message, options);
    this.name = 'AuthenticationError';
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * 速率限制与配额超限异常 (429 限流，支持自动重试触发)
 */
export class RateLimitError extends ModelCoreError {
  constructor(
    message: string,
    options: { provider: string; modelName: string; statusCode?: number; rawError?: unknown }
  ) {
    super(message, options);
    this.name = 'RateLimitError';
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * 模型调用超时异常
 */
export class ModelTimeoutError extends ModelCoreError {
  constructor(
    message: string,
    options: { provider: string; modelName: string; statusCode?: number; rawError?: unknown }
  ) {
    super(message, options);
    this.name = 'ModelTimeoutError';
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * 协议与数据双向转换异常 (包含 Tool JSON 反序列化失败、结构不匹配等)
 */
export class ProtocolTransformError extends ModelCoreError {
  constructor(
    message: string,
    options: { provider: string; modelName: string; statusCode?: number; rawError?: unknown }
  ) {
    super(message, options);
    this.name = 'ProtocolTransformError';
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * 将第三方原厂 SDK / 网络错误归一化包装为标准异常
 */
export function wrapProviderError(
  err: unknown,
  provider: string,
  modelName: string
): ModelCoreError {
  if (err instanceof ModelCoreError) {
    return err;
  }

  const anyError = err as { status?: number; statusCode?: number; message?: string; name?: string; code?: string };
  const statusCode = anyError?.status ?? anyError?.statusCode;
  const message = anyError?.message || String(err) || 'Unknown model invocation error';

  if (statusCode === 401 || statusCode === 403 || message.includes('API key') || message.includes('unauthorized')) {
    return new AuthenticationError(message, { provider, modelName, statusCode, rawError: err });
  }

  if (statusCode === 429 || message.includes('rate limit') || message.includes('Quota exceeded')) {
    return new RateLimitError(message, { provider, modelName, statusCode, rawError: err });
  }

  if (
    anyError?.code === 'ETIMEDOUT' ||
    anyError?.name === 'TimeoutError' ||
    message.includes('timeout') ||
    statusCode === 408 ||
    statusCode === 504
  ) {
    return new ModelTimeoutError(message, { provider, modelName, statusCode, rawError: err });
  }

  return new ModelCoreError(message, { provider, modelName, statusCode, rawError: err });
}
