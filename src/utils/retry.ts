import { AuthenticationError, RateLimitError } from './errors.js';

export interface RetryOptions {
  maxRetries?: number;
  initialDelayMs?: number;
  maxDelayMs?: number;
  backoffFactor?: number;
  shouldRetry?: (error: unknown) => boolean;
}

/**
 * 默认的可重试条件判断：
 * - 命中 RateLimitError (429) 重试
 * - 遇到临时服务端错误 (5xx) 或网络连接重置等临时错误重试
 * - 认证错误 (401/403) 绝对不重试
 */
export function defaultShouldRetry(error: unknown): boolean {
  if (error instanceof AuthenticationError) {
    return false;
  }
  if (error instanceof RateLimitError) {
    return true;
  }

  const anyErr = error as { status?: number; statusCode?: number; code?: string; message?: string };
  const status = anyErr?.status ?? anyErr?.statusCode;
  if (status === 429 || (status !== undefined && status >= 500 && status < 600)) {
    return true;
  }

  const code = anyErr?.code;
  if (code === 'ECONNRESET' || code === 'ETIMEDOUT' || code === 'ENOTFOUND' || code === 'ECONNABORTED') {
    return true;
  }

  const message = (anyErr?.message || '').toLowerCase();
  if (message.includes('rate limit') || message.includes('quota') || message.includes('overloaded')) {
    return true;
  }

  return false;
}

/**
 * 异步指数退避重试执行器
 */
export async function withExponentialBackoff<T>(
  operation: () => Promise<T>,
  options: RetryOptions = {}
): Promise<T> {
  const maxRetries = options.maxRetries ?? 3;
  const initialDelay = options.initialDelayMs ?? 1000;
  const maxDelay = options.maxDelayMs ?? 10000;
  const factor = options.backoffFactor ?? 2;
  const shouldRetry = options.shouldRetry ?? defaultShouldRetry;

  let attempt = 0;
  let delay = initialDelay;

  while (true) {
    try {
      return await operation();
    } catch (err) {
      attempt++;
      if (attempt > maxRetries || !shouldRetry(err)) {
        throw err;
      }

      // 添加轻微抖动 (Jitter) 避免共振惊群
      const jitter = Math.random() * 200;
      const sleepTime = Math.min(delay + jitter, maxDelay);

      await new Promise((resolve) => setTimeout(resolve, sleepTime));
      delay *= factor;
    }
  }
}
