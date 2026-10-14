import { ProtocolTransformError } from './errors.js';

export interface ThinkExtractionResult {
  content: string;
  reasoning?: string;
}

/**
 * 静态文本中的 <think>...</think> 标签清洗与推理过程剥离
 * 专门应对自部署开源模型 (如 DeepSeek-R1 / QwQ 等在 Ollama/vLLM 场景下直接输出在正文中的情况)
 */
export function stripThinkTags(rawText: string): ThinkExtractionResult {
  if (!rawText) {
    return { content: '' };
  }

  // 1. 检查是否存在成对的 <think>...</think>
  const thinkRegex = /<think>([\s\S]*?)<\/think>/gi;
  const thinkMatches: string[] = [];
  let match: RegExpExecArray | null;

  while ((match = thinkRegex.exec(rawText)) !== null) {
    if (match[1]) {
      thinkMatches.push(match[1].trim());
    }
  }

  // 移除所有完整的 <think>...</think> 标签块
  let cleanedContent = rawText.replace(thinkRegex, '');

  // 2. 检查是否有未闭合的 <think> (例如由于 max_tokens 截断导致的未闭合情况)
  const unclosedThinkIndex = cleanedContent.indexOf('<think>');
  if (unclosedThinkIndex !== -1) {
    const unclosedThinking = cleanedContent.slice(unclosedThinkIndex + '<think>'.length).trim();
    if (unclosedThinking) {
      thinkMatches.push(unclosedThinking);
    }
    cleanedContent = cleanedContent.slice(0, unclosedThinkIndex);
  }

  // 3. 处理可能存在的孤立 </think>
  cleanedContent = cleanedContent.replace(/<\/think>/gi, '').trim();

  const reasoning = thinkMatches.length > 0 ? thinkMatches.join('\n\n') : undefined;

  return {
    content: cleanedContent,
    reasoning,
  };
}

/**
 * 流式状态下的 <think> 标签状态机切分器
 * 解决大模型流式传输时，<think> 标签跨切片（如切片 1: "<th", 切片 2: "ink>内容"）的流式分离问题
 */
export class StreamThinkExtractor {
  private buffer = '';
  private inThinking = false;

  /**
   * 传入新的文本增量切片，返回已明确归属的正文增量与思考增量
   */
  public feed(delta: string): { deltaContent: string; deltaReasoning?: string } {
    this.buffer += delta;
    let deltaContent = '';
    let deltaReasoning = '';

    while (this.buffer.length > 0) {
      if (!this.inThinking) {
        // 当前处于正文输出状态，寻找 <think>
        const openIdx = this.buffer.indexOf('<think>');
        if (openIdx !== -1) {
          // <think> 之前的全部属于正文
          deltaContent += this.buffer.slice(0, openIdx);
          this.inThinking = true;
          this.buffer = this.buffer.slice(openIdx + '<think>'.length);
        } else {
          // 检查末尾是否可能是 <think> 的前缀（如 "<", "<t", "<th", "<thi", "<thin", "<think"）
          const partialMatch = this.findPartialOpenTag(this.buffer);
          if (partialMatch !== -1) {
            deltaContent += this.buffer.slice(0, partialMatch);
            this.buffer = this.buffer.slice(partialMatch);
            break; // 暂存疑似标签片段，等待下一 chunk 确认
          } else {
            deltaContent += this.buffer;
            this.buffer = '';
          }
        }
      } else {
        // 当前处于思考输出状态，寻找 </think>
        const closeIdx = this.buffer.indexOf('</think>');
        if (closeIdx !== -1) {
          deltaReasoning += this.buffer.slice(0, closeIdx);
          this.inThinking = false;
          this.buffer = this.buffer.slice(closeIdx + '</think>'.length);
        } else {
          // 检查末尾是否可能是 </think> 的前缀
          const partialMatch = this.findPartialCloseTag(this.buffer);
          if (partialMatch !== -1) {
            deltaReasoning += this.buffer.slice(0, partialMatch);
            this.buffer = this.buffer.slice(partialMatch);
            break; // 暂存疑似闭合标签片段
          } else {
            deltaReasoning += this.buffer;
            this.buffer = '';
          }
        }
      }
    }

    return {
      deltaContent,
      deltaReasoning: deltaReasoning || undefined,
    };
  }

  /**
   * 流结束时冲刷剩余缓冲区
   */
  public flush(): { deltaContent: string; deltaReasoning?: string } {
    const remaining = this.buffer;
    this.buffer = '';
    if (!remaining) {
      return { deltaContent: '' };
    }

    if (this.inThinking) {
      return { deltaContent: '', deltaReasoning: remaining };
    } else {
      return { deltaContent: remaining };
    }
  }

  private findPartialOpenTag(str: string): number {
    const tag = '<think>';
    for (let i = 1; i < tag.length; i++) {
      if (str.endsWith(tag.slice(0, i))) {
        return str.length - i;
      }
    }
    return -1;
  }

  private findPartialCloseTag(str: string): number {
    const tag = '</think>';
    for (let i = 1; i < tag.length; i++) {
      if (str.endsWith(tag.slice(0, i))) {
        return str.length - i;
      }
    }
    return -1;
  }
}

/**
 * 安全解析大模型返回的 Tool Arguments
 * 具备以下容错能力：
 * 1. 自动解开 Markdown 代码块（```json ... ```）
 * 2. 兼容已经是 Record<string, unknown> 的情况
 * 3. 自动修剪首尾空白
 * 4. 无法解析时抛出标准 ProtocolTransformError
 */
export function safeParseToolArguments(
  rawArgs: unknown,
  context?: { provider?: string; modelName?: string; toolName?: string }
): Record<string, unknown> {
  if (rawArgs === null || rawArgs === undefined) {
    return {};
  }

  if (typeof rawArgs === 'object' && !Array.isArray(rawArgs)) {
    return rawArgs as Record<string, unknown>;
  }

  if (typeof rawArgs !== 'string') {
    throw new ProtocolTransformError(
      `Invalid tool arguments format: expected string or object, got ${typeof rawArgs}`,
      {
        provider: context?.provider ?? 'unknown',
        modelName: context?.modelName ?? 'unknown',
        rawError: rawArgs,
      }
    );
  }

  let text = rawArgs.trim();

  // 若为空字符串直接返回空对象
  if (text === '' || text === '{}') {
    return {};
  }

  // 去除 markdown 代码块包裹
  if (text.startsWith('```')) {
    text = text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim();
  }

  try {
    const parsed = JSON.parse(text);
    if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return { value: parsed };
  } catch (err) {
    // 尝试二次清理常见模型幻觉语法（如末尾逗号等）
    try {
      const relaxed = text.replace(/,\s*([}\]])/g, '$1');
      const parsed = JSON.parse(relaxed);
      if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      // 容错失败，抛出统一异常
    }

    throw new ProtocolTransformError(
      `Failed to parse tool call arguments for "${context?.toolName ?? 'unknown'}": ${text}`,
      {
        provider: context?.provider ?? 'unknown',
        modelName: context?.modelName ?? 'unknown',
        rawError: err,
      }
    );
  }
}
