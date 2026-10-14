import { z } from 'zod';

/**
 * 统一消息角色类型
 */
export const UnifiedRoleSchema = z.enum(['system', 'user', 'assistant', 'tool']);
export type UnifiedRole = z.infer<typeof UnifiedRoleSchema>;

/**
 * 统一工具调用结构契约
 * 注意：arguments 必须解析为已结构化的 JS 对象，绝不能保留为未解析的 JSON 字符串
 */
export const UnifiedToolCallSchema = z.object({
  id: z.string(),
  name: z.string(),
  arguments: z.record(z.unknown()),
});
export type UnifiedToolCall = z.infer<typeof UnifiedToolCallSchema>;

/**
 * 统一外部工具声明规范 (遵循标准 JSON Schema)
 */
export const UnifiedToolSchema = z.object({
  name: z.string(),
  description: z.string(),
  parameters: z.record(z.unknown()),
});
export type UnifiedTool = z.infer<typeof UnifiedToolSchema>;

/**
 * 统一输入消息规范
 */
export const UnifiedMessageSchema = z.object({
  role: UnifiedRoleSchema,
  content: z.string(),
  name: z.string().optional(),
  toolCallId: z.string().optional(),
  toolCalls: z.array(UnifiedToolCallSchema).optional(),
});
export type UnifiedMessage = z.infer<typeof UnifiedMessageSchema>;

/**
 * 统一 Token 使用量统计
 */
export const UnifiedTokenUsageSchema = z.object({
  inputTokens: z.number().default(0),
  outputTokens: z.number().default(0),
  cachedTokens: z.number().optional(),
  totalTokens: z.number().default(0),
});
export type UnifiedTokenUsage = z.infer<typeof UnifiedTokenUsageSchema>;

/**
 * 统一结束原因枚举
 */
export const UnifiedStopReasonSchema = z.enum(['stop', 'tool_call', 'max_tokens', 'error']);
export type UnifiedStopReason = z.infer<typeof UnifiedStopReasonSchema>;

/**
 * 统一非流式聊天响应结果
 */
export const UnifiedChatResponseSchema = z.object({
  content: z.string(),
  reasoning: z.string().optional(),
  role: z.literal('assistant').default('assistant'),
  model: z.string(),
  stopReason: UnifiedStopReasonSchema,
  toolCalls: z.array(UnifiedToolCallSchema).optional(),
  usage: UnifiedTokenUsageSchema,
  raw: z.unknown().optional(),
});
export type UnifiedChatResponse = z.infer<typeof UnifiedChatResponseSchema>;

/**
 * 统一流式切片输出
 */
export const UnifiedStreamChunkSchema = z.object({
  deltaContent: z.string(),
  deltaReasoning: z.string().optional(),
  deltaToolCalls: z.array(UnifiedToolCallSchema.partial()).optional(),
  stopReason: UnifiedStopReasonSchema.nullable().optional(),
  usage: UnifiedTokenUsageSchema.optional(),
});
export type UnifiedStreamChunk = z.infer<typeof UnifiedStreamChunkSchema>;
