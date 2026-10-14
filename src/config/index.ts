import { z } from 'zod';

/**
 * 通用基础配置契约
 */
export const BaseConfigSchema = z.object({
  modelName: z.string().min(1, 'modelName is required'),
  temperature: z.number().min(0).max(2).default(0.7),
  maxTokens: z.number().positive().optional(),
  timeoutMs: z.number().positive().default(60000),
  maxRetries: z.number().int().min(0).default(3),
});
export type BaseConfig = z.infer<typeof BaseConfigSchema>;

/**
 * 官方云端模型配置契约 (category: 'official')
 */
export const OfficialModelConfigSchema = BaseConfigSchema.extend({
  category: z.literal('official'),
  vendor: z.enum(['openai', 'gemini', 'anthropic']),
  apiKey: z.string().min(1, 'apiKey is required for official providers'),
  thinkingBudget: z.number().positive().optional(),
  reasoningEffort: z.enum(['low', 'medium', 'high']).optional(),
});
export type OfficialModelConfig = z.infer<typeof OfficialModelConfigSchema>;

/**
 * 官方协议自部署配置契约 (category: 'protocol')
 * 兼容 vLLM, Ollama, SGLang, OneAPI 等自部署与网关架构
 */
export const ProtocolModelConfigSchema = BaseConfigSchema.extend({
  category: z.literal('protocol'),
  protocol: z.enum(['openai-compatible', 'anthropic-compatible']),
  baseURL: z.string().url('baseURL must be a valid URL'),
  apiKey: z.string().default('EMPTY'),
  customHeaders: z.record(z.string()).optional(),
  extraBody: z.record(z.unknown()).optional(),
});
export type ProtocolModelConfig = z.infer<typeof ProtocolModelConfigSchema>;

/**
 * 模型配置判别联合类型 (Discriminated Union)
 */
export const ModelConfigSchema = z.discriminatedUnion('category', [
  OfficialModelConfigSchema,
  ProtocolModelConfigSchema,
]);
export type ModelConfig = z.infer<typeof ModelConfigSchema>;

/**
 * 配置校验与默认值填充函数
 */
export function validateModelConfig(input: unknown): ModelConfig {
  return ModelConfigSchema.parse(input);
}
