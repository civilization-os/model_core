import type { ModelConfig } from '../config/index.js';
import type { BaseChatModel } from '../interfaces/base-model.js';
import { OpenAIOfficialModel } from './official/openai.official.js';
import { GeminiOfficialModel } from './official/gemini.official.js';
import { AnthropicOfficialModel } from './official/anthropic.official.js';
import { OpenAIProtocolModel } from './protocol/openai.protocol.js';
import { AnthropicProtocolModel } from './protocol/anthropic.protocol.js';

export * from './official/openai.official.js';
export * from './official/gemini.official.js';
export * from './official/anthropic.official.js';
export * from './protocol/openai.protocol.js';
export * from './protocol/anthropic.protocol.js';

/**
 * 统一模型工厂路由函数
 * 根据 ModelConfig 中的判别字段自动路由并实例化对应的模型适配器
 */
export function createChatModel(config: ModelConfig): BaseChatModel {
  if (config.category === 'official') {
    switch (config.vendor) {
      case 'openai':
        return new OpenAIOfficialModel(config);
      case 'gemini':
        return new GeminiOfficialModel(config);
      case 'anthropic':
        return new AnthropicOfficialModel(config);
      default: {
        const exhaustiveCheck: never = config.vendor;
        throw new Error(`Unsupported official vendor: ${exhaustiveCheck}`);
      }
    }
  }

  if (config.category === 'protocol') {
    switch (config.protocol) {
      case 'openai-compatible':
        return new OpenAIProtocolModel(config);
      case 'anthropic-compatible':
        return new AnthropicProtocolModel(config);
      default: {
        const exhaustiveCheck: never = config.protocol;
        throw new Error(`Unsupported protocol: ${exhaustiveCheck}`);
      }
    }
  }

  throw new Error(`Unknown model category: ${(config as { category: string }).category}`);
}
