import { describe, it, expect } from 'vitest';
import { validateModelConfig } from '../src/config/index.js';

describe('ModelConfig Configuration Center', () => {
  describe('Official Model Configuration', () => {
    it('should validate standard OpenAI official config', () => {
      const input = {
        category: 'official',
        vendor: 'openai',
        modelName: 'gpt-4o',
        apiKey: 'sk-test-key-12345',
      };

      const result = validateModelConfig(input);
      expect(result.category).toBe('official');
      if (result.category === 'official') {
        expect(result.vendor).toBe('openai');
        expect(result.modelName).toBe('gpt-4o');
        expect(result.apiKey).toBe('sk-test-key-12345');
        expect(result.temperature).toBe(0.7); // 默认值
        expect(result.timeoutMs).toBe(60000); // 默认值
        expect(result.maxRetries).toBe(3);    // 默认值
      }
    });

    it('should validate official Gemini config with thinking budget', () => {
      const input = {
        category: 'official',
        vendor: 'gemini',
        modelName: 'gemini-2.5-flash',
        apiKey: 'gemini-key',
        thinkingBudget: 2048,
        reasoningEffort: 'high',
      };

      const result = validateModelConfig(input);
      expect(result.category).toBe('official');
      if (result.category === 'official') {
        expect(result.thinkingBudget).toBe(2048);
        expect(result.reasoningEffort).toBe('high');
      }
    });

    it('should throw error when apiKey is missing in official config', () => {
      const input = {
        category: 'official',
        vendor: 'anthropic',
        modelName: 'claude-3-7-sonnet',
        apiKey: '',
      };

      expect(() => validateModelConfig(input)).toThrow();
    });
  });

  describe('Protocol Model Configuration (Self-hosted / Gateways)', () => {
    it('should validate openai-compatible protocol config and auto fill apiKey with EMPTY', () => {
      const input = {
        category: 'protocol',
        protocol: 'openai-compatible',
        modelName: 'deepseek-r1-70b',
        baseURL: 'http://localhost:8000/v1',
      };

      const result = validateModelConfig(input);
      expect(result.category).toBe('protocol');
      if (result.category === 'protocol') {
        expect(result.protocol).toBe('openai-compatible');
        expect(result.baseURL).toBe('http://localhost:8000/v1');
        expect(result.apiKey).toBe('EMPTY'); // 自动补齐占位符
      }
    });

    it('should support extraBody and customHeaders for private inference engines', () => {
      const input = {
        category: 'protocol',
        protocol: 'openai-compatible',
        modelName: 'qwen-2.5-72b',
        baseURL: 'https://gateway.internal.net/v1',
        apiKey: 'internal-token',
        customHeaders: {
          'X-Tenant-ID': 'ai-core-tenant-1',
        },
        extraBody: {
          top_k: 50,
          repetition_penalty: 1.1,
          min_p: 0.05,
        },
      };

      const result = validateModelConfig(input);
      expect(result.category).toBe('protocol');
      if (result.category === 'protocol') {
        expect(result.customHeaders?.['X-Tenant-ID']).toBe('ai-core-tenant-1');
        expect(result.extraBody?.['top_k']).toBe(50);
        expect(result.extraBody?.['min_p']).toBe(0.05);
      }
    });

    it('should throw error when baseURL is not a valid URL', () => {
      const input = {
        category: 'protocol',
        protocol: 'openai-compatible',
        modelName: 'llama-3.3-70b',
        baseURL: 'not-a-valid-url',
      };

      expect(() => validateModelConfig(input)).toThrow();
    });
  });
});
