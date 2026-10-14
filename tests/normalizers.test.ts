import { describe, it, expect } from 'vitest';
import {
  stripThinkTags,
  StreamThinkExtractor,
  safeParseToolArguments,
} from '../src/utils/normalizers.js';
import { ProtocolTransformError } from '../src/utils/errors.js';

describe('Normalizers & Transformers Utilities', () => {
  describe('stripThinkTags', () => {
    it('should strip complete <think>...</think> tags and extract reasoning', () => {
      const input = '<think>\nHere is my internal reasoning\nStep 1: Check parameters\n</think>\nFinal answer to the user.';
      const result = stripThinkTags(input);

      expect(result.content).toBe('Final answer to the user.');
      expect(result.reasoning).toBe('Here is my internal reasoning\nStep 1: Check parameters');
    });

    it('should handle multiple <think> blocks', () => {
      const input = '<think>Part 1</think>Interim text<think>Part 2</think>Final text';
      const result = stripThinkTags(input);

      expect(result.content).toBe('Interim textFinal text');
      expect(result.reasoning).toBe('Part 1\n\nPart 2');
    });

    it('should handle unclosed <think> tag caused by token truncation', () => {
      const input = 'Normal prefix<think>Truncated thinking process...';
      const result = stripThinkTags(input);

      expect(result.content).toBe('Normal prefix');
      expect(result.reasoning).toBe('Truncated thinking process...');
    });

    it('should return original text cleanly when no think tag is present', () => {
      const input = 'Hello! How can I assist you today?';
      const result = stripThinkTags(input);

      expect(result.content).toBe(input);
      expect(result.reasoning).toBeUndefined();
    });
  });

  describe('StreamThinkExtractor', () => {
    it('should properly split thinking and content across chunk boundaries', () => {
      const extractor = new StreamThinkExtractor();
      const chunks = [
        'Prefix: ',
        '<th',
        'ink>Thinking start ',
        'and ongoing',
        '</th',
        'ink>Actual answer',
      ];

      const deltas: { content: string; reasoning: string }[] = [];
      for (const chunk of chunks) {
        const res = extractor.feed(chunk);
        deltas.push({
          content: res.deltaContent,
          reasoning: res.deltaReasoning ?? '',
        });
      }
      const flushed = extractor.flush();

      const fullContent = deltas.map((d) => d.content).join('') + flushed.deltaContent;
      const fullReasoning = deltas.map((d) => d.reasoning).join('') + (flushed.deltaReasoning ?? '');

      expect(fullContent).toBe('Prefix: Actual answer');
      expect(fullReasoning).toBe('Thinking start and ongoing');
    });

    it('should output normal text continuously when no think tags occur', () => {
      const extractor = new StreamThinkExtractor();
      const res1 = extractor.feed('Hello ');
      const res2 = extractor.feed('world!');
      const flushed = extractor.flush();

      expect(res1.deltaContent).toBe('Hello ');
      expect(res2.deltaContent).toBe('world!');
      expect(flushed.deltaContent).toBe('');
      expect(res1.deltaReasoning).toBeUndefined();
      expect(res2.deltaReasoning).toBeUndefined();
    });
  });

  describe('safeParseToolArguments', () => {
    it('should return object directly if already an object', () => {
      const obj = { city: 'Beijing', days: 3 };
      const parsed = safeParseToolArguments(obj);
      expect(parsed).toEqual(obj);
    });

    it('should parse standard json string', () => {
      const jsonStr = '{"location": "Tokyo", "unit": "celsius"}';
      const parsed = safeParseToolArguments(jsonStr);
      expect(parsed).toEqual({ location: 'Tokyo', unit: 'celsius' });
    });

    it('should safely unwrap markdown fenced json code blocks', () => {
      const markdownJson = '```json\n{\n  "query": "deep learning architectures",\n  "limit": 10\n}\n```';
      const parsed = safeParseToolArguments(markdownJson);
      expect(parsed).toEqual({ query: 'deep learning architectures', limit: 10 });
    });

    it('should tolerate trailing commas from model hallucinations', () => {
      const trailingCommaJson = '{"key1": "val1", "key2": 123, }';
      const parsed = safeParseToolArguments(trailingCommaJson);
      expect(parsed).toEqual({ key1: 'val1', key2: 123 });
    });

    it('should throw ProtocolTransformError when json is completely malformed', () => {
      const invalid = 'Not a json at all {{{{';
      expect(() =>
        safeParseToolArguments(invalid, { provider: 'test-provider', toolName: 'search' })
      ).toThrow(ProtocolTransformError);
    });
  });
});
