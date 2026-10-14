/**
 * 端到端实机测试脚本 (可用于快速验证真实 API Key 或本地自部署服务)
 * 运行方式 (Windows PowerShell):
 *   node examples/demo.js
 */

import { createChatModel } from '../dist/index.js';

async function main() {
  console.log('====================================');
  console.log('   model_core 端到端测试运行器');
  console.log('====================================\n');

  // 1. 如果你有本地 Ollama (如 deepseek-r1 / qwen2.5 / llama3)
  //    默认端点: http://localhost:11434/v1
  const testOllama = process.env.TEST_OLLAMA === 'true';

  if (testOllama) {
    console.log('[测试 1] 正在连接本地 Ollama (openai-compatible 协议)...');
    try {
      const ollamaModel = createChatModel({
        category: 'protocol',
        protocol: 'openai-compatible',
        modelName: process.env.OLLAMA_MODEL || 'deepseek-r1:latest',
        baseURL: 'http://localhost:11434/v1',
      });

      console.log('发起流式提问: "你好，请用一句话介绍你自己"');
      const stream = ollamaModel.stream([
        { role: 'user', content: '你好，请用一句话介绍你自己' },
      ]);

      for await (const chunk of stream) {
        if (chunk.deltaReasoning) {
          process.stdout.write(`\x1b[33m[思考]\x1b[0m ${chunk.deltaReasoning}`);
        }
        if (chunk.deltaContent) {
          process.stdout.write(chunk.deltaContent);
        }
      }
      console.log('\n[Ollama 测试完成]\n');
    } catch (err) {
      console.error('[Ollama 测试失败]:', err.message);
    }
  }

  // 2. 如果配置了 GEMINI_API_KEY
  if (process.env.GEMINI_API_KEY) {
    console.log('[测试 2] 正在连接官方 Gemini 原厂模型...');
    try {
      const geminiModel = createChatModel({
        category: 'official',
        vendor: 'gemini',
        modelName: 'gemini-2.5-flash',
        apiKey: process.env.GEMINI_API_KEY,
        thinkingBudget: 1024,
      });

      const res = await geminiModel.invoke([
        { role: 'system', content: '你是一位精通 TypeScript 的助手。' },
        { role: 'user', content: '请输出一个快速排序的代码片段。' },
      ]);

      if (res.reasoning) {
        console.log('\x1b[33m[思考过程]:\x1b[0m\n', res.reasoning);
      }
      console.log('\x1b[32m[正文回答]:\x1b[0m\n', res.content);
      console.log('\x1b[36m[Token 消耗]:\x1b[0m', res.usage);
      console.log('\n[Gemini 测试完成]\n');
    } catch (err) {
      console.error('[Gemini 测试失败]:', err.message);
    }
  }

  // 3. 如果配置了 OPENAI_API_KEY
  if (process.env.OPENAI_API_KEY) {
    console.log('[测试 3] 正在连接官方 OpenAI 原厂模型...');
    try {
      const openaiModel = createChatModel({
        category: 'official',
        vendor: 'openai',
        modelName: 'gpt-4o-mini',
        apiKey: process.env.OPENAI_API_KEY,
      });

      const res = await openaiModel.invoke([
        { role: 'user', content: '你好，请回复 OK 即可。' },
      ]);

      console.log('\x1b[32m[回答]:\x1b[0m', res.content);
      console.log('\x1b[36m[Token 消耗]:\x1b[0m', res.usage);
      console.log('\n[OpenAI 测试完成]\n');
    } catch (err) {
      console.error('[OpenAI 测试失败]:', err.message);
    }
  }

  if (!testOllama && !process.env.GEMINI_API_KEY && !process.env.OPENAI_API_KEY) {
    console.log('💡 提示：当前未检测到 API Key 或 TEST_OLLAMA 环境变量。');
    console.log('若想运行真实模型调用测试，可以在 PowerShell 中传入环境变量后执行：');
    console.log('  例如测试 Gemini:');
    console.log('    $env:GEMINI_API_KEY="你的Key"; node examples/demo.js\n');
    console.log('  例如测试本地 Ollama:');
    console.log('    $env:TEST_OLLAMA="true"; node examples/demo.js\n');
  }
}

main().catch(console.error);
