import { createChatModel } from '../dist/index.js';

async function testDeepSeek() {
  const apiKey = 'sk-ec0c9302b0a64e59b6f94232b3f3ffa3';
  const baseURL = 'https://api.deepseek.com';

  console.log('==============================================');
  console.log('   正在测试 DeepSeek 真实 API 连接');
  console.log('==============================================\n');

  // 1. 测试常规对话模型: deepseek-chat
  console.log('▶ [测试 1] 调用 deepseek-chat (常规对话模型)...');
  try {
    const chatModel = createChatModel({
      category: 'protocol',
      protocol: 'openai-compatible',
      modelName: 'deepseek-chat',
      baseURL: baseURL,
      apiKey: apiKey,
      temperature: 0.7,
    });

    const res1 = await chatModel.invoke([
      { role: 'system', content: '你是一个专业的 AI 架构工程师，请简明扼要地回答。' },
      { role: 'user', content: '请用一句话告诉我：为什么大模型需要结构化输出？' },
    ]);

    console.log('\x1b[32m[deepseek-chat 回答]:\x1b[0m\n' + res1.content);
    console.log('\x1b[36m[Token 消耗]:\x1b[0m', res1.usage);
    console.log('\x1b[34m[结束原因]:\x1b[0m', res1.stopReason);
    console.log('----------------------------------------------\n');
  } catch (err) {
    console.error('\x1b[31m[deepseek-chat 测试异常]:\x1b[0m', err.message);
  }

  // 2. 测试推理模型: deepseek-reasoner (流式调用，测试思考过程剥离)
  console.log('▶ [测试 2] 调用 deepseek-reasoner (带深度思考链的模型)...');
  try {
    const reasonerModel = createChatModel({
      category: 'protocol',
      protocol: 'openai-compatible',
      modelName: 'deepseek-reasoner',
      baseURL: baseURL,
      apiKey: apiKey,
    });

    console.log('正在向 deepseek-reasoner 提问: "9.11 和 9.8 哪个数字更大？请给出你的详细分析。"');
    console.log('\n--- 开始流式接收 ---');

    const stream = reasonerModel.stream([
      { role: 'user', content: '9.11 和 9.8 哪个数字更大？请给出你的详细分析。' },
    ]);

    for await (const chunk of stream) {
      if (chunk.deltaReasoning) {
        process.stdout.write(`\x1b[33m${chunk.deltaReasoning}\x1b[0m`);
      }
      if (chunk.deltaContent) {
        process.stdout.write(`\x1b[32m${chunk.deltaContent}\x1b[0m`);
      }
    }
    console.log('\n\n--- 流式接收结束 ---');
    console.log('\x1b[32m[deepseek-reasoner 测试成功! 思考过程与正文已无缝分离输出]\x1b[0m');
  } catch (err) {
    console.error('\x1b[31m[deepseek-reasoner 测试异常]:\x1b[0m', err.message);
  }
}

testDeepSeek().catch(console.error);
