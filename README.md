# @civilization/model-core

`@civilization/model-core` 是底层模型连接核心包（Node.js + TypeScript），负责为上层 `@civilization/multi-agent`（多智能体系统）及业务层提供**完全无感的统一模型调用能力**。

[![CI](https://github.com/civilization-os/model_core/actions/workflows/ci.yml/badge.svg)](https://github.com/civilization-os/model_core/actions/workflows/ci.yml)
[![npm version](https://img.shields.io/npm/v/@civilization/model-core.svg)](https://www.npmjs.com/package/@civilization/model-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 核心设计哲学

本模块的设计哲学由三大支柱构成：

1. **官方模型 (`providers/official`)**：直连云端原厂 SDK（OpenAI、Gemini、Anthropic），释放厂商独占特性（如 Anthropic/Gemini/OpenAI 的深度思考预算控制、原生提示词缓存等）。
2. **官方协议自部署 (`providers/protocol`)**：基于行业标准协议连接内网/私有化推理容器（vLLM、Ollama、SGLang、OneAPI 等）及协议网关，强化内网安全、Header 控制与扩展参数（`top_k`、`repetition_penalty`、`min_p`）透传。
3. **内聚双向翻译 (`Bidirectional Translation`)**：每个 Provider 自身作为适配器（Adapter），内部消化“标准输入 $\rightarrow$ 厂商入参”与“厂商出参 $\rightarrow$ 标准输出”的双向转换，对外 100% 暴露统一的数据契约。

---

## 工程目录结构

```text
model_core/
├── package.json
├── tsconfig.json
├── README.md                           # 中文技术规格与使用文档
├── src/
│   ├── index.ts                        # 统一对外导出出口
│   ├── config/
│   │   └── index.ts                    # 基于 Zod 判别联合类型的配置中心
│   ├── schemas/
│   │   └── index.ts                    # 统一数据契约 (消息、工具、响应、流式切片、Token 统计)
│   ├── interfaces/
│   │   └── base-model.ts               # BaseChatModel 抽象基类与双向翻译方法契约
│   ├── providers/
│   │   ├── index.ts                    # createChatModel 统一工厂路由
│   │   ├── official/                   # 【核心部分 1】官方模型实现
│   │   │   ├── openai.official.ts      # OpenAI 官方原厂适配器
│   │   │   ├── gemini.official.ts      # Google Gemini 官方原厂适配器
│   │   │   └── anthropic.official.ts   # Anthropic Claude 官方原厂适配器
│   │   └── protocol/                   # 【核心部分 2】官方协议自部署实现
│   │       ├── openai.protocol.ts      # 兼容 vLLM / Ollama / SGLang / OneAPI
│   │       └── anthropic.protocol.ts   # 兼容 Anthropic 协议网关
│   └── utils/
│       ├── errors.ts                   # 统一异常体系封装与原厂错误包装
│       ├── retry.ts                    # 异步指数退避重试工具
│       └── normalizers.ts              # 【核心部分 3 辅助】清洗 <think> 标签、流式切分状态机、修复 Tool JSON
└── tests/
    ├── config.test.ts                  # 配置中心校验测试
    ├── normalizers.test.ts             # 标签清洗与容错解析测试
    └── providers.test.ts               # 双向转换与工厂路由测试
```

---

## 快速上手

### 1. 安装与构建

```bash
# 安装依赖
npm install

# 运行 TypeScript 严格编译
npm run build

# 执行单元测试
npm test
```

---

### 2. 基础使用示例

#### 示例 A：调用官方原厂模型（以 Gemini 2.5 为例）

```typescript
import { createChatModel } from '@civilization/model-core';

const model = createChatModel({
  category: 'official',
  vendor: 'gemini',
  modelName: 'gemini-2.5-flash',
  apiKey: process.env.GEMINI_API_KEY!,
  temperature: 0.7,
  thinkingBudget: 2048, // 启用官方思考预算
});

const response = await model.invoke([
  { role: 'system', content: '你是一位资深软件架构师。' },
  { role: 'user', content: '请简述微服务与单体架构的核心选型权衡。' },
]);

console.log('思考过程:', response.reasoning);
console.log('正文输出:', response.content);
console.log('Token消耗:', response.usage);
```

---

#### 示例 B：连接本地自部署开源模型（vLLM / Ollama 部署 DeepSeek-R1）

在自部署场景下，开源推理引擎通常会将思考过程混在输出正文的 `<think>...</think>` 中。`model_core` 会自动检测并干净剥离：

```typescript
import { createChatModel } from '@civilization/model-core';

const model = createChatModel({
  category: 'protocol',
  protocol: 'openai-compatible',
  modelName: 'deepseek-r1-70b',
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'EMPTY', // 可选，未传时自动填充为 'EMPTY' 防止 SDK 报错
  extraBody: {
    top_k: 50,
    repetition_penalty: 1.1, // 透传自部署私有超参
  },
});

const response = await model.invoke([
  { role: 'user', content: '用 TypeScript 实现一个支持 LRU 的缓存类。' },
]);

// content 100% 纯净，无任何 <think> 标签残留
console.log('纯净代码与正文:', response.content);
// 思考过程被自动归一化提取至 reasoning 字段
console.log('DeepSeek 内省推理:', response.reasoning);
```

---

#### 示例 C：工具调用 (Tool / Function Calling)

统一格式输入输出，框架负责抹平 OpenAI、Claude、Gemini 的工具结构差异，并自动将字符串参数反序列化为 JS 原生对象：

```typescript
import { createChatModel, type UnifiedTool } from '@civilization/model-core';

const weatherTool: UnifiedTool = {
  name: 'get_current_weather',
  description: '查询指定城市的实时天气数据',
  parameters: {
    type: 'object',
    properties: {
      city: { type: 'string', description: '城市名' },
      unit: { type: 'string', enum: ['celsius', 'fahrenheit'] },
    },
    required: ['city'],
  },
};

const model = createChatModel({
  category: 'official',
  vendor: 'openai',
  modelName: 'gpt-4o',
  apiKey: process.env.OPENAI_API_KEY!,
});

const response = await model.invoke(
  [{ role: 'user', content: '上海现在的气温怎么样？' }],
  { tools: [weatherTool] }
);

if (response.stopReason === 'tool_call' && response.toolCalls) {
  for (const call of response.toolCalls) {
    console.log('工具名称:', call.name);
    console.log('已解析的入参对象:', call.arguments); // 直接为 { city: '上海' } 对象
  }
}
```

---

#### 示例 D：流式输出与思考状态机切分

```typescript
const stream = model.stream([
  { role: 'user', content: '解释量子纠缠的基本原理。' },
]);

for await (const chunk of stream) {
  if (chunk.deltaReasoning) {
    process.stdout.write(`[思考] ${chunk.deltaReasoning}`);
  }
  if (chunk.deltaContent) {
    process.stdout.write(chunk.deltaContent);
  }
}
```

---

## 统一异常体系

当第三方原厂 SDK 或网络发生异常时，统一转换为标准异常类抛出：

| 异常类 | 触发场景 | 是否默认重试 |
| :--- | :--- | :--- |
| `AuthenticationError` | 401 / 403 API Key 无效或未授权 | 否 |
| `RateLimitError` | 429 配额耗尽或并发超限 | 是（指数退避重试） |
| `ModelTimeoutError` | 超出 `timeoutMs` 设置的时间（默认 60s） | 否 |
| `ProtocolTransformError` | 厂商回包格式不符或 Tool JSON 无法解析 | 否 |
| `ModelCoreError` | 统一基类，保留原始 `statusCode` 与 `rawError` | 根据状态码判定 |
