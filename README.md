# 多模态客服智能体 API

## 项目概述

这是一个基于 RAG + 多模态理解 的智能客服系统，支持文本和图片输入，能够从产品手册知识库中检索相关信息并生成准确回答。

## 核心能力

- **多模态理解**：支持文本和图片输入，准确识别用户意图
- **RAG知识库**：基于向量检索的产品手册问答系统
- **多轮对话**：支持会话上下文关联
- **幻觉抑制**：答案基于知识库，降低幻觉风险

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境

```bash
cp .env.example .env
# 编辑 .env，填入 API Key 等配置
```

### 3. 构建知识库

将产品手册放入 `knowledge_base/raw_documents/` 目录，然后运行：

```bash
python scripts/build_knowledge_base.py
```

### 4. 启动服务

```bash
python scripts/run_api.py
```

服务将在 `http://localhost:8000` 启动，API文档在 `http://localhost:8000/docs`

## API使用

### 文本问答

```bash
curl -X POST http://localhost:8000/chat/ \
  -H "Authorization: Bearer sk_customer_20260304" \
  -H "Content-Type: application/json" \
  -d '{"question": "DCB107电钻指示灯闪烁是什么意思？"}'
```

### 多模态问答（文本+图片）

```bash
curl -X POST http://localhost:8000/chat/ \
  -H "Authorization: Bearer sk_customer_20260304" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "请帮我看看这张图片显示什么故障",
    "images": ["data:image/png;base64,..."]
  }'
```

### 多轮对话

```bash
# 第一轮
curl -X POST http://localhost:8000/chat/ \
  -H "Authorization: Bearer sk_customer_20260304" \
  -d '{"question": "我想更换表带", "session_id": "my_session"}'

# 第二轮（追问）
curl -X POST http://localhost:8000/chat/ \
  -H "Authorization: Bearer sk_customer_20260304" \
  -d '{"question": "有其他尺寸吗？", "session_id": "my_session"}'
```

## 项目结构

```
.
├── api/                    # API服务层
│   ├── main.py            # FastAPI入口
│   ├── routes/           # 路由定义
│   ├── models/           # 数据模型
│   └── middleware/       # 中间件
├── core/                  # 核心业务逻辑
│   ├── multimodal/       # 多模态理解
│   ├── rag/              # RAG知识库
│   ├── dialogue/         # 对话管理
│   └── llm/              # LLM调用
├── config/                # 配置管理
├── scripts/               # 工具脚本
└── knowledge_base/       # 知识库存储
    ├── raw_documents/    # 原始文档
    └── chroma_db/        # 向量数据库
```

## 配置说明

主要环境变量（`.env`）：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_API_KEY` | LLM API密钥 | - |
| `LLM_MODEL` | 使用的模型 | gpt-4o-mini |
| `EMBEDDING_MODEL` | Embedding模型 | text-embedding-3-small |
| `API_TOKEN` | API认证Token | sk_customer_20260304 |

## 评分标准

回答质量由LLM裁判打分（1-5分）：

| 分数 | 描述 |
|------|------|
| 1 | 质量差：未回应问题，结构混乱 |
| 2 | 一般：部分回应，但不完整 |
| 3 | 中等：回应问题，缺乏深度 |
| 4 | 良好：清晰全面，图文结合合理 |
| 5 | 优秀：详细有深度，图文完美互补 |
