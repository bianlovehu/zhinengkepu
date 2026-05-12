# 多模态客服智能体 - 项目规格说明

## 1. 项目概述

### 项目名称
多模态客服智能体

### 项目类型
RAG + 多模态理解 的智能客服系统

### 核心功能
基于产品手册知识库的多模态问答系统，支持文本和图片输入，能够准确理解用户问题并从知识库中检索相关信息生成回答。

### 目标用户
- 客服系统集成方
- 电商平台
- 产品售后支持

---

## 2. 功能规格

### 2.1 核心功能

#### 2.1.1 多模态理解
- 文本输入解析
- 图片输入解析（Base64格式）
- 用户意图识别
- 关键信息提取

#### 2.1.2 RAG知识库
- 文档加载（TXT、PDF、DOCX、HTML、Markdown）
- 文本分块（支持语义分块和固定大小分块）
- 向量化存储（ChromaDB）
- 混合检索（向量检索 + 关键词检索）
- 图片索引和检索

#### 2.1.3 多轮对话
- 会话管理（创建、获取、更新）
- 对话历史记录
- 上下文关联
- 思维链拆解

#### 2.1.4 幻觉抑制
- 答案与知识库一致性检查
- 置信度评估
- 引用追溯

### 2.2 API接口

#### POST /chat
- **认证**：Bearer Token
- **请求体**：
  - `question`: string (必填)
  - `images`: string[] (可选，Base64图片)
  - `session_id`: string (可选)
  - `stream`: boolean (可选)
- **响应**：
  ```json
  {
    "code": 0,
    "msg": "success",
    "data": {
      "answer": "回答内容",
      "session_id": "会话ID",
      "timestamp": 1234567890
    }
  }
  ```

---

## 3. 技术架构

### 3.1 模块划分

```
┌─────────────────────────────────────────────────┐
│                   API Layer                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │
│  │ /chat   │  │ /health │  │ Auth Middleware │  │
│  └─────────┘  └─────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────┤
│                 Core Business                    │
│  ┌──────────────┐  ┌──────────────┐             │
│  │   Multimodal │  │     RAG      │             │
│  │  Understanding│  │  Retrieval  │             │
│  └──────────────┘  └──────────────┘             │
│  ┌──────────────┐  ┌──────────────┐             │
│  │   Dialogue   │  │     LLM      │             │
│  │  Management  │  │  Generation  │             │
│  └──────────────┘  └──────────────┘             │
├─────────────────────────────────────────────────┤
│              Knowledge Base Layer                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │Document  │  │  Vector  │  │    Image     │  │
│  │  Loader  │  │   Store  │  │    Index     │  │
│  └──────────┘  └──────────┘  └──────────────┘  │
└─────────────────────────────────────────────────┘
```

### 3.2 技术栈

| 层级 | 技术选型 |
|------|----------|
| API框架 | FastAPI + Uvicorn |
| LLM | OpenAI GPT-4o-mini / 智谱GLM / 通义千问 |
| Embedding | OpenAI text-embedding-3-small |
| 向量库 | ChromaDB |
| 文档解析 | PyPDF, python-docx, BeautifulSoup |
| 图片处理 | Pillow |

---

## 4. 数据规格

### 4.1 知识库数据

```
knowledge_base/
├── raw_documents/           # 原始文档
│   ├── 冰箱手册.pdf
│   ├── 吹风机手册.pdf
│   └── ...
├── processed/
│   ├── text_chunks/        # 分块后的文本
│   └── images/             # 提取的图片
│       ├── 冰箱手册_p1_img1.png
│       └── ...
└── chroma_db/               # ChromaDB向量库
```

### 4.2 会话数据（内存）

```python
{
    "session_id": "kf_xxx",
    "created_at": timestamp,
    "last_active": timestamp,
    "messages": [
        {"role": "user", "content": "...", "timestamp": ...},
        {"role": "assistant", "content": "...", "timestamp": ...}
    ],
    "context": {}
}
```

---

## 5. 配置规格

### 5.1 环境变量

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `LLM_PROVIDER` | string | openai | LLM提供商 |
| `LLM_API_KEY` | string | - | API密钥 |
| `LLM_MODEL` | string | gpt-4o-mini | 模型名称 |
| `VISION_MODEL` | string | gpt-4o-mini | 视觉模型 |
| `EMBEDDING_MODEL` | string | text-embedding-3-small | Embedding模型 |
| `EMBEDDING_DIMENSION` | int | 1536 | 向量维度 |
| `CHUNK_SIZE` | int | 500 | 分块大小 |
| `TOP_K` | int | 5 | 检索数量 |
| `API_TOKEN` | string | sk_customer_20260304 | 认证Token |

---

## 6. 验收标准

### 6.1 功能验收

- [x] API端点 `/chat` 正常工作
- [x] Bearer Token 认证通过
- [x] 文本问答返回正确格式
- [x] 多模态问答（文本+图片）正常工作
- [x] 多轮对话上下文关联正常
- [x] 知识库检索返回相关内容
- [x] 图片检索返回相关图片

### 6.2 性能要求

| 指标 | 要求 |
|------|------|
| 文本响应时间 | < 20s |
| 多模态响应时间 | < 30s |
| API可用性 | 99% |

### 6.3 评分目标

| 分数档 | 说明 |
|--------|------|
| 3分+ | 结构清晰，回应问题 |
| 4分+ | 全面准确，图文结合 |
| 5分 | 详细有深度，图文完美互补 |

---

## 7. 后续优化方向

1. **RAG优化**：混合检索策略、重排序、多跳推理
2. **多模态优化**：专用视觉模型微调
3. **幻觉抑制**：更严格的引用验证
4. **性能优化**：缓存、批量处理、流式输出
5. **扩展性**：支持更多文档格式、知识库类型
