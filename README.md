# 多模态客服智能体 API

## 项目概述

这是一个基于 RAG + 多模态理解 的智能客服系统，支持文本和图片输入，能够从产品手册与通用客服政策知识库中检索相关信息并生成准确回答。当前版本面向比赛评分做了专项优化：通用售后政策快答、产品手册 RAG、图片 `<PIC>` 保留、多问题拆解和批量提交闭环。

## 核心能力

- **多模态理解**：支持文本和图片输入，准确识别用户意图
- **RAG知识库**：基于向量检索的产品手册问答系统，包含客服政策手册
- **问题分类路由**：将问题分为 `policy`、`manual`、`mixed`，分别使用政策快答、产品手册检索或子问题拆解
- **图片命中增强**：对电钻指示灯、健身追踪器表带、airfryer 等高频题保留 `<PIC>` 并锚定关键图片
- **多轮对话**：支持会话上下文关联
- **幻觉抑制**：答案基于知识库，政策题使用本地规则模板，手册题提供 LLM 失败抽取式兜底
- **比赛闭环工具**：支持批量调用 API 生成提交 CSV，并检查空答案、占位话术、缺图和结构问题

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

当前仓库已包含 `knowledge_base/raw_documents/客服政策手册.txt`，重建知识库后会同时索引产品手册、客服政策和插图。

### 4. 启动服务

```bash
python scripts/run_api.py
```

服务将在 `http://localhost:8000` 启动，API文档在 `http://localhost:8000/docs`

### 5. 生成比赛提交文件

启动 API 后，可批量调用 public 问题并生成提交 CSV：

```bash
python scripts/run_public_questions.py \
  --input question_public.csv \
  --output submission.csv \
  --concurrency 4
```

可先小批量验证：

```bash
python scripts/run_public_questions.py --limit 60 --output submission_sample.csv
```

### 6. 检查答案质量

```bash
python scripts/evaluate_outputs.py \
  --questions question_public.csv \
  --answers submission.csv
```

检查项包括：空答案、少于 20 字、无效占位话术、API 错误串、多问题未编号、步骤题未编号、典型图片题缺少 `<PIC>`、产品题缺少产品关键词。

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
│   ├── dialogue/         # 对话管理、问题路由
│   └── llm/              # LLM调用
├── config/                # 配置管理
├── scripts/               # 工具脚本
└── knowledge_base/       # 知识库存储
    ├── raw_documents/    # 原始文档
    └── chroma_db/        # 向量数据库
```

## 比赛优化策略

### policy 路由

覆盖退换货、退款、运费、发票、物流、包装破损、少发错发、二手假货、质保维修、投诉安抚等通用客服问题。命中后优先使用本地模板生成答案，减少 LLM 幻觉并提升响应速度。

### manual 路由

面向产品说明书问题，使用混合检索和关键词加权定位章节；对需要图片的问题保留 `<PIC>` 标记。典型锚点包括：

- `DCB107/DCB112` 指示灯：`drill0_04`、`drill0_05`、`drill0_06`
- 健身追踪器表带尺寸：`Manual16_51`、`Manual16_52`
- `airfryer` 首次使用：`air_fryer_01`、`air_fryer_02`、`air_fryer_03`

### mixed 路由

对同时包含多个子问题且涉及不同知识来源的问题，先拆分子问题，再分别检索和生成，最终按编号合并回答，避免漏答。

## 配置说明

主要环境变量（`.env`）：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_API_KEY` | LLM API密钥 | - |
| `LLM_MODEL` | 使用的模型 | gpt-4o-mini |
| `EMBEDDING_MODEL` | Embedding模型 | text-embedding-3-small |
| `API_TOKEN` | API认证Token | sk_customer_20260304 |

## 工具脚本

| 脚本 | 作用 |
|------|------|
| `scripts/build_knowledge_base.py` | 构建文本和图片向量索引 |
| `scripts/run_api.py` | 启动 FastAPI 服务 |
| `scripts/test_api.py` | 基础 API 回归测试 |
| `scripts/test_retrieval.py` | 检查检索命中和图片 ID |
| `scripts/run_public_questions.py` | 批量调用 `/chat/` 并生成提交 CSV |
| `scripts/evaluate_outputs.py` | 检查提交答案中的低质量样本 |

## 评分标准

回答质量由LLM裁判打分（1-5分）：

| 分数 | 描述 |
|------|------|
| 1 | 质量差：未回应问题，结构混乱 |
| 2 | 一般：部分回应，但不完整 |
| 3 | 中等：回应问题，缺乏深度 |
| 4 | 良好：清晰全面，图文结合合理 |
| 5 | 优秀：详细有深度，图文完美互补 |
