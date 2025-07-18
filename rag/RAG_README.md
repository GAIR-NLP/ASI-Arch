# 基于OpenSearch的RAG服务

本项目实现了一个基于OpenSearch的检索增强生成（RAG）服务，专门用于搜索和检索认知数据库中的实验触发模式。

## 功能特性

- 🔍 **语义搜索**: 基于`EXPERIMENTAL_TRIGGER_PATTERNS`字段进行向量相似度搜索
- 📄 **文档管理**: 每个认知洞察作为独立文档，包含论文来源信息
- 🚀 **REST API**: 提供完整的HTTP API接口
- 🐳 **容器化部署**: 使用Docker Compose一键部署OpenSearch
- 📊 **统计分析**: 提供索引统计和搜索分析功能

## 系统架构

```
┌─────────────────┐    HTTP API    ┌─────────────────┐
│   客户端应用    │ ──────────────► │   Flask API     │
└─────────────────┘                 │   (rag_api.py)  │
                                    └─────────────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │   RAG Service   │
                                    │  (rag_service)  │
                                    └─────────────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │   OpenSearch    │
                                    │   (向量数据库)   │
                                    └─────────────────┘
```

## 快速开始

### 1. 环境准备

确保系统已安装：
- Python 3.8+
- Docker 和 Docker Compose
- 至少4GB可用内存

### 2. 安装依赖

```bash
cd mad-lab
pip3 install -r requirements_rag.txt
```

### 3. 启动服务

#### 方法一：使用启动脚本（推荐）

```bash
chmod +x start_rag_service.sh
./start_rag_service.sh
```

#### 方法二：手动启动

```bash
# 启动OpenSearch
docker-compose up -d opensearch

# 等待OpenSearch启动完成
sleep 30

# 启动RAG API服务
python3 rag_api.py
```

### 4. 验证服务

访问 http://localhost:5000 查看API文档

```bash
# 健康检查
curl http://localhost:5000/health

# 获取统计信息
curl http://localhost:5000/stats
```

## API 接口

### 搜索相似模式

```bash
POST /search
Content-Type: application/json

{
    "query": "模型在长序列上表现不佳",
    "k": 5,
    "similarity_threshold": 0.6
}
```

### 获取论文文档

```bash
GET /paper/arxiv.org_abs_1706.03762
```

### 获取统计信息

```bash
GET /stats
```

## 使用示例

### Python客户端

```python
import requests

# 搜索相似模式
response = requests.post('http://localhost:5000/search', json={
    "query": "attention mechanism computational complexity",
    "k": 3,
    "similarity_threshold": 0.7
})

results = response.json()
for result in results['results']:
    print(f"论文: {result['paper_key']}")
    print(f"相似度: {result['score']:.3f}")
    print(f"触发模式: {result['experimental_trigger_patterns']}")
    print("-" * 50)
```

### 测试客户端

运行内置的测试客户端：

```bash
python3 test_rag_client.py
```

## 数据结构

### 输入数据格式

每个JSON文件包含多个设计洞察对象：

```json
[
    {
        "DESIGN_INSIGHT": "设计洞察标题",
        "EXPERIMENTAL_TRIGGER_PATTERNS": "实验触发模式描述...",
        "BACKGROUND": "背景信息...",
        "ALGORITHMIC_INNOVATION": "算法创新...",
        "IMPLEMENTATION_GUIDANCE": "实现指导...",
        "DESIGN_AI_INSTRUCTIONS": "AI设计指令..."
    }
]
```

### 索引数据结构

每个文档包含：
- `paper_key`: 论文标识（JSON文件名）
- `design_insight`: 设计洞察
- `experimental_trigger_patterns`: 实验触发模式（用于embedding）
- `background`: 背景信息
- `algorithmic_innovation`: 算法创新
- `implementation_guidance`: 实现指导
- `design_ai_instructions`: AI设计指令
- `embedding`: 向量嵌入（384维）

## 配置说明

### OpenSearch配置

- 主机: localhost:9200
- 索引名: cognition_rag
- 分片数: 1
- 副本数: 0
- 禁用安全插件

### 嵌入模型配置

- 模型: all-MiniLM-L6-v2
- 维度: 384
- 语言: 多语言支持

### API配置

- 端口: 5000
- 跨域: 已启用
- 最大结果数: 50

## 性能优化

### 搜索优化

- 使用KNN向量搜索
- 可调节相似度阈值
- 支持结果数量限制

### 内存优化

- OpenSearch堆内存: 512MB
- 排除嵌入向量从搜索结果
- 使用流式处理大文件

## 故障排除

### 常见问题

1. **OpenSearch启动失败**
   ```bash
   # 检查内存使用
   docker stats opensearch-rag
   
   # 重启服务
   docker-compose restart opensearch
   ```

2. **API服务连接失败**
   ```bash
   # 检查OpenSearch状态
   curl http://localhost:9200/_cluster/health
   
   # 重新初始化RAG服务
   curl -X POST http://localhost:5000/reinit
   ```

3. **搜索结果为空**
   - 检查相似度阈值是否过高
   - 确认数据已正确索引
   - 尝试使用不同的查询词

### 日志查看

```bash
# OpenSearch日志
docker logs opensearch-rag

# API服务日志
# 直接在终端查看输出日志
```

## 扩展功能

### 自定义嵌入模型

修改 `rag_service.py` 中的模型配置：

```python
rag_service = OpenSearchRAGService(
    embedding_model="your-custom-model"
)
```

### 添加新的搜索字段

扩展索引映射以包含其他字段的向量搜索。

### 多语言支持

配置适合目标语言的分析器和嵌入模型。

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 许可证

本项目采用 MIT 许可证。

## 联系方式

如有问题或建议，请创建 Issue 或联系项目维护者。 