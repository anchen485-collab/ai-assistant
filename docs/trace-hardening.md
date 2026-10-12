# RAG Trace 加固说明

## 范围

只增强 Trace 的稳定性、诊断数据边界及访问控制。检索排序、知识治理和回答策略由对应模块维护。
核心实现是 Python 标准库；未增加运行时依赖。接口字段 trace_id / total_ms / steps 保持兼容，新增 dropped_steps 表示容量限制丢弃的步骤数。

## 稳定性约束

- 使用单调时钟计算耗时，结束后 total_ms 固定；重复结束不改变耗时。
- 缓存读写、淘汰、列表查询在同一锁内执行；返回 JSON 快照，调用方修改数据不会污染缓存。
- ContextVar 按请求隔离；StreamingResponse 消费时显式绑定，finally 收尾。预处理失败、流式异常、任务取消和生成器关闭均有记录。
- source_refs 容忍坏 chunk/分数/对象属性，过滤 NaN 和 Infinity；不保存知识块正文。
- 日志后端异常与额外字段序列化异常不向聊天链路传播。Trace INFO 日志不复制 extra 详情；其他模块日志有各自策略。
- 超限记录主动丢弃，不回压聊天；因此这是一份尽力保留的诊断缓存，不是完整审计日志。

默认限制（集中在 src/trace.py 顶部）：

| 对象 | 限制 |
|---|---|
| Trace 数量 | 500 条 |
| TTL | 创建后 1 小时，在后续读写时清理 |
| 每条 Trace 步骤 | 128 |
| 每步序列化载荷 | 16 KiB，超出时保留基本字段并标记 payload_truncated |
| 每条 Trace 载荷 | 256 KiB |
| 全部 Trace 载荷 | 8 MiB，超出时淘汰最旧记录 |
| 单次来源列表 | 前 64 项 |
| 字符串 / 容器 / 嵌套 | 2048 字符 / 64 项 / 深度限制与节点预算 |

载荷上限是序列化 UTF-8 字节数，不是进程 RSS；Python 对象、字符串及运行时另有开销。来源 source 最长 512 字符。

## 查询安全（使用方式有变化）

Trace 中有用户问题，查询接口仅供运维使用，不适合直接暴露给普通用户。

1. 在服务进程环境或现有 .env 中配置 TRACE_API_TOKEN，使用密码学随机值，至少 32 字符。勿提交真实令牌。
2. 未配置时两条查询路由返回 404；配置过短返回 503。
3. 请求头使用 Authorization: Bearer <运维令牌>。缺失/错误返回 401。
4. GET /api/traces?limit=50 支持 1～100，越界返回 422；单条 GET /api/traces/{trace_id} 不存在时返回 404。
5. 成功响应 Cache-Control: no-store；生产部署仍需 HTTPS、内网/网关限制及网关限流。不要把运维令牌放进前端代码或 URL。

显式 token/password/api_key 等字段会脱敏，但自由文本中的个人信息和密钥没有做语义识别。受保护查询仍应按敏感数据处理。
聊天 SSE、模型选择及检索策略未因接口访问校验而改变。

## 字段解释

- preprocess.question：用户原始问题。
- rag:search.original_query：RAG 工具收到的查询，可能已由 Agent 改写，并不必然等于用户原始问题。
- rag:search.query：工具内部 rewrite 后真正用于检索的文本。
- rag:search.query_rewritten：布尔值，表示工具内部是否改写，不是改写后的文本。
- rag:recall：各阶段候选来源定位信息。
- rag:quality_gate：门控结果及 selected；字段由检索模块上报。
- agent:stream.source_refs：现有消息提取逻辑返回的来源集合，不代表已经验证了每一句最终答案的引用。

## 验证

在 agent 目录、已有项目 Python 环境中执行：

```powershell
python -m unittest discover -s tests -p test_trace_hardening.py -v
```

测试依赖现有 FastAPI 与测试客户端 httpx（本次隔离环境验证使用 FastAPI 0.139.0）。使用真实 FastAPI TestClient 测试选定路由和 SSE 传输；app.py 的目标函数原样 AST 编译，隔离其他模块的启动导入。RAG 工具逻辑也原样编译并替换外部检索调用。核心 Trace 模块直接导入真实代码。

这批测试是单元/组件测试，不等于完整 app 导入、真实模型/向量库集成、服务器断网测试或生产压测。

## 部署边界

- 当前仅承诺单进程诊断缓存。多 worker/多实例各自持有独立缓存，查询可能落到另一进程；服务重启会清空记录。
- 本轮没有增加 Redis、数据库或集中式链路平台。需要跨实例和持久化时应另做方案。
- 正式上线前，使用真实配置补测模型/向量库、前端停止生成、网关鉴权与限流，并验证部署 worker 数量。
- 未配置令牌只关闭查询，不关闭记录，也不影响聊天。

## 并行修改与回滚

本次只写回 app.py、src/trace.py、tests/test_trace_hardening.py 和本说明。写回前比对基线 SHA256；其他 Agent 修改的检索/治理/同步文件不覆盖。
交付包包含 before/after 哈希、精确补丁、验证记录和 rollback.py。回滚先检查四个文件是否仍等于本次产物；出现后续修改则停止，不覆盖其他工作。先使用 --check 预检；支持 --root 指定复制出的验证目录。
