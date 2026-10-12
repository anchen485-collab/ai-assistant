# 混合检索加固说明

## 实现范围

`search.py` 的召回入口/RRF、`bm25.py`，新增 `keyword_cache.py`；未引入运行时依赖。
主目录同时合并了其他 Agent 的治理、Small-to-Big、同步锁、质量门控和 Trace。
本改动使用它们已有的接口，不替换它们的实现。

## 请求流程

1. 校验问题和 k；空问题及 k<=0 返回空列表，非法类型抛明确异常。
2. `knowledge_read_context()` 同步知识库并持有与写入共享的锁。
3. 两路共用同一个 store、`get_index_revision(store)` 和北京时间业务日期。
4. 分页读取有效文档，建立可复用的 BM25 倒排索引。
5. MMR 在有效 document_id 白名单中检索；BM25 只对有效文档计算词频。
6. 每条路线先去除重复父块，再用 RRF 融合；不同来源的同文内容保持独立。
7. 释放知识库锁后才请求 Rerank，继续使用现有门控/Trace。

一路召回失败时使用另一路；两路均失败时抛错，交给现有工具层处理。
知识库同步或完整性检查失败直接抛错，不伪装成“没有匹配内容”。

## 缓存与并发边界

- 每进程最多保留一个索引快照；查询只读，局部分数互不影响。
- 缓存键包括存储路径、collection UUID、提交版本、业务日期、文档数。
- 等量内容/治理更新通过 revision 立即失效；跨午夜通过日期失效；60秒 TTL 兜底。
- 冷启动每页500条，列长度/总数/重复 ID 校验通过后才发布完整快照。
- 知识库更新必须走项目同步入口；绕过同步锁、直接修改 Chroma 不在一致性契约内。
- 有效 document_id 白名单要求同一个已提交快照内，一个 document_id 对应同一版本/治理状态。
- 目前是单机中小型知识库方案。索引内存随语料增加，冷启动需扫描全库；知识库读锁会串行化召回。
  高并发/大语料上线前需用真实数据测内存、冷启动时间及 P95；不宣称已做生产容量认证。

中文仍是轻量双字切分，不是完整中文分词器；英文数字统一大小写和全半角。

## 验证

在 agent 目录、已有虚拟环境运行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_bm25.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_keyword_cache.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_hybrid*.py' -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

覆盖：原有中英文样例、公式对照、边界输入、并发查询、分页对齐、版本/日期失效、
单路故障、双路故障、父块去重、过期文档前置过滤、真实临时 Chroma 联调。
模型网络调用使用桩或本地 embedding，不消耗线上 API；这不等于已验证真实模型质量。

本次只修改本地文件，不提交或推送。备份、仅本次差异、字面测试输出和选择性回滚脚本
保存在本任务的 `.verification/hybrid-hardening-20260927` 中。
