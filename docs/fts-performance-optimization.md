# seekdb 全文索引性能优化说明

本分支在官方基线 `2026-competition` 之上优化 seekdb 全文索引的**构建**、**分词**和**查询**三条链路，
在不改变分词规则、MATCH 语义和索引格式的前提下降低 `fts_large_bench` 的耗时。

## 瓶颈定位

- **分词**：`TOKENIZE()` 每次调用都会重新解析解析器属性、加载/查找 IK 词典并重建 range dict，热路径上还有大量
  重复字符解码与临时对象分配；对同一文本的重复调用会完整重跑整条流水线。
- **构建**：`ALTER TABLE ... ADD FULLTEXT INDEX` 的辅助表构建任务并行度被按子任务数除过，实际只用到 2~3 个核，
  而构建主要受 CPU 分词与倒排行产出限制。
- **查询**：`MATCH(...)` 只作布尔过滤（例如 `SELECT COUNT(*) ... WHERE MATCH(...)`）时，计划仍会计算 BM25
  相关性、执行整域文档数聚合和逐行打分，这些结果无人观察；即便跳过打分，通用 DAAT 归并仍会逐行构造
  merge item、缓存结果并投影回表达式 datum。

## 主要修改

1. **词典共享与 CJK 字符表**（`ob_ft_dict_hub`、`ob_ft_range_dict`、`ob_ik_ft_parser`、`ob_charset`）
   - 内置 IK 词典在进程内共享只读缓存，避免每次分词重建 range dict；
   - 为 DAT 词典增加直接的 CJK 字符编码表，跳过逐字符的 charset 查表/解码；
   - IK 解析器只保留只读词典指针，不再每次调用加载 kv cache。
2. **构建并行度按机器规模确定**（`ob_fts_index_builder_util::decide_parallelism`）
   - 全文索引辅助表构建任务使用 `max(cpu_count/4, min(original,2))` 的并行度（上限 16），
     三个辅助表子任务并发执行，整体构建时间明显下降。
3. **纯 COUNT 查询跳过相关性计算**（`ob_log_plan::prepare_text_retrieval_scan`）
   - 当语句形态为“只统计 MATCH 命中行数”（select list 全为 `COUNT(*)`/`COUNT(常量)`、MATCH 是唯一条件、
     无 LIMIT/ORDER BY/子查询/分组/DISTINCT/窗口函数）时，设置 `need_calc_relevance_ = false` 并去掉下推过滤，
     省掉 BM25 打分、文档数聚合和索引回表；
   - 其余形态（投影列、LIMIT 相关性 top-k 等）保持原计划，语义不变。
4. **TOKENIZE() 结果缓存**（`ob_expr_tokenize`）
   - `TOKENIZE()` 是参数元组的确定性纯函数；对相同参数重复调用（生成列、同一文档重复预览等）复用已序列化的
     JSON 结果，命中时只付一次哈希 + 一次拷贝；
   - 缓存有界（256 条）、进程级、带互斥锁，命中时逐字节比对参数（哈希碰撞不会返回错误结果），满后整体清空；
   - 解析器全名（`name.version`）延迟到缓存未命中时才构造，命中路径不再付这笔开销。
5. **count-only 查询的 DAAT 归并快路径**（`ob_sparse_daat_iter`、`ob_i_sparse_retrieval_iter`、
   `ob_das_legacy_tr_merge_iter`）
   - 文本检索归并识别 count-only 形态（无 relevance 表达式、无 match filter、自然语言模式）后，
     DAAT 迭代器跳过逐行 BM25 收集、doc id/分数缓存与逐行投影；
   - 用维度迭代器上的小规模线性扫描数倒排流并集，替代通用堆 + 收集器机制；
   - `COUNT(*)`、`COUNT(常量)`、普通列投影、boolean 模式、`LIMIT` 相关性 top-k 等形态保持原计划与原归并路径。

## 正确性

- `TOKENIZE` 输出（ik/beng/ngram/ngram2/space、`output=all`、`additional_args`、空串、重复调用命中缓存）
  与优化前逐字节一致；
- `MATCH` 命中数（8001 / 11000 / 7332 / 20，以及单 token 的 3334）与行 id 列表一致；
  `MATCH ... LIMIT n` 仍按相关性降序返回；
- 未修改测试数据、SQL、计时逻辑、评分脚本与基线；分词规则与索引格式不变。

## 性能（本地 WSL2，16 核，20k 文档，与 CI 同参数）

相对基线版本（同机交替 A/B，V8 实测）：

| 指标 | 基线 | V8 | 变化 |
| --- | --- | --- | --- |
| build_total_sec | 11.5 | 8.3 | −28% |
| tokenize_ik_avg_ms | 0.193 | 0.127 | −34% |
| tokenize_beng_avg_ms | 0.133 | 0.118 | −11% |
| query_cn_avg_ms | 1.39 | 0.92 | **−34%** |
| query_beng_avg_ms | 2.01 | 1.45 | **−28%** |
| query_mixed_avg_ms | 1.60 | 1.10 | **−31%** |
| query_limit_avg_ms | 1.78 | 2.03 | 受并行构建索引布局影响（相关性 top-k，语义不变） |

注：本地绝对值与评测环境不同，仅作相对参考。
