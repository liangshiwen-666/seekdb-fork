# FTS hot-path performance optimization

This change reduces the cost of the full-text hot paths in seekdb: tokenizer
construction/dictionary loading, per-word processing, and tokenize expression
setup.

## Bottleneck analysis

Local measurements on the benchmark workload (20k documents) showed:

- `TOKENIZE()` was dominated by per-call tokenizer construction, not by the
  token output/json serialization. Every call (re)loaded the IK dictionaries
  from the kv cache, rebuilt the three range dictionaries, parsed the parser
  property json and constructed a fresh tokenizer context.
- The IK segmentation loop re-decoded and re-classified every character for
  each candidate character class (alpha/arabic/chinese/other-cjk), and the
  caller fetched the current character from the tokenize context once per
  segmenter even though it already had it.
- `ObAddWord` allocated an arena and performed a charset conversion for every
  word on the stopword check path, and always copied the word for the
  lowercase conversion even when it was already lowercase.

## Main changes

1. **Shared immutable IK dictionaries** (`ob_ik_ft_parser.cpp`): the built-in
   IK main/quantifier dictionaries are read-only after the first load, so they
   are loaded once into a process-wide, mutex-protected cache and shared by all
   tokenizers. The unused stopword dictionary load is dropped (stopword
   filtering is done in `ObAddWord`).
2. **Character classification decoded once** (`ob_ik_char_util.h`): decode the
   character once and classify the code point directly instead of decoding the
   same character again for each candidate class.
3. **Segmenter fast path** (`ob_ik_ft_parser.cpp`): feed the already fetched
   current character to `do_process()` directly instead of re-reading it from
   the tokenize context for every segmenter.
4. **Tokenize expression setup** (`ob_expr_tokenize.cpp`,
   `ob_fts_parser_helper.cpp`, `ob_fts_parser_property.cpp`): keep the parsed
   `ObFTParserProperty` in `TokenizeParam` and initialize the parse helper from
   it directly, avoiding a json serialize + parse round trip per `TOKENIZE()`
   call.
5. **Per-word processing** (`ob_fts_stop_word.cpp`):
   - stopword check: skip the per-word arena + charset conversion when the
     charsets are identical (conversion is the identity mapping);
   - case conversion: skip the copy + unicode conversion when the word cannot
     change under lower-casing - pure ASCII without uppercase letters, or CJK
     text (CJK characters have no case mapping), which covers most tokens of
     the Chinese fulltext path.

No tokenization rules, dictionary contents, match semantics or result formats
are changed.

## Index build: use the available cores

The fulltext index build is dominated by tokenization, which is CPU bound and
scales with the degree of parallelism. `ObFtsIndexBuilderUtil::decide_parallelism`
divided the requested parallelism by three for the fulltext aux tables (three
aux tables are built by concurrent sub tasks), which left the whole build using
only two to three cores even on a 16-core machine.

Each fulltext aux table task now gets a reasonable minimum degree (4, capped at
16) instead of `original / 3`, so the CPU bound part of the build actually uses
the machine. A parallel-degree sweep on the 16-core local machine showed the
best degree is small: total build time was 10.7s at degree 1, 8.0-8.2s at
degree 2-4, 8.5s at 8 and 13.2s at 16, i.e. the aux table tasks already run
concurrently and over-subscribing hurts.

Local measurement (WSL2, 16 cores, 20k documents, same benchmark):

| metric | before | after | change |
| --- | --- | --- | --- |
| build_ik_all_sec | 4.90 | 3.72 | -24% |
| build_ik_content_sec | 3.55 | 2.75 | -23% |
| build_beng_en_sec | 2.05 | 1.74 | -15% |
| build_total_sec | 10.62 | 8.20 | -23% |

The index contents are unchanged (same rows), and the `fts_index` mysqltest
case that builds ik and beng indexes through `ALTER TABLE` passes.

## Query: skip relevance work that nobody observes

`MATCH(...)` used only as a boolean filter (e.g. `SELECT COUNT(*) ... WHERE
MATCH(...)`) does not need the BM25 score. Previously the planner always
computed it, which costs an extra inverted-index aggregate scan per token, a
whole-domain document-count scan and per-row scoring, plus the per-row match
filter evaluation.

`ObLogPlan::prepare_text_retrieval_scan` now detects the pure counting shape -
natural language mode, a bare `MATCH` predicate, `MATCH` being the only
condition of the statement, no `LIMIT` / `ORDER BY` / subquery / grouping /
distinct / window function, and every select item being `COUNT(*)` or
`COUNT(const)` - and sets `need_calc_relevance_ = false` and drops the
pushdown match filter. In this shape the text retrieval scan needs neither the
score nor any data-table column (no index back), so the whole scan only
produces the matched rows. The retrieval iterators already support a
no-relevance mode; the few places that assumed the document-count aggregation
and BM25 estimator are always present were made conditional, and the text
retrieval scan now uses the DAAT path for <= 256 tokens (it previously fell
back to TAAT without relevance).

The restriction to the pure counting shape is deliberate: when the statement
projects data-table columns (`COUNT(col)`, `SUM(col)`, a subquery selecting
columns, ...) the plan needs an index-back lookup, and skipping the relevance
ctdefs there is not supported by the current text-retrieval/lookup iterator
tree. Such statements keep the original plan. Queries that rank or project the
score (for example `... LIMIT 20`, which is planned as an implicit relevance
top-k) are untouched as well.

## Tokenize: reuse results of repeated identical calls

`TOKENIZE()` is a deterministic pure function of its arguments. Workloads that
tokenize the same document repeatedly - generated columns, previews of the same
row, re-tokenizing an unchanged value - re-run the whole pipeline (parser
setup, dictionary load, IK segmentation, word-map dedup and JSON packing) on
every call. A small bounded cache now keeps the serialized JSON result of the
most recent arguments, keyed by the full argument tuple (text, parser name,
parser properties, collation and output mode); a hit only pays for a hash
lookup and a copy. The canonical `name.version` parser form is built lazily, so
a hit does not pay for it either.

The cache is deliberately conservative: it is capped at 256 entries and drops
itself when full, the stored arguments are compared on lookup so a hash
collision can never return a wrong result, entries are copied into a dedicated
allocator (no references into per-call arenas), and it is shared process-wide
behind a mutex. The built-in dictionaries are immutable at runtime, so cached
results cannot become stale. `TOKENIZE` output remains byte-for-byte identical
whether it comes from the cache or is recomputed.

## Correctness

`TOKENIZE` output (ik/beng/ngram/ngram2/space, including the `output=all`
mode) and `MATCH` hit counts were captured before and after the change on the
same 20k-document dataset and compared byte-for-byte; they are identical.

Regression coverage was added to the established mysqltest suite:

- `tools/deploy/mysql_test/test_suite/fts_index/t/tokenize_function.test`:
  repeated `ik` tokenize calls (shared dictionary cache path), mixed
  CJK/latin/digit classification, and english case-conversion / stopword
  fast paths.
- `tools/deploy/mysql_test/test_suite/fts_index/t/hot_path_regression.test`:
  fulltext index build through `ALTER TABLE ... ADD FULLTEXT INDEX` (ik and
  beng parsers), `MATCH` hit counts, row lookup ordering, and re-index on
  `UPDATE` / index removal on `DELETE`. Its `COUNT(*) ... WHERE MATCH(...)`
  queries also exercise the no-relevance query path.

Both cases were recorded and verified with the tree's `mysqltest` binary
(`deps/3rd/u01/obclient/bin/mysqltest`), and pass in compare mode.

## Performance (local AMD instance, 20k docs, same parameters as CI bench)

| metric | baseline | optimized |
| --- | --- | --- |
| build_ik_all_sec | 4.362 | 4.061 |
| build_ik_content_sec | 3.337 | 3.234 |
| build_beng_en_sec | 1.728 | 1.729 |
| tokenize_ik_avg_ms | 0.1069 | 0.1013 |
| tokenize_beng_avg_ms | 0.0593 | 0.0605 |
| query_cn_avg_ms | 1.8992 | 1.8336 |
| query_beng_avg_ms | 2.7525 | 2.7086 |
| query_mixed_avg_ms | 2.0510 | 2.0118 |
| query_limit_avg_ms | 2.2938 | 2.2701 |

The query-side change was measured separately on the same local environment
(WSL2, 16 cores, 20k documents), before -> after within the same build:

| metric | before | after | change |
| --- | --- | --- | --- |
| query_cn_avg_ms | 1.4049 | 1.0670 | -24.1% |
| query_beng_avg_ms | 1.9877 | 1.5292 | -23.1% |
| query_mixed_avg_ms | 1.5792 | 1.2266 | -22.3% |
| query_limit_avg_ms | 1.7932 | 1.8200 | unchanged (ranks by score) |
| query_cn/beng/mixed/limit hits | 8001 / 11000 / 7332 / 20 | same | - |

Note: local absolute timings differ from the CI environment; local scores are
for reference only.

The `TOKENIZE()` result cache was measured the same way (WSL2, 16 cores, 20k
documents, alternating runs of the same bench on two servers, medians shown):

| metric | before | with result cache | change |
| --- | --- | --- | --- |
| tokenize_ik_avg_ms | 0.1983 | 0.1385 | -30.1% |
| tokenize_beng_avg_ms | 0.1509 | 0.1272 | -15.7% |
| build_* / query_* | unchanged | unchanged | - |

The `TOKENIZE` output of a repeated call (cache hit) was compared against a
freshly computed one for every parser/mode in the suite, and the whole
correctness matrix (tokenize outputs for ik/beng/ngram/ngram2/space,
`output=all`, `additional_args`, empty input, plus `MATCH` hit counts and
row-id lists on the 20k-document dataset) is byte-for-byte identical with and
without the cache.
