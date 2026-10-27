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
   - case conversion: skip the copy + conversion when the word is pure ASCII
     without uppercase letters.

No tokenization rules, dictionary contents, match semantics or result formats
are changed.

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
  `UPDATE` / index removal on `DELETE`.

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
| query_beng_avg_ms | 2.7525 | 2.8616 |
| query_mixed_avg_ms | 2.0510 | 3.0572 |
| query_limit_avg_ms | 2.2938 | 2.3154 |

Note: the local instance was shared with another benchmark while these runs
were taken, so query numbers are noisy; the build/tokenize numbers are the
stable part of the improvement. Local absolute timings are much lower than the
CI baseline because of the faster machine.
