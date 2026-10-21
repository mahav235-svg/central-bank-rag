### Retrieval, 47 answerable questions, k=5

Generated 2026-10-17 22:38 UTC.

| variant | n | doc_recall@5 | hit@1 | hit@5 | mrr@10 | ndcg@5 |
|---|---|---|---|---|---|---|
| dense | 47 | 0.442 | 0.170 | 0.319 | 0.236 | 0.244 |
| bm25 | 47 | 0.521 | 0.340 | 0.553 | 0.440 | 0.459 |
| hybrid | 47 | 0.550 | 0.213 | 0.553 | 0.356 | 0.398 |
| hybrid+filters | 47 | 0.819 | 0.447 | 0.766 | 0.605 | 0.626 |
| hybrid+filters+rerank | 47 | 0.807 | 0.489 | 0.808 | 0.630 | 0.662 |
| hybrid+filters+bge_instruction | 47 | 0.798 | 0.425 | 0.745 | 0.596 | 0.618 |
