# Evaluation

Two suites, kept separate because one is free and deterministic and the other costs money.

## Golden set

`eval/golden.jsonl` has 52 questions written from the documents themselves:

| Category | Questions | Example |
|---|---|---|
| Fed statements | 10 | Who dissented at the March 2022 FOMC meeting? |
| RBI resolutions | 9 | What did the RBI decide at the off-cycle meeting in May 2022? |
| Minutes (both banks) | 10 | Which MPC members voted against the February 2023 hike? |
| Press conferences | 11 | What did Powell call the June 2021 meeting? |
| Governor's statements | 4 | What UPI limit change was announced in August 2024? |
| Cross-bank comparisons | 3 | How did both banks move rates in June 2022? |
| Unanswerable | 5 | What did the ECB decide in March 2024? |

Each answerable question lists:

* `doc_ids`: the documents that contain the answer (several where the same fact is repeated in a
  resolution, the minutes and the Governor's statement),
* `evidence`: short strings copied from those documents that a relevant passage must contain,
* `must_include`: facts a correct answer must state.

`tests/integration/test_golden.py` checks every evidence string against the built catalog, so a
parser or chunking change that breaks the ground truth fails the test suite instead of quietly
lowering scores.

Labelling relevance by evidence string rather than by chunk id is deliberate: chunk ids change
whenever chunking changes, and a stale label set would make every chunking experiment look worse
than it is.

## Retrieval suite (`cbrag eval retrieval`)

No LLM calls. For each variant, every answerable question is run and scored on:

* **hit@1, hit@5**: is any relevant passage at rank 1 / in the top 5;
* **MRR@10**: mean reciprocal rank of the first relevant passage;
* **nDCG@5**: rank-weighted relevance with binary gains;
* **doc_recall@5**: share of the gold documents represented in the top 5 (matters for comparisons).

Variants (each adds one component, so the table reads as an ablation):

| Variant | Dense | BM25 | Fusion | Metadata filters | Cross-encoder rerank |
|---|---|---|---|---|---|
| dense | yes | | | | |
| bm25 | | yes | | | |
| hybrid | yes | yes | RRF | | |
| hybrid+filters | yes | yes | RRF | yes | |
| hybrid+filters+rerank | yes | yes | RRF | yes | yes (scores header + text) |
| hybrid+filters+bge_instruction | yes, with BGE query prefix | yes | RRF | yes | |

Results are written to `docs/results/retrieval.md` and to the `eval_results` table in the
catalog, so runs can be compared in SQL:

```sql
SELECT label, metric, mean FROM eval_summary WHERE kind = 'retrieval' ORDER BY metric, mean DESC;
```

## Retrieval results (final run, 18 Oct 2026)

| Variant | hit@1 | hit@5 | MRR@10 | nDCG@5 | doc recall@5 |
|---|---|---|---|---|---|
| dense only (bge-small) | 0.17 | 0.32 | 0.24 | 0.24 | 0.44 |
| BM25 only | 0.34 | 0.55 | 0.44 | 0.46 | 0.52 |
| hybrid (RRF) | 0.21 | 0.55 | 0.36 | 0.40 | 0.55 |
| hybrid + metadata filters | 0.45 | 0.77 | 0.61 | 0.63 | 0.82 |
| **hybrid + filters + cross-encoder rerank (default)** | **0.49** | **0.81** | **0.63** | **0.66** | **0.81** |
| hybrid + filters, BGE query instruction | 0.43 | 0.74 | 0.60 | 0.62 | 0.80 |

Two findings. Metadata filters (bank, document type and dates read from the question) are the biggest
single gain: statements from different meetings are near-identical, so dates have to be enforced
rather than left to similarity. And the cross-encoder initially made results worse (hit@5 0.72 to
0.62) because it scored passage text alone and could not tell one meeting's boilerplate from
another's; scoring the provenance header (bank, type, date) with the text turned it into a small
gain. With 47 questions, differences under about 10 points are within noise.

## Answer suite (`cbrag eval answer`)

Runs the whole agent with the configured provider and scores:

* **answered**: share of answerable questions that got an answer (not withheld for review);
* **correct**: share whose answer contains every `must_include` fact (strict string match after
  normalisation; a paraphrase of "4-1/4 to 4-1/2 percent" as "4.25% to 4.5%" counts as wrong,
  which understates accuracy but never overstates it);
* **citation_precision**: share of cited documents that are gold documents;
* **abstained**: share of unanswerable questions where the agent declined;
* **tokens**: total tokens per question, a cost proxy.

Run it on purpose and record the provider and model in the label:

```bash
CBRAG_PROVIDER=anthropic cbrag eval answer --label claude-opus-5-5
CBRAG_PROVIDER=openai    cbrag eval answer --label gpt-5
```

## Known weaknesses of this evaluation

* 52 questions give wide confidence intervals; a difference of a few points between variants is
  noise. The retrieval table is useful for large differences (dense vs hybrid), not small ones.
* Questions were written by one person who knows the corpus, which favours phrasing close to the
  documents. Real users phrase questions less precisely.
* The groundedness judge uses the same model as the generator, which can share its blind spots.
