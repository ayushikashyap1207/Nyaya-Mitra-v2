# Evaluation

Run the deterministic evaluation harness:

```bash
python eval/run_eval.py
```

It measures:

- **Date extraction** — recall, precision, extra dates, latency
- **Document classification** — accuracy on labelled documents
- **Retrieval** — Hit@1, Hit@3, MRR, latency
- **Schema validation** — valid/invalid Pydantic outputs

## Current results

Dataset: 10 synthetic documents, 23 retrieval questions, 3 schema cases.

| Metric | Result |
|---|---|
| Date recall | 100% |
| Date precision | 100% |
| Document classification | 100% |
| Retrieval Hit@1 | 78.3% |
| Retrieval Hit@3 | 100% |
| Retrieval MRR | 0.891 |
| Schema validation | 100% |

### Retrieval improvement

| Version | Hit@1 | Hit@3 | MRR | Indic queries hit |
|---|---|---|---|---|
| English-only dense retrieval | 61.1% | 77.8% | 0.694 | 0 / 4 |
| Multilingual E5 + BM25 + reranker | **77.8%** | **100%** | **0.889** | **4 / 4** |

The full 23-query benchmark (adding five BNS/BNSS/BSA questions) scores **78.3% Hit@1, 100% Hit@3, and 0.891 MRR**.

The improvement came from:

- `intfloat/multilingual-e5-small` embeddings
- BM25 keyword retrieval
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` multilingual reranking
- reciprocal-rank fusion of dense, BM25, and reranker results

## Limitations

This is a **starter benchmark**:

- Documents are synthetic, not real anonymised case files.
- 10 documents and 23 retrieval questions are too few for statistical confidence.
- Legal correctness of expected sections has not been reviewed by a lawyer.
- Answer faithfulness is not yet measured with a human rubric.

To make this production-grade:

1. Add 50–100 anonymised real documents.
2. Add expected dates, document types, and legal sections reviewed by a lawyer.
3. Add answer-faithfulness and citation-precision scoring.
4. Track results by language and document type.
