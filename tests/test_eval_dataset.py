import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_eval_dataset_is_consistent():
    with open(ROOT / "eval" / "eval_dataset.json", encoding="utf-8") as handle:
        dataset = json.load(handle)
    with open(ROOT / "legal_corpus.json", encoding="utf-8") as handle:
        corpus_ids = {entry["id"] for entry in json.load(handle)}

    assert dataset["documents"]
    assert dataset["retrieval_questions"]

    for question in dataset["retrieval_questions"]:
        missing = set(question["expected_ids"]) - corpus_ids
        assert not missing, f"{question['id']} references missing corpus IDs: {missing}"

    for document in dataset["documents"]:
        assert document["expected_dates"], f"{document['id']} has no expected dates"
