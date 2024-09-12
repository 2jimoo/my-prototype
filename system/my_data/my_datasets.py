import json
from datasets import Dataset


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def read_doc_dataset(file_path="/Users/user/my-prototype/data/dummy_documents.jsonl"):
    data = read_jsonl(file_path)
    dataset_dict = {
        "doc_id": [item["doc_id"] for item in data],
        "text": [item["text"] for item in data],
    }
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def read_query_dataset(file_path="/Users/user/my-prototype/data/dummy_queries.jsonl"):
    data = read_jsonl(file_path)
    dataset_dict = {
        "qid": [item["qid"] for item in data],
        "text": [item["text"] for item in data],
    }
    dataset = Dataset.from_dict(dataset_dict)
    return dataset
