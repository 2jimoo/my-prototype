from model import DenseEncoder
import json
from my_data import read_query_dataset, dummy_generator
import torch
from collections import defaultdict
from functions import cosine_search, term_search, term_regl_search
from evaluate import evaluate_dataset

encoder = DenseEncoder()


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_result_file(rank_file_path, result):
    with open(rank_file_path, "w") as f:
        for key, values in result.items():
            line = f"{key} " + " ".join(map(str, values)) + "\n"
            f.write(line)


docs = read_jsonl("/Users/user/my-prototype/data/dummy_documents.jsonl")
doc_ids, doc_mean_embs, doc_token_embses = [], [], []
for doc in docs:
    doc_ids.append(doc["doc_id"])
    mean_emb, token_embs = encoder.encode(doc["text"])
    doc_mean_embs.append(mean_emb)
    doc_token_embses.append(token_embs)

test_queries = read_query_dataset()
qlen = len(test_queries)
q_ids, q_mean_embs, q_token_embses = [], [], []
for query in test_queries:
    q_ids.append(query["qid"])
    mean_emb, token_embs = encoder.encode(query["query"])
    q_mean_embs.append(mean_emb)
    q_token_embses.append(token_embs)


def generate_cosine_search_performance(k):
    print("====================cosine_search====================")
    rank_file_path = "./data/result/cosine_search.txt"
    result = defaultdict(list)
    for i in range(qlen):
        indices = cosine_search(query=q_mean_embs[i], k=k, documents=doc_mean_embs)
        found_doc_ids = [doc_ids[idx] for idx in indices]
        result[q_ids[i]] = found_doc_ids
    write_result_file(rank_file_path, result)
    evaluate_dataset(k, rankings_path=rank_file_path)


def generate_term_search_performance(k):
    print("====================term_search====================")
    rank_file_path = "./data/result/term_search.txt"
    result = defaultdict(list)
    for i in range(qlen):
        indices = term_search(query=q_token_embses[i], k=k, documents=doc_token_embses)
        found_doc_ids = [doc_ids[idx] for idx in indices]
        result[q_ids[i]] = found_doc_ids
    write_result_file(rank_file_path, result)
    evaluate_dataset(k, rankings_path=rank_file_path)


def generate_term_regl_search_performance(k):
    print("====================term_regl_search====================")
    rank_file_path = "./data/result/term_regl_search.txt"
    result = defaultdict(list)
    for i in range(qlen):
        indices = term_regl_search(
            query=q_token_embses[i], k=k, documents=doc_token_embses
        )
        found_doc_ids = [doc_ids[idx] for idx in indices]
        result[q_ids[i]] = found_doc_ids
    write_result_file(rank_file_path, result)
    evaluate_dataset(k, rankings_path=rank_file_path)


if __name__ == "__main__":
    k = 5
    generate_cosine_search_performance(k)
    generate_term_search_performance(k)
    generate_term_regl_search_performance(k)
    # dummy_generator()
