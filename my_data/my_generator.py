import json
import os
import random
from collections import defaultdict


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def sample_data(data, percentage):
    sample_size = int(len(data) * percentage)
    return random.sample(data, sample_size)


def save_jsonl(data, file_name):
    with open(file_name, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def generate():
    common_domains = ["technology", "writing"]
    emerging_domains = ["lifestyle", "recreation", "science"]
    domains = common_domains + emerging_domains

    data = defaultdict(list)
    queries_data = defaultdict(list)

    for domain in domains:
        # 문서 데이터 로드
        doc_file_path = f"/Users/user/dais/raw/{domain}_dev_collection.jsonl"
        if os.path.exists(doc_file_path):
            data[domain] = read_jsonl(doc_file_path)

        # 도메인별 쿼리 데이터 로드
        query_file_path = f"/User/user/dais/raw/{domain}_dev_qas.search.jsonl"
        if os.path.exists(query_file_path):
            queries_data[domain] = read_jsonl(query_file_path)

    # 세션 생성
    sessions = {}

    # D0 세션 생성 - 공통 도메인에서 70%, 급성장 도메인에서 40% 샘플링
    D0_docs = []
    D0_queries = []

    for domain in common_domains:
        sampled_docs = sample_data(data[domain], 0.7)
        D0_docs += sampled_docs
        # D0에 해당하는 쿼리들 추가
        D0_queries += [
            query
            for query in queries_data[domain]
            if any(doc["doc_id"] in query["answer_pids"] for doc in sampled_docs)
        ]

    for domain in emerging_domains:
        sampled_docs = sample_data(data[domain], 0.4)
        D0_docs += sampled_docs
        D0_queries += [
            query
            for query in queries_data[domain]
            if any(doc["doc_id"] in query["answer_pids"] for doc in sampled_docs)
        ]

    sessions["D0"] = {"documents": D0_docs, "queries": D0_queries}

    # D1, D2, D3 세션 생성
    for i in range(1, 4):
        session_key = f"D{i}"
        session_docs = []
        session_queries = []

        # 급성장 도메인 지정
        emerging_domain = emerging_domains[i - 1]

        # 급성장 도메인 50% 샘플링
        sampled_docs = sample_data(data[emerging_domain], 0.5)
        session_docs += sampled_docs
        session_queries += [
            query
            for query in queries_data[emerging_domain]
            if any(doc["doc_id"] in query["answer_pids"] for doc in sampled_docs)
        ]

        # 공통 도메인에서 10% 샘플링
        for domain in common_domains:
            sampled_docs = sample_data(data[domain], 0.1)
            session_docs += sampled_docs
            session_queries += [
                query
                for query in queries_data[domain]
                if any(doc["doc_id"] in query["answer_pids"] for doc in sampled_docs)
            ]

        # 급성장 도메인을 제외한 나머지 도메인에서 5% 샘플링
        for domain in emerging_domains:
            if domain != emerging_domain:
                sampled_docs = sample_data(data[domain], 0.05)
                session_docs += sampled_docs
                session_queries += [
                    query
                    for query in queries_data[domain]
                    if any(
                        doc["doc_id"] in query["answer_pids"] for doc in sampled_docs
                    )
                ]

        sessions[session_key] = {"documents": session_docs, "queries": session_queries}

    for session_key, session_data in sessions.items():
        save_jsonl(
            session_data["documents"],
            f"/Users/user/my-prototype/data/{session_key}_documents.jsonl",
        )
        save_jsonl(
            session_data["queries"],
            f"/Users/user/my-prototype/data/{session_key}_queries.jsonl",
        )


def dummy_generator():
    domain_a = "technology"
    domain_b = "lifestyle"
    data = defaultdict(list)
    queries_data = defaultdict(list)

    for domain in [domain_a, domain_b]:
        doc_file_path = f"./raw/{domain}_dev_collection.jsonl"
        data[domain] = read_jsonl(doc_file_path)
        query_file_path = f"./raw/{domain}_dev_qas.search.jsonl"
        queries_data[domain] = read_jsonl(query_file_path)

    D0_docs = []
    D0_queries = []
    OFFSET = 0
    sampled_doc_ids = set()
    for domain in [domain_a, domain_b]:
        sampled_queries = queries_data[domain][:25]

        for query in sampled_queries:
            query["qid"] = OFFSET + query["qid"]
            answer_pids = set(query["answer_pids"])
            matching_docs = [
                doc for doc in data[domain] if doc["doc_id"] in answer_pids
            ]
            for idx, answer_pid in enumerate(query["answer_pids"]):
                query["answer_pids"][idx] = OFFSET + answer_pid
            filtered_docs = []
            for doc in matching_docs:
                doc["doc_id"] = OFFSET + doc["doc_id"]
                if not doc["doc_id"] in sampled_doc_ids:
                    filtered_docs.append(doc)
                sampled_doc_ids.add(doc["doc_id"])
            D0_docs.extend(filtered_docs)
        D0_queries.extend(sampled_queries)
        OFFSET += 10000000
        print(f"D0_queries:{len(D0_queries)}, D0_docs:{len(D0_docs)}")

    save_jsonl(D0_docs, f"../my-prototype/data/dummy_documents.jsonl")
    save_jsonl(D0_queries, f"../my-prototype/data/dummy_queries.jsonl")
