# domain_a = "technology"
# domain_b = "science"
# data = defaultdict(list)
# queries_data = defaultdict(list)

# for domain in [domain_a, domain_b]:
#     doc_file_path = f"./raw/{domain}_dev_collection.jsonl"
#     data[domain] = read_jsonl(doc_file_path)
#     query_file_path = f"./raw/{domain}_dev_qas.search.jsonl"
#     queries_data[domain] = read_jsonl(query_file_path)

# D0_docs = []
# D0_queries = []
# OFFSET = 0
# sampled_doc_ids =set()
# for domain in [domain_a, domain_b]:
#     sampled_queries = queries_data[domain][:25]

#     for query in sampled_queries:
#         query["qid"] = OFFSET + query["qid"]
#         query["question_author"] = ""
#         answer_pids = set(query["answer_pids"])
#         matching_docs = [
#             doc for doc in data[domain] if doc["doc_id"] in answer_pids
#         ]
#         for idx, answer_pid in enumerate(query["answer_pids"]):
#             query["answer_pids"][idx] = OFFSET + answer_pid
#         filtered_docs=[]
#         for doc in matching_docs:
#             doc["doc_id"] = OFFSET + doc["doc_id"]
#             doc["author"] = ""
#             if not doc["doc_id"] in sampled_doc_ids:
#                 filtered_docs.append(doc)
#             sampled_doc_ids.add(doc["doc_id"])
#         D0_docs.extend(filtered_docs)
#     D0_queries.extend(sampled_queries)
#     OFFSET += 10000000
#     print(f"D0_queries:{len(D0_queries)}, D0_docs:{len(D0_docs)}")

# random.shuffle(D0_docs)
# save_jsonl(D0_docs, f"../my-prototype/data/dummy_documents.jsonl")
# save_jsonl(D0_queries, f"../my-prototype/data/dummy_queries.jsonl")
