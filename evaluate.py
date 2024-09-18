from collections import defaultdict
import json
from typing import List
from my_data import read_query_dataset
import torch

torch.autograd.set_detect_anomaly(True)
"""
MRR (Mean Reciprocal Rank)
- 각 쿼리마다 정답 문서가 처음 등장하는 순위를 계산하고, 그 역수를 구해 누적합니다. 
- 전체 쿼리에 대한 평균을 최종 MRR로 계산합니다.

Forget
- 이전에 성공했던 정답이 이번에는 성공하지 못한 경우를 측정합니다. 
- 이전과 이번의 차이를 비율로 계산해 누적한 뒤 평균을 내어 최종 Forget 값을 구합니다.

FWT (Forward Transfer)
- 새로운 학습이 이전 학습에 긍정적인 영향을 주는지 평가합니다. 
- 이번 학습에서 성공한 정답이 이전 학습에서 실패한 경우, FWT로 측정합니다. 
- 이전과 이번의 차이를 비율로 계산해 누적한 뒤 평균을 내어 최종 FWT 값을 구합니다.
"""


def evaluate_dataset(k, rankings_path, previous_rankings_path=None):
    eval_queries = read_query_dataset()

    rankings = defaultdict(list)
    with open(rankings_path, "r") as f:
        for line in f:
            items = line.strip().split()
            qid: int = int(items[0])
            pids: List[int] = list(map(int, items[1:]))
            rankings[qid].extend(pids)

    # previous_rankings = defaultdict(list)
    # if previous_rankings_path:
    #     with open(previous_rankings_path, "r") as f:
    #         for line in f:
    #             items = line.strip().split()
    #             qid: int = int(items[0])
    #             pids: List[int] = list(map(int, items[1:]))
    #             rankings[qid].extend(pids)

    success = 0
    num_q = 0
    recall = 0.0
    mrr = 0.0
    forget = 0.0
    fwt = 0.0

    for query in eval_queries:
        num_q += 1
        qid = query["qid"]
        answer_pids = query["answer_pids"]
        hit = set(rankings[qid][:k]).intersection(answer_pids)

        if len(hit) > 0:
            success += 1
            recall += len(hit) / len(answer_pids)

        # for i, pid in enumerate(rankings[qid][:k]):
        #     if pid in answer_pids:
        #         mrr += 1 / (i + 1)
        #         break

        # if previous_rankings_path:
        #     previous_hit = set(previous_rankings[qid][:k]).intersection(answer_pids)
        #     if previous_hit:
        #         forget += (len(previous_hit) - len(hit)) / len(answer_pids)
        #         fwt += (len(hit) - len(previous_hit)) / len(answer_pids)

    num_rankings = len(rankings)

    print(
        f"# query:  {num_q}\n",
        f"Avg Success@{k}: {success / num_q * 100:.1f}\n",
        f"Acg Recall@{k}: {recall / num_q * 100:.1f}\n",
        # f"MRR@{k}: {mrr / num_rankings:.4f}\n",
    )

    if previous_rankings_path:
        print(
            f"Forget: {forget / num_rankings * 100:.1f}\n",
            f"FWT: {fwt / num_rankings * 100:.1f}\n",
        )
