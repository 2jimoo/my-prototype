from collections import defaultdict
from .data_structure import (
    ClusterInstance,
    ActiveClusterFeatureVector,
    DeactiveClusterFeatureVector,
)
from config import Strategy
from typing import List
from utils import print_dicts, print_dict
import torch
from torch import Tensor

torch.autograd.set_detect_anomaly(True)


class ClusterManager:
    def __init__(self, strategy: Strategy, init_cluster_num):
        self.time_step: int = 0
        self.init_centroid_num: int = init_cluster_num
        self.centroid_id: int = 0
        self.assignment_table = defaultdict(list)  # 클러스터 id: 할당된 인스턴스 번호
        self.instance_memory = {}  # 인스턴스 id : 인스턴스 데이터
        self.centroid_memory = {}  # 클러스터 id: 클러스터 데이터
        self.active_threshold = 0.01
        self.strategy = strategy
        self.deactive_cluster_manager = DeactivedClusterManager(strategy=strategy)

    def find_closest_centroid(self, x: ClusterInstance) -> ActiveClusterFeatureVector:
        I = self.find_closest_centroid_ids(x, 1)[0]
        return self.centroid_memory[I]

    def find_closest_centroid_ids(self, x: ClusterInstance, k) -> List[int]:
        I = self.strategy.get_closest_cluster_indice(
            query=x, k=k, data=self.centroid_memory.values()
        )
        return I

    def _find_prototype(self, cfv: ActiveClusterFeatureVector) -> ClusterInstance:
        instances = [
            self.instance_memory[int(x_id)]
            for x_id in self.assignment_table[int(cfv.centroid_id)]
        ]
        I = self.strategy.get_closet_instance_indice(query=cfv, k=1, data=instances)[0]
        # print(f'manager _find_prototype I:{I}')
        return instances[I]

    def _evict_cluster(self):
        # deactivate
        centroid_memory_keys = list(self.centroid_memory.keys())
        for c_id in centroid_memory_keys:
            cfv: ActiveClusterFeatureVector = self.centroid_memory[c_id]
            if cfv.get_weight(self.time_step) < self.active_threshold:
                print(f"Deactivate {c_id}")
                prototype: ClusterInstance = self._find_prototype(cfv)
                self.deactive_cluster_manager.update(cfv, prototype=prototype)
                del self.centroid_memory[c_id]
        # discard
        discarded_clusters = self.deactive_cluster_manager.discard()
        for c_id in discarded_clusters:
            print(f"Discard {c_id}")
            if c_id in self.assignment_table.keys():
                for i_id in self.assignment_table[int(c_id)]:
                    del self.instance_memory[int(i_id)]
                del self.assignment_table[int(c_id)]
            if c_id in self.centroid_memory.keys():
                del self.centroid_memory[int(c_id)]

    def _recall_cluster(self, centroid_id):
        # TODO 도착시간을 갱신시켜야할까..?
        reactivated_centroid = self.deactive_cluster_manager.reactivate(centroid_id)
        self.centroid_memory[int(centroid_id)] = reactivated_centroid

    def _add_centroid(self, x: ClusterInstance, current_time_step):
        # print(f"seed x: {x.id}")
        new_id = int(self.centroid_id)
        self.centroid_memory[new_id] = self.strategy.build_ActiveClusterFeatureVector(
            centroid_id=new_id,
            centroid=x,
            current_time_step=current_time_step,
        )
        self.assignment_table[new_id].append(int(x.id))
        self.centroid_id += 1

    def _assign_instance(
        self,
        x: ClusterInstance,
        centroid_id,
        current_time_step,
        is_prototype_updated: bool,
    ):
        self.assignment_table[centroid_id].append(int(x.id))
        centroid: ActiveClusterFeatureVector = self.centroid_memory[int(centroid_id)]
        centroid.update(x, current_time_step)
        if is_prototype_updated:
            prototype = self._find_prototype(self.centroid_memory[int(centroid_id)])
            centroid.update_prototype(prototype)

    def _assign(self, x_id, x_passage, x_mean_emb, x_token_embs, current_time_step):
        self._evict_cluster()

        x = ClusterInstance(x_id, x_passage, x_mean_emb, x_token_embs)
        self.instance_memory[int(x.id)] = x

        # TODO 초기 클러스터 어떻게 구성할지 고민 필요...
        # TODO 다 탈락하는 경우 생각 못 한 threshold가 하나 더..
        if len(self.centroid_memory.keys()) < self.init_centroid_num:
            print(
                f"new centroid initially {len(self.centroid_memory.keys())}/{self.init_centroid_num}"
            )
            self._add_centroid(x, current_time_step)
        else:
            new_centroid = self.find_closest_centroid(x)
            new_distance = self.strategy.get_distance(x, new_centroid)
            # old가 아직 없을 수 있음
            old_centroid = None
            if len(self.deactive_cluster_manager.centroid_memory.keys()):
                old_centroid = self.deactive_cluster_manager.find_closest_centroid(x)
                old_distance = self.strategy.get_distance(x, old_centroid)
                centroid = new_centroid if new_distance < old_distance else old_centroid
                distance = min(new_distance, old_distance)
            else:
                centroid = new_centroid
                distance = new_distance

            if self.strategy.is_assigned(centroid, distance):
                if old_centroid and centroid.centroid_id == old_centroid.centroid_id:
                    print("recall cluster")
                    self._recall_cluster(centroid.centroid_id)
                print("assign x to cluster")
                self._assign_instance(
                    x,
                    centroid.centroid_id,
                    current_time_step,
                    centroid.get_std_norm() > distance,  # get_std()
                )
            else:
                print(f"new centroid | distance: {distance}")
                self._add_centroid(x, current_time_step)

    def assign(self, x_id, x_passage, mean_embedding: Tensor, token_embedding: Tensor):
        self._assign(x_id, x_passage, mean_embedding, token_embedding, self.time_step)
        # self.time_step += 1


class DeactivedClusterManager:
    def __init__(self, strategy: Strategy):
        self.strategy = strategy
        self.centroid_memory = {}

    def update(self, cfv: ActiveClusterFeatureVector, prototype: ClusterInstance):
        deactivated = self.strategy.build_DeactiveClusterFeatureVector(
            centroid_id=cfv.centroid_id,
            n=cfv.n,
            V1=cfv.V1,
            V2=cfv.V2,
            prototype=prototype,
            t=cfv.t,
        )
        self.centroid_memory[deactivated.centroid_id] = deactivated

    def find_closest_centroid(self, x: ClusterInstance) -> DeactiveClusterFeatureVector:
        I = self.strategy.get_closest_cluster_indice(
            query=x, k=1, data=self.centroid_memory.values()
        )[0]
        return self.centroid_memory[I]

    def discard(self) -> List[int]:
        discarded_clusters = []
        centroid_memory_keys = list(self.centroid_memory.keys())
        for c_id in centroid_memory_keys:
            _discarded: DeactiveClusterFeatureVector = self.centroid_memory[int(c_id)]
            prototype: ClusterInstance = _discarded.prototype
            if self.strategy.is_repr_drifted(_discarded, prototype):
                discarded_clusters.append(int(c_id))
                del self.centroid_memory[int(c_id)]
        return discarded_clusters

    def reactivate(self, centroid_id) -> ActiveClusterFeatureVector:
        centroid: DeactiveClusterFeatureVector = self.centroid_memory[centroid_id]
        reactivated_centroid = centroid.build_ACFV()
        del self.centroid_memory[centroid_id]
        return reactivated_centroid
