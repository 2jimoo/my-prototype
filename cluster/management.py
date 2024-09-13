from collections import defaultdict
from .data_structure import (
    ClusterInstance,
    ActiveClusterFeatureVector,
    DeactiveClusterFeatureVector,
)
from config import Strategy
from typing import List


class ClusterManager:
    def __init__(self, strategy: Strategy, init_cluster_num):
        self.time_step = 0
        self.init_centroid_num = init_cluster_num
        self.centroid_id = 0
        self.assignment_table = defaultdict(list)  # 클러스터 id: 할당된 인스턴스 번호
        self.instance_memory = {}  # 인스턴스 id : 인스턴스 데이터
        self.centroid_memory = {}  # 클러스터 id: 클러스터 데이터
        self.active_threshold = 0.1
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
            self.instance_memory[x_id]
            for x_id in self.assignment_table[cfv.centroid_id]
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
                prototype: ClusterInstance = self._find_prototype(cfv)
                self.deactive_cluster_manager.update(cfv, prototype=prototype)
                del self.centroid_memory[c_id]
        # discard
        discarded_clusters = self.deactive_cluster_manager.discard()
        for c_id in discarded_clusters:
            if c_id in self.assignment_table.keys():
                for i_id in self.assignment_table[c_id]:
                    del self.instance_memory[i_id]
                del self.assignment_table[c_id]
            if c_id in self.centroid_memory.keys():
                del self.centroid_memory[c_id]

    def _recall_cluster(self, centroid_id):
        # TODO 도착시간을 갱신시켜야할까..?
        reactivated_centroid = self.deactive_cluster_manager.reactivate(centroid_id)
        self.centroid_memory[centroid_id] = reactivated_centroid

    def _add_centroid(self, x: ClusterInstance, current_time_step):
        self.centroid_memory[self.centroid_id] = ActiveClusterFeatureVector(
            centroid_id=self.centroid_id,
            centroid=x,
            current_time_step=current_time_step,
        )
        self.assignment_table[self.centroid_id].append(x.id)
        self.centroid_id += 1

    def _assign_instance(
        self,
        x: ClusterInstance,
        centroid_id,
        current_time_step,
        is_prototype_updated: bool,
    ):
        self.assignment_table[centroid_id].append(x.id)
        centroid: ActiveClusterFeatureVector = self.centroid_memory[centroid_id]
        centroid.update(x.mean_emb, current_time_step)
        if is_prototype_updated:
            prototype = self._find_prototype(self.centroid_memory[centroid_id])
            centroid.update_prototype(prototype)

    def _assign(self, x_id, x_passage, x_mean_emb, x_token_embs, current_time_step):
        self._evict_cluster()

        x = ClusterInstance(x_id, x_passage, x_mean_emb, x_token_embs)
        self.instance_memory[x.id] = x

        # TODO 초기 클러스터 어떻게 구성할지 고민 필요...
        # TODO 다 탈락하는 경우 생각 못 한 threshold가 하나 더..
        if len(self.centroid_memory.keys()) < self.init_centroid_num:
            self._add_centroid(x, current_time_step)
        else:
            new_centroid = self.find_closest_centroid(x)
            old_centroid = self.deactive_cluster_manager.find_closest_centroid(x)
            new_distance = self.strategy.get_distance(x, new_centroid)
            old_distance = self.strategy.get_distance(x, old_centroid)
            centroid = new_centroid if new_distance < old_distance else old_centroid
            distance = min(new_distance, old_distance)

            if centroid.get_rms() < distance:
                self._add_centroid(x, current_time_step)
            else:
                if centroid.centroid_id == old_centroid.centroid_id:
                    self._recall_cluster(centroid.centroid_id)
                self._assign_instance(
                    x,
                    centroid.centroid_id,
                    current_time_step,
                    centroid.get_std_norm() > distance,  # get_std()
                )

    def assign(self, x_id, x_passage, mean_embedding, token_embedding):
        self._assign(x_id, x_passage, mean_embedding, token_embedding, self.time_step)
        self.time_step += 1


class DeactivedClusterManager:
    def __init__(self, strategy: Strategy):
        self.strategy = strategy
        self.centroid_memory = {}

    def update(self, cfv: ActiveClusterFeatureVector, prototype: ClusterInstance):
        deactivated = DeactiveClusterFeatureVector(
            centroid_id=cfv.centroid_id,
            n=cfv.n,
            S1=cfv.S1,
            S2=cfv.S2,
            prototype=prototype,
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
            _discarded: DeactiveClusterFeatureVector = self.centroid_memory[c_id]
            prototype: ClusterInstance = _discarded.prototype
            if _discarded.get_std_norm() <= self.strategy.get_repr_drift(prototype):
                discarded_clusters.append(c_id)
                del self.centroid_memory[c_id]
        return discarded_clusters

    def reactivate(self, centroid_id) -> ActiveClusterFeatureVector:
        centroid: DeactiveClusterFeatureVector = self.centroid_memory[centroid_id]
        reactivated_centroid = centroid.build_ACFV()
        del self.centroid_memory[centroid_id]
        return reactivated_centroid
