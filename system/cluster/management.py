from collections import defaultdict
from functions import (
    faiss_search,
    ncl_lsh,
)
from cluster import (
    ClusterInstance,
    ActiveClusterFeatureVector,
    DeactiveClusterFeatureVector,
)
from model import DenseEncoder
from typing import List
import numpy as np


class ClusterManager:
    def __init__(self, encoder: DenseEncoder, similarity_func, init_cluster_num=10):
        self.time_step = 0
        self.encoder = encoder
        self.init_centroid_num = init_cluster_num
        self.centroid_id = 0
        self.assignment_table = defaultdict(list)  # 클러스터 id: 할당된 인스턴스 번호
        self.instance_memory = {}  # 인스턴스 id : 인스턴스 데이터
        self.centroid_memory = {}  # 클러스터 id: 클러스터 데이터
        self.active_threshold = 0.1
        self.similarity_func = similarity_func
        self.deactive_cluster_manager = DeactivedClusterManager(
            encoder=self.encoder, similarity_func=self.similarity_func
        )

    def find_closest_centroid(self, x_mean_emb) -> ActiveClusterFeatureVector:
        return self.find_closest_centroids(x_mean_emb=x_mean_emb, k=1)[0]

    def find_closest_centroids(self, x_mean_emb, k) -> ActiveClusterFeatureVector:
        centroids = self.centroid_memory.values()
        embeddings = np.array([c.get_mean() for c in centroids])
        I = faiss_search(query=x_mean_emb, k=k, data=embeddings)
        return [centroids[I[i]].centroid_id for i in I]

    def _find_prototype(self, cfv: ActiveClusterFeatureVector) -> ClusterInstance:
        instances = [
            self.instance_memory[x_id]
            for x_id in self.assignment_table[cfv.centroid_id]
        ]
        embeddings = np.array([x.mean_emb for x in instances])
        I = faiss_search(query=cfv.get_mean(), k=1, data=embeddings)
        return instances[I]

    def _evict_cluster(self):
        # deactivate
        for c_id in self.centroid_memory.keys():
            cfv: ActiveClusterFeatureVector = self.centroid_memory[c_id]
            if cfv.get_weight() < self.active_threshold:
                prototype: ClusterInstance = self._find_prototype(cfv)
                self.deactive_cluster_manager.update(cfv, prototype=prototype)
                del self.centroid_memory[c_id]
        # discard
        discarded_clusters = self.deactive_cluster_manager.discard()
        for c_id in discarded_clusters:
            for i_id in self.assignment_table[c_id]:
                del self.instance_memory[i_id]
            del self.assignment_table[c_id]
            del self.centroid_memory[c_id]

    def _recall_cluster(self, centroid_id):
        # TODO 도착시간을 갱신시켜야할까..?
        reactivated_centroid = self.deactive_cluster_manager.reactivate(centroid_id)
        self.centroid_memory[centroid_id] = reactivated_centroid

    def _add_centroid(self, x: ClusterInstance, current_time_step):
        x.cluster_id = self.centroid_id
        self.centroid_memory[self.centroid_id] = ActiveClusterFeatureVector(
            centroid_id=self.centroid_id,
            centroid_mean_emb=x.mean_emb,
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
        x.cluster_id = centroid_id
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
        if self.centroid_id < self.init_centroid_num:
            self._add_centroid(x, current_time_step)
        else:
            new_centroid = self.find_closest_centroid(x.mean_emb)
            old_centroid = self.deactive_cluster_manager.find_closest_centroid(
                x.mean_emb
            )
            new_distance = self.similarity_func(x.mean_emb, new_centroid.get_mean())
            old_distance = self.similarity_func(x.mean_emb, old_centroid.get_mean())
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
                    centroid.get_std() > distance,
                )

    def assign(self, x_id, x_passage, mean_embedding, token_embedding):
        self._assign(x_id, x_passage, mean_embedding, token_embedding, self.time_step)
        self.time_step += 1


class DeactivedClusterManager:
    def __init__(self, encoder: DenseEncoder, similarity_func):
        self.encoder = encoder
        self.similarity_func = similarity_func
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

    def find_closest_centroid(self, x_mean_emb) -> DeactiveClusterFeatureVector:
        centroids = self.centroid_memory.values()
        embeddings = np.array([c.get_mean() for c in centroids])
        I = faiss_search(query=x_mean_emb, k=1, data=embeddings)
        return centroids[I].centroid_id

    def discard(self) -> List[int]:
        discarded_clusters = []
        for c_id in self.centroid_memory.keys():
            _discarded: DeactiveClusterFeatureVector = self.centroid_memory[c_id]
            E0, E_cur = (
                _discarded.prototype.mean_emb,
                self.encoder.encode(_discarded.prototype.passage)[0],
            )  # self._get_current_prototype_embedding(_discarded.prototype)
            if _discarded.get_std() <= self.similarity_func(E0, E_cur):
                discarded_clusters.append(c_id)
                del self.centroid_memory[c_id]
        return discarded_clusters

    def reactivate(self, centroid_id) -> ActiveClusterFeatureVector:
        centroid: DeactiveClusterFeatureVector = self.centroid_memory[centroid_id]
        reactivated_centroid = centroid.build_ACFV()
        del self.centroid_memory[centroid_id]
        return reactivated_centroid
