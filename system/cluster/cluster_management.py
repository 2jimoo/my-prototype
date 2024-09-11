from collections import defaultdict
from functions import calculate_cosine_similarity, calculate_term, calculate_term_regl
from system.cluster_data_structure import (
    ClusterInstance,
    ActiveClusterFeatureVector,
    DeactiveClusterFeatureVector,
)
from system.dpr.model import DenseEncoder
from typing import List

similarity_func = {
    "cosine": calculate_cosine_similarity,
    "term": calculate_term,
    "term-regl": calculate_term_regl,
}


# TODO faiss
class ClusterManager:
    def __init__(
        self, encoder: DenseEncoder, init_cluster_num=10, similarity_func_name="cosine"
    ):
        self.time_step = 0
        self.encoder = encoder
        self.init_centroid_num = init_cluster_num
        self.centroid_id = 0
        self.assignment_table = defaultdict(list)  # 클러스터 id: 할당된 인스턴스 번호
        self.instance_memory = {}  # 인스턴스 id : 인스턴스 데이터
        self.centroid_memory = {}  # 클러스터 id: 클러스터 데이터
        self.active_threshold = 0.1
        if similarity_func_name == "cosine":
            self.similarity_func = calculate_cosine_similarity
        else:
            raise Exception(f"upsupported similarity function {similarity_func_name}")
        self.deactive_cluster_manager = DeactivedClusterManager(
            self.encoder, self.active_threshold
        )

    def _get_prototype_and_embedding(self, cfv: ActiveClusterFeatureVector):
        pass

    def _evict_cluster(self):
        # deactivate
        for c_id in self.centroid_memory.keys():
            cfv = self.centroid_memory[c_id]
            if cfv.get_weight() < self.active_threshold:
                prototype, E0 = self._get_prototype_and_embedding(cfv)
                self.deactive_cluster_manager.update(cfv, prototype=prototype, E0=E0)
                del self.centroid_memory[c_id]
        # discard
        discarded_clusters = self.deactive_cluster_manager.evict()
        for c_id in discarded_clusters:
            for i_id in self.assignment_table[c_id]:
                del self.instance_memory[i_id]
            del self.assignment_table[c_id]
            del self.centroid_memory[c_id]

    def _recall_cluster(self, centroid_id):
        reacteivated = self.deactive_cluster_manager.reactivate(centroid_id)
        self.centroid_memory[centroid_id] = reacteivated

    def _add_centroid(self, x: ClusterInstance, current_time_step):
        self.centroid_memory[self.centroid_id] = ActiveClusterFeatureVector(
            centroid_id=self.centroid_id,
            centroid_mean_emb=x.mean_emb,
            current_time_step=current_time_step,
        )
        self.assignment_table[self.centroid_id].append(x.id)
        self.centroid_id += 1

    def get_closest_centroid(self, x: ClusterInstance) -> ActiveClusterFeatureVector:
        pass

    def _assign_instance(self, x: ClusterInstance, centroid_id, current_time_step):
        self.assignment_table[centroid_id].append(x)
        self.centroid_memory[centroid_id].update(x.mean_emb, current_time_step)

    def _assign(self, x_id, x_passage, x_mean_emb, x_token_embs, current_time_step):
        self._evict_cluster()

        x = ClusterInstance(x_id, x_passage, x_mean_emb, x_token_embs)
        self.instance_memory[x.id] = x

        # TODO 초기 클러스터 어떻게 구성할지 고민 필요...
        if self.centroid_id < self.init_centroid_num:
            self._add_centroid(x, current_time_step)
        else:
            new_centroid = self.get_closest_centroid(x)
            old_centroid = self.deactive_cluster_manager.get_closest_centroid(x)
            new_distance = self.similarity_func(x.mean_emb, new_centroid.get_mean())
            old_distance = self.similarity_func(x.mean_emb, old_centroid.get_mean())
            centroid = new_centroid if new_distance < old_distance else old_centroid
            distance = min(new_distance, old_distance)

            if centroid.get_std() < distance:
                self._add_centroid(x, current_time_step)
            else:
                if centroid.centroid_id == old_centroid.centroid_id:
                    self._recall_cluster(centroid.centroid_id)
                # mean 내면 그냥 할당, 아니면 prototype 갱신(일단은 mean vector이므로 할일X)
                self._assign_instance(x, centroid.centroid_id, current_time_step)

    def assign(self, x_id, x_passage):
        embs = self.encoder.encode(x_passage)
        mean_embedding = embs.last_hidden_state.mean(dim=1)
        token_embedding = embs.last_hidden_state.squeeze(0)
        self._assign(x_id, x_passage, mean_embedding, token_embedding, self.time_step)


class DeactivedClusterManager:
    def __init__(self, encoder: DenseEncoder, similarity_func):
        self.encoder = encoder
        self.similarity_func = similarity_func
        self.centroid_memory = {}

    def update(self, cfv: ActiveClusterFeatureVector, prototype, E0, instances):
        deactivated = DeactiveClusterFeatureVector(
            centroid_id=cfv.centroid_id,
            n=cfv.n,
            S1=cfv.S1,
            S2=cfv.S2,
            w=cfv.get_weight(),
            prototype=prototype,
            E0=E0,
            instances=instances,
        )
        self.centroid_memory[deactivated.centroid_id] = deactivated

    def get_closest_centroid(self, x: ClusterInstance) -> DeactiveClusterFeatureVector:
        # min_id, min_value= -1, float('inf')
        # for c_id in self.centroid_memory.keys():
        #     E0=  self.centroid_memory[c_id].E0

        # return
        pass

    def _get_current_prototype_embedding(self, prototype):
        embs = self.encoder.encode(prototype)
        mean_embedding = embs.last_hidden_state.mean(dim=1)
        token_embedding = embs.last_hidden_state.squeeze(0)
        return mean_embedding, token_embedding

    def discard(self) -> List[int]:
        discarded_clusters = []
        for c_id in self.centroid_memory.keys():
            _discarded = self.centroid_memory[c_id]
            E0, E_cur = (
                _discarded.E0,
                _discarded.get_mean(),
            )  # self._get_current_prototype_embedding(_discarded.prototype)
            if self.similarity_func(_discarded.E0, E_cur) < self.get_std():
                discarded_clusters.append(c_id)
                del self.centroid_memory[c_id]
        return discarded_clusters

    def reactivate(self, centroid_id) -> ActiveClusterFeatureVector:
        centroid = self.centroid_memory[centroid_id]
        reacted = ActiveClusterFeatureVector(
            centroid_id=centroid.centroid_id,
            n=centroid.n,
            S1=centroid.S1,
            S2=centroid.S2,
            t=centroid.t,
        )
        del self.centroid_memory[centroid_id]
        return reacted
