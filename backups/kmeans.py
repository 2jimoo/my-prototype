# def _debug_assign_kmeans(self, x_id, x_passage, x_mean_emb, x_token_embs, current_time_step):
#     x = ClusterInstance(x_id, x_passage, x_mean_emb, x_token_embs)
#     self.instance_memory[int(x.id)] = x
#     if len(self.instance_memory.keys()) == self.init_centroid_num:
#         print("clustering")
#         instances = list(self.instance_memory.values())
#         mean_embeddings = torch.stack([instance.mean_emb.cpu().squeeze() for instance in instances]).numpy()
#         # print(f"mean_embedding shape: {mean_embeddings[0].shape}")
#         kmeans = KMeans(n_clusters=4, random_state=42) # 2개 클러스터
#         kmeans.fit(mean_embeddings)
#         for i, label in enumerate(kmeans.labels_):
#             self.assignment_table[label].append(instances[i].id)
#             if not label in self.centroid_memory.keys():
#                 self.centroid_memory[label] = self.strategy.build_ActiveClusterFeatureVector(
#                     centroid_id=label,
#                     centroid=instances[i],
#                     current_time_step=current_time_step)
#             else:
#                 self.centroid_memory[label].update(instances[i], current_time_step)
#     elif len(self.instance_memory.keys()) > self.init_centroid_num:
#         print("assign")
#         centroid = self.find_closest_centroid(x)
#         distance = self.strategy.get_distance(x, centroid)
#         self._assign_instance(
#                 x,
#                 centroid.centroid_id,
#                 current_time_step,
#                 centroid.get_std_norm() > distance,  # get_std()
#         )
