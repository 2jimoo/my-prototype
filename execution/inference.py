from system import ClusterManager, DenseEncoder

encoder = DenseEncoder()
cluster_manager = ClusterManager(encoder=encoder)


def inference():
    dummy_query = {"qid": 4, "query": "are alpha and beta glucose geometric isomers?"}
    cluster_manager.assign(dummy_query["qid"], dummy_query["query"])
