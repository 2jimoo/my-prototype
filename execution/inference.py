from system import ClusterManager, DenseEncoder, NCLSampler


def inference():
    encoder = DenseEncoder()
    cluster_manager = ClusterManager(encoder=encoder)
    sampler = NCLSampler(cluster_manager=cluster_manager)

    dummy_query = {"qid": 4, "query": "are alpha and beta glucose geometric isomers?"}
    cluster_manager.assign(dummy_query["qid"], dummy_query["query"])
