import faiss


def faiss_search(query, k, data, vector_dim=768):
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(vector_dim)

    # gpu 인덱스로 옮기기
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(data)
    D, I = gpu_index_flat.search(query, k)
    return D, I


def ncl_lsh(query, k, data, vector_dim=768):
    pass
