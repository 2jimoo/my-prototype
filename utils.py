import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def visualize_clusters(
    data, labels, method="tsne", perplexity=30, n_components=2, random_state=42
):
    """
    클러스터링 결과를 2D로 시각화하는 함수.

    Parameters:
    - data: ndarray of shape (n_samples, n_features), 데이터 벡터
    - labels: ndarray of shape (n_samples,), 클러스터 레이블
    - method: str, 'tsne' 또는 'pca'를 선택하여 차원 축소 방법 선택
    - perplexity: float, t-SNE의 perplexity 값 (method='tsne'일 때만 사용)
    - n_components: int, 축소할 차원의 수 (기본값: 2)
    - random_state: int, 무작위 상태 시드 (기본값: 42)

    Returns:
    - None
    """
    # 차원 축소
    if method == "tsne":
        reducer = TSNE(
            n_components=n_components, perplexity=perplexity, random_state=random_state
        )
        reduced_data = reducer.fit_transform(data)
    elif method == "pca":
        reducer = PCA(n_components=n_components)
        reduced_data = reducer.fit_transform(data)
    else:
        raise ValueError("Invalid method. Choose either 'tsne' or 'pca'.")

    # 클러스터링 시각화
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap(
        "tab10", len(unique_labels)
    )  # 레이블 개수에 맞는 색상 팔레트 설정

    for i, label in enumerate(unique_labels):
        cluster_points = reduced_data[labels == label]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=[colors(i)],
            label=f"Cluster {label}",
            alpha=0.7,
        )

    plt.title(f"Cluster Visualization using {method.upper()}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()
