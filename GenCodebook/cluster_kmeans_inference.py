import numpy as np
import joblib
import os
from typing import Dict, List
from tqdm import tqdm
import math

def load_model_components(kmeans_path: str):
    """Load the saved PCA and cluster centroids"""
    model_data = joblib.load(kmeans_path)
    pca_list = model_data["pca"]     # List of PCA objects
    centers_list = model_data["centers"]  # list of list of centers: [ [K, D], [K, D], ... ]
    return pca_list, centers_list

def predict_residual_codes(
    new_data: Dict[str, List[np.ndarray]],
    pca_list: List,
    centers_list: List[List[np.ndarray]]
) -> Dict[str, List[int]]:
    
    clustered_labels = {item: [] for item in new_data}
    new_embeddings = {item: [] for item in new_data}
    num_gist = len(pca_list)
    num_dept = len(centers_list) // num_gist

    for i in tqdm(range(num_gist), desc="Predicting residual cluster codes"):
        # Get all samples' i-th vector
        matrix = np.array([new_data[item][i] for item in new_data])  # shape: (N, D)

        # Step 1: Compression
        compressed_matrix = pca_list[i].transform(matrix)

        for nd in range(num_dept):
            index = i * num_dept + nd
            centers = centers_list[index]  # shape: (n_clusters, D)
            distances = np.linalg.norm(compressed_matrix[:, None, :] - centers[None, :, :], axis=-1)
            labels = np.argmin(distances, axis=1)  # shape: (N,)

            # Restore using the inverse transformation of PCA
            if nd == 0:
                reconstructed_data = matrix
            else:
                reconstructed_data = pca_list[i].inverse_transform(compressed_matrix)
                
            # Record current layer labels
            for idx, item in enumerate(new_data):
                clustered_labels[item].append(int(labels[idx]))
                new_embeddings[item].append(reconstructed_data[idx])
            
            # The residuals are calculated for the next layer.
            compressed_matrix = compressed_matrix - centers[labels]

    return clustered_labels, new_embeddings

def main():
    sign = "8fa057"
    new_sign = "98a29b"
    kmeans_model_path = f"/data/home/zdhs0021/Projects/Tools/StoreCodebook/tuning/Qwen3/{sign}@32-512-2.kmeans"  # The path you have saved
    new_embedding_path = f"/data/home/zdhs0021/Projects/Tools/StoreCodebook/tuning/new/{new_sign}.npy"  # The path of new embedding data
    output_code_path = f"/data/home/zdhs0021/Projects/Tools/StoreCodebook/tuning/new/{new_sign}@32-512-2.code"  # Output predicted labels
    output_embeddings_path = f"/data/home/zdhs0021/Projects/Tools/StoreCodebook/tuning/new/{new_sign}_reconstructed_32.npy"

    print("🧠 Loading PCA and cluster centers...")
    pca_list, centers_list = load_model_components(kmeans_model_path)

    print("📦 Loading new embedding data...")
    new_data = np.load(new_embedding_path, allow_pickle=True).item()

    print("🔁 Running residual inference...")
    predicted_codes, label_embeddings = predict_residual_codes(new_data, pca_list, centers_list)

    with open(output_code_path, "w") as f:
        json.dump(predicted_codes, f, indent=2)
    print(f"✅ Done! Code saved to {output_code_path}")
    
    np.save(output_embeddings_path, label_embeddings, allow_pickle=True)
    print(f'saving embeddings to {output_embeddings_path}')

if __name__ == "__main__":
    import json
    main()
