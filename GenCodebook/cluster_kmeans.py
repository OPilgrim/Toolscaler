import json
import os
from typing import Type, cast
import joblib
import numpy as np
import pigmento
from pigmento import pnt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

from loader.class_hub import ClassHub
from model.base_model import BaseModel
from utils.config_init import ConfigInit


class Cluster:
    def __init__(self, conf):
        self.conf = conf
        self.conf.model = self.conf.model.replace('.', '').lower()
        self.model_class = self.load_model()  # type: Type[BaseModel]

        self.conf.sign = str(self.conf.sign).replace('@', '')
        self.log_dir = os.path.join('tuning', self.model_class.get_name())
        self.path = os.path.join(self.log_dir, f'{self.conf.sign}.npy')

        self.data = np.load(self.path, allow_pickle=True).item()   # {"N53526": (10, 4096) array,...}, 10 is the gist number I set
        self.data = cast(dict, self.data)

    def load_model(self):
        models = ClassHub.models()
        if self.conf.model not in models:
            raise ValueError(f'Unknown model: {self.conf.model}')
        return models[self.conf.model]  # type: Type[BaseModel]

    def run(self):
        num_gist = len(next(iter(self.data.values())))  # 10

        # Create a dictionary to store the cluster IDs for each item
        clustered_dict = {item: [] for item in self.data}
        new_data = {item: [] for item in self.data}
        
        # The compressor and centroid of each gist
        pca_list = []
        centers_list = []

        for i in tqdm(range(num_gist), total=num_gist):
            # Create a matrix for the i-th vector of all items
            matrix = np.array([self.data[item][i] for item in self.data])

            # Apply PCA to compress the dimensions
            pca = PCA(n_components=self.conf.num_comp)
            compressed_matrix = pca.fit_transform(matrix)
            pca_list.append(pca)  # Save each round of PCA
            
            for nd in range(self.conf.num_dept):
                # Cluster the compressed vectors
                # kmeans = KMeans(n_clusters=self.conf.num_clst)
                kmeans = MiniBatchKMeans(n_clusters=self.conf.num_clst, batch_size=100, random_state=0)
                kmeans.fit(compressed_matrix)
                labels = kmeans.labels_.tolist()

                centers = kmeans.cluster_centers_
                centers_list.append(centers)  # Save the center of mass of each layer
                
                # Use PCA inverse transformation to restore
                if nd == 0:
                    reconstructed_data = matrix
                else:
                    reconstructed_data = pca.inverse_transform(compressed_matrix)
                
                compressed_matrix -= centers[labels]

                # Assign the cluster labels to the corresponding item in the dictionary
                for idx, item in enumerate(self.data):
                    clustered_dict[item].append(labels[idx])
                    new_data[item].append(reconstructed_data[idx])

        # Save the clustered dictionary as a JSON file
        path = self.path.replace('.npy', f'@{self.conf.num_comp}-{self.conf.num_clst}-{self.conf.num_dept}.code')
        with open(path, 'w') as f:
            json.dump(clustered_dict, f, indent=2)
        # Save the new embedding
        np.save(os.path.join(self.log_dir, f'{self.conf.sign}_reconstructed_{self.conf.num_comp}.npy'), new_data, allow_pickle=True)
        pnt(f'saving embeddings to {self.log_dir}/{self.conf.sign}_reconstructed_{self.conf.num_comp}.npy')
        # Save the centroid and PCA model
        joblib.dump({'pca': pca_list, 'centers': centers_list}, self.path.replace(".npy", f"@{self.conf.num_comp}-{self.conf.num_clst}-{self.conf.num_dept}.kmeans"))
        
        code_set = set()
        for item in clustered_dict:
            code_set.add(tuple(clustered_dict[item]))
        print(len(code_set))
        print(len(clustered_dict))
        print('collapse rate', 1 - len(code_set) / len(clustered_dict))

        return clustered_dict




import argparse
import os
from oba import Obj

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--model', required=True, type=str, help='Model name (e.g., llama3)')
    parser.add_argument('--sign', required=True, type=str, help='Signature or ID string')

    # Optional arguments with defaults
    parser.add_argument('--num_comp', type=int, default=32, help='Number of compressed dimensions')
    parser.add_argument('--num_clst', type=int, default=512, help='Number of clusters (gist size)')
    parser.add_argument('--num_dept', type=int, default=1, help='Hierarchy depth (gist_num * num_dept)')

    args = parser.parse_args()
    return Obj(vars(args))

if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )
    
    configuration = parse_arguments()

    cluster = Cluster(configuration)
    cluster.run()