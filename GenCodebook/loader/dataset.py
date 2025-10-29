import pandas as pd
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    def __init__(self, datalist: pd.DataFrame):
        self.datalist = datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        values = self.datalist.iloc[idx]
        return {column: values[column] for column in self.datalist.columns}
