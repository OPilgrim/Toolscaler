import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader as BaseDataLoader


class DataLoader(BaseDataLoader):
    skip_padding_columns = ['gist_positions', 'gist_lengths', 'item_id']

    def __init__(self, **kwargs):
        super().__init__(collate_fn=self.stack, **kwargs)

    @classmethod
    def stack(cls, batch):
        data = dict()
        for key in batch[0].keys():
            value = [item[key] for item in batch]
            if key in cls.skip_padding_columns:
                value = torch.tensor(value, dtype=torch.long)
            else:
                value = pad_sequence([torch.tensor(x, dtype=torch.long) for x in value], batch_first=True)
            data[key] = value
        return data