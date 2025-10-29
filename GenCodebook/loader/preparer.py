import os

import pandas as pd
from pigmento import pnt
from tqdm import tqdm

from loader.vocab import Vocab
from model.base_model import BaseModel
from process.base_processor import BaseProcessor


class Preparer:
    def __init__(self, processor: BaseProcessor, model: BaseModel, train_ratio, sign):
        self.processor = processor
        self.model = model
        self.train_ratio = train_ratio
        self.sign = sign

        self.store_dir = os.path.join(
            'prepare',
            f'{self.processor.get_name()}_{self.model.get_name()}',
            sign,
        )
        os.makedirs(self.store_dir, exist_ok=True)

        self.train_datapath = os.path.join(self.store_dir, 'train.parquet')
        self.valid_datapath = os.path.join(self.store_dir, 'valid.parquet')
        self.export_datapath = os.path.join(self.store_dir, 'export.parquet')

        self.has_training_data = os.path.exists(self.train_datapath) and os.path.exists(self.valid_datapath)
        self.has_export_data = os.path.exists(self.export_datapath)

    def tokenize(self):
        items = self.processor.items  # type: pd.DataFrame
        tokens = []

        for index, row in items.iterrows():
            current_tokens = dict()
            for attr in self.processor.ATTRS:
                key = self.processor.META.get_natural_key(attr)
                input_ids = self.model.generate_input_ids(f'{key}: {row[attr]}')
                max_len = self.processor.META.get_maxlen(attr)
                if max_len > 0:
                    input_ids = input_ids[:max_len]
                current_tokens[attr] = input_ids
            tokens.append(current_tokens)

        return tokens

    def _concatenate_values(self, values):
        value = []
        for index, v in enumerate(values):
            value.extend(v)
            if index < len(values) - 1:
                value.append(self.model.sep_token)
        return value

    def _build_single_training_sample(self, source_values, target_values, task_id, gist_length):
        # encoder: SOURCE_VALUE, N_GIST, <G>, ..., <G>
        # decoder: N_GIST, <G>, ..., <G>, <T>, TARGET_VALUE
        source_value = self._concatenate_values(source_values)
        target_value = self._concatenate_values(target_values)

        encoder_input_ids = [*source_value, gist_length - 1, *list(range(gist_length))]
        encoder_input_vocabs = [Vocab.LLM] * len(source_value) + [Vocab.SPC] + [Vocab.GST] * gist_length
        encoder_attention_mask = [1] * len(encoder_input_ids)
        encoder_labels = source_value[1:] + [-100] * (gist_length + 2)
        gist_position = len(source_value) + 1
        decoder_input_ids = [gist_length - 1, *list(range(gist_length)), task_id, *target_value]
        decoder_input_vocabs = [Vocab.SPC] + [Vocab.GST] * gist_length + [Vocab.TSK] + [Vocab.LLM] * len(target_value)
        decoder_attention_mask = [1] * len(decoder_input_ids)
        decoder_labels = [-100] * (gist_length + 1) + target_value + [self.model.eos_token]
        return dict(
            encoder_input_ids=encoder_input_ids,
            encoder_input_vocabs=encoder_input_vocabs,
            encoder_attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_input_vocabs=decoder_input_vocabs,
            decoder_attention_mask=decoder_attention_mask,
            gist_positions=gist_position,
            gist_lengths=gist_length,
            encoder_labels=encoder_labels,
            decoder_labels=decoder_labels,
        )
    
    def _build_single_export_sample(self, attr_values, gist_length, item_id):
        # encoder: SOURCE_VALUE, N_GIST, <G>, ..., <G>
        # decoder: N_GIST, <G>, ..., <G>, <T>, TARGET_VALUE
        attr_value = self._concatenate_values(attr_values)

        encoder_input_ids = [*attr_value, gist_length - 1, *list(range(gist_length))]
        encoder_input_vocabs = [Vocab.LLM] * len(attr_value) + [Vocab.SPC] + [Vocab.GST] * gist_length
        encoder_attention_mask = [1] * len(encoder_input_ids)
        gist_position = len(attr_value) + 1
        return dict(
            encoder_input_ids=encoder_input_ids,
            encoder_input_vocabs=encoder_input_vocabs,
            encoder_attention_mask=encoder_attention_mask,
            gist_positions=gist_position,
            gist_lengths=gist_length,
            item_id=item_id
        )

    def load_training_datalist(self):
        items = self.tokenize()
        datalist = []
        for item in tqdm(items):
            # pnt(item)  # {'cat': [7663, 29901, 9763], 'subcat': [1014, 7663, 29901, 9763, 459, 262, 291], 'title': [3611, 29901, 2664, 3197, 23072, 29901, 15549, 8156, 29915, 29879, 4802, 5401, 313, 392, 825, 372, 2794, 363, 27504, 29897], 'abs': [9846, 29901, 15549, 8156, 5131, 263, 1560, 17223, 13736, 322, 289, 27494, 21603, 2304, 1434, 8401, 6375, 411, 263, 11719, 373, 527, 412, 25117, 29892, 322, 1183, 925, 1122, 679, 902, 982, 29889]}
            for task_tuple, task_id in self.processor.META.get_tasks().items():
                source_attrs, target_attrs = task_tuple
                source_values = [item[attr] for attr in source_attrs]
                target_values = [item[attr] for attr in target_attrs]
                datalist.append(self._build_single_training_sample(source_values, target_values, task_id, self.model.num_gist))

        return datalist

    def load_export_datalist(self):
        items = self.tokenize()
        datalist = []
        attr_tuple = list(map(lambda x: x[0], self.processor.EXPORT))
        for index, item in tqdm(enumerate(items)):
            # pnt(item)  # {'cat': [7663, 29901, 9763], 'subcat': [1014, 7663, 29901, 9763, 459, 262, 291], 'title': [3611, 29901, 2664, 3197, 23072, 29901, 15549, 8156, 29915, 29879, 4802, 5401, 313, 392, 825, 372, 2794, 363, 27504, 29897], 'abs': [9846, 29901, 15549, 8156, 5131, 263, 1560, 17223, 13736, 322, 289, 27494, 21603, 2304, 1434, 8401, 6375, 411, 263, 11719, 373, 527, 412, 25117, 29892, 322, 1183, 925, 1122, 679, 902, 982, 29889]}
            item_id = self.processor.items[self.processor.AID_COL].values[index] 
            item_id = self.processor.item_vocab[item_id]
            attr_values = [item[attr] for attr in attr_tuple]
            datalist.append(self._build_single_export_sample(attr_values, self.model.num_gist, item_id))

        return datalist

    def split_datalist(self, datalist):
        total_items = len(self.processor.items)
        total_tasks = len(self.processor.META.get_tasks())
        assert len(datalist) == total_items * total_tasks

        train_size = int(total_items * self.train_ratio)

        train_datalist = datalist[:train_size * total_tasks]
        valid_datalist = datalist[train_size * total_tasks:]

        return train_datalist, valid_datalist

    @staticmethod
    def display_one_sample(datalist: pd.DataFrame):
        pnt('one sample:')
        columns = datalist.columns
        sample = datalist.sample(1)
        for column in columns:
            pnt(f'{column}: {sample[column].values[0]}')

    def load_or_generate_training_data(self):
        if self.has_training_data:
            pnt(f'loading prepared {self.processor.get_name()} dataset')

            train_datalist = pd.read_parquet(self.train_datapath)
            valid_datalist = pd.read_parquet(self.valid_datapath)

            self.display_one_sample(train_datalist)
            return train_datalist, valid_datalist

        datalist = self.load_training_datalist()
        train_datalist, valid_datalist = self.split_datalist(datalist)

        train_datalist = pd.DataFrame(train_datalist)
        valid_datalist = pd.DataFrame(valid_datalist)

        train_datalist.to_parquet(self.train_datapath)
        valid_datalist.to_parquet(self.valid_datapath)

        self.display_one_sample(train_datalist)
        return train_datalist, valid_datalist

    def load_or_generate_export_data(self):
        if self.has_export_data:
            pnt(f'loading prepared {self.processor.get_name()} dataset')

            export_datalist = pd.read_parquet(self.export_datapath)
            self.display_one_sample(export_datalist)
            return export_datalist

        datalist = self.load_export_datalist()

        export_datalist = pd.DataFrame(datalist)
        export_datalist.to_parquet(self.export_datapath)

        self.display_one_sample(export_datalist)
        return export_datalist

