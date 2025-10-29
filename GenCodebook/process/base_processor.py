import os
from typing import List, Optional, Dict, Tuple

import pandas as pd
from UniTok import Vocab
from pigmento import pnt

class Attr(tuple):
    _ATTR_COUNT = 0

    def __new__(cls, *args):
        instance = super().__new__(cls, args)
        instance.attr_id = cls._ATTR_COUNT
        cls._ATTR_COUNT += 1
        return instance

    def __init__(self, *args):
        super().__init__()


class Task:
    _TASK_COUNT = 0

    def __init__(self, source_attrs, target_attrs):
        if not isinstance(source_attrs[0], tuple):
            source_attrs = (source_attrs,)
        if not isinstance(target_attrs[0], tuple):
            target_attrs = (target_attrs,)
        self.source_attrs = tuple(map(lambda x: x[0], source_attrs))
        self.target_attrs = tuple(map(lambda x: x[0], target_attrs))
        self.task_id = Task._TASK_COUNT
        Task._TASK_COUNT += 1

    def get_signature(self):
        return self.source_attrs, self.target_attrs

    # def get_signature(self, meta: 'BaseMeta'):
    #     map_ = meta.get_attr_to_key_map()
    #     source_keys = []
    #     for attr in self.source_attrs:
    #         source_keys.append(map_[attr[0]])
    #     source_keys = tuple(source_keys)
    #
    #     target_keys = []
    #     for attr in self.target_attrs:
    #         target_keys.append(map_[attr[0]])
    #     target_keys = tuple(target_keys)
    #
    #     return source_keys, target_keys

class BaseMeta:
    __tasks = dict()

    @classmethod
    def _get_keys(cls):
        return filter(lambda x: x.isupper(), cls.__dict__.keys())

    @classmethod
    def get_natural_key(cls, attr: str):
        key = cls.get_attr_to_key_map()[attr]
        return key.replace('_', ' ').lower()

    @classmethod
    def get_attr(cls, key: str):
        return getattr(cls, key)[0]

    @classmethod
    def _get_attr_to_info_map(cls):
        keys = cls._get_keys()
        return dict(map(lambda x: (getattr(cls, x)[0], getattr(cls, x)), keys))

    @classmethod
    def get_attr_to_key_map(cls):
        keys = cls._get_keys()
        return dict(map(lambda x: (getattr(cls, x)[0], x), keys))

    @classmethod
    def get_attrs(cls):
        keys = cls._get_keys()
        # return list(map(lambda x: getattr(cls, x)[0], keys))
        attrs = list(map(lambda x: getattr(cls, x), keys))  # type: List[Attr]
        # sort with attr_id
        attrs.sort(key=lambda x: x.attr_id)
        return [attr[0] for attr in attrs]

    @classmethod
    def get_maxlen(cls, attr: str):
        if attr.isupper():
            return getattr(cls, attr)[1]
        return cls._get_attr_to_info_map()[attr][1]

    @classmethod
    def register_tasks(cls, *tasks: Task):
        # map_ = cls.get_attr_to_key_map()
        # tasks = set(map(lambda x: tuple(map(lambda y: map_[y[0]], x)), tasks))
        # for task in tasks:
        #     if task not in cls.__tasks:
        #         cls.__tasks[task] = len(cls.__tasks)
        # tasks = set(map(lambda x: x.get_signature(), tasks))
        for task in tasks:
            sign = task.get_signature()
            cls.__tasks[sign] = task.task_id
            pnt(f'registered task: {sign} with id {cls.__tasks[sign]}')

    @classmethod
    def get_tasks(cls):
        return cls.__tasks

# If the id of the item already exists, read from the specified address and assign it to self.item_vocab. If not, remove duplicates from item[AID_COL] and use it as item_vocab.
class BaseProcessor:
    META = BaseMeta

    AID_COL: str
    ATTR_DICT: dict
    ATTRS: List[str]
    EXPORT: tuple

    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.store_dir = os.path.join('data', self.get_name())
        os.makedirs(self.store_dir, exist_ok=True)

        self.items: Optional[pd.DataFrame] = None

        self.item_path = os.path.join(self.store_dir, 'item.parquet')
        self.has_generated = os.path.exists(self.item_path)

        self.item_vocab = Vocab(name='item')
        self.has_vocab = os.path.exists(self.item_vocab.get_store_path(self.store_dir))

    @classmethod
    def get_name(cls):
        return cls.__name__.replace('Processor', '').lower()

    def get_path(self, obj):
        return os.path.join(self.store_dir, f'{obj}.parquet')

    def load_items(self) -> pd.DataFrame:
        raise NotImplemented

    def load_or_generate(self):
        if self.has_generated:
            self.items = pd.read_parquet(self.item_path)
        else:
            self.items = self.load_items()
            self.items.to_parquet(self.item_path)

        if self.has_vocab:
            self.item_vocab = self.item_vocab.load(self.store_dir)
        else:
            item_ids = self.items[self.AID_COL].unique()
            self.item_vocab.extend(item_ids)
            self.item_vocab.save(self.store_dir)
        self.item_vocab.deny_edit()
