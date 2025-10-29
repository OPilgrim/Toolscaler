import os

import pandas as pd

from process.base_processor import BaseProcessor, BaseMeta, Task, Attr

class TBMeta(BaseMeta):
    CATEGORY = Attr('cat', 10)   # category_name
    NAMEDESC = Attr('namedesc', 50)   # tool_name and api_name and api_description
    APICODE = Attr('acode', 256)   # code
    APIEXAM = Attr('aexam', 100)   # examples, query

TBMeta.register_tasks(
    Task((TBMeta.NAMEDESC, TBMeta.APICODE, TBMeta.APIEXAM), TBMeta.CATEGORY),
    Task((TBMeta.NAMEDESC, TBMeta.APICODE, TBMeta.APIEXAM), TBMeta.NAMEDESC),
    Task((TBMeta.NAMEDESC, TBMeta.APICODE, TBMeta.APIEXAM), TBMeta.APICODE),
    Task((TBMeta.NAMEDESC, TBMeta.APICODE, TBMeta.APIEXAM), TBMeta.APIEXAM),
)

class TBProcessor(BaseProcessor):
    META = TBMeta

    AID_COL = 'aid'
    ATTRS = TBMeta.get_attrs()   # ['cat'，'namedesc'，'acode'，'aexam']
    EXPORT = (TBMeta.NAMEDESC, TBMeta.APICODE, TBMeta.APIEXAM)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # TODO
    def _load_items(self, mode: str) -> pd.DataFrame:
        return pd.read_csv(
            filepath_or_buffer=f"data/{self.data_dir}_{mode}/tools.tsv",
            sep='\t',
            names=[self.AID_COL, *self.ATTRS],   # TODO If there are extra columns in the tsv file, add them here. However, the subsequent processing will not use these columns. This is merely done to prevent read_csv from throwing an error.
            usecols=[self.AID_COL, *self.ATTRS],
        )

    def load_items(self) -> pd.DataFrame:
        train_df = self._load_items('train')  # TODO
        dev_df = self._load_items('dev')

        return pd.concat([train_df, dev_df]).drop_duplicates([self.AID_COL])



class TB_1Meta(BaseMeta):
    CATEGORY = Attr('cat', 10)
    NAMEDESC = Attr('namedesc', 50)
    APICODE = Attr('acode', 256)
    APIEXAM = Attr('aexam', 100)

TB_1Meta.register_tasks(
    Task((TB_1Meta.NAMEDESC, TB_1Meta.APICODE, TB_1Meta.APIEXAM), TB_1Meta.CATEGORY),
    Task((TB_1Meta.NAMEDESC, TB_1Meta.APICODE, TB_1Meta.APIEXAM), TB_1Meta.NAMEDESC),
    Task((TB_1Meta.NAMEDESC, TB_1Meta.APICODE, TB_1Meta.APIEXAM), TB_1Meta.APICODE),
    Task((TB_1Meta.NAMEDESC, TB_1Meta.APICODE, TB_1Meta.APIEXAM), TB_1Meta.APIEXAM),
)

class TB_1Processor(BaseProcessor):
    META = TB_1Meta

    AID_COL = 'aid'
    ATTRS = TB_1Meta.get_attrs()
    EXPORT = (TB_1Meta.NAMEDESC, TB_1Meta.APICODE, TB_1Meta.APIEXAM)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # TODO
    def _load_items(self, mode: str) -> pd.DataFrame:
        return pd.read_csv(
            filepath_or_buffer=f"data/{self.data_dir}_{mode}/tools.tsv",
            sep='\t',
            names=[self.AID_COL, *self.ATTRS],
            usecols=[self.AID_COL, *self.ATTRS],
        )

    def load_items(self) -> pd.DataFrame:
        train_df = self._load_items('train')
        dev_df = self._load_items('dev')

        return pd.concat([train_df, dev_df]).drop_duplicates([self.AID_COL])
    


class TB_10Meta(BaseMeta):
    CATEGORY = Attr('cat', 10)
    NAMEDESC = Attr('namedesc', 50)
    APICODE = Attr('acode', 256)
    APIEXAM = Attr('aexam', 100)

TB_10Meta.register_tasks(
    Task((TB_10Meta.NAMEDESC, TB_10Meta.APICODE, TB_10Meta.APIEXAM), TB_10Meta.CATEGORY),
    Task((TB_10Meta.NAMEDESC, TB_10Meta.APICODE, TB_10Meta.APIEXAM), TB_10Meta.NAMEDESC),
    Task((TB_10Meta.NAMEDESC, TB_10Meta.APICODE, TB_10Meta.APIEXAM), TB_10Meta.APICODE),
    Task((TB_10Meta.NAMEDESC, TB_10Meta.APICODE, TB_10Meta.APIEXAM), TB_10Meta.APIEXAM),
)

class TB_10Processor(BaseProcessor):
    META = TB_10Meta

    AID_COL = 'aid'
    ATTRS = TB_10Meta.get_attrs()
    EXPORT = (TB_10Meta.NAMEDESC, TB_10Meta.APICODE, TB_10Meta.APIEXAM)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # TODO
    def _load_items(self, mode: str) -> pd.DataFrame:
        return pd.read_csv(
            filepath_or_buffer=f"data/{self.data_dir}_{mode}/tools.tsv",
            sep='\t',
            names=[self.AID_COL, *self.ATTRS],
            usecols=[self.AID_COL, *self.ATTRS],
        )

    def load_items(self) -> pd.DataFrame:
        train_df = self._load_items('train')
        dev_df = self._load_items('dev')

        return pd.concat([train_df, dev_df]).drop_duplicates([self.AID_COL])


class TB_50Meta(BaseMeta):
    CATEGORY = Attr('cat', 10)
    NAMEDESC = Attr('namedesc', 50)
    APICODE = Attr('acode', 256)
    APIEXAM = Attr('aexam', 100)

TB_50Meta.register_tasks(
    Task((TB_50Meta.NAMEDESC, TB_50Meta.APICODE, TB_50Meta.APIEXAM), TB_50Meta.CATEGORY),
    Task((TB_50Meta.NAMEDESC, TB_50Meta.APICODE, TB_50Meta.APIEXAM), TB_50Meta.NAMEDESC),
    Task((TB_50Meta.NAMEDESC, TB_50Meta.APICODE, TB_50Meta.APIEXAM), TB_50Meta.APICODE),
    Task((TB_50Meta.NAMEDESC, TB_50Meta.APICODE, TB_50Meta.APIEXAM), TB_50Meta.APIEXAM),
)

class TB_50Processor(BaseProcessor):
    META = TB_50Meta

    AID_COL = 'aid'
    ATTRS = TB_50Meta.get_attrs()
    EXPORT = (TB_50Meta.NAMEDESC, TB_50Meta.APICODE, TB_50Meta.APIEXAM)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # TODO
    def _load_items(self, mode: str) -> pd.DataFrame:
        return pd.read_csv(
            filepath_or_buffer=f"data/{self.data_dir}_{mode}/tools.tsv",
            sep='\t',
            names=[self.AID_COL, *self.ATTRS],
            usecols=[self.AID_COL, *self.ATTRS],
        )

    def load_items(self) -> pd.DataFrame:
        train_df = self._load_items('train')
        dev_df = self._load_items('dev')

        return pd.concat([train_df, dev_df]).drop_duplicates([self.AID_COL])