from typing import cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.base_model import BaseModel

class BaseQwenModel(BaseModel):
    model: AutoModelForCausalLM

    def _get_max_len(self) -> int:
        return 4_000

    def _get_token_embeddings(self) -> torch.nn.Embedding:
        return cast(torch.nn.Embedding, self.model.get_input_embeddings())

class Qwen14Model(BaseQwenModel):
    KEY = 'Qwen/Qwen2-14B'

class Qwen7Model(BaseQwenModel):
    KEY = 'Qwen/Qwen2-7B'

class Qwen3Model(BaseQwenModel):
    KEY = 'Qwen/Qwen2-3B'
    
class Qwen15Model(BaseQwenModel):
    KEY = 'Qwen/Qwen2-1.5B'

class Qwen05Model(BaseQwenModel):
    KEY = 'Qwen/Qwen2-0.5B'