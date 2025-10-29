from typing import cast

import torch
from transformers import OPTForCausalLM, LlamaForCausalLM

from model.base_model import BaseModel


class BaseLlamaModel(BaseModel):
    model: LlamaForCausalLM

    def _get_max_len(self) -> int:
        return 4_000

    def _get_token_embeddings(self) -> torch.nn.Embedding:
        return cast(torch.nn.Embedding, self.model.get_input_embeddings())

class Llama3Model(BaseLlamaModel):
    KEY = 'meta-llama/Llama-3-8b'
