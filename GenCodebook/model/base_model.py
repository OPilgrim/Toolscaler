import abc
from typing import cast, Optional

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from loader.vocab import Vocab
from utils.auth import HF_KEY


class BaseModel(nn.Module):
    KEY = None

    def __init__(
            self,
            num_gist: int,
            num_task: int,
            lora_config: LoraConfig,
            device,

            warmup: bool,
            num_code: int,
            hidden_layers: list,
    ):
        super().__init__()

        if self.KEY is None:
            raise ValueError("KEY attribute must be set for the model")

        self.warmup = warmup

        if self.KEY == "meta-llama/Llama-2-7b-hf":
            self.model = AutoModelForCausalLM.from_pretrained(
                "/data1/suyunyue/models/Llama-2-7b-hf",
                token=HF_KEY,
                torch_dtype=torch.bfloat16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "/data1/suyunyue/models/Llama-2-7b-hf",
                token=HF_KEY
            )
        elif self.KEY == "meta-llama/Llama-3-8b":
            self.model = AutoModelForCausalLM.from_pretrained(
                "models/llama-3-8b",
                token=HF_KEY,
                torch_dtype=torch.bfloat16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "models/llama-3-8b",
                token=HF_KEY
            )
        elif self.KEY == "Qwen/Qwen2-14B":
            self.model = AutoModelForCausalLM.from_pretrained(
                "models/qwen2.5-14B",
                token=HF_KEY,
                torch_dtype=torch.bfloat16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "models/qwen2.5-14B",
                token=HF_KEY
            )
        elif self.KEY == "Qwen/Qwen2-7B":
            self.model = AutoModelForCausalLM.from_pretrained(
                "models/qwen2.5-7B",
                token=HF_KEY,
                torch_dtype=torch.bfloat16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "models/qwen2.5-7B",
                token=HF_KEY
            )
        elif self.KEY == "Qwen/Qwen2-3B":
            self.model = AutoModelForCausalLM.from_pretrained(
                "models/qwen2.5-3B",
                token=HF_KEY,
                torch_dtype=torch.bfloat16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "models/qwen2.5-3B",
                token=HF_KEY
            )
        elif self.KEY == "Qwen/Qwen2-1.5B":
            self.model = AutoModelForCausalLM.from_pretrained(
                "models/qwen2.5-1.5B",
                token=HF_KEY,
                torch_dtype=torch.bfloat16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "models/qwen2.5-1.5B",
                token=HF_KEY
            )
        elif self.KEY == "Qwen/Qwen2-0.5B":
            self.model = AutoModelForCausalLM.from_pretrained(
                "models/qwen2.5-0.5B",
                token=HF_KEY,
                torch_dtype=torch.bfloat16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "models/qwen2.5-0.5B",
                token=HF_KEY
            )
        else:
            raise NotImplementedError

        self.encoder = get_peft_model(self.model, lora_config)
        self.decoder = get_peft_model(self.model, lora_config)

        self.num_gist = num_gist
        self.num_task = num_task
        self.num_code = num_code

        self.max_len = self._get_max_len()
        self.tkn_embeddings = self._get_token_embeddings()   # content block
        self.gst_embeddings = nn.Embedding(self.num_gist, self.tkn_embeddings.embedding_dim, dtype=torch.bfloat16)  # token block
        self.tsk_embeddings = nn.Embedding(self.num_task, self.tkn_embeddings.embedding_dim, dtype=torch.bfloat16)  # task block
        self.spc_embeddings = nn.Embedding(self.num_gist, self.tkn_embeddings.embedding_dim, dtype=torch.bfloat16)  # placeholder block
        self.eos_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        
        try:
            self.sep_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        except:
            print("Warning: SEP token not found in tokenizer, using default comma token.")
            self.sep_token = None
            
        if self.sep_token is None:
            self.sep_token = self.tokenizer.convert_tokens_to_ids([','])[0]

        self.device = device
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        self.index = 0

        # initialization
        self.gst_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.tsk_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.spc_embeddings.weight.data.uniform_(-0.1, 0.1)

    @classmethod
    def get_name(cls):
        return cls.__name__.replace('Model', '')

    @abc.abstractmethod
    def _get_max_len(self) -> int:
        pass

    @abc.abstractmethod
    def _get_token_embeddings(self) -> torch.nn.Embedding:
        pass

    def generate_input_ids(self, content) -> torch.Tensor:
        return self.tokenizer.encode(content, return_tensors=None, add_special_tokens=False)

    def save(self, path):
        encoder_state_dict = dict()
        for k, v in self.encoder.state_dict().items():
            if 'lora' in k:
                encoder_state_dict[k] = v
        # torch.save(encoder_state_dict, encoder_path)

        decoder_state_dict = dict()
        for k, v in self.decoder.state_dict().items():
            if 'lora' in k:
                decoder_state_dict[k] = v
        # torch.save(decoder_state_dict, decoder_path)

        embeddings = dict(
            tkn_embeddings=self.tkn_embeddings.state_dict(),
            gst_embeddings=self.gst_embeddings.state_dict(),
            tsk_embeddings=self.tsk_embeddings.state_dict(),
            spc_embeddings=self.spc_embeddings.state_dict(),
        )

        state_dict = dict(
            encoder=encoder_state_dict,
            decoder=decoder_state_dict,
            embeddings=embeddings,
        )

        torch.save(state_dict, path)

    def load(self, path):
        state_dict = torch.load(path, map_location='cpu')

        self.encoder.load_state_dict(state_dict['encoder'], strict=False)
        self.decoder.load_state_dict(state_dict['decoder'], strict=False)
        self.gst_embeddings.load_state_dict(state_dict['embeddings']['gst_embeddings'])
        self.tsk_embeddings.load_state_dict(state_dict['embeddings']['tsk_embeddings'])
        self.spc_embeddings.load_state_dict(state_dict['embeddings']['spc_embeddings'])

    def _get_input_embeddings(self, input_ids: torch.Tensor, input_vocabs: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch_size, seq_len)
        # input_vocabs: (batch_size, seq_len), 0 for PAD, 1 for LM, 2 for GIST

        llm_tokens = cast(torch.Tensor, input_vocabs == Vocab.LLM)
        gst_tokens = cast(torch.Tensor, input_vocabs == Vocab.GST)
        spc_tokens = cast(torch.Tensor, input_vocabs == Vocab.SPC)

        llm_input_ids = input_ids * llm_tokens
        gst_input_ids = input_ids * gst_tokens
        spc_input_ids = input_ids * spc_tokens

        llm_embeddings = self.tkn_embeddings(llm_input_ids) * llm_tokens.unsqueeze(-1)
        gst_embeddings = self.gst_embeddings(gst_input_ids) * gst_tokens.unsqueeze(-1)
        spc_embeddings = self.spc_embeddings(spc_input_ids) * spc_tokens.unsqueeze(-1)

        return llm_embeddings + gst_embeddings + spc_embeddings

    def encode(self, batch: dict):
        # encoder_input_ids: [TASK, A, B, C, <N>, <G>, <G>, <G>]
        encoder_input_ids = batch['encoder_input_ids'].to(self.device)
        encoder_input_vocabs = batch['encoder_input_vocabs'].to(self.device)
        encoder_attention_mask = batch['encoder_attention_mask'].to(self.device)

        encoder_input_embeds = self._get_input_embeddings(encoder_input_ids, encoder_input_vocabs)
        output = self.encoder(
            inputs_embeds=encoder_input_embeds,
            attention_mask=encoder_attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        return output.hidden_states[-1], output.logits

    def get_gist_embeddings(self, batch: dict, last_hidden_states: torch.Tensor):
        gist_lengths = batch['gist_lengths'].to(self.device)  # [B]
        gist_positions = batch['gist_positions'].to(self.device)  # [B]
        batch_size = last_hidden_states.size(0)

        max_length = gist_lengths.max()
        indices = torch.arange(max_length).unsqueeze(0).expand(batch_size, -1).to(self.device)
        mask = cast(torch.Tensor, indices < gist_lengths.unsqueeze(1))
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, max_length).to(self.device)
        seq_indices = gist_positions.unsqueeze(1) + indices

        gist_embeddings = last_hidden_states[batch_indices, seq_indices] * mask.unsqueeze(2)
        return gist_embeddings, mask

    def build_weight_matrix(self, batch):
        decoder_attention_mask = batch['decoder_attention_mask'].to(self.device)
        gist_lengths = batch['gist_lengths'].to(self.device)  # [B]
        batch_size = gist_lengths.size(0)
        seq_length = decoder_attention_mask.size(1)

        decoder_lengths = decoder_attention_mask.sum(dim=1) - 2 - gist_lengths
        start_positions = gist_lengths + 2

        indices = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1).to(self.device)
        start_mask = cast(torch.Tensor, indices >= start_positions.unsqueeze(1))
        end_mask = cast(torch.Tensor, indices < (start_positions + decoder_lengths).unsqueeze(1))
        mask = start_mask & end_mask

        weights = torch.zeros(batch_size, seq_length, device=self.device)
        relative_positions = (indices - start_positions.unsqueeze(1)).clamp(min=0)
        decay_values = torch.exp(torch.linspace(0, -2, seq_length, device=self.device))
        weights[mask] = decay_values[relative_positions[mask]]
        return weights

    def decode(self, batch: dict, latent_embeds):
        # decoder_input_ids: [TASK, <N>, <G>, <G>, <G>, D, E, F]
        gist_lengths = batch['gist_lengths'].to(self.device)  # [B]
        batch_size = gist_lengths.size(0)
        max_length = gist_lengths.max()
        embedding_dim = self.tkn_embeddings.embedding_dim

        indices = torch.arange(max_length).unsqueeze(0).expand(batch_size, -1).to(self.device)
        mask = cast(torch.Tensor, indices < gist_lengths.unsqueeze(1))

        decoder_input_ids = batch['decoder_input_ids'].to(self.device)
        decoder_input_vocabs = batch['decoder_input_vocabs'].to(self.device)
        decoder_attention_mask = batch['decoder_attention_mask'].to(self.device)

        decoder_input_embeds = self._get_input_embeddings(decoder_input_ids, decoder_input_vocabs)

        mask_expanded = mask.unsqueeze(2).expand(-1, -1, embedding_dim)
        decoder_input_embeds[:, 1: max_length + 1, :] = decoder_input_embeds[:, 1: max_length + 1, :] * ~mask_expanded + latent_embeds

        output = self.decoder(
            inputs_embeds=decoder_input_embeds,
            attention_mask=decoder_attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        return output.logits

    def forward(self, batch: dict, get_gist_embeddings=False, decoder_prediction_only=True):
        loss = torch.tensor(0.0, device=self.device)
        last_hidden_states, encoder_logits = self.encode(batch)
        if not decoder_prediction_only:
            encoder_labels = batch['encoder_labels'].to(self.device)
            loss += self.loss_fct(encoder_logits.view(-1, encoder_logits.size(-1)), encoder_labels.view(-1)).mean()

        gist_embeddings, gist_mask = self.get_gist_embeddings(batch, last_hidden_states)

        quantizer_output = None
        
        self.index += 1

        if get_gist_embeddings:
            return gist_embeddings

        logits = self.decode(batch, gist_embeddings)
        decoder_labels = batch['decoder_labels'].to(self.device)

        loss += self.loss_fct(logits.view(-1, logits.size(-1)), decoder_labels.view(-1)).mean()

        if quantizer_output:
            loss += quantizer_output.loss

        return loss, quantizer_output
