import copy
import hashlib
import json
import os
import numpy as np
import pigmento
import torch
from tqdm import tqdm
from typing import Type, cast
from oba import Obj
from peft import LoraConfig
from pigmento import pnt

from loader.class_hub import ClassHub
from loader.dataloader import DataLoader
from loader.dataset import Dataset
from loader.preparer import Preparer
from model.base_model import BaseModel
from process.base_processor import BaseProcessor
from utils.config_init import ConfigInit
from utils.function import seeding
from utils.gpu import GPU
from utils.monitor import Monitor


class Worker:
    def __init__(self, conf):
        self.conf = conf
        self.conf.model = self.conf.model.replace('.', '').lower()

        self.meta = self.get_meta()
        self.sign = self.get_signature()
        pnt(f'current worker signature: {self.sign}')

        if self.conf.warmup:  # Use this mode
            assert self.conf.num_code == 0, 'warmup mode does not support vector quantization'   # Vector quantization refers to VQ-VAE and RQ-VAE
            assert self.conf.layers is None, 'warmup mode does not support layer definition'
        else:
            assert self.conf.load is not None, 'vector quantization should be loaded from previous model'
            assert self.conf.layers is not None, 'hidden layers should be provided for vector quantization'
            self.conf.layers = list(map(int, cast(str, self.conf.layers).split('+')))

        self.device = self.get_device()
        self.device_ids = None
        if isinstance(self.device, tuple):
            self.device, self.device_ids = self.device

        self.processor = self.load_processor()  # type: BaseProcessor
        self.lora_config = LoraConfig(
            inference_mode=False,
            r=conf.lora_r,
            lora_alpha=conf.lora_alpha,
            lora_dropout=conf.lora_dropout
        )
        self.base_model = self.model = self.load_model()  # type: BaseModel

        self.log_dir = os.path.join('tuning', self.model.get_name())
        os.makedirs(self.log_dir, exist_ok=True)

        self.meta_path = os.path.join(self.log_dir, f'{self.sign}.json')
        self.log_path = os.path.join(self.log_dir, f'{self.sign}.log')
        json.dump(self.meta, open(self.meta_path, 'w'), indent=2)
        pigmento.add_log_plugin(self.log_path)

        if self.conf.load is not None:
            self.conf.load = str(self.conf.load)
            self.conf.load = os.path.join('tuning', self.model.get_name(), self.conf.load + '.json')
            self.load_meta = Obj(json.load(open(self.conf.load)))
            required_args = ['lora_r', 'lora_alpha', 'lora_dropout']
            for arg in required_args:
                assert arg in self.load_meta, f'{arg} is required in model loader configuration'
                assert self.load_meta[arg] == self.conf[arg], f'{arg} should be consistent with previous model'
            self.model.load(self.conf.load.replace('.json', '.pt'))

        self.model.to(self.device)
        if self.device_ids is not None:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)

        self.optimizer = torch.optim.AdamW(
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.conf.lr,
            weight_decay=self.conf.weight_decay,
        )
        self.monitor = Monitor(patience=self.conf.patience)

    def load_processor(self):
        processors = ClassHub.processors()
        if self.conf.data not in processors:
            raise ValueError(f'Unknown dataset: {self.conf.data}')
        processor_class = processors[self.conf.data]
        pnt(f'loading {processor_class.get_name()} processor')
        processor = processor_class(data_dir=self.conf.data)  # type: BaseProcessor
        processor.load_or_generate()
        return processor

    def load_model(self):
        models = ClassHub.models()
        if self.conf.model not in models:   # TODO The "model" parameter should be consistent with the name of the checkpoint used for the call.
            print("required models:", models)
            raise ValueError(f'Unknown model: {self.conf.model}')
        model_class = models[self.conf.model]  # type: Type[BaseModel]
        pnt(f'loading {model_class.get_name()} model')

        return model_class(
            num_gist=self.conf.num_gist,
            num_task=len(self.processor.META.get_tasks()),
            lora_config=self.lora_config,
            device=self.device,
            warmup=self.conf.warmup,
            num_code=self.conf.num_code,
            hidden_layers=Obj.raw(self.conf.layers),
        )

    def get_meta(self):
        conf = copy.deepcopy(Obj.raw(self.conf))
        del conf['gpu']
        del conf['mode']
        return conf

    def get_signature(self):
        keys = sorted(self.meta.keys())
        key = '-'.join([f'{k}={self.meta[k]}' for k in keys])
        md5 = hashlib.md5(key.encode()).hexdigest()
        return md5[:6]

    def get_device(self):
        if self.conf.gpu is None:
            return GPU.auto_choose(torch_format=True)
        if self.conf.gpu == -1:
            pnt('manually choosing CPU device')
            return 'cpu'

        pnt(f'manually choosing {self.conf.gpu}-th GPU')
        if isinstance(self.conf.gpu, int):
            return f'cuda:{self.conf.gpu}'
        gpus = list(map(int, self.conf.gpu.split('+')))
        return f'cuda:{gpus[0]}', gpus

    def list_tunable_parameters(self):
        pnt('tunable parameters:')
        for name, param in self.model.named_parameters():
            if self.conf.train_vq_only and 'quantizer' not in name:
                param.requires_grad = False
            if param.requires_grad:
                pnt(f'{name}: {param.size()}')

    def get_eval_interval(self, total_train_steps):
        if self.conf.eval_interval == 0:
            self.conf.eval_interval = -1

        if self.conf.eval_interval < 0:
            return total_train_steps // -self.conf.eval_interval

        return self.conf.eval_interval

    def evaluate(self, valid_dl, epoch):
        total_valid_steps = (len(valid_dl.dataset) + self.conf.batch_size - 1) // self.conf.batch_size

        loss_list = []
        # indices_set = set()   # TODO
        self.model.eval()
        with torch.no_grad():
            for index, batch in tqdm(enumerate(valid_dl), total=total_valid_steps):
                loss, q = self.model(batch)
                loss_list.append(loss.item())
                # codes = q.codes.cpu().detach().tolist()
                # for code in codes:
                #     indices_set.add(tuple(code))

        loss_avg = sum(loss_list) / len(loss_list)
        # collision_rate = (len(valid_dl.dataset) - len(list(indices_set))) * 1.0 / len(valid_dl.dataset)
        # pnt(f'(epoch {epoch}) validation loss: {loss_avg:.4f} with collision rate: {collision_rate:.4f}')
        pnt(f'(epoch {epoch}) validation loss: {loss_avg:.4f} with collision rate: not ok')

        self.model.train()

        action = self.monitor.push(loss_avg, minimize=True)
        if action is self.monitor.BEST:
            self.model.save(os.path.join(self.log_dir, f'{self.sign}.pt'))
            pnt(f'saving best model to {self.log_dir}/{self.sign}.pt')
        return action

    def train(self):
        preparer = Preparer(
            processor=self.processor,
            model=self.base_model,
            train_ratio=self.conf.train_ratio,
            sign=self.sign,
        )
        train_df, valid_df = preparer.load_or_generate_training_data()
        train_ds, valid_ds = Dataset(train_df), Dataset(valid_df)
        train_dl = DataLoader(dataset=train_ds, batch_size=self.conf.batch_size, shuffle=True)
        valid_dl = DataLoader(dataset=valid_ds, batch_size=self.conf.batch_size, shuffle=False)

        self.list_tunable_parameters()
        total_train_steps = (len(train_ds) + self.conf.batch_size - 1) // self.conf.batch_size

        eval_interval = self.get_eval_interval(total_train_steps)

        for epoch in range(100):    # TODO
            self.model.train()
            accumulate_step = 0
            self.optimizer.zero_grad()
            for index, batch in tqdm(enumerate(train_dl), total=total_train_steps):
                loss, q = self.model(batch)
                loss.backward()

                accumulate_step += 1
                if accumulate_step == self.conf.acc_batch:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    accumulate_step = 0

                if (index + 1) % eval_interval == 0:
                    action = self.evaluate(valid_dl, epoch)
                    if action is self.monitor.STOP:
                        pnt('early stopping')
                        return

    def export(self):   # This function is used to extract the dense vector embedding obtained from the first step of training.
        preparer = Preparer(
            processor=self.processor,
            model=self.base_model,
            train_ratio=1,
            sign=self.sign,
        )
        item_vocab = self.processor.item_vocab

        export_df = preparer.load_or_generate_export_data()
        export_ds = Dataset(export_df)
        export_dl = DataLoader(dataset=export_ds, batch_size=self.conf.batch_size, shuffle=False)

        self.model.eval()
        total_export_steps = (len(export_ds) + self.conf.batch_size - 1) // self.conf.batch_size
        embeds = dict()
        with torch.no_grad():
            for _, batch in tqdm(enumerate(export_dl), total=total_export_steps):
                gists = self.model(batch, get_gist_embeddings=True).float().cpu().detach().numpy()
                items = batch['item_id'].cpu().detach().tolist()    # In the preparation phase, the sample dict was renamed, with item_id corresponding to AID_COL.

                for index, item in enumerate(items):
                    embeds[item_vocab[item]] = gists[index]
        np.save(os.path.join(self.log_dir, f'{self.sign}.npy'), embeds, allow_pickle=True)
        pnt(f'saving embeddings to {self.log_dir}/{self.sign}.npy')

    def run(self):
        assert self.conf.mode in ['train', 'export']

        if self.conf.mode == 'train':
            return self.train()
        return self.export()


if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    seeding(2024)

    # The ConfigInit function is located in the utils folder. It integrates three types of *_args. In fact, it also calls argparse. Therefore, the items listed here are all the parameters that need to be provided when invoking the command line.
    configuration = ConfigInit(
        required_args=['model', 'data', 'batch_size', 'num_gist', 'warmup'],
        default_args=dict(
            num_code=0,
            gpu=None,
            valid_metric='GAUC',
            train_ratio=0.8,
            lora_r=32,
            lora_alpha=128,
            lora_dropout=0.1,
            acc_batch=1,
            eval_interval=0,
            patience=2,
            load=None,
            mode='train',
            weight_decay=1e-4,
            lr=1e-3,
            layers=None,
            train_vq_only=False,
        ),
        makedirs=[]
    ).parse()

    worker = Worker(configuration)
    worker.run()
