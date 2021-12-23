
from typing import List

import torch
from transformers import AutoModelForMaskedLM  # Has MLM head
from transformers import AutoConfig, AutoTokenizer

from autocomplete import AutoComplete


class BertAutoComplete(AutoComplete):
    def __init__(self, args):
        '''
        Set from_file = "./models_dir/" if the model was saved in the models_dir directory before.
        '''
        super().__init__()

        self.MODEL_NAME = args.transformer_model_name
        self.DEVICE = args.device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.config = AutoConfig.from_pretrained(args.from_file or self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.from_file or self.MODEL_NAME)
        # TODO: .to(device) ? is it enough? how will loading and saving using different devices affect saving and loading
        self.model = AutoModelForMaskedLM.from_pretrained(
            args.from_file or self.MODEL_NAME).to(self.DEVICE)

    def load(self, dir_path: str = './models_dir/'):
        self.config = AutoConfig.from_pretrained(dir_path)
        self.tokenizer = AutoTokenizer.from_pretrained(dir_path)
        self.model = AutoModelForMaskedLM.from_pretrained(dir_path)

    def save(self, dir_path: str = './models_dir/'):
        self.config.save_pretrained(dir_path)
        self.tokenizer.save_pretrained(dir_path)
        self.model.save_pretrained(dir_path)

    def complete(self, sent: str):
        return sent

    def topk(self, sent: str, k: int = 10):
        mask_idx = self.tokenizer.tokenize(sent).index('[MASK]') + 1
        out = self.model(torch.IntTensor(self.tokenizer.encode(
            sent), device=self.DEVICE).unsqueeze(0))
        out = out['logits'].squeeze(0)[mask_idx, :].cpu()
        _, out = out.topk(k)
        return self.tokenizer.decode(out)

    def train(self, args):
        # TODO
        self.save(args.models_dir)
