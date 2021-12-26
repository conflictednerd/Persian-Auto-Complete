
import math
import os
from typing import Dict, List, Tuple
from multiprocess import set_start_method

import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForMaskedLM  # Has MLM head
from transformers import (AutoConfig, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

from autocomplete import AutoComplete


class BertAutoComplete(AutoComplete):
    def __init__(self, args):
        '''
        Set args.models_dir = "./models_dir/" if the model was saved in the models_dir directory before.
        '''
        super().__init__()
        # so that multi-processing with cuda is possible
        set_start_method('spawn')

        self.MODEL_NAME = args.transformer_model_name
        self.MODEL_DIR = args.models_dir
        self.TRAIN_DATA_PATH = args.train_data_path
        self.CLEANIFY = args.cleanify
        self.DEVICE = args.device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.config = AutoConfig.from_pretrained(
            self.MODEL_DIR if args.load else self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_DIR if args.load else self.MODEL_NAME)
        # TODO: .to(device) ? is it enough? how will loading and saving using different devices affect saving and loading
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.MODEL_DIR if args.load else self.MODEL_NAME).to(self.DEVICE)

    def load(self, dir_path: str = None):
        dir_path = dir_path or self.MODEL_DIR
        self.config = AutoConfig.from_pretrained(dir_path)
        self.tokenizer = AutoTokenizer.from_pretrained(dir_path)
        self.model = AutoModelForMaskedLM.from_pretrained(dir_path)

    def save(self, dir_path: str = None):
        dir_path = dir_path or self.MODEL_DIR
        self.config.save_pretrained(dir_path)
        self.tokenizer.save_pretrained(dir_path)
        self.model.save_pretrained(dir_path)

    def complete(self, sent: str, num_suggestions: int = 5) -> List[str]:
        '''
        The sentence must contain exactly one incomplete word determined using three consecutive dots(...)
        Examples:
        "This sentence should be com..."
        "این جمله باید کامل ش..."
        '''
        # Find the incomplete word and replace it with [MASK]
        incomplete_word = ''
        words = sent.split(' ')
        for word in words:
            if '...' in word:
                incomplete_word = word[:-3]
                break
        words = ['[MASK]' if '...' in word else word for word in words]
        sent = ' '.join(words)
        suggestions = self.topk(sent, k=200)
        suggestions = [(word, score, self.prefix_distance(
            word, incomplete_word)) for word, score in suggestions]
        # Sort by prefix distance first and then score
        suggestions.sort(key=lambda x: x[1] - 1000000*x[2], reverse=True)
        suggestions = suggestions[:num_suggestions]
        suggestions = [word for word, score, d in suggestions]
        return suggestions

    def topk(self, sent: str, k: int = 10) -> List:
        mask_idx = self.tokenizer.tokenize(sent).index('[MASK]') + 1
        out = self.model(torch.tensor(self.tokenizer.encode(sent)).int().to(self.DEVICE).unsqueeze(0))
        out = out['logits'].squeeze(0)[mask_idx, :].cpu()
        scores, tokens = out.topk(k)
        # TODO: check output type(tensor?)
        return [(self.tokenizer.decode(tokens[i]), scores[i]) for i in range(len(scores))]

    def create_dataset(self, args):
        '''
        This function will use the data provided in "data.txt" located at self.TRAIN_DATA_PATH to create a dataset for the "masked language modeling" task.
        This dataset will then be used by the train function to fine-tune the model.
        '''
        print('Creating dataset for training...')
        s = ''
        with open(os.path.join(self.TRAIN_DATA_PATH, 'data.txt'), 'r', encoding='utf-8') as f:
            s += self.clean(f.read()) if self.CLEANIFY else f.read()
        parags = s.split('\n')
        train, test = train_test_split(parags, test_size=args.test_size)

        # Train and test data after the split will be saved for further analysis in future
        with open(os.path.join(self.TRAIN_DATA_PATH, 'train.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(train))

        with open(os.path.join(self.TRAIN_DATA_PATH, 'test.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(test))

        dataset = load_dataset('text', data_files={
                               'train': os.path.join(self.TRAIN_DATA_PATH, 'train.txt'), 'test': os.path.join(self.TRAIN_DATA_PATH, 'test.txt')})
        # This will tokenize our dataset
        tokenized_dataset = dataset.map(lambda x: self.tokenizer(
            x['text']), batched=True, num_proc=args.num_proc, remove_columns=['text'])
        ''' TODO: REMOVE THIS?
        tokenized_dataset is now like a dict with three keys: "attention_mask", "input_ids", and "token_type_ids".
        We will mainly work with input_ids which is a list of list of ids. E.g.,
        {'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1]],
         'input_ids': [[2, 2458, 28016, 17861, 23497, 1203, 24578, 2049, 4],
               [2, 2046, 2458, 7694, 2083, 16411, 2678, 2049, 4],
               [2, 316, 2166, 4723, 15009, 89094, 36793, 4],
               [2, 2046, 3327, 11350, 2030, 77946, 2614, 2049, 4],
               [2, 30721, 29932, 331, 2166, 2036, 35224, 77080, 4]],
         'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]]}
        '''
        # We should now create sequences of equal length from our dataset and add labels
        # For this, we will use the following function:
        seq_len = args.seq_len

        def f(examples: Dict) -> Dict:
            # Concatenate all the tokenized paragraphs along with their attention masks and etc.
            concat_examples = {k: sum(examples[k], [])
                               for k in examples.keys()}
            # See how many sequences of seq_len length we can make
            num_sequences = len(
                concat_examples[list(examples.keys())[0]]) // seq_len
            # Split into num_sequences chunks of length seq_len
            result = {
                k: [t[i*seq_len: (i+1)*seq_len] for i in range(num_sequences)]
                for k, t in concat_examples.items()
            }
            # For MLM task, labels are the same as inputs
            result['labels'] = result['input_ids'].copy()
            return result

        mlm_dataset = tokenized_dataset.map(
            f, batched=True, batch_size=1000, num_proc=args.num_proc)
        print('Dataset creation is done.')
        return mlm_dataset

    def train(self, args):
        print('Training the model has started...')
        mlm_dataset = self.create_dataset(args)
        training_args = TrainingArguments(
            f'{self.MODEL_NAME}-finetuned',
            evaluation_strategy='epoch',
            num_train_epochs=args.ft_epochs,
            per_device_train_batch_size=args.ft_batch_size,
            learning_rate=args.ft_lr,
            weight_decay=args.ft_wd,
            logging_dir=args.logging_dir,
        )
        print(f'Training will be done on {training_args.device}.')
        # We also need a data_collator to randomly mask some words for the MLM task.
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm_probability=args.ft_mlm_prob)
        # Now we just have to pass everything to the trainer and start training
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=mlm_dataset['train'],
            eval_dataset=mlm_dataset['test'],
            data_collator=data_collator,
        )
        eval_results = trainer.evaluate()
        print(
            f"Perplexity before training: {math.exp(eval_results['eval_loss']):.2f}")
        trainer.train()
        eval_results = trainer.evaluate()
        print(
            f"Perplexity after training: {math.exp(eval_results['eval_loss']):.2f}")
        print('Training is finished.')

        self.save(args.models_dir)
