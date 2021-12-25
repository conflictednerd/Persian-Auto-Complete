import argparse
from pprint import pprint

from ngrammodels import NGramAutoComplete
from transformermodels import BertAutoComplete

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='bert',
                        help='type of autocomplete model: bert or ngram')
    parser.add_argument('--transformer_model_name',
                        default='HooshvareLab/albert-fa-zwnj-base-v2')
    parser.add_argument('--load', action='store_true', default=False,
                        help='use if you want to load the model; otherwise the model will be downloaded')
    parser.add_argument('--device', default=None)
    parser.add_argument('--train', action='store_true', default=False,
                        help='use if you want to fine-tune the model')
    parser.add_argument('--train_data_path', default='./data/',
                        help='directory where training data resides under the name "data.txt"')
    parser.add_argument('--cleanify', action='store_true',
                        default=False, help='use if training data needs cleaning')
    parser.add_argument('--models_dir', default='./models_dir/',
                        help='directory used for loading and saving models')
    parser.add_argument('--logging_dir', default='./logs',
                        help='directory used for logging fine-tunning process')
    parser.add_argument('--test_size', default=.1,
                        help='portion of data that should be used for testing (validation during fine-tuning)')
    parser.add_argument('--num_proc', default=4,
                        help='number of processes (for now only used in tokenizing dataset)')
    parser.add_argument('--seq_len', default=128,
                        help='length of each sequence in a batch used in training the model')
    parser.add_argument('--ft_epochs', default=3,
                        help='number of epochs of fine-tunning')
    parser.add_argument('--ft_batch_size', default=16,
                        help='batch-size used in fine-tunning')  # Huggingface default was 8
    parser.add_argument('--ft_lr', default=2e-5,
                        help='learning rate used in fine-tunning')
    parser.add_argument('--ft_wd', default=1e-2,
                        help='weight decay coefficient used in fine-tunning')
    parser.add_argument('--ft_mlm_prob', default=0.15,
                        help='probability of masking a word during fine-tunning')

    args = parser.parse_args()
    pprint(f'Arguments are:{vars(args)}')  # For debugging
    if args.model == 'bert':
        autocomplete = BertAutoComplete(args)
    else:
        autocomplete = NGramAutoComplete(args)

    if args.train:
        autocomplete.train(args)
    else:
        print('No Training')

    while True:
        print(autocomplete.complete(input()))
