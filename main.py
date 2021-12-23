import argparse
from pprint import pprint
from ngrammodels import NGramAutoComplete

from transformermodels import BertAutoComplete

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='bert')
    parser.add_argument('--transformer_model_name', default='HooshvareLab/albert-fa-zwnj-base-v2')
    parser.add_argument('--from_file', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--train_data_path', default='./data/data.txt')
    parser.add_argument('--models_dir', default='./models_dir/')

    args = parser.parse_args()
    pprint(f'Arguments are:{vars(args)}') # For debugging
    if args.model == 'bert':
        autocomplete = BertAutoComplete(args)
    else:
        autocomplete = NGramAutoComplete(args)

    if args.train:
        autocomplete.train(args)
        print('Training Complete!')
    else:
        print('No Training')

    while True:
        print(autocomplete.complete(input()))
