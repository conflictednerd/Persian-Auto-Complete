from __future__ import unicode_literals
from collections import Counter
import numpy as np
import pickle
import math
import nltk
import itertools
import os
import string
import re
from typing import Dict, List, Tuple
from hazm import *
import random
from autocomplete import AutoComplete
from sklearn.model_selection import train_test_split


class NGramAutoComplete(AutoComplete):
    UNK = "<UNK>"
    SOS = "<S>"
    EOS = "</S>"

    def __init__(
            self,
            args,
    ):
        '''
        Set from_file = "./models_dir/" if the model was saved in the models_dir directory before.
        '''
        super().__init__()

        # so that multi-processing with cuda is possible
        # set_start_method('spawn')

        self.MODEL_NAME = 'ngram'
        self.MODEL_DIR = args.models_dir
        self.TRAIN_DATA_PATH = args.train_data_path
        self.CLEANIFY = args.cleanify
        self.k = args.k
        self.n = args.n
        self.mode = args.ngram_mode
        self.vocab = dict()
        if not args.load_data:
            self.create_dataset(args)
        else:
            self.load_prepared_data()

        self.unigrams = self.create_vocab(obj='train')
        self.vocab = nltk.FreqDist(self.unigrams)
        self.model = self.train() if args.train else self.load()  # if we don't want to train, we want to load the model, right?

        if args.train:
            self.save()

    def load_prepared_data(self):
        s = ''
        with open(os.path.join(self.TRAIN_DATA_PATH, 'train.txt'), 'r', encoding='utf-8') as f:
            s += f.read()
        self.train_dataset = self.init_cleaning(s.split('\n'))
        # print(self.train_dataset[0])
        # print(word_tokenize(self.train_dataset[0]))

        s = ''
        with open(os.path.join(self.TRAIN_DATA_PATH, 'test.txt'), 'r', encoding='utf-8') as f:
            s += f.read()
        self.test_dataset = self.init_cleaning(s.split('\n'))
        print(self.test_dataset[0].split(' ')[:-1])
        # print(self.test_dataset[0])
        # print(self.create_test(self.test_dataset[0]))

    def sentence_padding(self, sentences):
        sos = ' '.join([NGramAutoComplete.SOS] * (self.n - 1)) if self.n > 1 else NGramAutoComplete.SOS
        return ['{} {} {}'.format(sos, s, NGramAutoComplete.EOS) for s in sentences]

    def init_cleaning(self, data):
        final_data = []
        for datum in data:
            for sen in sent_tokenize(datum):
                final_data.append(sen)
        return self.sentence_padding(final_data)

    def create_test(self, sent):
        words_init = sent.split(' ')
        sen_len = len(words_init)
        word_idx = random.randint(1, sen_len) - 1
        while len(words_init[word_idx]) <= 1:
            word_idx = random.randint(1, sen_len) - 1
        char_idx = random.randint(2, len(words_init[word_idx])) - 1
        reconstructed_sent = ''.join(x + ' ' for x in words_init[:word_idx + 1])
        reconstructed_sent += words_init[word_idx][:char_idx]
        return reconstructed_sent

    def create_dataset(self, args):
        print('Creating dataset for training...')
        s = ''
        with open(os.path.join(self.TRAIN_DATA_PATH, 'data.txt'), 'r', encoding='utf-8') as f:
            s += self.clean(f.read()) if self.CLEANIFY else f.read()
        parags = s.split('\n')
        print('done reading and normalizing!')
        parags = random.sample(parags, int(len(parags) / 10))
        train_init, test_init = train_test_split(parags, test_size=args.test_size)


        with open(os.path.join(self.TRAIN_DATA_PATH, 'train.txt'), 'w', encoding='utf-8') as f:
            for i in range(len(train_init)):
                f.write(train_init[i] + '\n')
        #
        with open(os.path.join(self.TRAIN_DATA_PATH, 'test.txt'), 'w', encoding='utf-8') as f:
            for i in range(len(test_init)):
                f.write(self.create_test(test_init[i]) + '\n')

        self.train_dataset = self.init_cleaning(train_init)
        self.test_dataset = self.init_cleaning(test_init)
        print(self.test_dataset[0])
        # print(train[1])

        # with open(os.path.join(self.TRAIN_DATA_PATH, 'train_ngram.txt'), 'w', encoding='utf-8') as f:
        #     for i in range(len(train)):
        #         f.write(train[i] + '\n')
        # #
        # with open(os.path.join(self.TRAIN_DATA_PATH, 'test_ngram.txt'), 'w', encoding='utf-8') as f:
        #     for i in range(len(test)):
        #         f.write(test[i] + '\n')

    def create_vocab(self, obj='train'):
        tokens = []
        if obj == 'train':
            tokens = [word_tokenize(para) for para in self.train_dataset]
        else:
            tokens = [word_tokenize(para) for para in self.test_dataset]
        unigrams = []
        for sen in tokens:
            for w in sen:
                unigrams.append(w)
        self.vocab = nltk.FreqDist(unigrams)
        return [token if self.vocab[token] > 1 else NGramAutoComplete.UNK for token in unigrams]

    def smoothed(self, m):

        vocab_size = len(self.vocab)
        m_grams = nltk.ngrams(self.unigrams, m)  # longer
        m_vocab = nltk.FreqDist(m_grams)

        p_grams = nltk.ngrams(self.unigrams, m - 1)  # shorter, to condition on
        p_vocab = nltk.FreqDist(p_grams)

        def smoothed_count(m_gram, m_count):
            p_gram = m_gram[:-1]
            p_count = p_vocab[p_gram]
            return (m_count + self.k) / (p_count + self.k * vocab_size)

        return {n_gram: smoothed_count(n_gram, count) for n_gram, count in m_vocab.items()}

    def load(self, dir_path: str = './models_dir/', name='ngram_model.pkl'):
        with open(dir_path + name, 'rb') as f:
            return pickle.load(f)

    def save(self, dir_path: str = './models_dir/', name='ngram_model.pkl'):
        with open(dir_path + name, 'wb') as f:
            pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)

    def complete(self, sent: str, num_suggestions: int = 5) -> List[str]:
        incomplete_word = ''
        words = sent.split(' ')
        for word in words:
            if '...' in word:
                incomplete_word = word[:-3]
                break
        words = ['' if '...' in word else word for word in words]
        sent = ' '.join(words)
        suggestions = self.topk(sent, k_=200)
        suggestions = [(word, score, self.prefix_distance(
            word, incomplete_word)) for word, score in suggestions]
        # Sort by prefix distance first and then score
        suggestions.sort(key=lambda x: x[1] - 1000000 * x[2], reverse=True)
        suggestions = suggestions[:num_suggestions]
        suggestions = [word for word, score, d in suggestions]
        return suggestions

    def topk(self, sent: str, k_: int = 10):
        sent = [NGramAutoComplete.SOS] * (max(1, self.n - 1)) + word_tokenize(sent)
        polished_sent = []
        for w in sent:
            if w in self.unigrams:
                polished_sent.append(w)
            else:
                polished_sent.append("<UNK>")
        sen_len = len(polished_sent)
        if self.mode == 'bo':
            found = False
            m = self.n
            while not found and m >= 1:
                candidates = list(((mgram[-1], prob) for mgram, prob in self.model[m].items() if
                                   mgram[:-1] == tuple(polished_sent[sen_len - m: sen_len - 1])))
                if len(candidates) == 0:
                    m -= 1
                    continue
                else:
                    found = True
                    words = [x for (x, y) in candidates]
                    for ill in ["<UNK>", "<S>", "</S>"]:
                        if ill in words:
                            words.remove(ill)
                    probs = np.array([y for (x, y) in candidates])
                    probs = probs / np.sum(probs)
                    idx = (-probs).argsort()[:min(k_, len(words))]
                    k_words = [words[int(x)] for x in idx]
                    k_probs = probs[idx.astype(int)]
                    return [(k_words[i], k_probs[i]) for i in range(len(k_words))]
        else:
            scores = {w: 0 for w in self.unigrams}
            # print(len(self.unigrams))
            # print(self.unigrams[:30])
            # print('wwww')
            for m in reversed(range(1, self.n + 1)):
                rel_score = math.pow(2, m)
                candidates = list(((mgram[-1], prob) for mgram, prob in self.model[m].items() if
                                   mgram[:-1] == tuple(polished_sent[sen_len - m: sen_len - 1])))
                if len(candidates) == 0:
                    continue
                else:
                    words = [x for (x, y) in candidates]
                    probs = np.array([y for (x, y) in candidates])
                    probs = probs / np.sum(probs)
                    for i in range(len(words)):
                        scores[words[i]] = scores[words[i]] + probs[i] * rel_score
            words = list(scores.keys())
            for ill in ["<UNK>", "<S>", "</S>"]:
                if ill in words:
                    words.remove(ill)

            probs = np.array([scores[w] for w in words])
            probs = probs / np.sum(probs)
            idx = (-probs).argsort()[:k_]
            k_words = [words[int(x)] for x in idx]
            k_probs = probs[idx.astype(int)]
            return [(k_words[i], k_probs[i]) for i in range(len(k_words))]

    def train(self):
        print(Counter(self.unigrams).most_common(20))
        num_tokens = len(self.unigrams)
        unigram_dict = {(unigram,): count / num_tokens for unigram, count in self.vocab.items()}
        all_dicts = {1: unigram_dict}

        for i in range(self.n - 1):
            all_dicts[i + 2] = self.smoothed(i + 2)
            print(i + 2)
            print("##########")
        print('done training!')
        # print(Counter(self.unigrams).most_common(20))
        return all_dicts

    def evaluate(self):
        print("evaluating model on test data...")
        in_suggestions = 0

        for datum in self.test_dataset:
            # print('hi')
            # first_ = self.sentence_padding([datum])[0]
            test_tokens = datum.split(' ')[:-1]  # dropping EOS
            unfinished_word = test_tokens[
                                  -1] + '...'  # last word is unfinished, the word before that is the finished word (ground truth)
            last_word = test_tokens[
                -2]  ## for example: <S> <S> As I was moving ahead occasionally I saw brief glimpses of beauty bea -> beauty is our last word, bea is passed for testing
            test_tokens[-2] = unfinished_word
            reconstructed_sent = ' '.join(test_tokens[:-1])
            in_suggestions += last_word in self.complete(reconstructed_sent)

        return in_suggestions / len(self.test_dataset)
