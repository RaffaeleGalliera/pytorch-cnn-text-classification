from torchtext import data
import torch
import gensim
import torchtext.vocab as vocab
import inquirer
import os
import json
import string


def load_dataset():
    if not (os.path.exists('dataset_folder/test')
            and os.path.exists('dataset_folder/train')
            and os.path.exists('dataset_folder/validation')):
        print('Dataset not available for JSON exporting, please extract dataset_folder to the root folder'
              'dataset_folder/test, dataset_folder/train, dataset_folder/validation are required')
        exit()

    dict1 = {}

    if not os.path.exists('data'):
        os.makedirs('data')

    dict_fields = ['text', 'label']
    data_root = 'dataset_folder'

    for root_folder in os.listdir(data_root):
        data_folder = os.path.join(data_root, root_folder)
        for test in os.listdir(data_folder):
            folder = os.path.join(data_folder, test)
            for filename in os.listdir(folder):
                file = os.path.join(folder, filename)
                with open(file) as fh:
                    description = fh.read()

                    # punctuations = '''!()-[]{};:'"\,<>/@#$%^&*_~'''
                    for x in description.lower():
                        if x in string.punctuation:
                            description = description.replace(x, "")

                    dict1[dict_fields[0]] = description.strip().split()
                    dict1[dict_fields[1]] = test

                    out_file = open(f'data/{root_folder}.json', "a+")
                    json.dump(dict1, out_file, ensure_ascii=False)
                    out_file.write('\n')
                    out_file.close()


if not (os.path.isfile('data/test.json')
        and os.path.isfile('data/test.json')
        and os.path.isfile('data/test.json')):
    print("Not found test, training and validation data in proper 'data' directory")
    quest = [
        inquirer.Confirm('continue',
                         message="Would you like to load them from 'dataset_folder'?", default=True),
    ]

    answers = inquirer.prompt(quest)

    if answers.get('continue'):
        load_dataset()
        print('Exporting dataset to JSON and loading vocab Vectors...')

    else:
        exit()

torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy', tokenizer_language='it_core_news_sm', batch_first=True)
LABEL = data.LabelField()
fields = {'text': ('text', TEXT), 'label': ('label', LABEL)}

train_data, valid_data, test_data = data.TabularDataset.splits(
    path='data',
    train='train.json',
    validation='validation.json',
    test='test.json',
    format='json',
    fields=fields
)

MAX_VOCAB_SIZE = 25_000

# Necessary code for Glove models to be convert to w2c
# model = gensim.models.Word2Vec.load('home/berardi/glove_WIKI')
# model.wv.save_word2vec_format('vector_cache/glove_WIKI_w2v/wv_cnr_ITA')
# vectors = vocab.Vectors(name='wv_cnr_ITA', cache='vector_cache/glove_WIKI_w2v/')

vectors = vocab.Vectors(name='model.txt', cache='vector_cache/word2vec_CoNLL17')

TEXT.build_vocab(train_data,
                 max_size=MAX_VOCAB_SIZE,
                 vectors=vectors,
                 unk_init=torch.Tensor.normal_)

LABEL.build_vocab(train_data)


def print_dataset_details():
    print(f'Dataset details: ')
    print(f'    Number of training examples: {len(train_data)}')
    print(f'    Number of testing examples: {len(test_data)}')
    print(f'    Number of validation examples: {len(valid_data)}')


def dataset_info():
    print(f'Number of labels {len(LABEL.vocab)}')
    for label in LABEL.vocab.itos:
        c = 0
        k = 0
        z = 0

        print(f'Examples of {label}:')
        for train in train_data.examples:
            if train.label == label:
                c = c + 1
        for test in test_data.examples:
            if test.label == label:
                k = k + 1
        for validation in valid_data.examples:
            if validation.label == label:
                z = z + 1
        print(f'    Training: {c}')
        print(f'    Testing: {k}')
        print(f'    Validation: {z}')
