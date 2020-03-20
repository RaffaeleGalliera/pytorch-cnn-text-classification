import dataset as dataset
import model as m
import torch
import torch.optim as optim
import torch.nn as nn
import train
import spacy
import inquirer
import os
import sys
import nlpaug.augmenter.word as naw
from tensorboardX import SummaryWriter
from torchsummary import summary
from torchtext import data

sys.path.append(os.path.realpath('.'))

dataset.print_dataset_details()

if os.path.isfile('ezmath-model.pt'):
    print('Model found: Loading ezmath-model.pt')
    choices = ['Evaluate model',
               'Evaluate and plot PR curves - TensorboardX',
               'More info about dataset',
               'Make Prediction',
               'Plot embedding space - TensorboardX',
               'Print model details',
               'Test Textual Augmenter by word2vec similarity',
               'Train model',
               'Exit']
else:
    print('ezmath-model.pt not found, run the training first and then restart.')
    choices = ['Train model',
               'Exit']

questions = [
    inquirer.List('choice',
                  message="What would you like to do?",
                  choices=choices,
                  ),
]

answers = inquirer.prompt(questions)

while answers.get('choice') != 'Exit':

    INPUT_DIM = len(dataset.TEXT.vocab)
    EMBEDDING_DIM = 100
    N_FILTERS = 400
    FILTER_SIZES = [2, 3, 4]
    OUTPUT_DIM = len(dataset.LABEL.vocab)
    DROPOUT = 0.5
    PAD_IDX = dataset.TEXT.vocab.stoi[dataset.TEXT.pad_token]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 32
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (dataset.train_data, dataset.valid_data, dataset.test_data),
        sort=False,  # don't sort test/validation data
        batch_size=BATCH_SIZE,
        device=device)

    model = m.CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

    pretrained_embeddings = dataset.TEXT.vocab.vectors

    model.embedding.weight.data.copy_(pretrained_embeddings)

    UNK_IDX = dataset.TEXT.vocab.stoi[dataset.TEXT.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    optimizer = optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    if answers.get('choice') == 'Train model':
        train.train_and_evaluate(model, train_iterator, valid_iterator, optimizer, criterion)
        test_loss, test_acc = train.evaluate(model, test_iterator, criterion)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.3f}%')

    if answers.get('choice') == 'Evaluate model':
        model.load_state_dict(torch.load('ezmath-model_83.pt'))
        test_loss, test_acc = train.evaluate(model, test_iterator, criterion)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.3f}%')

    if answers.get('choice') == 'Make Prediction':
        model.load_state_dict(torch.load('ezmath-model_83.pt'))
        nlp = spacy.load('it_core_news_sm')
        string = input("Please insert the exercise text: ")
        print('Making prediction for: ')
        print(string)
        pred_class = model.predict_class(string, nlp, dataset, device)
        print(f'Predicted class is: {pred_class} = {dataset.LABEL.vocab.itos[pred_class]}')

    if answers.get('choice') == 'Print model details':
        print(f'Embedding dimension: {EMBEDDING_DIM} ')
        print(f'N. of filters: {N_FILTERS} ')
        print(f'Vocab dimension: {INPUT_DIM} ')
        print('Categories:')
        [print(f'   {cat}') for cat in dataset.LABEL.vocab.itos]
        summary(model, (32,))

    if answers.get('choice') == 'Plot embedding space - TensorboardX':
        writer = SummaryWriter('tensorboard/embeddings')
        writer.add_embedding(pretrained_embeddings, metadata=dataset.TEXT.vocab.itos, tag='Embedding')
        writer.close()
        print('Remember to run Tensorboard thru: tensorboard --logdir=tensorboard')

    if answers.get('choice') == 'More info about dataset':
        dataset.dataset_info()
        dataset.print_dataset_details()

    if answers.get('choice') == 'Evaluate and plot PR curves - TensorboardX':
        model.load_state_dict(torch.load('ezmath-model_83.pt'))
        test_loss, test_acc = train.evaluate_with_pr_plotting(model, test_iterator, criterion, dataset.LABEL.vocab.itos)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.3f}%')
        print('Remember to run Tensorboard thru: tensorboard --logdir=tensorboard')

    if answers.get('choice') == 'Test Textual Augmenter by word2vec similarity':
        print('Loading Word2Vec model...')
        aug = naw.WordEmbsAug(
            model_type='word2vec', model_path='vector_cache/word2vec_CoNLL17/model.bin',
            action="substitute")
        text = input("Please insert the exercise text to augment: ")
        augmented_text = aug.augment(text)
        print("Original:")
        print(text)
        print("Augmented Text:")
        print(augmented_text)

    answers = inquirer.prompt(questions)
