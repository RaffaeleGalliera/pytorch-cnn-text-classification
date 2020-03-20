import torch
import time
import dataset
import numpy as np
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import pdb


def train_and_evaluate(model, train_iterator, valid_iterator, optimizer, criterion):
    n_epochs = 15
    best_valid_loss = float('inf')
    writer = SummaryWriter('tensorboard/acc_loss')

    for epoch in range(n_epochs):

        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, writer)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'ezmath-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss:: {train_loss:.3f}\t|\tTrain Acc: {train_acc * 100:.3f}%')
        print(f'\tValid Loss: {valid_loss:.3f}\t|\tValid Acc: {valid_acc * 100:.3f}%')

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/valid', valid_acc, epoch)

    writer.close()


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text)

            loss = criterion(predictions, batch.label)

            acc = categorical_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate_with_pr_plotting(model, iterator, criterion, classes):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    writer = SummaryWriter('tensorboard/pr_curve')

    class_probs = []
    class_preds = []

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text)

            loss = criterion(predictions, batch.label)

            acc = categorical_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            class_probs_batch = [F.softmax(el, dim=0) for el in predictions]
            _, class_preds_batch = torch.max(predictions, 1)

            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)

    def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
        '''
        Takes in a "class_index" from 0 to 7 and plots the corresponding
        precision-recall curve
        '''
        tensorboard_preds = test_preds == class_index
        tensorboard_probs = test_probs[:, class_index]
        writer.add_pr_curve(classes[class_index],
                            tensorboard_preds,
                            tensorboard_probs,
                            global_step=global_step)

    # plot all the pr curves
    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, test_probs, test_preds)

    writer.close()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train(model, iterator, optimizer, criterion, writer):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text)

        loss = criterion(predictions, batch.label)

        acc = categorical_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])
