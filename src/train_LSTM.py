import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from sklearn.metrics import accuracy_score

from lstm import LSTMClassifier
device = -1
if torch.cuda.is_available():
    device = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', action='store', dest='train_path',
                        help='Path to train data')
    parser.add_argument('--test_path', action='store', dest='test_path',
                        help='Path to test data')
    parser.add_argument('--log-every', type=int, default=10000, help='Steps for each logging.')

    parser.add_argument('--batch-size', action='store', default=16, type=int,
                        help='Mini batch size.')

    return parser.parse_args()

def evaluate(model, batch):
    inputs=batch.text #F.pad(batch.text.transpose(0,1), (0,sentence_len-len(batch.text)))
    preds =model(inputs)
    #print(preds.data.cpu().numpy(), batch.label.data.cpu().numpy())
    eval_acc=sum([1 if preds.data.cpu().numpy()[i][j]>-0.693147181 else 0 for i,j in enumerate(batch.label.data.cpu().numpy()[0])])
    return eval_acc



def main():

    opt = parse_args()

    src_field = data.Field()
    label_field = data.Field(pad_token=None, unk_token=None)
    train = data.TabularDataset(
        path=opt.train_path, format='tsv',
        fields=[('text', src_field), ('label', label_field)]
    )
    test = data.TabularDataset(
        path=opt.test_path, format='tsv',
        fields=[('text', src_field), ('label', label_field)]
    )
    src_field.build_vocab(train, max_size=100000, min_freq=2, vectors="glove.6B.300d")
    label_field.build_vocab(train)

    print("Training size: {0}, Testing size: {1}".format(len(train), len(test)))

    classifier = LSTMClassifier(300, 512, len(label_field.vocab), src_field.vocab.vectors)

    if torch.cuda.is_available():
        classifier.cuda()

    train_iter = data.BucketIterator(
        dataset=train,
        batch_size=opt.batch_size,
        device=device,
        repeat=False
    )
    test_iter = data.BucketIterator(
        dataset=test,
        batch_size=5,
        device=device,
        repeat=False
    )
    for param in classifier.parameters():
        param.data.uniform_(-0.08, 0.08)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters())

    step = 0
    for epoch in range(15):
        test_acc=0
        for batch in test_iter:
            test_acc+=evaluate(classifier, batch)
        print('Test accuracy: {0}'.format(test_acc/len(test)))
        running_loss = 0.0
        for batch in train_iter:
            optimizer.zero_grad()
            pred = classifier(batch.text)
            loss = criterion(pred, batch.label.view(-1))
            running_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            step += 1
            if step % opt.log_every == 0:
                print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, step + 1, running_loss / opt.log_every))
                running_loss = 0.0
        torch.save(classifier, os.path.join("model_{0}".format(epoch + 1)))

if __name__ == '__main__':
    main()


