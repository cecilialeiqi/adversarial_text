import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from sklearn.metrics import accuracy_score
from lstm import LSTMClassifier
from evaluate import evaluate
import torch.nn.functional as F

device = -1
if torch.cuda.is_available():
    device = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', action='store', dest='train_path',
                        help='Path to train data')
    parser.add_argument('--test_path', action='store', dest='test_path',
                        help='Path to test data')
    parser.add_argument('--log-every', type=int, default=5, help='Steps for each logging.')

    parser.add_argument('--batch-size', action='store', default=16, type=int,
                        help='Mini batch size.')

    return parser.parse_args()

class CNN(nn.Module):
    def __init__(self, sentence_len=200, kernel_sizes=[3,4,5], num_filters=100, embedding_dim=300, pretrained_embeddings=None):
        super(CNN, self).__init__()
        self.sentence_len=sentence_len
        use_cuda = torch.cuda.is_available()
        self.kernel_sizes = kernel_sizes
        vocab_size=len(pretrained_embeddings)
        print(vocab_size)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = False #mode=="nonstatic"
        if use_cuda:
            self.embedding = self.embedding.cuda()
        conv_blocks = []
        for kernel_size in kernel_sizes:
            # maxpool kernel_size must <= sentence_len - kernel_size+1, otherwise, it could output empty
            maxpool_kernel_size = sentence_len - kernel_size +1
            conv1d = nn.Conv1d(in_channels = 1, out_channels = num_filters, kernel_size = kernel_size*embedding_dim, stride = embedding_dim)

            component = nn.Sequential(
                conv1d,
                nn.ReLU(),
                nn.MaxPool1d(kernel_size = maxpool_kernel_size)
            )
            if use_cuda:
                component = component.cuda()

            conv_blocks.append(component)

            '''conv_blocks.append(
                nn.Sequential(
                    conv1d,
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size = maxpool_kernel_size)
                ).cuda()
            )'''
        self.conv_blocks = nn.ModuleList(conv_blocks)   # ModuleList is needed for registering parameters in conv_blocks
        self.fc = nn.Linear(num_filters*len(kernel_sizes), 2)

    def forward(self, x):       # x: (batch, sentence_len)
        x = self.embedding(x)   # embedded x: (batch, sentence_len, embedding_dim)
        #    input:  (batch, in_channel=embedding_dim, in_length=sentence_len),
        #    output: (batch, out_channel=num_filters, out_length=sentence_len-...)
        x = x.view(x.size(0), 1, -1)  # needs to convert x to (batch, 1, sentence_len*embedding_dim)
        x_list= [conv_block(x) for conv_block in self.conv_blocks]
        out = torch.cat(x_list, 2)
        out = out.view(out.size(0), -1)
        feature_extracted = out
        out = F.dropout(out, p=0.3, training=self.training)
        return F.softmax(self.fc(out), dim=1), feature_extracted

def evaluate_cnn(model, batch, sentence_len):
    inputs=F.pad(batch.text.transpose(0,1), (0,sentence_len-len(batch.text)))
    preds,_ =model(inputs)
    #print(preds.data.cpu().numpy(), batch.label.data.cpu().numpy())
    eval_acc=sum([1 if preds.data.cpu().numpy()[i][j]>0.5 else 0 for i,j in enumerate(batch.label.data.cpu().numpy()[0])])
    return eval_acc

def evaluate_lstm(model, batch):
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
    print(type(train))
    max_s=100000
    src_field.build_vocab(train, max_size=max_s, min_freq=2, vectors="glove.6B.300d")
    label_field.build_vocab(train)
    sentence_len=max([len(v.text) for v in train])
    print('sentence_len', sentence_len)
    if sentence_len>5000: sentence_len=5000
    print("Training size: {0}, Testing size: {1}, sentence_len: {2}".format(len(train), len(test), sentence_len))
    embedding_dim = 300
    num_filters = 100
    kernel_sizes = [3]
    print(src_field.vocab.vectors)
    classifier = CNN(sentence_len, kernel_sizes, num_filters, embedding_dim, src_field.vocab.vectors)    
    if torch.cuda.is_available():
        classifier.cuda()

    train_iter = data.BucketIterator(
        dataset=train,
        batch_size=opt.batch_size,
        device=device,
        repeat=False
    )
    test_iter = data.BucketIterator(
        dataset=test, batch_size=20, device=device, repeat=False)
    print("\n{}\n".format(str(classifier)))
    parameters = filter(lambda p: p.requires_grad, classifier.parameters())
    optimizer = optim.Adam(parameters)  #classifier.parameters())
    print('set up optimizer over')
    for param in parameters: #classifier.parameters():
        param.data.uniform_(-0.08, 0.08)
    criterion = nn.CrossEntropyLoss()
    print('started training')
    step = 0
    opt.log_every=9999
    for epoch in range(15):
        #train_accu = evaluate(classifier, train, opt.batch_size)
        #test_accu = evaluate(classifier, test, opt.batch_size)
        test_acc=0
        for batch in test_iter:
             test_acc+=evaluate_cnn(classifier, batch, sentence_len)
        print('Test accuracy: {0}'.format(test_acc/len(test)))
        #print('Train accuracy: {0}, Test accuracy: {1}'.format(train_accu, test_accu))
        running_loss = 0.0
        #total_acc=0
        #total_n=0
        for batch in train_iter:
            #total_n+=16
            optimizer.zero_grad()
            if len(batch.text)>sentence_len:
                inputs=batch.text[:sentence_len].transpose(0,1)
            elif len(batch.text)<sentence_len:
                inputs=F.pad(batch.text.transpose(0,1), (0,sentence_len-len(batch.text)))
            else:
                inputs=batch.text.transpose(0,1)
            pred,_ = classifier(inputs)
            #acc=sum([1 if pred.data.cpu().numpy()[i,j]>-0.693147 else 0 for i,j in enumerate(batch.label.data.cpu().numpy()[0])])
            #total_acc+=acc
            loss = criterion(pred, batch.label.view(-1))
            running_loss += loss.data[0]
            loss.backward()
            optimizer.step()

            step += 1

            if step % (opt.log_every) == 0:
                print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, step + 1, running_loss / opt.log_every)) #, total_acc*1.0/total_n))
                running_loss = 0.0
        if epoch%5==4:
            torch.save(classifier, os.path.join("model_{0}".format(epoch + 1)))
if __name__ == '__main__':
    main()
