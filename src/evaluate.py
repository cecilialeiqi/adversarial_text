import argparse

import torch
from torchtext import data
from sklearn.metrics import accuracy_score

device = -1
if torch.cuda.is_available():
    device = None

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('train_path', action='store', help='Path to train data')
    parser.add_argument('test_path', action='store', help='Path to test data')
    parser.add_argument('model_path', action='store', help='Path to pre-trained classifier model')
    parser.add_argument('--batch-size', action='store', default=16, type=int,
                        help='Mini batch size.')

    return parser.parse_args()

def evaluate(model, dataset, batch_size):
    iterator = data.BucketIterator(
            dataset=dataset,
            batch_size=batch_size,
            device=device,
            train=False
    )
    model.eval()

    predlist = []
    labellist = []
    for batch in iterator:
        pred = model(batch.text)
        predlist += pred.max(dim=1)[1].data.tolist()
        labellist += batch.label.view(-1).data.tolist()
    return accuracy_score(labellist, predlist)

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
    src_field.build_vocab(train, max_size=100000, min_freq=1, vectors="glove.6B.300d")
    label_field.build_vocab(train)

    classifier = torch.load(opt.model_path, map_location=lambda storage, loc: storage)

    if torch.cuda.is_available():
        classifier.cuda()

    test_accu = evaluate(classifier, test, opt.batch_size)

    print("Test accuracy: %f", test_accu)

if __name__ == '__main__':
    main()
