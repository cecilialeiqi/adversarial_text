import csv
from torchtext import data
import torch
from torch.autograd import Variable
from math import exp

def read_data(path, pos):
    x, y = [], []
    count=0
    with open(path) as fin:
        reader = csv.reader(fin, delimiter='\t')
        for row in reader:
            count+=1
            if count>20000: break
            x.append(row[0][:])#.decode('utf-8'))
            y.append(1 if row[1]==pos else 0)
    return x, y

def dump_data(path, x, y):
    with open(path, 'w') as fout:
        writer = csv.writer(fout, delimiter="\t")
        for doc, label in zip(x, y):
            writer.writerow([u" ".join(doc), label])

def dump_row(path, doc, label):
    with open(path, 'a') as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow([u" ".join(doc), label])

def v2l(v):
    return 'FAKE' if v else 'REAL'

def dump_p(path, p):
    with open(path, 'w') as fout:
        writer = csv.writer(fout,delimiter="\t")
        for count, doc, pred, pred_prob, changed_pos in p:
            writer.writerow([count, u" ".join(doc), v2l(pred), pred_prob, ",".join([str(i) for i in changed_pos])])

def dump_p_row(path, p):
    count, doc, pred, pred_prob, changed_pos=p[0], p[1], p[2], p[3], p[4]
    with open(path, 'a') as fout:
        writer = csv.writer(fout,delimiter="\t")
        writer.writerow([count, u" ".join(doc), v2l(pred), pred_prob, ",".join([str(i) for i in changed_pos])])

def text_to_var(docs, vocab):
    tensor = []
    max_len = max([len(doc) for doc in docs])
    for doc in docs:
        vec = []
        for tok in doc:
            vec.append(vocab.stoi[tok])
        if len(doc) < max_len:
            vec += [vocab.stoi['<pad>']] * (max_len - len(doc))
        tensor.append(vec)
    try:
        var = Variable(torch.LongTensor(tensor)).transpose(0, 1)
    except:
        print(tensor, docs)
        exit()
    if torch.cuda.is_available():
        var = var.cuda()
    return var

def classify(var, model):
    if 'CNN' in str(type(model)):
        pred,_ = model(var)
        prob, clazz = pred[0].max(dim=0)
        return prob.data[0], clazz.data[0]
    else:
        pred = model(var)
        prob, clazz = pred[0].max(dim=0)
        return exp(prob.data[0]), clazz.data[0]

def negative_score(var, model, y):
    if 'CNN' in str(type(model)):
        pred_probs,_ = model(var)
        c_prob = pred_probs[:, 1-y].data[0]
    else:
        pred_probs = model(var)
        c_prob = exp(pred_probs[:, 1-y].data[0])
    return c_prob


