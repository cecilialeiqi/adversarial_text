from __future__ import print_function
import csv
import logging
import argparse
from math import exp
import math
from copy import copy
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
from torchtext import data
import nltk
from nltk import word_tokenize
from gensim.models.keyedvectors import KeyedVectors
import time
from lm import NGramLangModel
import sys
sys.path.append('../../paraphraser/paraphraser')
from util import *
import spacy
from inference import *
import wmd
import re
import torch.nn.functional as F
# load the paraphraser
paraphraser = Paraphraser('../../paraphraser/train-20180325-001253/model-171856')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
nlp = spacy.load('en', create_pipeline=wmd.WMD.create_spacy_pipeline)
NGRAM = 3
N_NEIGHBOR = 15
TAU = 0.7
TAU_sim=0.6
N_REPLACE = 5

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('sentence_delta', help= 'percentage of allowed sentence paraphasing')
    parser.add_argument('word_delta', help= 'percentage of allowed word paraphasing')
    parser.add_argument('model', help='model: either CNN or LSTM')
    parser.add_argument('train_path', help='Path to training data')
    parser.add_argument('test_path', help='Path to testing data')
    parser.add_argument('output_path', help='Path to output changed test data')
    parser.add_argument('--embedding_path', action='store', dest='embedding_path',
        help='Path to pre-trained embedding data')
    parser.add_argument('--model_path', action='store', dest='model_path',
        help='Path to pre-trained classifier model')
    parser.add_argument('max_size', help='max amount of data to be processed by the model')
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
        # dropout is used in training but not in inference
        # out = F.dropout(out, p=0.3, training=self.training)
        return F.softmax(self.fc(out), dim=1), feature_extracted




class Attacker(object):
    ''' main part of the attack model '''
    def __init__(self, X, opt):
        self.opt=opt
        self.suffix=str(opt.sentence_delta)+'-'+str(opt.word_delta)
        self.DELTA_W=int(opt.word_delta)*0.1
        self.DELTA_S=int(opt.sentence_delta)*0.1
        self.TAU_2=2
        self.TAU_wmd_s = 0.8
        self.TAU_wmd_w=0.75
        # want do sentence level paraphrase first
        X=[doc.split() for doc in X]
        logging.info("Initializing language model...")
        print("Initializing language model...")
        self.lm = NGramLangModel(X, NGRAM)
        logging.info("Initializing word vectors...")
        print("Initializing word vectors...")
        self.w2v = KeyedVectors.load_word2vec_format(opt.embedding_path, binary=False)
        logging.info("Loading pre-trained classifier...")
        print("Loading pre-trained classifier...")
        self.model = torch.load(opt.model_path, map_location=lambda storage, loc: storage)
        if torch.cuda.is_available():
            self.model.cuda()
        logging.info("Initializing vocabularies...")
        print("Initializing vocabularies...")
        self.src_vocab, self.label_vocab = self.load_vocab(opt.train_path)
        # to compute the gradient, we need to set up the optimizer first
        self.criterion = nn.CrossEntropyLoss()

    def word_paraphrase(self, words, poses, list_neighbors, y):
            candidates = [words]
            j=1
            if self.opt.model=='LSTM':
                max_size=int(self.opt.max_size)//len(words)
            else:
                max_size=int(self.opt.max_size)//self.model.sentence_len
            for pos in poses:
                    closest_neighbors=list_neighbors[pos]
                    if not closest_neighbors:
                        j+=1
                        continue
                    current_candidates= copy(candidates)
                    for repl in closest_neighbors:
                        for c in candidates:
                            if len(current_candidates)>max_size:
                                break
                            corrupted = copy(c)
                            corrupted[pos] = repl
                            current_candidates.append(corrupted)
                    candidates=copy(current_candidates)
                    if len(candidates)>max_size:
                        break
                    j+=1
            if candidates:
                if self.opt.model=='LSTM':
                    candidate_var = text_to_var(candidates, self.src_vocab)
                    pred_probs = self.model(candidate_var)
                    log_pred_prob, best_candidate_id = pred_probs[:, 1-y].max(dim=0)
                    new_words = candidates[best_candidate_id.data[0]]
                    pred_prob = exp(log_pred_prob.data[0])
                elif self.opt.model=='CNN':
                    candidate_var = self.text_to_var_CNN(candidates, self.src_vocab)
                    pred_probs,_ = self.model(candidate_var)
                    log_pred_prob, best_candidate_id = pred_probs[:, 1-y].max(dim=0)
                    new_words = candidates[best_candidate_id.data[0]]
                    pred_prob = log_pred_prob.data[0]
            else:
                print('empty candidates!')
            return new_words, pred_prob, j

    def hidden(self, hidden_dim):
        if torch.cuda.is_available():
            h0=Variable(torch.zeros(1,1,hidden_dim).cuda())
            c0=Variable(torch.zeros(1,1,hidden_dim).cuda())
        else:
            h0=Variable(torch.zeros(1,1,hidden_dim))
            c0=Variable(torch.zeros(1,1,hidden_dim))
        return (h0,c0)

    def forward_lstm(self, embed,model):  #copying the structure of LSTMClassifer, just omitting the first embedding layer
        lstm_out, hidden0= model.rnn(embed, self.hidden(512))
        y=model.linear(lstm_out[-1])
        return y
    def forward_cnn(self,embed,model):
        x_list= [conv_block(embed) for conv_block in model.conv_blocks]
        out = torch.cat(x_list, 2)
        out = out.view(out.size(0), -1)
        return F.softmax(model.fc(out), dim=1)
    def text_to_var_CNN(self, docs, vocab):
        tensor = []
        max_len = self.model.sentence_len 
        for doc in docs:
            vec = []
            for tok in doc:
                vec.append(vocab.stoi[tok])
            if len(doc) < max_len:
                vec += [0]*(max_len-len(doc))   
            else:
                vec=vec[:max_len]
            tensor.append(vec)
        var = Variable(torch.LongTensor(tensor))
        if torch.cuda.is_available():
            var = var.cuda()
        return var

    
    def sentence_paraphrase(self, y, sentences, changed_pos, list_closest_neighbors):
            candidates = []
            responding_pos = [] # the index of the changed sentence
            for i, sentence in enumerate(sentences):
                if i in changed_pos:
                    continue
                j=0
                for p in list_closest_neighbors[i]:
                    new_sentence=copy(sentences)
                    new_sentence[i]=p
                    new_sentence=(" ".join(new_sentence)).split()
                    candidates.append(new_sentence)
                    responding_pos.append((i,j))
                    j+=1

            if candidates:
                m=len(candidates)
                if self.opt.model=='LSTM':
                    n=max([len(candidates[i]) for i in range(m)])
                else: n=self.model.sentence_len
                b=np.random.permutation(m)[:int(self.opt.max_size)//n]
                candidates=[candidates[i] for i in b]
                responding_pos= [responding_pos[i] for i in b]
                if self.opt.model=='LSTM':
                    candidate_var = text_to_var(candidates, self.src_vocab)
                    pred_probs = self.model(candidate_var)
                    log_pred_prob, best_candidate_id = pred_probs[:, 1-y].max(dim=0)
                    final_pos=responding_pos[best_candidate_id.data[0]][0]
                    final_choice=responding_pos[best_candidate_id.data[0]][1]
                    pred_prob = exp(log_pred_prob.data[0])
                else:
                    candidate_var = self.text_to_var_CNN(candidates, self.src_vocab)
                    pred_probs,_ = self.model(candidate_var)
                    log_pred_prob, best_candidate_id = pred_probs[:, 1-y].max(dim=0)
                    final_pos=responding_pos[best_candidate_id.data[0]][0]
                    final_choice=responding_pos[best_candidate_id.data[0]][1]
                    pred_prob = log_pred_prob.data[0]                        
                print('final changed pos '+str(final_pos)+' from '+sentences[final_pos]+' ------->>>>> '+list_closest_neighbors[final_pos][final_choice]+', score='+str(pred_prob))
                sentences[final_pos]=list_closest_neighbors[final_pos][final_choice]
                return sentences, final_pos, pred_prob
            else:
                return sentences, -1, 0
    def load_vocab(self, path):
        src_field = data.Field()
        label_field = data.Field(pad_token=None, unk_token=None)
        dataset = data.TabularDataset(
            path=path, format='tsv',
            fields=[('text', src_field), ('label', label_field)]
        )
        if self.opt.model=='LSTM':
            src_field.build_vocab(dataset, max_size=100000, min_freq=2, vectors="glove.6B.300d")
        else:
            if self.opt.data=='news':
                src_field.build_vocab(dataset, max_size=10000, min_freq=1, vectors="glove.6B.300d")
            else:
                src_field.build_vocab(dataset, max_size=10000, min_freq=2, vectors="glove.6B.300d")

        label_field.build_vocab(dataset)
        return src_field.vocab, label_field.vocab

    def attack(self, count, doc, y):
        best_score=0.0
        st=time.time()
        #--------------------------------------------sentence paraphrasing--------------------------------------------#
        sentences=tokenizer.tokenize(doc)
        print('before classification')
        if not(doc.split()): return doc, 0, -1
        if self.opt.model=='CNN':    
            doc_var = self.text_to_var_CNN([doc.split()], self.src_vocab)
        else:
            doc_var = text_to_var([doc.split()],self.src_vocab)
        orig_prob, orig_pred = classify(doc_var, self.model)
        pred, pred_prob = orig_pred, orig_prob
        if not (pred == y or pred_prob < TAU):
            return doc.split(), pred_prob, -1
        num_replaced=0
        changed_pos=set()
        # first get all the paraphrases for each sentence
        list_closest_neighbors=[]
        for i, sentence in enumerate(sentences):
            doc1=nlp(sentence)
            closest_neighbors=[]
            sentence=re.sub("[^a-zA-Z0-9@()*.,-:\?!/ ]","",sentence)
            valid_words=[self.src_vocab.stoi[w] for w in word_tokenize(sentence)]
            bad_words=  sum([i==0 for i in valid_words]) 
            if (count,i) in set(pairs) or len(word_tokenize(sentence))>60 or bad_words>=0.2*len(valid_words) or bad_words>=3 or len(sentence)>500:
                 paraphrases=[]
            else:
                print(count,i,sentence)
                paraphrases = paraphraser.sample_paraphrase(sentence, sampling_temp=0.75, how_many=N_NEIGHBOR)
            for p in paraphrases:
                doc2=nlp(p)
                score=doc1.similarity(doc2)
                if score>=self.TAU_wmd_s: 
                    closest_neighbors.append(p)
            list_closest_neighbors.append(closest_neighbors)
        while (pred == y or pred_prob < TAU) and time.time()-st<3600 \
                and num_replaced < self.DELTA_S * len(sentences): 
            new_sentences, pos, pred_prob = self.sentence_paraphrase(y, sentences, changed_pos, list_closest_neighbors)
            if pos==-1 or pred_prob<best_score:
                print('sentence paraphraser over')
                break
            changed_pos.add(pos)
            if pred_prob>=best_score: sentences=copy(new_sentences)
            best_score=max(best_score,pred_prob)
            s=(" ".join(new_sentences)).split()
            if self.opt.model=='LSTM':
                var=text_to_var([s], self.src_vocab)
            else:
                var=self.text_to_var_CNN([s], self.src_vocab)
            ns=negative_score(var, self.model, y)
            best_score = min(ns,best_score)
            pred_prob=best_score
            num_replaced += 1
            if pred_prob>0.5:
                pred=1-y
        #---------------------------------word paraphrasing----------------------------------------------#
        doc=" ".join(sentences)
        words=doc.split()
        words_before=copy(words)
        best_words=copy(words)
        # check if the value of this doc to be right
        if self.opt.model=='LSTM':
            doc_var = text_to_var([words], self.src_vocab)
        else:
            doc_var = self.text_to_var_CNN([words], self.src_vocab)
        c_prob = negative_score(doc_var, self.model, y)
        ### turns out they are different, weird, will fix that
        best_score=c_prob
        # wanna save the following things: [document, pred, changed_pos] after sentence paraphrasing, as well as after word paraphrasing
        dump_p_row(self.opt.output_path+'_per_sentence'+self.suffix+'.csv',[count, doc, pred, pred_prob, list(changed_pos)])
        if not (pred == y or pred_prob < TAU):
            return words, pred_prob, 0 
        # now word level paraphrasing
        list_closest_neighbors=[]
        for pos, w in enumerate(words):
            if self.opt.model=='CNN' and pos>=self.model.sentence_len: break
            try:
                closest_neighbors = self.w2v.most_similar(positive=[w.lower()], topn=N_NEIGHBOR)
            except:
                closest_neighbors=[]
            closest_paraphrases=[]
            closest_paraphrases.extend(closest_neighbors)
            # check if the words make sense
            valid_paraphrases=[]
            doc1=nlp(w)
            for repl,repl_sim in closest_paraphrases:
                doc2=nlp(repl)  #' '.join(repl_words))
                score=doc1.similarity(doc2)
                syntactic_diff = self.lm.log_prob_diff(words, pos, repl)
                logging.debug("Syntactic difference: %f", syntactic_diff)
                if score>=self.TAU_wmd_w and syntactic_diff <= self.TAU_2:
                    valid_paraphrases.append(repl)
            list_closest_neighbors.append(valid_paraphrases) #closest_neighbors)
            if not closest_paraphrases: #neighbors:
                print('find no neighbor for word: '+w)
        changed_pos=set()
        iteration=0
        recompute=True
        n_change=0
        if self.opt.model=='CNN':
            lword=min(len(words), self.model.sentence_len)
        else: lword=len(words)
        while (pred == y or pred_prob < TAU) and time.time()-st<3600 \
                and n_change < self.DELTA_W * lword and len(changed_pos)+N_REPLACE<len(words):
            iteration+=1
            if recompute:  # when words are changed, the gradient wrt other words might change as well
                if self.opt.model=='LSTM':
                    doc_var = text_to_var([words], self.src_vocab)
                    embed_doc = self.model.embedding(doc_var)
                    embed_doc = Variable(embed_doc.data, requires_grad=True) # make it a leaf node and requires gradient
                    output = self.forward_lstm(embed_doc, self.model) 
                elif self.opt.model=='CNN':
                    doc_var = self.text_to_var_CNN([words], self.src_vocab)
                    embed_doc = self.model.embedding(doc_var)
                    embed_doc = embed_doc.view(embed_doc.size(0),1,-1)
                    embed_doc = Variable(embed_doc.data, requires_grad=True) # make it a leaf node and requires gradient
                    output = self.forward_cnn(embed_doc, self.model)
                if torch.cuda.is_available():
                    loss = self.criterion(output, Variable(torch.LongTensor([y])).cuda())
                else:
                    loss = self.criterion(output, Variable(torch.LongTensor([y])))
                loss.backward()
                # obtained the gradient with respect to the per word embedding, \
                # for each word, we need to compute the dot product between the embedding of each possible replacements
                # and the gradient, and replace the most negative one
                score = np.zeros(len(words)) #,1+N_NEIGHBOR*2))
                # save the score of the nearest paraphrases and the original word
                if self.opt.model=='CNN':
                    grad_data=embed_doc.grad.data[0,0,:].view(-1,300)
                for pos, w in enumerate(words):
                    if self.opt.model=='CNN' and pos>=self.model.sentence_len: break
                    if pos in changed_pos or not list_closest_neighbors[pos]:
                        continue   # don't want to change again, or if there's no choice of replacement
                    if self.opt.model=='CNN':
                        a=grad_data[pos,:]
                    else:
                        a=embed_doc.grad.data[pos,0,:].view(300)
                    score[pos]=torch.dot(a,a)
            min_score=[]
            valid_n=0
            for i in range(len(list_closest_neighbors)):
                if list_closest_neighbors[i] and not i in changed_pos:
                    min_score.append(-score[i]) 
                    valid_n+=1
                else:
                    min_score.append(10000)
            indices=np.argsort(min_score)
            if valid_n<N_REPLACE: break
            words, pred_prob, N_CHANGE=self.word_paraphrase(words, indices[:N_REPLACE], list_closest_neighbors, y)
            for i in indices[:N_CHANGE]: changed_pos.add(i)
            if pred_prob>best_score:
                best_words=copy(words)
                best_score=pred_prob
            else:
                words=copy(best_words)
                recompute=False
            if pred_prob>0.5:
                pred=1-y
            n_change=sum([0 if words_before[i]==words[i] else 1 for i in range(len(words))])
        dump_p_row(self.opt.output_path+'_per_word'+self.suffix+'.csv', [count, best_words, pred, pred_prob, list(changed_pos)])
        print('after change:',' '.join(best_words),best_score)
        print(n_change, len(words_before), len(best_words), lword)
        return best_words, pred_prob, n_change*10.0/lword if best_words else 0

def main():
    opt = parse_args()
    if 'yelp' in opt.train_path: 
        opt.data='yelp'
    elif 'trec07p' in opt.train_path: 
        opt.data='email'
    else:
        opt.data='news'
    X_train, y_train =read_data(opt.train_path)
    X, y = read_data(opt.test_path)
    attacker=Attacker(X_train,opt)
    del X_train 
    del y_train
    suc=0
    suffix=str(opt.sentence_delta)+'-'+str(opt.word_delta)
    for count, doc in enumerate(X):
        logging.info("Processing %d/%d documents", count + 1, len(X))
        print("Processing %d/%d documents, success %d/%d", count+1, len(X), suc, count)
        if opt.data=='email':
            changed_doc, flag, num_changed = attacker.attack(count, doc,1 if y[count]=='REAL' else 0)
        else:
            changed_doc, flag, num_changed = attacker.attack(count, doc,0 if y[count]=='REAL' else 1)
        try:
            v=float(flag)
            if v>0.7:
                suc+=1
                changed_y='REAL' if y[count]=='FAKE' else 'FAKE'
            else:
                changed_y='REAL' if y[count]=='REAL' else 'FAKE'
        except:
            changed_y='REAL' if y[count]=='FAKE' else 'FAKE'
        dump_row(opt.output_path+suffix+'.tsv', changed_doc, changed_y)
        fout = open(opt.output_path+'_count'+suffix+'.csv','a')
        fout.write(str(count)+','+str(flag)+','+str(num_changed)+'\n')
        fout.close()
if __name__ == '__main__':
    main()
