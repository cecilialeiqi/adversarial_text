# adversarial_text

 - Qi Lei, Lingfei Wu, Pin-Yu Chen, Alexandros G. Dimakis, Inderjit S. Dhillon, Michael Witbrock. "Discrete Adversarial Attacks and Submodular Optimization with Applications to Text Classification” Systems and Machine Learning (sysML). 2019 ([arXiv](https://arxiv.org/abs/1812.00151),[slides](http://users.oden.utexas.edu/~leiqi/discrete_attack.pdf)) 

 - Press coverage: <[Nature Story](https://www.nature.com/articles/d41586-019-01510-1?utm_source=twt_nnc&utm_medium=social&utm_campaign=naturenews&sf212595612=1)> <[Vecturebeat](https://venturebeat.com/2019/04/01/text-based-ai-models-are-vulnerable-to-paraphrasing-attacks-researchers-find/)> <[Tech Talks](https://bdtechtalks.com/2019/04/02/ai-nlp-paraphrasing-adversarial-attacks/)> <[机器之心](https://www.jiqizhixin.com/articles/2019-03-27-10?from=synced&keyword=SysML%202019)> 

## step 1: train the original model
 - download training/testing dataset and put it in ./data/train.tsv and ./data/test.tsv, each line should consist of the text and the label, seprated by \t
 - cd src/
 - make train_LSTM (to train LSTM classifier)
 - make train_CNN (to train the word-level CNN classifier)
 - Move the models to targeted directory, e.g. ../model/model_lstm.pt and ../model/model_cnn.pt
## step 2: set up word embeddings model
 - Download paragram_300_sl999 file from https://github.com/recski/wordsim
 - change in the Makefile the embedding_path to be the directory of the above file

## step 3 (optional): set up sentence paraphraser (it will take up very large memory)
 - Download the sentence paraphrasing model from https://github.com/vsuthichai/paraphraser
 - put it the the same parent path as the text_adversarial repository

## step 4: generate adversarial examples
 - In the Makefile, change the input parameter model_path to the above generated models; also, change the input parameter first_label to the first label name (e.g. FAKE for the news data) appeared in the training file. (Otherwise the model doesn't distinguish positive and negative labels)
 - "make attack_cnn" to generate adversarial examples of the wcnn model
 - "make attack_lstm" to generate adversarial examples of the lstm classifier

 - To use joint sentence and word level attacks, do step 3 and run the following
 - make attack_cnn_joint
 - make attack_lstm_joint


## Datasets:
 - Finally, the datasets we used could be obtained from https://www.dropbox.com/sh/jdkhvdgzmytu78i/AACo53pUyerYO6jwVds5SZyPa?dl=0
 - The dataset in ./data folder is the fake news dataset  
