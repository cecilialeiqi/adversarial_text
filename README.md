# adversarial_text

## step 1: train the original model
 - download training/testing dataset and put it in ./data/train.tsv and ./data/test.tsv, each line should consist of the text and the label, seprated by \t
 - cd src/
 - make train_LSTM (to train LSTM classifier)
 - make train_CNN (to train the word-level CNN classifier)

## step 2: set up word embeddings model
 - Download paragram_300_sl999 file from https://github.com/recski/wordsim
 - change in the Makefile the embedding_path to be the directory of the above file

## step 3 (optional): set up sentence paraphraser (it will take up very large memory)
 - Download the sentence paraphrasing model from https://github.com/vsuthichai/paraphraser
 - put it the the same parent path as the text_adversarial repository

## step 4: generate adversarial examples
 - In the Makefile, change the model_path to the above generated models
 - "make attack_cnn" to generate adversarial examples of the wcnn model
 - "make attack_lstm" to generate adversarial examples of the lstm classifier

 - To use joint sentence and word level attacks, run the following
 - make attack_cnn_joint
 - make attack_lstm_joint
