# adversarial_text

## train the original model
 - download training/testing dataset and put it in ./data/train.tsv and ./data/test.tsv, each line should consist of the text and the label, seprated by \t
 - cd src/
 - sh run_LSTM.sh (to train LSTM classifier)
 - sh run_CNN.sh (to train the word-level CNN classifier)

## set up word embeddings model
 - Download paragram_300_sl999 file from https://github.com/recski/wordsim
 - change in the Makefile the embedding_path to be the directory of the above file

## set up paraphraser
 - Download the sentence paraphrasing model from https://github.com/vsuthichai/paraphraser
 - put it the the same parent path as the text_adversarial repository 

## generate adversarial examples
 - in the Makefile, change the model_path to the above generated model
 - make attack_cnn to generate adversarial examples of the wcnn model
 - make attack_lstm to generate adversarial examples of the lstm classifier



