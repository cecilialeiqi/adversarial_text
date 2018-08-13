#! /bin/sh

TRAIN=../data/mixed_train_lstm.tsv
TEST=../data/test.tsv

python train_LSTM.py --train_path $TRAIN --test_path $TEST

