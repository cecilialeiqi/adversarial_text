#! /bin/sh
TRAIN=../data/train.tsv
TEST=../data/test.tsv

python train_CNN.py --method CNN --retrain False --train_path $TRAIN --test_path $TEST


