TRAIN=../data/train.tsv
TEST=../data/test.tsv
	
train_lstm:
	python train_LSTM.py --train_path $(TRAIN) --test_path $(TEST)

train_cnn:
	python train_CNN.py --train_path $(TRAIN) --test_path $(TEST)

attack_lstm:
	python3 discrete_attack.py 2 LSTM $(TRAIN) $(TEST) ../data/changed_lstm --embedding_path c.txt --model_path ../model/model_lstm.pt 20000 FAKE

attack_cnn:
	python3 discrete_attack.py 2 CNN $(TRAIN) $(TEST) ../data/changed_cnn_train --embedding_path c.txt   --model_path ../model/model_cnn.pt 20000 FAKE

attack_lstm_joint:
	python3 joint_attack.py 2 2 LSTM $(TRAIN) $(TEST) ../data/changed_lstm --embedding_path c.txt --model_path ../model/model_lstm.pt 20000 FAKE

attack_cnn_joint:
	python3 joint_attack.py 2 2 CNN $(TRAIN) $(TEST) ../data/changed_cnn_train --embedding_path c.txt  --model_path ../model/model_cnn.pt 20000 FAKE



