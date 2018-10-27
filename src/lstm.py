import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, num_class, pretrained_embed):
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=pretrained_embed.size(0),
            embedding_dim=300
        )
        self.embedding.weight.data.copy_(pretrained_embed)

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size)

        self.linear = nn.Linear(hidden_size, num_class)

    def forward(self, input_seq):
        embed = self.embedding(input_seq)
        output, hidden = self.rnn(embed)
        # output = (seq_len, batch_size, hidden_size)
        feature = output.mean(dim=0)
        feature = self.linear(feature)
        out = F.log_softmax(feature)
        return out
