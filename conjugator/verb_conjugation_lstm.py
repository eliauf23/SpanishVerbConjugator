import torch.nn as nn


class VerbConjugationLSTM(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_size, n_layers):
        super(VerbConjugationLSTM, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        output = self.fc(output)
        return output
