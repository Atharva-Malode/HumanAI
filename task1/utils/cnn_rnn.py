import torch.nn as nn

class CNN_RNN_Model(nn.Module):
    def __init__(self, input_size=1032, hidden_size=256, num_classes=50):
        super(CNN_RNN_Model, self).__init__()
        """
        Initializes the BiLSTM model.
        """
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        """
        Fully connected layer for classification.
        """
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        """
        Forward pass through LSTM and classification layer.
        """
        rnn_out, _ = self.rnn(x)
        final_feature = rnn_out[:, -1, :]
        output = self.fc(final_feature)
        return output