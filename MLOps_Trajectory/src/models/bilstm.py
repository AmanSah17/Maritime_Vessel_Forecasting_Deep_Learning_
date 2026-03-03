import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_rate=0.5):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout_rate,
            bidirectional=True
        )
        
        # Linear output layer (x2 because bidirectional concatenates forward + backward hidden states)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        x shape: [batch_size, seq_len, input_size]
        """
        self.lstm.flatten_parameters()
        
        # lstm_out shape: [batch_size, seq_len, hidden_size * 2]
        lstm_out, _ = self.lstm(x)
        
        # Extract the predictions for each timestep
        # output shape: [batch_size, seq_len, output_size]
        output = self.linear(lstm_out)
        
        return output
