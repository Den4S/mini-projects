import torch
from tqdm import tqdm
from torch import nn


INIT_MASK_SIZE = 64

# Convolutional encoder fixed params
CHANNELS_1 = 4
CHANNELS_2 = 8
POOL_SIZE = 2


class CNNEncoder(nn.Module):
    
    def __init__(
        self, feature_dim, 
        input_size=(1, INIT_MASK_SIZE, INIT_MASK_SIZE)
    ):
        super(CNNEncoder, self).__init__()
        C, H, W = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(C, CHANNELS_1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(POOL_SIZE),
            nn.Conv2d(CHANNELS_1, CHANNELS_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(POOL_SIZE),
        )

        # Calculate the size after conv layers dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, C, H, W)
            conv_output = self.conv(dummy_input)
            conv_out_dim = conv_output.view(1, -1).size(1)

        self.fc = nn.Linear(conv_out_dim, feature_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # make a vector
        x = self.fc(x)
        return x


class ConvRecClassifier(nn.Module):
    def __init__(
        self, 
        feature_dim,  # number of features after a convolutional encoder
        hidden_dim, 
        num_classes, 
        input_size=(1, INIT_MASK_SIZE, INIT_MASK_SIZE)
    ):
        super(ConvRecClassifier, self).__init__()
        self.cnn_encoder = CNNEncoder(feature_dim, input_size=input_size)
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_seq):
        # x_seq: (batch_size, seq_len, C, H, W)
        batch_size, seq_len, C, H, W = x_seq.shape
        # encoding
        x_seq = x_seq.view(batch_size * seq_len, C, H, W)
        features = self.cnn_encoder(x_seq)  # (batch_size * seq_len, feature_dim)
        features = features.view(batch_size, seq_len, -1)  # (batch_size, seq_len, feature_dim)
        # recurrent part
        lstm_out, _ = self.lstm(features)
        last_hidden = lstm_out[:, -1, :]
        # final layer
        out = self.fc(last_hidden)
        return out