
from torch.nn import functional as F
from torch.nn import Module
import torch.nn as nn
import torch


class CNNClassifier(Module):
    """
    PyTorch classification convolutional neural network
    """
    def __init__(self, lookup_table):
        """
        Constructs layers of the CNN
        """
        super(CNNClassifier, self).__init__()

        self.embed_words = nn.Embedding(lookup_table.shape[0], lookup_table.shape[1])
        lookup_table = torch.tensor(lookup_table, dtype=torch.float64)
        self.embed_words.weight = nn.Parameter(lookup_table)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=2).double(), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3).double(),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=4).double(),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(300, 50).float()
        self.out = nn.Linear(50, 1).float()

    def forward(self, indices):
        """
        Performs forward pass of CNN
        :param indices: tensor of indices to embed
        :return:
        """
        indices = torch.tensor(indices, dtype=torch.long)
        embeddings = self.embed_words(indices)
        embeddings = embeddings.permute(0, 2, 1)

        convolved1 = self.conv1(embeddings)
        convolved2 = self.conv2(embeddings)
        convolved3 = self.conv3(embeddings)

        pooled_1 = F.max_pool1d(convolved1, convolved1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(convolved2, convolved2.shape[2]).squeeze(2)
        pooled_3 = F.max_pool1d(convolved3, convolved3.shape[2]).squeeze(2)

        cat = self.dropout(torch.cat((pooled_1, pooled_2, pooled_3), dim=1)).to(torch.float32)

        linear = self.fc1(cat)
        linear = F.relu(linear)
        return self.out(linear).squeeze(1).to(torch.float64)