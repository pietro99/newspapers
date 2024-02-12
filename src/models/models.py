import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.a1(self.l1(x))
        x = self.l2(x)
        return self.softmax(x)
    
class NearestMeanClassifier(nn.Module):
    def __init__(self, num_classes, num_features):
        super().__init__()
        # Initialize centroids as learnable parameters
        self.num_classes = num_classes
        self.num_features = num_features
        self.centroids = nn.Parameter(torch.randn(num_classes, num_features))

    def compute_means(self, X, y):
        # Compute centroids based on input data and labels
        with torch.no_grad():
            class_counts = torch.zeros(self.num_classes)
            class_sums = torch.zeros(self.num_classes, self.num_features)
            for i in range(len(X)):
                class_idx = int(y[i])
                class_counts[class_idx] += 1
                class_sums[class_idx] += X[i]

            for i in range(self.num_classes):
                self.centroids[i] = class_sums[i] / class_counts[i]

    def forward(self, x):
        # Compute distances to centroids
        distances = -torch.norm(x[:, None] - self.centroids, dim=-1)
        return distances
    


class WordEmbeddings(nn.Module):
    def __init__(self, vocab_lenght, embed_dim, hidden, l1_dim, out_dim, num_lstm_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_lenght+1,embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=num_lstm_layers, batch_first=True )
        with torch.no_grad():
            self.embedding.weight[0] = torch.zeros(embed_dim) #padding index

        self.l1 = nn.Linear(hidden, l1_dim)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(l1_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x:torch.Tensor):
        embedding = self.embedding(x)
        embedding = nn.functional.normalize(embedding, dim=2) #embeddings should have norm of 1.
        
        lengths = torch.sum(torch.any((embedding != 0), dim=2), dim=1)
        packed_embeddings = pack_padded_sequence(input=embedding, lengths=lengths, batch_first=True, enforce_sorted=False)
        lstm_out, (a, b) = self.lstm(packed_embeddings)
        padded_output, _ = pad_packed_sequence(lstm_out, batch_first=True)
        idx = (lengths - 1).view(-1, 1).expand(len(lengths), padded_output.size(2)).unsqueeze(1)
        padded_output = padded_output.gather(1, idx).squeeze()
        out = self.r1(self.l1(padded_output))
        out = self.l2(out) 
        return  self.softmax(out)


class DocEmbeddings(nn.Module):
    def __init__(self, vocab_lenght, embed_dim, hidden, l1_dim, out_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_lenght+1,embed_dim, padding_idx=0)
        with torch.no_grad():
            self.embedding.weight[0] = torch.zeros(embed_dim)
        self.l1 = nn.Linear(hidden, l1_dim)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(l1_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x:torch.Tensor):
        embedding = self.embedding(x)
        torch.sum(embedding, dim=1)        
        norm_embed_sum = nn.functional.normalize(torch.sum(embedding, dim=1) )#embeddings should have norm of 1.
        out = self.r1(self.l1(norm_embed_sum))
        out = self.l2(out) 
        return  self.softmax(out)
