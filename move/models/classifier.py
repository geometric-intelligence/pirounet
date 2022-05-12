import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearClassifier(nn.Module):
    def __init__(self, input_features, h_features_loop, label_features, seq_len):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(Classifier, self).__init__()
        self.dense = nn.Linear(input_features, h_features_loop)
        self.logits = nn.Linear(h_features_loop, label_features)

    def forward(self, x):
        x = F.relu(self.dense(x))
        x = F.softmax(self.logits(x), dim=-1)
        return x

# class TransformerClassifier(PositionalEncoding):
#     def __init__(self, input_features, h_features_loop, label_features, n_t_class_layers, latent_dim, dropout):
#         """
#         Transformer classifier
#         with softmax output.
#         """
#         super(TransformerClassifier, self).__init__()

#         self.dense = nn.Linear(input_features, h_features_loop)
#         self.sequence_pos_encoder = PositionalEncoding(d_model = )
#         self.embed = nn.Embedding(input_vocab_size, d_model)

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=512, 
#             nhead=8, 
#             batch_first=True
#         )
#         self.logits = nn.TransformerEncoder(
#             encoder_layer, 
#             num_layers = n_t_class_layers,

#         )

#     def forward(self, x):
#         x = F.relu(self.dense(x))
#         x = self.sequence_pos_encoder(x)
#         x = F.softmax(self.logits(x), dim=-1)
#         return x

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
        
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         # not used in the final model
#         x = x + self.pe[:x.shape[0], :]
#         return self.dropout(x)