# credit: copied from UMI
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def get_shape(inputs):
    dynamic_shape = list(inputs.shape)
    static_shape = inputs.size()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])
    return shape

class GCNEmbedding(nn.Module):
    def __init__(self, num_layer, embedding_dim, se_num, batch_size, seq_len, layer_size=[64, 64, 64]):
        super(GCNEmbedding, self).__init__()
        self.num_layer = num_layer
        self.embedding_dim = embedding_dim
        self.se_num = se_num
        self.batch_size = batch_size
        self.seq_len = seq_len
        # self.layer_size = layer_size
        # self.weights_size_list = [embedding_dim] + layer_size
        # self.all_weights = nn.ParameterDict()
        
        # for lay in range(num_layer):
        #     self.all_weights['W_gc%d' % lay] = nn.Parameter(torch.randn(self.weights_size_list[lay], self.weights_size_list[lay+1]) * 0.01)
        #     self.all_weights['B_gc%d' % lay] = nn.Parameter(torch.randn(1, self.weights_size_list[lay+1]) * 0.01)

    def forward(self, A, x):
        A = A.to(dtype = torch.float32)
        # all_embeddings = [x[:, self.se_num:self.se_num+self.seq_len, :self.embedding_dim]]
        all_embeddings = []
        embeddings = x
        for k in range(self.num_layer):
            embeddings = torch.matmul(A, embeddings)
            # embeddings = F.leaky_relu(torch.matmul(embeddings, self.all_weights['W_gc%d' % k]) + self.all_weights['B_gc%d' % k])
            all_embeddings.append(embeddings[:, self.se_num:self.se_num+self.seq_len, :self.embedding_dim])
        return all_embeddings

def hard_attention(interests, item_embeddings):
    atten = torch.matmul(item_embeddings, interests.transpose(1, 2))
    atten = F.softmax(atten, dim=-1)
    atten = (atten == atten.max(dim=-1, keepdim=True)[0]).float()
    readout = torch.matmul(atten, interests)
    return readout

def normalize_adj_tensor(adj, seq_len, device):
    adj = adj + torch.unsqueeze(torch.from_numpy(np.eye(seq_len)), dim = 0).to(device=device)
    rowsum = torch.sum(adj, dim = 1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    candidate_a = torch.zeros_like(d_inv_sqrt)
    d_inv_sqrt = torch.where(torch.isinf(d_inv_sqrt), candidate_a, d_inv_sqrt)
    d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
    norm_adg = torch.matmul(d_mat_inv_sqrt, adj)
    return norm_adg


class UserProfileInterestAttention(nn.Module):
    def __init__(self, emb, user_attention_units=[8, 3]):
        super(UserProfileInterestAttention, self).__init__()
        self.user_attention_units = user_attention_units
        self.fc_layers = nn.ModuleList()
        self.dropout = nn.Dropout(0.3)
        for i, units in enumerate(user_attention_units):
            input_dim = emb * 2 if i == 0 else user_attention_units[i-1]
            activation_fn = (nn.ReLU() if i < len(user_attention_units) - 1 else nn.Sigmoid())
            self.fc_layers.append(nn.Linear(in_features = input_dim, out_features = units))
            self.fc_layers.append(activation_fn)

    def forward(self, user_interests, user_profile):
        bs, num_all_interest, user_emb_size = get_shape(user_interests)
        user_attention_weights = torch.cat([user_profile, user_interests], dim=-1)
        for layer in self.fc_layers:
            user_attention_weights = layer(user_attention_weights)
            user_attention_weights = self.dropout(user_attention_weights)
            
        user_multi_features = user_profile.view(bs, num_all_interest, user_emb_size)
        pad_size = int(user_multi_features.size(2) - user_attention_weights.size(2))
        user_attention_weights_padded = F.pad(user_attention_weights, (0, pad_size), "constant", 0)
        
        user_attended_features = user_multi_features * user_attention_weights_padded
        user_attended_profile = user_attended_features.view(bs, num_all_interest, user_emb_size)
        
        return user_attended_profile
