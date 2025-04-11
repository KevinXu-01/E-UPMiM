import os
import torch
from torch import nn
import torch.nn.functional as F
from utils import hard_attention, UserProfileInterestAttention, GCNEmbedding
import numpy as np
from torch.distributions import Categorical

class Model_Comi_Rec(nn.Module):
    def __init__(self, n_user, n_mid, embedding_dim, hidden_size, batch_size, num_interest, num_layer, seq_len=10, hard_readout=True, relu_layer=False, device="cuda:0"):
        super(Model_Comi_Rec, self).__init__()
        self.device = device
        self.reg = False
        self.batch_size = batch_size
        self.n_uid = n_user # total number of users in the dataset
        self.n_mid = n_mid # total number of items in the dataset
        self.neg_num = 10
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.num_interest = num_interest
        self.hidden_size = hidden_size
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        
        self.mid_embeddings_var = nn.Embedding(self.n_mid, self.embedding_dim) # item embedding matrix
        nn.init.xavier_normal_(self.mid_embeddings_var.weight)

        self.mid_embeddings_bias = nn.Embedding(self.n_mid, 1)
        nn.init.zeros_(self.mid_embeddings_bias.weight)
        self.mid_embeddings_bias.weight.requires_grad = False

        self.user_embeddings_var = nn.Embedding(self.n_uid, self.embedding_dim)
        nn.init.xavier_normal_(self.user_embeddings_var.weight)

        self.capsule_network = CapsuleNetwork(self.hidden_size, seq_len, bilinear_type=2, num_interest=self.num_interest, hard_readout=hard_readout, relu_layer=relu_layer)

    def build_sampled_softmax_loss(self, item_id, user_emb):
        # Sample negative indices
        neg_samples = torch.randint(0, self.n_mid, (self.batch_size * self.neg_num,), device=user_emb.device)
        # Combine positive and negative samples
        all_samples = torch.cat([item_id.clone().detach(), neg_samples])
        # Gather selected embeddings and biases
        selected_emb = self.mid_embeddings_var.weight[all_samples].reshape(self.batch_size, self.neg_num+1, -1)
        selected_bias = self.mid_embeddings_bias.weight[all_samples].reshape(self.batch_size, self.neg_num+1, -1)
        # Compute logits
        user_embedding_expanded = user_emb.unsqueeze(2)  # (batch_size, 1, emb_dim)
        logits = torch.bmm(selected_emb, user_embedding_expanded) + selected_bias
        logits = logits.reshape(self.batch_size, self.neg_num+1)
        # Create labels (first batch_size items are positives, rest are negatives)
        labels = torch.zeros(self.batch_size, self.neg_num + 1, device=user_emb.device)
        labels[:, 0] = 1  # first column is positive
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        return loss

    def forward(self, flag, nick_id, user_age, user_gender, user_occup, item_id, hist_item, hist_mask):
        nick_id = torch.tensor(nick_id).to(device = self.device)
        item_id = torch.tensor(item_id).to(device = self.device)
        hist_item = torch.tensor(hist_item).to(device = self.device)
        hist_mask = torch.tensor(hist_mask).to(device = self.device)
        self.uid_batch_embedded = self.user_embeddings_var.weight[nick_id]
        self.item_eb = self.mid_embeddings_var.weight[item_id]
        self.item_his_eb = self.mid_embeddings_var.weight[hist_item] * hist_mask.reshape(-1, self.seq_len, 1)

        item_his_emb = self.item_his_eb
        self.user_eb, self.readout = self.capsule_network(item_his_emb, self.item_eb, hist_mask)
        if flag == "train":
            loss = self.build_sampled_softmax_loss(item_id, self.readout)
        else:
            loss = 0
        return self.user_eb, loss

    def output_item(self):
        item_embs = self.mid_embeddings_var.weight
        return item_embs

    def output_user(self, user_id):
        user_embs = self.user_embeddings_var.weight[user_id]
        return user_embs

def get_shape(inputs):
    dynamic_shape = list(inputs.shape)
    static_shape = inputs.size()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])
    return shape


class CapsuleNetwork(nn.Module):
    def __init__(self, dim, seq_len, bilinear_type=2, num_interest=4, hard_readout=True, relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.bilinear_type = bilinear_type
        self.num_interest = num_interest
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True

        if self.bilinear_type == 2:
            self.weights = nn.Parameter(torch.randn(1, self.seq_len, self.num_interest * self.dim, self.dim))

        if self.relu_layer:
            self.proj = nn.Linear(self.dim, self.dim)
        
        if self.bilinear_type == 0:
            self.fc1 = nn.Linear(self.dim, self.dim, bias=False)
        elif self.bilinear_type == 1:
            self.fc1 = nn.Linear(self.dim, self.dim * self.num_interest, bias=False)
        else:
            self.fc1 = None
            
    def forward(self, item_his_emb, item_eb, mask):
        if self.bilinear_type == 0:
            item_emb_hat = self.fc1(item_his_emb)
            item_emb_hat = item_emb_hat.unsqueeze(2).repeat(1, 1, self.num_interest, 1)
        elif self.bilinear_type == 1:
            item_emb_hat = self.fc1(item_his_emb)
            item_emb_hat = item_emb_hat.view(-1, self.seq_len, self.num_interest, self.dim)
        else:
            u = item_his_emb.unsqueeze(2)
            item_emb_hat = torch.sum(self.weights[:, :self.seq_len, :, :] * u, dim=3)

        item_emb_hat = item_emb_hat.reshape(-1, self.seq_len, self.num_interest, self.dim)
        item_emb_hat = item_emb_hat.permute(0, 2, 1, 3)
        item_emb_hat = item_emb_hat.reshape(-1, self.num_interest, self.seq_len, self.dim)

        if self.stop_grad:
            item_emb_hat_iter = item_emb_hat.detach()
        else:
            item_emb_hat_iter = item_emb_hat

        if self.bilinear_type > 0:
            capsule_weight = torch.zeros(item_his_emb.size(0), self.num_interest, self.seq_len, device=item_his_emb.device).detach()
        else:
            capsule_weight = torch.randn(item_his_emb.size(0), self.num_interest, self.seq_len, device=item_his_emb.device).detach()

        for i in range(3):
            atten_mask = mask.unsqueeze(1).repeat(1, self.num_interest, 1)
            paddings = torch.zeros_like(atten_mask)

            capsule_softmax_weight = F.softmax(capsule_weight, dim=1)
            capsule_softmax_weight = torch.where(atten_mask == 0, paddings, capsule_softmax_weight)
            capsule_softmax_weight = capsule_softmax_weight.unsqueeze(2)

            if i < 2:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_emb_hat_iter)
                cap_norm = torch.sum(interest_capsule ** 2, dim=-1, keepdim=True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = torch.matmul(item_emb_hat_iter, interest_capsule.transpose(2, 3))
                delta_weight = delta_weight.view(-1, self.num_interest, self.seq_len)
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_emb_hat)
                cap_norm = torch.sum(interest_capsule ** 2, dim=-1, keepdim=True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = interest_capsule.view(-1, self.num_interest, self.dim)

        if self.relu_layer:
            interest_capsule = self.proj(interest_capsule)
            interest_capsule = F.relu(interest_capsule)
        
        atten = torch.matmul(interest_capsule, torch.reshape(item_eb, [-1, self.dim, 1]))
        
        atten = torch.nn.functional.softmax(torch.pow(torch.reshape(atten, [-1, self.num_interest]), 1))

        if self.hard_readout:
            readout = torch.reshape(interest_capsule, [-1, self.dim])[torch.argmax(atten, dim=1) + torch.arange(item_his_emb.shape[0]) * self.num_interest]
        else:
            readout = torch.matmul(torch.reshape(atten, [get_shape(item_his_emb)[0], 1, self.num_interest]), interest_capsule)
            readout = torch.reshape(readout, [get_shape(item_his_emb)[0], self.dim])

        return interest_capsule, readout