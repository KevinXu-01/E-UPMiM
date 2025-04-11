# model file: a base model class for embedding, and our e-upmim model that inherits from the base model

import os
import torch
from torch import nn
import torch.nn.functional as F
from utils import hard_attention, UserProfileInterestAttention, GCNEmbedding
import numpy as np
from torch.distributions import Categorical

class Model_Comi_Rec(nn.Module):
    def __init__(self, n_user, n_mid, embedding_dim, hidden_size, batch_size, num_interest, num_layer, seq_len=10, hard_readout=True, relu_layer=True, device="cuda:0"):
        super(Model_Comi_Rec, self).__init__()
        self.device = device
        self.reg = False
        self.batch_size = batch_size
        self.n_uid = n_user # total number of users in the dataset
        self.n_mid = n_mid # total number of items in the dataset
        self.neg_num = 10
        self.seq_len = seq_len
        self.num_layer = num_layer
        self.embedding_dim = embedding_dim
        self.num_interest = num_interest
        self.hidden_size = hidden_size
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        
        self.mid_embeddings_var = nn.Embedding(self.n_mid, self.embedding_dim) # item embedding matrix
        nn.init.xavier_normal_(self.mid_embeddings_var.weight)

        # 来自Comi_Rec, 未实际用到，但在sampled_softmax中需要有对应tensor名
        self.mid_embeddings_bias = nn.Embedding(self.n_mid, 1)
        nn.init.zeros_(self.mid_embeddings_bias.weight)
        self.mid_embeddings_bias.weight.requires_grad = False

        self.user_embeddings_var = nn.Embedding(self.n_uid, self.embedding_dim)
        nn.init.xavier_normal_(self.user_embeddings_var.weight)

        self.capsule_network = CapsuleNetwork(self.hidden_size, self.seq_len, bilinear_type=2, num_interest=self.num_interest, hard_readout=hard_readout, relu_layer=relu_layer)
        # self.multi_interest_network = AutoregressiveMultiInterest(input_dim=self.embedding_dim, hidden_dim=self.hidden_size, num_interests=self.num_interest)
        

    def build_softmax_ce_loss(self, item_emb, user_emb):
        # parameter loss
        l2_loss = 1e-5 * sum(torch.sum(torch.pow(p.float(), 2)) * 0.5 for p in self.parameters())
        
        # adj_loss
        # adj_l1_loss = 1e-5 * self.adj_l1
        
        # sparse softmax cross entropy with logits loss
        neg_sampling_loss = torch.mean(F.cross_entropy(input = user_emb, target = item_emb))
        
        loss = l2_loss + neg_sampling_loss
        return loss
    

    def sampled_softmax_loss(self, weights, biases, inputs, labels, num_sampled, num_classes, temperature=1.0):
        """
        Sampled softmax loss implementation in PyTorch.

        Args:
            weights (torch.Tensor): Embedding weights of shape [num_classes, embedding_dim].
            biases (torch.Tensor): Biases of shape [num_classes].
            inputs (torch.Tensor): Input embeddings of shape [batch_size, embedding_dim].
            labels (torch.Tensor): Target labels of shape [batch_size].
            num_sampled (int): Number of negative samples to draw.
            num_classes (int): Total number of classes (items).
            temperature (float): Temperature parameter for softmax.

        Returns:
            torch.Tensor: Sampled softmax loss.
        """
        batch_size = int(labels.size(0))
        # import pdb; pdb.set_trace()

        # Sample negative labels
        negative_distribution = torch.ones(num_classes) / num_classes  # Uniform distribution
        negative_samples = Categorical(probs=negative_distribution).sample(torch.Size([batch_size, num_sampled]))  # [batch_size, num_sampled]
        # Combine positive and negative samples
        # import pdb; pdb.set_trace()
        all_samples = torch.cat([labels, negative_samples], dim=1)  # [batch_size, 1 + num_sampled]

        # Gather the weights and biases for the sampled classes
        sampled_weights = weights[all_samples]  # [batch_size, 1 + num_sampled, embedding_dim]
        sampled_biases = biases[all_samples]  # [batch_size, 1 + num_sampled]

        # Compute logits
        logits = torch.einsum('bd,bnd->bn', inputs, sampled_weights) + sampled_biases  # [batch_size, 1 + num_sampled]
        logits /= temperature  # Apply temperature scaling

        # Labels are always the first sample (positive sample)
        true_labels = torch.zeros(batch_size, dtype=torch.long, device=inputs.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, true_labels, reduction='mean')

        return loss

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
    def __init__(self, dim, seq_len, bilinear_type=0, num_interest=4, hard_readout=True, relu_layer=False):
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
        item_eb = item_eb.reshape(-1, self.dim, 1)
        atten = torch.matmul(interest_capsule, item_eb)
        
        atten = torch.nn.functional.softmax(torch.pow(torch.reshape(atten, [-1, self.num_interest]), 1))

        if self.hard_readout:
            readout = torch.reshape(interest_capsule, [-1, self.dim])[torch.argmax(atten, dim=1) + torch.arange(item_his_emb.shape[0]) * self.num_interest]
        else:
            readout = torch.matmul(torch.reshape(atten, [get_shape(item_his_emb)[0], 1, self.num_interest]), interest_capsule)
            readout = torch.reshape(readout, [get_shape(item_his_emb)[0], self.dim])

        return interest_capsule, readout

class AutoregressiveMultiInterest(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, num_interests=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.K = num_interests
        
        # 历史行为编码层 (可替换为Transformer/GRU)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
            )
        
        # 门控注意力参数
        self.W_q = nn.Linear(hidden_dim, hidden_dim)  # 查询变换
        self.W_h = nn.Linear(hidden_dim, hidden_dim)  # 历史行为变换
        self.W_a = nn.Linear(hidden_dim, 1)          # 注意力得分
        
        # 自回归查询生成MLP
        self.query_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear((k + 2) * hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for k in range(self.K - 1)
        ])
        
    def forward(self, X, seq_mask=None):
        """
        输入: 
            X: [batch_size, seq_len, input_dim] 历史行为序列
            seq_mask: [batch_size, seq_len] 序列有效位mask (1=真实行为, 0=填充位置)
        输出:
            V: [batch_size, K, hidden_dim] K个解耦的兴趣向量
            attn_weights: [batch_size, K, seq_len] 注意力权重（可选）
        """
        batch_size, seq_len, dim = X.shape
        
        # 1. 编码历史行为
        H = self.encoder(X)  # [batch, seq_len, hidden_dim]
        
        # 2. 全局上下文向量 (平均池化)
        if seq_mask is None:
            seq_mask = torch.ones(batch_size, seq_len, device=X.device)
        seq_mask = seq_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        c = (H * seq_mask).sum(dim=1) / (seq_mask.sum(dim=1) + 1e-6)  # [batch, hidden_dim]

        # 3. 自回归生成K个兴趣
        V = []
        for k in range(self.K):
            # 生成当前兴趣的查询向量
            if k == 0:
                q_k = c  # 第一个兴趣仅用全局上下文
            else:
                # 拼接已有兴趣和全局上下文 [batch, k*hidden_dim + hidden_dim]
                prev_V = torch.cat([v.squeeze(1) for v in V], dim=-1)
                combined = torch.cat([prev_V, c], dim=-1)
                q_k = self.query_mlps[k-1](combined)  # 使用对应的MLP层
            # 门控注意力机制
            # 计算注意力得分 [batch, seq_len, 1]
            scores = self.W_a(torch.tanh(
                self.W_q(q_k).unsqueeze(1) + self.W_h(H)
            ))
            
            # 关键步骤：padding位置赋极小值
            scores = scores.masked_fill(seq_mask == 0, -1e9)
            alpha = F.softmax(scores, dim=1)  # [batch, seq_len, 1]
            
            # 生成兴趣向量 [batch, hidden_dim]
            v_k = torch.sum(alpha * H * seq_mask, dim=1)
            
            # 正交化处理 (Gram-Schmidt)
            if k > 0:  # 第一个兴趣无需正交化
                # 获取所有已生成兴趣（确保每个v_j是[batch, hidden_dim]）
                prev_interests = torch.cat([v.squeeze(1) for v in V], dim=0)  # [k*batch, hidden_dim]
                prev_interests = prev_interests.view(k, -1, self.hidden_dim)  # [k, batch, hidden_dim]
            
            # 逐兴趣正交化
            for j in range(k):
                v_j = prev_interests[j]  # [batch, hidden_dim]
                dot_product = torch.sum(v_k * v_j, dim=1, keepdim=True)  # [batch, 1]
                norm_sq = torch.sum(v_j * v_j, dim=1, keepdim=True) + 1e-6  # [batch, 1]
                proj = dot_product / norm_sq  # [batch, 1]
                v_k = v_k - proj * v_j  # [batch, hidden_dim]
            # 存储时增加维度
            V.append(v_k.unsqueeze(1))  # [batch, 1, hidden_dim]
        return torch.cat(V, dim=1)  # [batch, K, hidden_dim]


def normalize_adj_tensor(adj, seq_len, device):
    adj = adj + torch.unsqueeze(torch.from_numpy(np.eye(seq_len)), dim = 0).to(device=device)
    rowsum = torch.sum(adj, dim = 1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    candidate_a = torch.zeros_like(d_inv_sqrt)
    d_inv_sqrt = torch.where(torch.isinf(d_inv_sqrt), candidate_a, d_inv_sqrt)
    d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
    norm_adg = torch.matmul(d_mat_inv_sqrt, adj)
    return norm_adg

