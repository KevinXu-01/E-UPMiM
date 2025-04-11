# model file: E-UPMiM

import torch
from torch import nn
import torch.nn.functional as F
from utils import hard_attention, UserProfileInterestAttention, GCNEmbedding, normalize_adj_tensor
import numpy as np

class Model_E_UPMiM(nn.Module):
    def __init__(self, n_user, n_mid, embedding_dim, hidden_size, batch_size, num_interest, num_layer, seq_len=10, hard_readout=True, relu_layer=False, device="cuda:0"):
        super(Model_E_UPMiM, self).__init__()
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
        
        # Embedding layer
        self.user_age_embedding_matrix = nn.Embedding(57, self.embedding_dim)
        nn.init.xavier_normal_(self.user_age_embedding_matrix.weight)
        self.user_gender_embedding_matrix = nn.Embedding(2, self.embedding_dim)
        nn.init.xavier_normal_(self.user_gender_embedding_matrix.weight)
        self.user_occup_embedding_matrix = nn.Embedding(21, self.embedding_dim)
        nn.init.xavier_normal_(self.user_occup_embedding_matrix.weight)
        self.mid_embeddings_var = nn.Embedding(self.n_mid, self.embedding_dim) # item embedding matrix
        nn.init.xavier_normal_(self.mid_embeddings_var.weight)
        self.dropout = nn.Dropout(0.3)

        self.mid_embeddings_bias = nn.Embedding(self.n_mid, 1)
        nn.init.zeros_(self.mid_embeddings_bias.weight)
        self.mid_embeddings_bias.weight.requires_grad = False

        self.user_embeddings_var = nn.Embedding(self.n_uid, self.embedding_dim)
        nn.init.xavier_normal_(self.user_embeddings_var.weight)
        # self.capsule_network = CapsuleNetwork(self.hidden_size, seq_len, bilinear_type=2, num_interest=self.num_interest, hard_readout=hard_readout, relu_layer=relu_layer)
        self.multi_interest_network = AutoregressiveMultiInterest(input_dim=self.embedding_dim, hidden_dim=self.hidden_size, num_interests=self.num_interest)
        self.linear_units = [self.hidden_size // 2, self.hidden_size]
        self.fc_layers = nn.ModuleList()
        for i, units in enumerate(self.linear_units):
            input_dim = self.embedding_dim * 2 if i < len(self.linear_units) - 1 else self.linear_units[i - 1]
            layer = nn.Linear(in_features = input_dim, out_features=units, bias=True)
            nn.init.zeros_(layer.bias)
            self.fc_layers.append(layer)
        self._user_profile_interest_attention = UserProfileInterestAttention(emb = self.embedding_dim)
        self.gcn_embedding = GCNEmbedding(num_layer = self.num_layer - 1, embedding_dim = self.embedding_dim, se_num = 0, batch_size = self.batch_size, 
                                          seq_len = self.seq_len, layer_size = [self.embedding_dim, self.embedding_dim, self.embedding_dim])

    def build_softmax_ce_loss(self, item_emb, user_emb):
        # parameter loss
        l2_loss = 1e-5 * sum(torch.sum(torch.pow(p.float(), 2)) * 0.5 for p in self.parameters())
        
        # adj_loss
        adj_l1_loss = 1e-5 * self.adj_l1
        
        # sparse softmax cross entropy with logits loss

        neg_sampling_loss = torch.mean(F.cross_entropy(input = user_emb, target = item_emb))
        
        loss = l2_loss + adj_l1_loss + neg_sampling_loss
        return loss
    
    def vectorized_sample_negative_items(self, embedding, history, target, num_negatives=10):
        batch_size, seq_length = history.shape
        num_items = embedding.size(0)
        
        # 合并需要排除的item
        exclude = torch.cat([history, target.unsqueeze(1)], dim=1)  # (batch_size, seq_length+1)
        
        # 生成所有可能的item ID
        all_items = torch.arange(num_items, device=history.device).unsqueeze(0).expand(batch_size, -1)
        
        # 创建mask标记哪些item可以采样
        mask = torch.ones((batch_size, num_items), dtype=torch.bool, device=history.device)
        
        # 使用scatter_将需要排除的item置为False
        # 需要将exclude转换为适合scatter的索引
        batch_indices = torch.arange(batch_size, device=history.device).unsqueeze(1).expand(-1, exclude.size(1))
        mask.scatter_(1, exclude, False)
        
        # 为每个用户采样负样本
        # 由于每个用户的候选数量不同，我们需要使用torch.multinomial
        # 首先为所有候选item生成权重
        weights = mask.float()
        
        # 采样
        sampled_indices = torch.multinomial(weights, num_negatives, replacement=False)
        
        # 获取负样本的embedding
        neg_emb = embedding[sampled_indices]  # (batch_size, num_negatives, emb_dim)
        
        return sampled_indices, neg_emb


    def forward(self, flag, nick_id, user_age, user_gender, user_occup, item_id, hist_item, hist_mask):
        nick_id = torch.tensor(nick_id).to(device = self.device)
        user_age = torch.tensor(user_age).to(device = self.device)
        user_gender = torch.tensor(user_gender).to(device = self.device)
        user_occup = torch.tensor(user_occup).to(device = self.device)
        item_id = torch.tensor(item_id).to(device = self.device)
        hist_item = torch.tensor(hist_item).to(device = self.device)
        hist_mask = torch.tensor(hist_mask).to(device = self.device)
        self.uid_batch_embedded = self.user_embeddings_var.weight[nick_id]
        self.user_age_eb = self.user_age_embedding_matrix.weight[user_age]
        self.user_gender_eb = self.user_gender_embedding_matrix.weight[user_gender]
        self.user_occup_eb = self.user_occup_embedding_matrix.weight[user_occup]
        self.item_eb = self.mid_embeddings_var.weight[item_id]
        self.item_his_eb = self.mid_embeddings_var.weight[hist_item] * hist_mask.reshape(-1, self.seq_len, 1)
        if flag == "train":
            self.neg_item_id, self.neg_item_emb = self.vectorized_sample_negative_items(self.mid_embeddings_var.weight, hist_item, item_id, self.neg_num)
        user_profile = torch.stack([self.uid_batch_embedded, self.user_gender_eb, self.user_age_eb, self.user_occup_eb], dim=1)  # [batch_size, emb_size]
        user_profile = torch.mean(user_profile, dim = 1, keepdim = True)
        user_profile = torch.squeeze(user_profile, dim = 1)
        
        # from MGNM
        adj_l = torch.unsqueeze(self.item_his_eb, dim=2).repeat(1, 1, self.seq_len, 1)
        adj_r = torch.unsqueeze(self.item_his_eb, dim=1).repeat(1, self.seq_len, 1, 1)
        # apply user_emb
        adj_node = torch.multiply(adj_l, adj_r)
        adj_user = torch.unsqueeze(torch.unsqueeze(user_profile, dim=1), dim=2)
        adj = torch.sigmoid(torch.sum(adj_node*adj_user, dim=-1))
        adj = adj * torch.unsqueeze(hist_mask, dim=1)
        adj = adj * torch.unsqueeze(hist_mask, dim=2)
        self.adj_l1 = torch.norm(adj, p=1)

        adj = normalize_adj_tensor(adj, self.seq_len, device = self.device)
        # GCN layer
        all_embeddings = self.gcn_embedding(adj, self.item_his_eb)
        interest_list = []
        # readout_list = []
        user_profile = torch.unsqueeze(user_profile, dim=1).repeat(1, self.num_interest, 1)
        for l in range(self.num_layer - 1):
            interest = self.multi_interest_network(all_embeddings[l], hist_mask) # [batch_size, num_interest, embedding_dim]
            # interest = self.capsule_network(all_embeddings[l], self.item_eb, hist_mask)
            interest_list.append(interest)
            # readout_list.append(readout)
        interest_list_stack = torch.stack(interest_list)
        # readout_list_stack = torch.stack(readout_list)
        mean_interest = torch.mean(interest_list_stack, dim = 0) # meaning_pooling
        # mean_readout = torch.mean(readout_list_stack, dim = 0)
        user_attended_profile = self._user_profile_interest_attention(mean_interest, user_profile)
        self.user_eb = torch.concat([user_attended_profile, mean_interest], dim = -1)

        for i, units in enumerate(self.linear_units):
            # activation_fn = (tf.nn.relu if i < len(linear_units) - 1 else lambda x: x)
            self.user_eb = self.fc_layers[i](self.user_eb)
            if i < len(self.linear_units) - 1:
                self.user_eb = torch.relu(self.user_eb)
                self.dropout(self.user_eb)
            else:
                pass
        if flag == "train":
            self.item_pos_and_neg = torch.cat([torch.unsqueeze(self.item_eb, 1), self.neg_item_emb], dim=1)
            self.readout = hard_attention(self.user_eb, self.item_pos_and_neg)
            # compute loss
            user_item_product = torch.multiply(self.readout, self.item_pos_and_neg)
            self.distance = torch.sum(user_item_product, 2)
            self.sample_label = torch.reshape(torch.zeros_like(torch.sum(self.distance, 1), dtype=torch.int64), [-1])
            loss = self.build_softmax_ce_loss(self.sample_label, self.distance)
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
        return interest_capsule

class AutoregressiveMultiInterest(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, num_interests=3, encoder_type = "Transformer"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.K = num_interests
        self.encoder_type = encoder_type

        # 历史行为编码层
        if self.encoder_type == "Linear":
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            )
        elif self.encoder_type == "Transformer":
            encoder_layers = nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=4,
                dim_feedforward=4*hidden_dim,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)
        
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
        if self.encoder_type == "Linear":
            H = self.encoder(X)  # [batch, seq_len, hidden_dim]
            if seq_mask is None:
                seq_mask = torch.ones(batch_size, seq_len, device=X.device)
            seq_mask = seq_mask.unsqueeze(-1)  # [batch, seq_len, 1]
            c = (H * seq_mask).sum(dim=1) / (seq_mask.sum(dim=1) + 1e-6)  # [batch, hidden_dim]
        elif self.encoder_type == "Transformer":
            if seq_mask is None:
                src_key_padding_mask = None
            else:
                src_key_padding_mask = (seq_mask == 0)  # Transformer需要的mask格式
            
            H = self.encoder(X, src_key_padding_mask=src_key_padding_mask)  # [batch, seq_len, hidden_dim]
            
            if seq_mask == None:
                c = H.mean(dim=1)
            else:
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