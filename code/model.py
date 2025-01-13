# model file: a base model class for embedding, and our e-upmim model that inherits from the base model

import os
import tensorflow as tf
from tensorflow import contrib
from utils import hard_attention, dual_softmax_routing, _user_profile_interest_attention, attn, _create_gcn_emb


class Model(object):
    def __init__(self, n_uid, n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="E-UPMiM"):
        self.model_flag = flag
        self.reg = False
        self.batch_size = batch_size
        self.n_uid = n_uid # total number of users in the dataset
        self.n_mid = n_mid # total number of items in the dataset
        self.neg_num = 5
        with tf.name_scope('Inputs'):
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.user_age = tf.placeholder(tf.int32, [None, ], name='user_age')
            self.user_gender = tf.placeholder(tf.int32, [None, ], name='user_gender')
            self.user_occup = tf.placeholder(tf.int32, [None, ], name='user_occup')
            
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask_batch_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, None], name='target_ph')

            self.lr = tf.placeholder(tf.float64, [])
        self.mask_length = tf.cast(tf.reduce_sum(self.mask, -1), dtype=tf.int32)

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.user_age_embedding_matrix = tf.get_variable("user_age_embedding_matrix", [n_uid, embedding_dim],
                                                             trainable=True)
            
            self.user_gender_embedding_matrix = tf.get_variable("user_gender_embedding_matrix",
                                                                [n_uid, embedding_dim],
                                                                trainable=True)

            self.user_occup_embedding_matrix = tf.get_variable("user_occup_embedding_matrix",
                                                               [n_uid, embedding_dim],
                                                               trainable=True)
            
            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, embedding_dim], trainable=True) # item embedding matrix

            # 来自Comi_Rec, 未实际用到，但在sampled_softmax中需要有对应tensor名
            self.mid_embeddings_bias = tf.get_variable("bias_lookup_table", [n_mid], initializer=tf.zeros_initializer(), trainable=False)
            
            self.user_embeddings_var = tf.get_variable('uid_embedding_var', [n_uid, embedding_dim], trainable=True)
            
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.user_embeddings_var, self.uid_batch_ph)
            self.user_age_eb = tf.nn.embedding_lookup(self.user_age_embedding_matrix, self.user_age)
            self.user_gender_eb = tf.nn.embedding_lookup(self.user_gender_embedding_matrix, self.user_gender)
            self.user_occup_eb = tf.nn.embedding_lookup(self.user_occup_embedding_matrix, self.user_occup)

        self.item_eb = self.mid_batch_embedded
        self.item_his_eb = self.mid_his_batch_embedded * tf.reshape(self.mask, (-1, seq_len, 1))


    def build_softmax_ce_loss(self, item_emb, user_emb):
        # parameter loss
        l2_loss = 1e-5 * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
        
        # adj_loss
        adj_l1_loss = 1e-5 * self.adj_l1
        
        # sparse softmax cross entropy with logits loss
        neg_sampling_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=item_emb, logits=user_emb))
        
        self.loss = l2_loss + adj_l1_loss + neg_sampling_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)


    def train(self, sess, inps):
        feed_dict = {
            self.uid_batch_ph: inps[0],
            self.user_age: inps[1],
            self.user_gender: inps[2],
            self.user_occup: inps[3],
            self.mid_batch_ph: inps[4], # target item embedding
            self.mid_his_batch_ph: inps[5], # historical item embedding
            self.mask: inps[6],
            self.lr: inps[7]
        }
        loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    def output_item(self, sess):
        item_embs = sess.run(self.mid_embeddings_var)
        return item_embs

    def output_user(self, sess, inps):
        user_embs = sess.run(self.user_eb, feed_dict={
            self.uid_batch_ph: inps[0],
            self.user_age: inps[1],
            self.user_gender: inps[2],
            self.user_occup: inps[3],
            self.mid_his_batch_ph: inps[4],
            self.mask: inps[5],
        })
        return user_embs
    
    
    def save(self, sess, path):
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, path + 'model.ckpt')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path + 'model.ckpt')
        print('model restored from %s' % path)


def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])
    return shape


class CapsuleNetwork(tf.layers.Layer):
    def __init__(self, dim, seq_len, bilinear_type=2, num_interest=4, hard_readout=True, relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.bilinear_type = bilinear_type
        self.num_interest = num_interest
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True

    def call(self, item_his_emb, item_eb, mask):
        with tf.variable_scope('bilinear'):
            if self.bilinear_type == 0:
                item_emb_hat = tf.layers.dense(item_his_emb, self.dim, activation=None, bias_initializer=None)
                item_emb_hat = tf.tile(item_emb_hat, [1, 1, self.num_interest])
            elif self.bilinear_type == 1:
                item_emb_hat = tf.layers.dense(item_his_emb, self.dim * self.num_interest, activation=None, bias_initializer=None)
            else:
                w = tf.get_variable(
                    'weights', shape=[1, self.seq_len, self.num_interest * self.dim, self.dim],
                    initializer=tf.random_normal_initializer())
                u = tf.expand_dims(item_his_emb, axis=2)
                item_emb_hat = tf.reduce_sum(w[:, :self.seq_len, :, :] * u, axis=3)

        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.seq_len, self.num_interest, self.dim])
        item_emb_hat = tf.transpose(item_emb_hat, [0, 2, 1, 3])
        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.num_interest, self.seq_len, self.dim])

        if self.stop_grad:
            item_emb_hat_iter = tf.stop_gradient(item_emb_hat, name='item_emb_hat_iter')
        else:
            item_emb_hat_iter = item_emb_hat

        if self.bilinear_type > 0:
            capsule_weight = tf.stop_gradient(tf.zeros([get_shape(item_his_emb)[0], self.num_interest, self.seq_len]))
        else:
            capsule_weight = tf.stop_gradient(tf.truncated_normal([get_shape(item_his_emb)[0], self.num_interest, self.seq_len], stddev=1.0))

        for i in range(3):
            atten_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_interest, 1])
            paddings = tf.zeros_like(atten_mask)

            capsule_softmax_weight = tf.nn.softmax(capsule_weight, axis=1)
            capsule_softmax_weight = tf.where(tf.equal(atten_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = tf.expand_dims(capsule_softmax_weight, 2)

            if i < 2:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat_iter)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = tf.matmul(item_emb_hat_iter, tf.transpose(interest_capsule, [0, 1, 3, 2]))
                delta_weight = tf.reshape(delta_weight, [-1, self.num_interest, self.seq_len])
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = tf.reshape(interest_capsule, [-1, self.num_interest, self.dim])

        if self.relu_layer:
            interest_capsule = tf.layers.dense(interest_capsule, self.dim, activation=tf.nn.relu, name='proj')
        return interest_capsule

def normalize_adj_tensor(adj, seq_len):
    adj = adj + tf.expand_dims(tf.eye(seq_len), axis=0)
    rowsum = tf.reduce_sum(adj, axis=1)
    d_inv_sqrt = tf.pow(rowsum, -0.5)
    candidate_a = tf.zeros_like(d_inv_sqrt)
    d_inv_sqrt = tf.where(tf.math.is_inf(d_inv_sqrt), candidate_a, d_inv_sqrt)
    d_mat_inv_sqrt = tf.matrix_diag(d_inv_sqrt)
    norm_adg = tf.matmul(d_mat_inv_sqrt, adj)
    return norm_adg

class Model_E_UPMiM(Model):
    def __init__(self, n_user, n_mid, embedding_dim, hidden_size, batch_size, num_interest, num_layer, seq_len=10, hard_readout=True, relu_layer=False):
        super(Model_E_UPMiM, self).__init__(n_user, n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="Model_E_UPMiM")
        user_profile = tf.stack([self.uid_batch_embedded, self.user_gender_eb, self.user_age_eb, self.user_occup_eb], axis=1)  # [batch_size, emb_size]
        user_profile = tf.reduce_mean(user_profile, axis = 1, keepdims = True)
        user_profile = tf.squeeze(user_profile, axis = 1)
        item_his_emb = self.item_his_eb
        capsule_network = CapsuleNetwork(hidden_size, seq_len, bilinear_type=2, num_interest=num_interest, hard_readout=hard_readout, relu_layer=relu_layer)
        # from MGNM
        self.num_layer = num_layer
        adj_l = tf.tile(tf.expand_dims(self.item_his_eb, axis=2), [1, 1, seq_len, 1])
        adj_r = tf.tile(tf.expand_dims(self.item_his_eb, axis=1), [1, seq_len, 1, 1])
        # apply user_emb
        adj_node = tf.multiply(adj_l, adj_r)
        adj_user = tf.expand_dims(tf.expand_dims(user_profile, axis=1), axis=2)
        adj = tf.nn.sigmoid(tf.reduce_sum(adj_node*adj_user, axis=-1))
        adj = adj * tf.expand_dims(self.mask, axis=1)
        adj = adj * tf.expand_dims(self.mask, axis=2)
        self.adj_l1 = tf.norm(adj, ord=1)

        adj = normalize_adj_tensor(adj, seq_len)
        # GCN layer
        all_embeddings = _create_gcn_emb(adj, self.mid_his_batch_embedded, num_layer-1, embedding_dim, 0, batch_size, seq_len, layer_size=[embedding_dim, embedding_dim, embedding_dim])
        interest_list = []
        user_profile = tf.tile(tf.expand_dims(user_profile, axis=1), [1, num_interest, 1])
        for l in range(num_layer):
            interest = capsule_network(all_embeddings[l], self.item_eb, self.mask) # [batch_size, num_interest, embedding_dim]
            interest_list.append(interest)
        interest_list_stack = tf.stack(interest_list)
        mean_interest = tf.reduce_mean(interest_list_stack, axis = 0) # meaning_pooling
        user_attended_profile = _user_profile_interest_attention(mean_interest, user_profile)
        self.user_eb = tf.concat([user_attended_profile, mean_interest], axis=-1)
        with tf.variable_scope("linear_user_net", reuse=tf.AUTO_REUSE):
            linear_units = [int(hidden_size // 2), hidden_size]
            bias = True
            biases_initializer = tf.zeros_initializer() if bias else None

            for i, units in enumerate(linear_units):
                activation_fn = (tf.nn.relu if i < len(linear_units) - 1
                                else lambda x: x)
                self.user_eb = contrib.layers.fully_connected(
                    self.user_eb, units, activation_fn=None,
                    biases_initializer=biases_initializer)
                self.user_eb = activation_fn(self.user_eb)
        self.readout = hard_attention(self.user_eb, tf.expand_dims(self.mid_batch_embedded, 1))

        # compute loss
        user_item_product = tf.multiply(self.readout, tf.expand_dims(self.mid_batch_embedded, 1))
        self.distance = tf.reduce_sum(user_item_product, 2)
        self.sample_label = tf.reshape(tf.zeros_like(tf.reduce_sum(self.distance, 1), dtype=tf.int64), [-1])

        self.build_softmax_ce_loss(self.sample_label, self.distance)
        
