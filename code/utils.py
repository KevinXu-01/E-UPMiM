# credit: copied from UMI

import tensorflow as tf
from tensorflow import contrib


def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape


def squash(vector, _, epsilon=1e-9):
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -1, True)
    scalar_factor = (vec_squared_norm / (1 + vec_squared_norm)
                     / tf.sqrt(vec_squared_norm + epsilon))
    vec_squashed = scalar_factor * vector  # element-wise
    return vec_squashed


routing_activations = {
    "squash": squash
}


#from MGNM
def _create_gcn_emb(A, x, num_layer, embedding_dim, se_num, batch_size, seq_len, layer_size=[64, 64, 64]):
    initializer = tf.random_normal_initializer(stddev=0.01)
    weights_size_list = [embedding_dim] + layer_size
    all_weights = {}
    with tf.variable_scope("weights", reuse=tf.AUTO_REUSE):
        for lay in range(num_layer):
            all_weights['W_gc%d' % lay] = tf.Variable(
                initializer([weights_size_list[lay], weights_size_list[lay+1]]), name='W_gc%d'%lay
            )
            all_weights['B_gc%d' % lay] = tf.Variable(
                initializer([1, weights_size_list[lay+1]]), name='b_gc%d'%lay
            )

    # gcn has num_layer layers
    all_embeddings = [tf.slice(x, [0, se_num, 0], [batch_size, seq_len, embedding_dim])]
    for k in range(num_layer):
        embeddings = tf.matmul(A, x)
        embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, all_weights['W_gc%d' % k]) + all_weights['B_gc%d' % k])
        all_embeddings.append(tf.slice(embeddings, [0, se_num, 0], [batch_size, seq_len, embedding_dim]))

    return all_embeddings



def attn(query, key):
    # key [batch_size, short_seq_length, units]
    # query [batch_size, units]
    alpha = tf.expand_dims(tf.nn.softmax(tf.reduce_sum(key * tf.expand_dims(query, axis=1), axis=-1), axis=-1), axis=-1) # [b_s, shrot_seq_length, 1]
    res = tf.reduce_sum(key * alpha, axis=1)
    return res

def hard_attention(interests, item_embeddings):
    atten = tf.matmul(item_embeddings, tf.transpose(interests, perm=[0, 2, 1]))
    atten = tf.nn.softmax(atten)

    atten = tf.cast(
        tf.equal(atten, tf.reduce_max(atten, -1, True)),
        dtype=tf.float32)
    readout = tf.matmul(atten, interests)
    return readout

##带有UGANet
def dual_softmax_routing(inputs, user_profile, num_caps, dim_caps, inputs_mask, stddev_b=1, routing_iter=3,
                         inner_activation="squash",
                         last_activation="squash", scope="routing", bilinear_type=1):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        N, T, C = get_shape(inputs)

        with tf.variable_scope("user_attention_net", reuse=tf.AUTO_REUSE):
            user_profile = tf.tile(tf.expand_dims(user_profile, axis=1), [1, T, 1])
            second_attention_weights = tf.concat([inputs, user_profile], axis=-1)
            second_attention_units = [8, 1]
            bias = True
            biases_initializer = tf.zeros_initializer() if bias else None
            for i, units in enumerate(second_attention_units):
                second_attention_weights = contrib.layers.fully_connected(
                    second_attention_weights, units, activation_fn=None,
                    biases_initializer=biases_initializer)
                if i == 0:
                    activation_fn = tf.nn.relu
                else:
                    activation_fn = tf.nn.sigmoid
                second_attention_weights = activation_fn(second_attention_weights)
            second_attention_weights = tf.expand_dims(second_attention_weights, -1) #UGANet

        inputs_mask = tf.reshape(inputs_mask, [N, T, 1, 1])
        if bilinear_type == 0:
            u_hat = contrib.layers.fully_connected(inputs, dim_caps, activation_fn=None,
                                           biases_initializer=None)
            u_hat = tf.tile(u_hat, [1, 1, num_caps])
            u_hat = tf.reshape(u_hat, [N, T, num_caps, dim_caps])
        else:
            u_hat = contrib.layers.fully_connected(inputs, num_caps * dim_caps, activation_fn=None,
                                           biases_initializer=None)
            u_hat = tf.reshape(u_hat, [N, T, num_caps, dim_caps])
        u_hat_stopped = tf.stop_gradient(u_hat, name='u_hat_stop')
        b_ij = tf.truncated_normal([N, T, num_caps, 1], stddev=stddev_b)
        b_ij = tf.stop_gradient(b_ij)
        second_attention_weights_stopped = tf.stop_gradient(second_attention_weights)

        for i in range(routing_iter):
            with tf.variable_scope('iter_' + str(i)):
                c_ij = tf.nn.softmax(b_ij, 2)
                c_ij = c_ij * inputs_mask

                if i < routing_iter - 1:
                    s_j = tf.multiply(c_ij, u_hat_stopped)
                    s_j = tf.multiply(s_j, second_attention_weights_stopped)
                    s_j = tf.reduce_sum(s_j, 1, True)
                    v_j = routing_activations[inner_activation](s_j, c_ij)

                    u_produce_v = u_hat_stopped * v_j
                    b_ij = tf.reduce_sum(u_produce_v, -1, True)
                else:
                    s_j = tf.multiply(c_ij, u_hat)
                    s_j = tf.multiply(s_j, second_attention_weights)
                    s_j = tf.reduce_sum(s_j, 1, True)
                    v_j = routing_activations[last_activation](s_j, c_ij)
        caps = tf.squeeze(v_j, axis=1)
        return caps


def _user_profile_interest_attention(user_interests, user_profile):
    bs, num_all_interest, user_emb_size = get_shape(user_interests)
    user_attention_weights = tf.concat([user_profile, user_interests], axis=-1)  # [None, 6, 40 + 352]
    user_attention_units = [8, 3]

    with tf.variable_scope("user_attention_net", reuse=tf.AUTO_REUSE):
        biases_initializer = tf.zeros_initializer()

        for i, units in enumerate(user_attention_units):
            activation_fn = (tf.nn.relu if i < len(user_attention_units) - 1
                             else tf.nn.sigmoid)
            user_attention_weights = contrib.layers.fully_connected(
                user_attention_weights, units, activation_fn=None,
                biases_initializer=biases_initializer)
            user_attention_weights = activation_fn(user_attention_weights)
    # print(user_attention_weights.shape)
    user_multi_features = tf.reshape(user_profile, [bs, num_all_interest, user_emb_size])  # [None, 6, 5, 8]
    # print(user_multi_features.shape)

    pad_size = int(user_multi_features.shape[2] - user_attention_weights.shape[2])
    pad_spec = tf.constant([[0, 0], [0, 0], [0, pad_size]])
    user_attention_weights_padded = tf.pad(user_attention_weights, pad_spec, "CONSTANT")

    user_attended_features = tf.multiply(user_multi_features, user_attention_weights_padded)  # [None, 6, 5, 1]

    user_attended_profile = tf.reshape(user_attended_features, [bs, num_all_interest, user_emb_size])

    return user_attended_profile
