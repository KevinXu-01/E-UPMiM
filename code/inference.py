# using this file, you can get the target user's recommendation results (ids)

import faiss
import tensorflow as tf
from model import *
from data_iterator import DataIterator
import numpy as np

def get_model(model_type, user_count, item_count, batch_size, maxlen):
    if model_type == 'E-UPMiM':
        model = Model_E_UPMiM(user_count, item_count, 32, 32, batch_size, 3, 4, maxlen)
    else:
        print("Invalid model_type : %s", model_type)
        return
    return model

def prepare_test_data(src, target):
    nick_id, user_age, user_gender, user_occup, item_id = src
    hist_item, hist_mask = target
    return nick_id, user_age, user_gender, user_occup, item_id, hist_item, hist_mask

def inference(sess, batch_size, num_interest, test_data, model, topN = 10):
    item_embs = model.output_item(sess)
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 7
    # return
    try:
        gpu_index = faiss.GpuIndexFlatIP(res, 32, flat_config)
        gpu_index.add(item_embs)
    except Exception as e:
        return {}
    iii = 0
    for src, tgt in test_data:
        if iii >= 1:
            break
        nick_id, user_age, user_gender, user_occup, item_id, hist_item, hist_mask = prepare_test_data(src, tgt)
        user_embs = model.output_user(sess, [nick_id, user_age, user_gender, user_occup, hist_item, hist_mask])
        user_index = 0
        I = []
        while user_index < batch_size:
            user_index += 1
            preds = []
            for i in range(num_interest):
                _user_embs = user_embs[user_index:user_index+1, i, :]
                d, index = gpu_index.search(_user_embs, topN)
                preds.append(index)
            I.append(np.concatenate(preds))
        for i, iid_list in enumerate(item_id):
            item_list_set = set()
            true_item_set = set(iid_list)
            for no, iid in enumerate(I[i]):
                iid = iid.tolist()
                for item in iid:
                    if item in true_item_set:
                        item_list_set.add(item)
            print(f"user id: {nick_id[i]}, predictions: {item_list_set}")
        iii += 1


if __name__ == '__main__':
    gpu_options = tf.GPUOptions(allow_growth=True)
    n_uid = 6040
    n_mid = 3417
    batch_size = 128
    max_len = 10
    num_interest = 3
    model = get_model('E-UPMiM', n_uid, n_mid, batch_size, max_len)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, '../best_model/movielens_E-UPMiM_b128_lr0.001_d32_len10_topN10/')
        test_file = "../dataset/movielens_data/movielens_test.txt"
        test_data = DataIterator(test_file, batch_size, max_len, train_flag = 2)
        inference(sess, batch_size, num_interest, test_data, model)
        
