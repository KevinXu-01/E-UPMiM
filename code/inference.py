# using this file, you can get the target user's recommendation results (ids)

import torch
from model import *
from comi_rec import *
from data_iterator import DataIterator
import numpy as np
from ann_search import find_topN_items

def get_model(model_type, user_count, item_count, batch_size, maxlen, device):
    if model_type == 'E-UPMiM':
        model = Model_E_UPMiM(user_count, item_count, 32, 32, batch_size, 3, 4, maxlen, device = device)
    elif model_type == 'Comi_Rec':
        model = Model_Comi_Rec(user_count, item_count, 64, 64, batch_size, 4, 4, maxlen, device = device)
    else:
        print("Invalid model_type : %s", model_type)
        return
    return model

def prepare_test_data(src, target):
    nick_id, user_age, user_gender, user_occup, item_id = src
    hist_item, hist_mask = target
    return nick_id, user_age, user_gender, user_occup, item_id, hist_item, hist_mask

def inference(batch_size, num_interest, test_data, model, topN = 10):
    model.eval()
    item_embs = model.output_item()
    
    for src, tgt in test_data:
        nick_id, user_age, user_gender, user_occup, item_id, hist_item, hist_mask = prepare_test_data(src, tgt)
        user_embs, _ = model("test", nick_id, user_age, user_gender, user_occup, hist_item, hist_item, hist_mask)
        user_index = 0
        I = []
        while user_index < batch_size:
            preds = []
            for i in range(num_interest):
                _user_embs = user_embs[user_index:user_index+1, i, :]
                d, index = find_topN_items(_user_embs.detach().cpu().numpy(), item_embs.detach().cpu().numpy(), topN)
                preds.append(index)
            print(f"user id: {nick_id[user_index]}, history item: {hist_item[user_index]}, GT: {item_id[user_index]}, predictions: {preds}")
            I.append(np.stack(preds, axis = 0))
            user_index += 1
        for i, iid_list in enumerate(item_id):
            item_list_set = set()
            true_item_set = set(iid_list)
            for no, iid in enumerate(I[i]):
                iid = iid.tolist()
                for item in iid:
                    if item in true_item_set:
                        item_list_set.add(item)
            hit_ratio = 0 if len(item_list_set) == 0 else len(item_list_set) / len(item_id[i])
            print(f"user id: {nick_id[i]}, true-positive predictions: {item_list_set}, hit rate for the user: {hit_ratio}")


if __name__ == '__main__':
    n_uid = 6040
    n_mid = 3417
    batch_size = 128
    max_len = 50
    num_interest = 3
    model = get_model('E-UPMiM', n_uid, n_mid, batch_size, max_len, "cpu")
    model.load_state_dict(torch.load("../best_model/movielens_E-UPMiM_b128_lr0.001_d32_len10_topN10/model.pt"))
    model.to("cpu")
    test_file = "../dataset/movielens_data/movielens_test.txt"
    test_data = DataIterator(test_file, batch_size, max_len, train_flag = 2)
    inference(batch_size, num_interest, test_data, model, topN = 10)