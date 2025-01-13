# training file

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #设置输出信息：ERROR + FATAL，隐藏tf的Warning

import argparse
import random
# import sys
import time

import tensorflow as tf
from data_iterator import DataIterator
from model import *
from tensorboardX import SummaryWriter

import faiss
import math
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type = str, default = 'train', help = 'train | test')
parser.add_argument('--dataset', type = str, default = 'movielens', help = 'movielens | fliggy')
parser.add_argument('--random_seed', type = int, default = 321)
parser.add_argument('--embedding_dim', type = int, default = 32)
parser.add_argument('--hidden_size', type = int, default = 32)
parser.add_argument('--num_interest', type = int, default = 3)
parser.add_argument('--num_layer', type = int, default = 4)
parser.add_argument('--model_type', type = str, default = 'E-UPMiM', help = 'E-UPMiM | ..')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = '')
parser.add_argument('--max_iter', type = int, default = 500, help = '(k)')
parser.add_argument('--patience', type = int, default = 20)
parser.add_argument('--coef', default = None)
parser.add_argument('--topN', type = int, default = 10, help = '10 | 50')

best_metric = 0
args = parser.parse_args()

def prepare_train_data(src, target):
    nick_id, user_age, user_gender, user_occup, item_id = src
    hist_item, hist_mask = target
    return nick_id, user_age, user_gender, user_occup, item_id, hist_item, hist_mask

def prepare_test_data(src, target):
    nick_id, user_age, user_gender, user_occup, item_id = src
    hist_item, hist_mask = target
    return nick_id, user_age, user_gender, user_occup, item_id, hist_item, hist_mask

def evaluate_full(sess, num_interest, valid_file, model, model_path, batch_size, maxlen, train_flag, save=True, coef=None):
    valid_data = DataIterator(valid_file, batch_size, maxlen, train_flag = train_flag)
    topN = args.topN

    item_embs = model.output_item(sess)
    

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    try:
        gpu_index = faiss.GpuIndexFlatIP(res, args.embedding_dim, flat_config)
        gpu_index.add(item_embs)
    except Exception as e:
        return {}

    total = 0
    total_recall = 0.0
    total_precision = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    for src, tgt in valid_data:
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
            recall = 0
            dcg = 0.0
            true_item_set = set(iid_list)
            for iid in I[i]:
                iid = iid.tolist()
                for no, item in enumerate(iid):
                    if item in true_item_set:
                        recall += 1
                        dcg += 1.0 / math.log(no + 2, 2)
            idcg = 0.0
            for no in range(recall):
                idcg += 1.0 / math.log(no + 2, 2)
            total_recall += recall * 1.0 / len(iid_list)
            total_precision += recall * 1.0 / topN
            if recall > 0:
                total_ndcg += dcg / idcg
                total_hitrate += 1
        
        total += len(item_id)
    
    hitrate = total_hitrate / total
    recall = total_recall / total
    precision = total_precision / total
    ndcg = total_ndcg / total
    print('total:', total)
    print('topN:', topN)

    if save:
        return {'hitrate': hitrate, 'recall': recall, 'precision': precision, 'ndcg': ndcg}
    return {'hitrate': hitrate,'recall': recall, 'precision': precision, 'ndcg': ndcg}


def get_exp_name(dataset, model_type, batch_size, lr, maxlen, topN, save=True):
    para_name = '_'.join([dataset, model_type, 'b' + str(batch_size), 'lr' + str(lr), 'd' + str(args.embedding_dim),
                          'len' + str(maxlen), 'topN' + str(topN)])
    exp_name = para_name
    return exp_name

def get_model(model_type, user_count, item_count, batch_size, maxlen):
    if model_type == 'E-UPMiM':
        model = Model_E_UPMiM(user_count, item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, args.num_layer, maxlen)
    else:
        print("Invalid model_type : %s", model_type)
        return
    return model


def train(
        train_file,
        test_file,
        valid_file,
        user_count, 
        item_count,
        dataset="movielens",
        batch_size = 128,
        maxlen = 10,
        test_iter = 50,
        model_type = 'E-UPMiM',
        lr = 0.001,
        max_iter = 1000,
        patience = 20,
        topN = 10
):
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen, topN)

    best_model_path = "../best_model/" + exp_name + '/'

    gpu_options = tf.GPUOptions(allow_growth=True)

    writer = SummaryWriter('../runs/' + exp_name)


    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, batch_size, maxlen, train_flag=0)
        

        model = get_model(model_type, user_count, item_count, batch_size, maxlen)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print('training begin')

        start_time = time.time()
        iter = 0
        try:
            loss_sum = 0.0
            trials = 0

            for src, tgt in train_data:
                # user_id / user_age / user_gender / occupation / target_item_id / history_items_id / masks
                data_iter = prepare_train_data(src, tgt)
                loss = model.train(sess, list(data_iter) + [float(lr)])

                loss_sum += loss
                iter += 1
                if iter % test_iter == 0:
                    metrics = evaluate_full(sess, args.num_interest, valid_file, model, best_model_path, batch_size, maxlen, train_flag = 1)
                    log_str = 'iter: %d, train loss: %.4f' % (iter, loss_sum / test_iter)
                    if metrics != {}:
                        log_str += ', ' + ', '.join(
                            ['valid ' + key + ': %.6f' % value for key, value in metrics.items()])
                    print(exp_name)
                    print(log_str)

                    writer.add_scalar('train/loss', loss_sum / test_iter, iter)
                    if metrics != {}:
                        for key, value in metrics.items():
                            writer.add_scalar('eval/' + key, value, iter)

                    if 'recall' in metrics:
                        recall = metrics['recall']
                        global best_metric
                        if recall > best_metric:
                            best_metric = recall
                            model.save(sess, best_model_path)
                            trials = 0
                        else:
                            trials += 1
                            if trials > patience:
                                print('early stopping...')
                                break

                    loss_sum = 0.0
                    test_time = time.time()
                    print("time interval: %.4f min" % ((test_time - start_time) / 60.0))

                if iter >= max_iter * 1000:
                    break
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        model.restore(sess, best_model_path)
        metrics = evaluate_full(sess, args.num_interest, valid_file, model, best_model_path, batch_size, maxlen, train_flag = 1, save=False)
        print(', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()]))

        metrics = evaluate_full(sess, args.num_interest, test_file, model, best_model_path, batch_size, maxlen, train_flag = 2, save=False)
        print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))

def test(
        test_file,
        user_count, 
        item_count,
        dataset = "movielens",
        batch_size = 128,
        maxlen = 10,
        model_type = 'E-UPMiM',
        lr = 0.001,
        topN = 10
):
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen, topN, save=False)
    best_model_path = "../best_model/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)
    model = get_model(model_type, user_count, item_count, batch_size, maxlen)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)
        metrics = evaluate_full(sess, args.num_interest, test_file, model, best_model_path, batch_size, maxlen, train_flag = 2, save=False, coef=args.coef)
        print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))

def output(
        user_count,
        item_count,
        dataset,
        batch_size = 128,
        maxlen = 10,
        model_type = 'E-UPMiM',
        lr = 0.001
):
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=False)
    best_model_path = "../best_model/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)
    model = get_model(dataset, model_type, user_count, item_count, batch_size, maxlen)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)
        item_embs = model.output_item(sess)
        np.save('../output/' + exp_name + '_emb.npy', item_embs)


if __name__ == '__main__':
    args = parser.parse_args()
    SEED = args.random_seed

    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_name = 'train'
    valid_name = 'valid'
    test_name = 'test'

    if args.dataset == 'movielens':
        path = '../dataset/movielens_data/'
        # the following values are consistent across all experiments on the same dataset.
        user_count = 6040
        item_count = 3417
        batch_size = 128
        maxlen = 10
        test_iter = 50
    elif args.dataset == 'fliggy':
        path = '../dataset/fliggy_data/'
        # the following values are consistent across all experiments on the same dataset.
        user_count = 277662
        item_count = 39785
        batch_size = 256
        maxlen = 20
        test_iter = 50

    train_file = path + args.dataset + '_train.txt'
    valid_file = path + args.dataset + '_valid.txt'
    test_file = path + args.dataset + '_test.txt'
    
    dataset = args.dataset
    if args.mode == 'train':
        train(train_file=train_file, valid_file=valid_file, test_file=test_file, user_count = user_count, 
              item_count=item_count, dataset=dataset, batch_size=batch_size, maxlen=maxlen, test_iter=test_iter, 
              model_type=args.model_type, lr=args.learning_rate, max_iter=args.max_iter, patience=args.patience, topN=args.topN)
    elif args.mode == 'test':
        test(test_file=test_file, user_count = user_count, item_count=item_count, dataset=dataset, batch_size=batch_size, 
             maxlen=maxlen, model_type=args.model_type, lr=args.learning_rate, topN=args.topN)
    elif args.mode == 'output':
        output(user_count = user_count, item_count=item_count, dataset=dataset, batch_size=batch_size, maxlen=maxlen, 
               model_type=args.model_type, lr=args.learning_rate)
    else:
        print('do nothing...')
