# training file

import os
import argparse
import random
import time

import torch
from data_iterator import DataIterator
from model import *
from comi_rec import *
from tensorboardX import SummaryWriter
from ann_search import find_topN_items
import math
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type = str, default = 'train', help = 'train | test | output')
parser.add_argument('--dataset', type = str, default = 'movielens', help = 'movielens | fliggy')
parser.add_argument('--random_seed', type = int, default = 321)
parser.add_argument('--embedding_dim', type = int, default = 32)
parser.add_argument('--hidden_size', type = int, default = 32)
parser.add_argument('--num_interest', type = int, default = 3)
parser.add_argument('--num_layer', type = int, default = 4)
parser.add_argument('--model_type', type = str, default = 'E-UPMiM', help = 'E-UPMiM | Comi_Rec | ..')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = '')
parser.add_argument('--max_iter', type = int, default = 500, help = '(k)')
parser.add_argument('--patience', type = int, default = 20)
parser.add_argument('--coef', default = None)
parser.add_argument('--topN', type = int, default = 10, help = '10 | 50')
parser.add_argument('--device', type = str, default = "cpu", help = 'cpu | cuda:0')


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

def evaluate_full(num_interest, valid_file, model, model_path, batch_size, maxlen, train_flag, save=True, coef=None):
    model.eval()
    valid_data = DataIterator(valid_file, batch_size, maxlen, train_flag = train_flag)
    topN = args.topN
    item_embs = model.output_item()

    total = 0
    total_recall = 0.0
    total_precision = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    for src, tgt in valid_data:
        nick_id, user_age, user_gender, user_occup, item_id, hist_item, hist_mask = prepare_test_data(src, tgt)

        user_embs, _ = model("test", nick_id, user_age, user_gender, user_occup, hist_item, hist_item, hist_mask)
        # user_embs = model.output_user(nick_id)
        user_index = 0
        I = []
        while user_index < batch_size:
            preds = []
            for i in range(num_interest):
                _user_embs = user_embs[user_index:user_index+1, i, :]
                # _user_embs = user_embs[user_index:user_index+1, :]
                d, index = find_topN_items(_user_embs.detach().cpu().numpy(), item_embs.detach().cpu().numpy(), topN)
                preds.append(index)
            user_index += 1
            I.append(np.stack(preds, axis = 0))
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
        model = Model_E_UPMiM(user_count, item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, args.num_layer, maxlen, device = args.device)
    elif model_type == 'Comi_Rec':
        model = Model_Comi_Rec(user_count, item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, args.num_layer, maxlen, device = args.device)
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

    best_model_path = "../best_model/" + exp_name

    writer = SummaryWriter('../runs/' + exp_name)

    train_data = DataIterator(train_file, batch_size, maxlen, train_flag=0)
    
    model = get_model(model_type, user_count, item_count, batch_size, maxlen).to(device = args.device)

    print('training begin')
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # 打印可训练的参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of trainable params: {trainable_params}")

    start_time = time.time()
    iter = 0
    try:
        loss_sum = 0.0
        trials = 0

        for src, tgt in train_data:
            model.train()
            model.to(device = args.device)
            # user_id / user_age / user_gender / occupation / target_item_id / history_items_id / masks
            nick_id, user_age, user_gender, user_occup, item_id, hist_item, hist_mask = prepare_train_data(src, tgt)
            optimizer.zero_grad()
            # 前向传播
            _, loss = model("train", nick_id, user_age, user_gender, user_occup, item_id, hist_item, hist_mask)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            loss_sum += loss
            iter += 1
            if iter % test_iter == 0:
                metrics = evaluate_full(args.num_interest, valid_file, model, best_model_path, batch_size, maxlen, train_flag = 1)
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
                        if not os.path.exists(best_model_path):
                            os.makedirs(best_model_path, exist_ok = True)
                        torch.save(model.cpu().state_dict(), f"{best_model_path}/model.pt")
                        # model.save(best_model_path)
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

    model.load_state_dict(torch.load(f"{best_model_path}/model.pt"))
    metrics = evaluate_full(args.num_interest, valid_file, model, best_model_path, batch_size, maxlen, train_flag = 1, save=False)
    print(', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()]))

    metrics = evaluate_full(args.num_interest, test_file, model, best_model_path, batch_size, maxlen, train_flag = 2, save=False)
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
    best_model_path = "../best_model/" + exp_name + '/model.pt'
    model = get_model(model_type, user_count, item_count, batch_size, maxlen).to(device = args.device)
    model.load_state_dict(torch.load(best_model_path))
    metrics = evaluate_full(args.num_interest, test_file, model, best_model_path, batch_size, maxlen, train_flag = 2, save=False, coef=args.coef)
    print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))

def output(
        user_count,
        item_count,
        dataset,
        batch_size = 128,
        maxlen = 10,
        model_type = 'E-UPMiM',
        lr = 0.001,
        topN = 10
):
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen, topN, save=False)
    best_model_path = "../best_model/" + exp_name + '/model.pt'
    model = get_model(model_type, user_count, item_count, batch_size, maxlen).to(device = args.device)
    model.load_state_dict(torch.load(best_model_path))
    item_embs = model.output_item()
    if not os.path.exists('../output/'):
        os.makedirs('../output/', exist_ok = True)
    np.save('../output/' + exp_name + '_emb.npy', item_embs.detach().cpu().numpy())
    print("item embedding has been saved!")


if __name__ == '__main__':
    args = parser.parse_args()
    SEED = args.random_seed

    torch.manual_seed(SEED)
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