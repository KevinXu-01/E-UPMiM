# dataloader / data iterator

import random


class DataIterator:

    def __init__(self, source,
                 batch_size=128,
                 maxlen=10,
                 train_flag=0
                ):
        self.read(source)
        self.users = list(self.users)
        
        self.batch_size = batch_size
        self.eval_batch_size = batch_size
        self.train_flag = train_flag
        self.maxlen = maxlen
        self.index = 0

    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self.users)

    def next(self):
        return self.__next__()

    def read(self, source):
        self.graph = {}
        self.users = set()
        self.items = set()
        self.ages = {}
        self.genders = {}
        self.occupations = {}
        with open(source, 'r') as f:
            for line in f:
                conts = line.strip().split(',')
                user_id = int(conts[0])
                age = int(conts[1])
                gender = int(conts[2])
                occupation = int(conts[3])
                if occupation < 0:
                    occupation = 0
                item_id = int(conts[4])
                time_stamp = int(conts[5])
                self.users.add(user_id)
                self.items.add(item_id)
                if user_id not in self.graph:
                    self.graph[user_id] = []
                self.graph[user_id].append((item_id, time_stamp))

                if user_id not in self.ages:
                    self.ages[user_id] = []
                    self.ages[user_id].append(age)

                if user_id not in self.genders:
                    self.genders[user_id] = []
                    self.genders[user_id].append(gender)

                if user_id not in self.occupations:
                    self.occupations[user_id] = []
                    self.occupations[user_id].append(occupation)

        for user_id, value in self.graph.items():
            value.sort(key=lambda x: x[1])
            self.graph[user_id] = [x[0] for x in value]
        self.users = list(self.users)
        self.items = list(self.items)
    
    def __next__(self):
        if self.train_flag == 0:
            user_id_list = random.sample(self.users, self.batch_size)
        else:
            total_user = len(self.users)
            if self.index >= total_user:
                prev_index = self.index - self.eval_batch_size
                self.eval_batch_size = total_user - prev_index
                self.index = prev_index
                raise StopIteration
            user_id_list = self.users[self.index: self.index + self.eval_batch_size]
            self.index += self.eval_batch_size

        age_list = []
        gender_list = []
        occupation_list = []
        item_id_list = []
        hist_item_list = []
        hist_mask_list = []
        for user_id in user_id_list:

            age_str = str(self.ages[user_id]).replace("[", "").replace("]","")
            age = int(age_str)
            age_list.append(age)

            gender_str = str(self.genders[user_id]).replace("[", "").replace("]","")
            gender = int(gender_str)
            gender_list.append(gender)

            occupation_str = str(self.occupations[user_id]).replace("[", "").replace("]","")
            occupation = int(occupation_str)
            occupation_list.append(occupation)

            # randomly select a number and set item[k] as target, items before
            # item[k] are training items; while after are used for prediction
            item_list = self.graph[user_id]
            if self.train_flag == 0:
                k = random.choice(range(4, len(item_list)))
                item_id_list.append(item_list[k])
            else:
                k = int(len(item_list) * 0.8)
                item_id_list.append(item_list[k:])
            if k >= self.maxlen:
                hist_item_list.append(item_list[k-self.maxlen: k])
                hist_mask_list.append([1.0] * self.maxlen)
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.maxlen - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.maxlen - k))
        return (user_id_list, age_list, gender_list, occupation_list, item_id_list), (hist_item_list, hist_mask_list)

def prepare_train_data(src, target):
    nick_id, user_age, user_gender, user_occup, item_id = src
    hist_item, hist_mask = target
    return nick_id, user_age, user_gender, user_occup, item_id, hist_item, hist_mask

def prepare_test_data(src, target):
    nick_id, user_age, user_gender, user_occup, item_id = src
    hist_item, hist_mask = target
    return nick_id, user_age, user_gender, user_occup, item_id, hist_item, hist_mask

if __name__ == "__main__":
    test_file = '../dataset/movielens_data/movielens_test.txt'
    test_data = DataIterator(test_file, batch_size=128, maxlen=50, train_flag=0)
    for src, tar in test_data:
        nick_id, user_age, user_gender, user_occup, item_id, hist_item, hist_mask = prepare_test_data(src, tar)
        for i in range(128):
            print(len(hist_item[i]))
            print(len(item_id[i]))
