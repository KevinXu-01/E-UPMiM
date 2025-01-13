# The file has been used; and it is no longer useful as we do not release the original data; processed data is in the dataset dir.

import random
from collections import defaultdict

random.seed(1230)

users = defaultdict(list)
item_count = defaultdict(int)
filter_size = 5
user_profile = defaultdict(list)

def read_from_movielens(source1, source2):
    f = open(source2, 'r')#rating
    f_ = open(source1, 'r')#user profile
    for line in f:
        contents=line.strip().split('::')
        user_id = int(contents[0])
        if contents[1] == 'F':
            gender = 0
        elif contents[1] == 'M':
            gender = 1
        age = int(contents[2])
        occupation = int(contents[3])
        user_profile[user_id].append((age, gender, occupation))
    
    for line in f_:
        contents = line.strip().split('::')
        user_id = int(contents[0])
        movie_id = int(contents[1])
        rating = int(contents[2])
        #if rating <= 1:
        #    continue
        item_count[movie_id] += 1
        timestamp = int(contents[3])
        users[user_id].append((movie_id, timestamp))

def read_from_fliggy(source1, source2):
    i = 0
    f_ = open(source2, 'r')#profile
    for line in f_:
        contents = line.strip().split(',')
        user_id = int(contents[0])
        age = int(contents[1])
        gender = int(contents[2])
        occupation = int(contents[3])
        user_profile[user_id].append((age, gender, occupation))

    f = open(source1, 'r')#rating
    for line in f:
        if i == 0:
            i += 1
            continue

        contents = line.strip().split(',')
        user_id = int(contents[0])
        product_id = int(contents[1])
        if contents[2] != 'pay':
            continue
        item_count[product_id] += 1
        timestamp = int(contents[3])
        users[user_id].append((product_id, timestamp))

#read_from_movielens("/code/e-upmim/dataset/movielens/ratings.dat", "/code/e-upmim/dataset/movielens/users.dat")
read_from_fliggy("/code/e-upmim/dataset/fliggy/user_item_behavior_history.csv", "/code/e-upmim/dataset/fliggy/user_profile.csv")

items = list(item_count.items())
items.sort(key=lambda x:x[1], reverse=True)

item_total = 0
for index, (movie_id, num) in enumerate(items):
    if num >= filter_size:
        item_total = index + 1
    else:
        break

print(item_total)

item_map = dict(zip([items[i][0] for i in range(item_total)], list(range(1, item_total+1))))  

user_ids = list(users.keys())
filter_user_ids = []
filter_user_profiles = []
for user in user_ids:
    item_list = users[user]
    index = 0
    for item, timestamp in item_list:
        if item in item_map:
            index += 1
    if index >= filter_size:
        filter_user_ids.append(user)
        
user_ids = filter_user_ids
random.shuffle(user_ids)
num_users = len(user_ids)
user_map = dict(zip(user_ids, list(range(num_users))))
split_1 = int(num_users * 0.8)
split_2 = int(num_users * 0.9)
train_users = user_ids[:split_1]
valid_users = user_ids[split_1:split_2]
test_users = user_ids[split_2:]
def export_map(name, map_dict):
    with open(name, 'w') as f:
        for key, value in map_dict.items():
            f.write('%s,%d\n' % (key, value))

def export_data(name, user_list):
    total_data = 0
    with open(name, 'w') as f:
        for user in user_list:
            if user not in user_map:
                continue
            item_list = users[user]
            item_list.sort(key=lambda x:x[1])
            index = 0
            for item, timestamp in item_list:
                if item in item_map:
                    profile = str(user_profile[user])
                    profile = profile.replace("[(", "")
                    profile = profile.replace(")]", "")
                    profiles = profile.strip().split(', ')
                    age = int(profiles[0])
                    gender = int(profiles[1])
                    occupation = int(profiles[2])
                    f.write('%d,%d,%d,%d,%d,%d\n' % (user_map[user], age, gender, occupation, item_map[item], index))
                    index += 1
                    total_data += 1
    return total_data


        
export_map('/code/e-upmim/dataset/fliggy/fliggy_user_map.txt', user_map)
export_map('/code/e-upmim/dataset/fliggy/fliggy_item_map.txt', item_map)

total_train = export_data('/code/e-upmim/dataset/fliggy/fliggy_train.txt', train_users)
total_valid = export_data('/code/e-upmim/dataset/fliggy/fliggy_valid.txt', valid_users)
total_test = export_data('/code/e-upmim/dataset/fliggy/fliggy_test.txt', test_users)
print('total behaviors: ', total_train + total_valid + total_test)


