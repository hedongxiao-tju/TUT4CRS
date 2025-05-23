import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import json

from utils.utils import *

import utils.user_att_count as user_att_count
import utils.user_att_count_yelp as user_att_count_yelp
from utils.global_variable import *

random.seed(1895)


def pad_list_of_list(list_of_list, pad_idx=0):
    maxlen = max([len(_) for _ in list_of_list])
    if maxlen == 0:
        maxlen = 1
    padded_list_of_list = np.full((len(list_of_list), maxlen), pad_idx)
    mask_list_of_list = np.full((len(list_of_list), maxlen), False)
    for i in range(len(list_of_list)):
        padded_list_of_list[i][:len(list_of_list[i])] = list_of_list[i]
        mask_list_of_list[i][:len(list_of_list[i])] = True
    return torch.from_numpy(padded_list_of_list), torch.from_numpy(mask_list_of_list)


def item_att_collate_fn(batch):
    user, item, item_pos_att, item_neg_att, item_neg_train1, item_neg_train2, \
    att_pos_att, att_neg_att, att_pos_train, att_neg_train, att, pos, neg, item_neg_item, date_list = zip(*batch)

    user_list = torch.tensor(user)
    pos_item_list = torch.tensor(item)
    item_pos_att_list, item_pos_att_mask = pad_list_of_list(item_pos_att)
    item_neg_att_list, item_neg_att_mask = pad_list_of_list(item_neg_att)
    neg_item_list1, neg_item_mask1 = pad_list_of_list(item_neg_train1)
    neg_item_list2, neg_item_mask2 = pad_list_of_list(item_neg_train2)

    att_pos_att_list, att_pos_att_mask = pad_list_of_list(att_pos_att)
    att_neg_att_list, att_neg_att_mask = pad_list_of_list(att_neg_att)
    att_pos_train_list = torch.tensor(att_pos_train)
    att_neg_train_list = torch.tensor(att_neg_train)

    att_list = torch.tensor(att)
    pos_list = torch.tensor(pos)
    neg_list = torch.tensor(neg)

    item_neg_item_list, item_neg_item_list_mask = pad_list_of_list(item_neg_item)

    return user_list, pos_item_list, \
           item_pos_att_list, item_pos_att_mask, item_neg_att_list, item_neg_att_mask, \
           neg_item_list1, neg_item_mask1, neg_item_list2, neg_item_mask2, \
           att_pos_att_list, att_pos_att_mask, att_neg_att_list, att_neg_att_mask, att_pos_train_list, att_neg_train_list, \
            att_list, pos_list, neg_list, item_neg_item_list, item_neg_item_list_mask, date_list


class ItemAttTrainGenerator(Dataset):
    def __init__(self, name, train_info_path, user_info, item_info, att_tree_dict,
                 att_num, item_num, epoch_num, use_gpu=True):
        self.data_name = name
        if name == 'YELP2018_STAR':
            train_info_path = train_info_path + str(epoch_num % 50) + '.pkl'
        else:
            train_info_path = train_info_path + str(epoch_num % 250) + '.pkl'
        self.train_info = load_pickle(train_info_path)

        self.user_info = user_info  # dict  int user: dict{'items'=set items , 'interactInfo_map'=dict{inter_item -> dict{date,review_id}}}

        self.item_info = item_info    # key: int item   value:  set(att)

        self.att_tree_dict = att_tree_dict
        for key in self.att_tree_dict:
            self.att_tree_dict[key] = set(self.att_tree_dict[key])

        self.att_num = att_num
        self.item_num = item_num
        self.len = len(self.train_info[0])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        '''
            :return: 5 tensors:
            (1) user id
            (2) positive item id that has interacted with user
            (3) negative item id that has not interacted with user   # I/I_u   I_u is the set of items historically interacted by user u
            (4) negative item id that has not interacted with the user in the candidate item set   I_{cand} / I_u
            (5) preference attribute ids that confirmed by user in current turn
        '''

        # -----------item item------------
        user = self.train_info[0][index]   #user
        item = self.train_info[1][index]   #ground-truth item

        date = self.user_info[user]['interactInfo_map'][item]['date']

        item_neg_train1 = [self.train_info[2][index]]   #  I/I_u
        if self.train_info[3][index] is None:
            item_neg_train2 = [self.train_info[2][index]]
        else:
            item_neg_train2 = [self.train_info[3][index]]   #I_{cand} / I_u

        item_pos_att = self.train_info[4][index]   #  preference attribute ids that confirmed by user in current turn

        neg_pre_len = random.sample(range(1, 4), 1)[0]

        #ground-truth item
        item_neg_att = random.sample(set(range(self.att_num)) - self.item_info[item], neg_pre_len)
        
        item_neg_item = []
        not_target_set = set(item_neg_train2).union(set(item_neg_train1)) - set([item])
        if len(not_target_set) > 0:
            for _ in not_target_set:
                item_neg_item.append(_)

        # -----------att att------------  att_pos_train  att_neg_train
        att_pos_att = item_pos_att   #preference attribute ids that confirmed by user in current turn
        att_neg_att = item_neg_att   #ground-truth item
        neg_att_set = set(range(self.att_num)) - self.item_info[item] - set(att_neg_att)
        if len(neg_att_set) == 0:
            neg_att_set = set(range(self.att_num)) - self.item_info[item]

        if len(self.item_info[item] - set(att_pos_att)) == 0:
            att_pos_train = random.sample(self.item_info[item], 1)[0]
            att_pos_att.remove(att_pos_train)
        else:
            att_pos_train = random.sample(self.item_info[item] - set(att_pos_att), 1)[0]

        att_neg_train = random.sample(neg_att_set, 1)[0]

        # -----------att item------------
        att = random.sample(range(self.att_num), 1)[0]
        if self.data_name == YELP:
            att_gt = user_att_count_yelp.get_att_item_list(att)
            att_reverse_gt = user_att_count_yelp.get_att_reverse_item(att)
        else:
            att_gt = user_att_count.get_att_item_list(att)
            att_reverse_gt = user_att_count.get_att_reverse_item(att)
        pos_len = len(att_gt)
        pos_num = random.randint(0, pos_len - 1)
        pos = att_gt[pos_num]
        neg_len = len(att_reverse_gt)
        neg_num = random.randint(0, neg_len - 1)
        neg = att_reverse_gt[neg_num]

        # item item
        # att  att
        # att item
        return user, item, item_pos_att, item_neg_att, item_neg_train1, item_neg_train2, \
               att_pos_att, att_neg_att, att_pos_train, att_neg_train,\
               att, pos, neg, \
               item_neg_item, date


def build_item_att_loader(name, train_info_path, user_info, item_info, att_tree_dict,
                          att_num, item_num, epoch_num, use_gpu=True, batch_size=1, shuffle=True, num_threads=4):

    # Dataset
    train_generator = ItemAttTrainGenerator(name, train_info_path, user_info, item_info, att_tree_dict,
                                            att_num, item_num, epoch_num, use_gpu=use_gpu)
    return DataLoader(
        train_generator,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_threads,
        collate_fn=item_att_collate_fn
    )
