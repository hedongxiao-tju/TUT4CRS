import os
from datetime import datetime

import torch

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
from utils.global_variable import *
import json
from utils.utils import *


class ConfigYelp2018Star:
    def __init__(self):
        self.name = 'YELP2018_STAR'
        self.use_gpu = True
        self.user_num = 27675
        self.item_num = 70311
        self.attribute_num = 590
        self.parent_attribute_num = 590
        self.interaction_num_train = 945968

        ################time information###############
        self.year_start = 2004
        self.year_num = 15  # 2004 - 2018
        self.month_num = 12
        self.day_num = 31
        self.week_num = 7
        self.hour_num = 0   #YELP2018 no hour information

        self.time_type_num = 4
        ######################################
        self.nlayer = 2
        self.conv_name = 'gcn'
        self.n_heads = 1
        self.drop = 0.1
        self.max_rec_item_num = 10
        self.top_taxo = 3
        self.feedback_aggregate = "3loss_not_equal"
        self.layer_aggregate = "mean"

        self.att_num = self.attribute_num

        self.batch_size = 1024
        self.item_lr = 0.0001
        self.att_lr = 0.0001
        self.weight_decay = 1e-5
        self.train_shuffle = True
        self.num_threads = 4
        self.add_neg_item_num_max = 100
        self.epoch_num = 10000
        self.test_epoch_num = 10000

        self.input_dim = 4 + 15 + 8
        self.hidden_dim = 64
        self.output_dim = 2
        self.time_dim = 16

        self.rec_model_path = "../recommendersystem/recmodel/yelp2018-star"
        self.global_3_emb_path = "../recommendersystem/global_3_Emb/yelp2018-star"
        self.global_emb_path = "../recommendersystem/globalEmb/yelp2018-star"

        self.user_info = None
        self.item_info = None  # key: int item   value:  set(att)
        self.att_info = None  # key int att    value set items
        self.att_reverse_info = None  # key int att    value list items
        self.friend_info = None
        self.adj_index = [[], []]
        self.edge_attr = []
        self.time_node = []
        self.att_tree_dict = None
        self._data_init()

    def _data_init(self):
        print('start _data_init....')
        ################################################ user_item.json #############################################
        new_user_info = {}
        if os.path.exists('utils/' + self.name + '/new_user_info.pkl'):
            with open('utils/' + self.name + '/new_user_info.pkl', 'rb') as f:
                new_user_info = pickle.load(f)
        else:
            with open(DATA_DIR[self.name] + '/' + GRAPH_GENERATE_DATA + '/' + "/user_item_train_withTime.json", "r") as f:
                user_info = json.load(f)
            num_of_interaction_train = 0
            for user in user_info:
                inter_item_set = set()
                item_date_reviewId_map = {}
                for inter_info in user_info[user]:  # item  date   review_id
                    inter_item = inter_info['item']  # int

                    inter_item_set.add(inter_item)
                    if inter_item not in item_date_reviewId_map:
                        item_date_reviewId_map[inter_item] = {}
                    item_date_reviewId_map[inter_item]['date'] = inter_info['date']
                    item_date_reviewId_map[inter_item]['review_id'] = inter_info['review_id']

                if int(user) not in new_user_info:
                    new_user_info[int(user)] = {}
                new_user_info[int(user)]['items'] = inter_item_set
                num_of_interaction_train += len(inter_item_set)
                new_user_info[int(user)]['interactInfo_map'] = item_date_reviewId_map
            with open('utils/' + self.name + '/new_user_info.pkl', 'wb') as f:
                pickle.dump(obj=new_user_info, file=f)

        self.user_info = new_user_info
        ################################################################################################################

        ################################################ item_dict.json #############################################
        new_item_info = {}  # key: int item   value:  set(att)
        att_info = {}  # key int att    value set items
        att_reverse_info = {}  # key int att    value list items
        if os.path.exists('utils/' + self.name + '/att_reverse_info.txt'):
            with open('utils/' + self.name + '/new_item_info.pkl', 'rb') as f:
                new_item_info = pickle.load(f)

            with open('utils/' + self.name + '/att_info.txt', 'rb') as f:
                att_info = pickle.load(f)

            with open('utils/' + self.name + '/att_reverse_info.txt', 'rb') as f:
                att_reverse_info = pickle.load(f)
        else:
            with open(DATA_DIR[self.name] + '/' + GRAPH_GENERATE_DATA + '/' + "/item_dict-original_tag.json",
                      "r") as f:  # item_dict.json
                item_info = json.load(f)  # item_dict.json

            for item in item_info:
                new_item_info[int(item)] = set(
                    item_info[item][ATT_KEY])  # list转set   # key: int item   value:  set(att)

                for att in item_info[item][ATT_KEY]:
                    if len(att_info.get(int(att), set())) == 0:
                        att_info[int(att)] = set()
                        att_info[int(att)].add(int(item))
                    else:
                        att_info[int(att)].add(int(item))

                for att in set(range(self.att_num)) - set(item_info[item][ATT_KEY]):
                    if len(att_reverse_info.get(int(att), set())) == 0:
                        att_reverse_info[int(att)] = list()
                        att_reverse_info[int(att)].append(int(item))
                    else:
                        att_reverse_info[int(att)].append(int(item))
            with open('utils/' + self.name + '/att_info.txt', 'wb') as f:
                pickle.dump(att_info, f)
            with open('utils/' + self.name + '/att_reverse_info.txt', 'wb') as f:
                pickle.dump(att_reverse_info, f)
            with open('utils/' + self.name + '/new_item_info.pkl', 'wb') as f:
                pickle.dump(new_item_info, f)
        self.item_info = new_item_info  # key: int item   value:  set(att)
        self.att_info = att_info
        self.att_reverse_info = att_reverse_info
        ################################################################################################################

        ############################################# build graph ##############################################################
        if os.path.exists('utils/' + self.name + '/adj_index.pkl'):
            with open('utils/' + self.name + '/adj_index.pkl', 'rb') as f:
                self.adj_index = pickle.load(f)
            with open('utils/' + self.name + '/time_node.pkl', 'rb') as f:
                self.time_node = pickle.load(f)
        else:
            #  user <-> item
            index_of_interaction = 0
            for user in self.user_info:
                interactInfo_map = self.user_info[user]['interactInfo_map']
                for item in self.user_info[user]['items']:
                    self.adj_index[0].append(user)
                    self.adj_index[1].append(
                        index_of_interaction + self.user_num + self.item_num + self.att_num)
                    self.adj_index[1].append(user)
                    self.adj_index[0].append(index_of_interaction + self.user_num + self.item_num + self.att_num)

                    self.adj_index[0].append(item + self.user_num)
                    self.adj_index[1].append(index_of_interaction + self.user_num + self.item_num + self.att_num)
                    self.adj_index[1].append(item + self.user_num)
                    self.adj_index[0].append(index_of_interaction + self.user_num + self.item_num + self.att_num)

                    index_of_interaction += 1

                    date_string = interactInfo_map[item]['date']
                    time_att_index = self._date2array(date_string)
                    self.time_node.append(time_att_index)

            #  item <-> att
            for item in self.item_info:
                for att in self.item_info[item]:
                    self.adj_index[0].append(item + self.user_num)
                    self.adj_index[1].append(att + self.item_num + self.user_num)
                    self.adj_index[1].append(item + self.user_num)
                    self.adj_index[0].append(att + self.item_num + self.user_num)

            self.adj_index = torch.tensor(self.adj_index)
            self.time_node = torch.tensor(self.time_node)

            with open('utils/' + self.name + '/adj_index.pkl', 'wb') as f:
                pickle.dump(obj=self.adj_index, file=f)
            with open('utils/' + self.name + '/time_node.pkl', 'wb') as f:
                pickle.dump(obj=self.time_node, file=f)
        if self.use_gpu:
            self.adj_index = self.adj_index.cuda()
            self.time_node = self.time_node.cuda()
        ################################################################################################################

        if os.path.exists('utils/' + self.name + '/att_tree_dict.pkl'):
            with open('utils/' + self.name + '/att_tree_dict.pkl', 'rb') as f:
                self.att_tree_dict = pickle.load(f)
        else:
            with open(DATA_DIR[self.name] + '/' + GRAPH_GENERATE_DATA + '/' + "attribute_tree_dict.json", "r") as f:
                attribute_tree_dict = json.load(f)
            self.att_tree_dict = trans_index(attribute_tree_dict)
            with open('utils/' + self.name + '/att_tree_dict.pkl', 'wb') as f:
                pickle.dump(obj=self.att_tree_dict, file=f)


    def _date2array(self, date_string):
        date_object = datetime.strptime(date_string, "%Y-%m-%d").timetuple()
        # （tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, tm_wday, tm_yday, tm_isdst）
        year = date_object.tm_year
        month = date_object.tm_mon
        day = date_object.tm_mday
        weekday = date_object.tm_wday
        return [year - self.year_start,
                month - 1 + self.year_num,
                day - 1 + self.year_num + self.month_num,
                weekday + self.year_num + self.month_num + self.day_num]

    def _getZeroArray(self, dim):
        ans = []
        for _ in range(dim):
            ans.append(0)
        return ans