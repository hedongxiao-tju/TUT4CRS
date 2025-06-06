import json
import numpy as np
import os
import random
from utils import *
from torch import nn
from tkinter import _flatten
from collections import Counter


class TimeAwareBinaryRecommendEnv(object):
    def __init__(self, kg, dataset, data_name, embed, seed=1, max_turn=15, cand_num=10, cand_item_num=10, attr_num=20,
                 mode='train', ask_num=1, entropy_way='weight entropy', fm_epoch=0, args = None):
        self.data_name = data_name
        self.mode = mode
        self.seed = seed
        self.max_turn = max_turn  # MAX_TURN
        self.attr_state_num = attr_num
        self.kg = kg
        self.dataset = dataset
        self.feature_length = getattr(self.dataset, 'feature').value_len
        self.user_length = getattr(self.dataset, 'user').value_len
        self.item_length = getattr(self.dataset, 'item').value_len

        # action parameters
        self.ask_num = ask_num
        self.rec_num = 10
        self.random_sample_feature = False
        self.random_sample_item = False
        if cand_num == 0:
            self.cand_num = 10
            self.random_sample_feature = True
        else:
            self.cand_num = cand_num
        if cand_item_num == 0:
            self.cand_item_num = 10
            self.random_sample_item = True
        else:
            self.cand_item_num = cand_item_num
        #  entropy  or weight entropy
        self.ent_way = entropy_way

        # user's profile
        self.reachable_feature = []  # user reachable feature
        self.user_acc_feature = []  # user accepted feature which asked by agent
        self.user_rej_feature = []  # user rejected feature which asked by agent
        self.cand_items = []  # candidate items
        self.rej_items = []
        self.item_feature_pair = {}  # key:item   value: list(set(fea_belong_items) - set(self.user_rej_feature))
        self.cand_item_score = []

        # user_id  item_id   cur_step   cur_node_set
        self.user_id = None
        self.target_item = None
        self.cur_conver_step = 0  # the number of conversation in current step
        self.cur_node_set = []  # maybe a node or a node set  /   normally save feature node

        self.target_item_date = None  # data time (type = str)
        self.short_target_item_date = None  # short id of date(type = list)
        self.time_node = None   # full id of date(type = list)

        # state veactor
        self.user_embed = None
        self.conver_his = []  # conversation_history
        self.attr_ent = []  # attribute entropy

        self.ui_dict = self.__load_rl_data__(data_name, mode=mode)  # np.array [ u i weight]
        self.user_weight_dict = dict()
        self.user_items_dict = dict()

        # init seed & init user_dict
        set_random_seed(self.seed)  # set random seed
        if mode == 'train':
            self.__user_dict_init__()  # init self.user_weight_dict  and  self.user_items_dict
        elif mode == 'test':
            self.ui_array = None  # u-i array [ [userID1, itemID1], ...,[userID2, itemID2]]
            self.__test_tuple_generate__()
            self.test_num = 0

        embeds = load_timetrain_embed(data_name, args=args)

        if embeds:
            self.ui_embeds = embeds['ui_emb']
            self.feature_emb = embeds['feature_emb']
            self.time_emb = embeds['time_emb']                      # time_num * 16
            self.time_to_hidden = embeds['time_to_hidden']          # time_type * hidden
            self.user_time_transfer = embeds['user_time_transfer']  #  2xhidden * hidden
        else:
            raise Exception('no embedding....')

        self.action_space = 2

        self.reward_dict = {
            'ask_suc': 0.01,
            'ask_fail': -0.1,
            'rec_suc': 1,
            'rec_fail': -0.1,
            'until_T': -0.3,  # MAX_Turn
            'cand_none': -0.1
        }
        self.history_dict = {
            'ask_suc': 1,
            'ask_fail': -1,
            'rec_scu': 2,
            'rec_fail': -2,
            'until_T': 0
        }
        self.attr_count_dict = dict()  # This dict is used to calculate entropy

    def __load_rl_data__(self, data_name, mode):
        if mode == 'train':
            with open(os.path.join(DATA_DIR[data_name], 'UI_Interaction_data/review_dict_valid.json'),
                      encoding='utf-8') as f:
                print('train_data: load RL valid data')
                mydict = json.load(f)
        elif mode == 'test':
            with open(os.path.join(DATA_DIR[data_name], 'UI_Interaction_data/review_dict_test.json'),
                      encoding='utf-8') as f:
                print('test_data: load RL test data')
                mydict = json.load(f)
        return mydict

    def __user_dict_init__(self):  # Calculate the weight of the number of interactions per user
        ui_nums = 0
        for items in self.ui_dict.values():
            ui_nums += len(items)
        for user_str in self.ui_dict.keys():
            user_id = int(user_str)
            self.user_weight_dict[user_id] = len(self.ui_dict[user_str]) / ui_nums
        print('user_dict init successfully!')

    def __test_tuple_generate__(self):
        ui_list = []
        for user_str, items in self.ui_dict.items():
            user_id = int(user_str)
            for item_info in items:
                item_id = item_info['item']
                interact_date = item_info['date']
                ui_list.append([user_id, item_id, interact_date])
        self.ui_array = np.array(ui_list)
        np.random.shuffle(self.ui_array)

    def reset(self, embed=None):
        if embed is not None:
            self.ui_embeds = embed[:self.user_length + self.item_length]
            self.feature_emb = embed[self.user_length + self.item_length:]
        # init  user_id  item_id  cur_step   cur_node_set
        self.cur_conver_step = 0  # reset cur_conversation step
        self.cur_node_set = []
        if self.mode == 'train':
            users = list(self.user_weight_dict.keys())
            # self.user_id = np.random.choice(users, p=list(self.user_weight_dict.values())) # select user  according to user weights
            self.user_id = np.random.choice(users)
            target_item_info = np.random.choice(self.ui_dict[str(self.user_id)])
            self.target_item_date = target_item_info['date']  # str
            self.short_target_item_date = date2array(self.target_item_date,date_name=self.data_name)  # short_id
            self.time_node = self._map_to_all_id(self.short_target_item_date, old_type='time')  # full_id
            self.target_item = target_item_info['item']

        elif self.mode == 'test':
            self.user_id = int(self.ui_array[self.test_num, 0])
            self.target_item = int(self.ui_array[self.test_num, 1])
            self.target_item_date = self.ui_array[self.test_num, 2]  #str
            self.short_target_item_date = date2array(self.target_item_date, date_name=self.data_name)
            self.time_node = self._map_to_all_id(self.short_target_item_date, old_type='time')  # full_id
            self.test_num += 1

        # init user's profile
        print('-----------reset state vector------------')
        print('user_id:{}, target_item:{}'.format(self.user_id, self.target_item))
        self.reachable_feature = []  # user reachable feature in cur_step
        self.user_acc_feature = []  # user accepted feature which asked by agent
        self.user_rej_feature = []  # user rejected feature which asked by agent
        self.cand_items = list(range(self.item_length))
        self.rej_items = []

        # init state vector
        # self.user_embed = self.ui_embeds[self.user_id].tolist()  # init user_embed   np.array---list
        origin_user_emb = self.ui_embeds[self.user_id]  #np.array
        cur_time_emb = np.dot(self.time_emb[self.short_target_item_date].reshape(-1),self.time_to_hidden) #np.array
        origin_user_time = np.concatenate((origin_user_emb,cur_time_emb)) #ndarry 2*hidden
        self.user_embed = np.dot(origin_user_time,self.user_time_transfer).tolist()  #list 64dim
        self.conver_his = [0] * self.max_turn  # conversation_history
        self.attr_ent = [0] * self.attr_state_num  # attribute entropy

        # ===============    Transition Stage ===========
        # initialize dialog by randomly asked a question from ui interaction
        user_like_random_fea = random.choice(self.kg.G['item'][self.target_item]['belong_to'])
        self.user_acc_feature.append(user_like_random_fea)  # update user acc_fea
        self.cur_node_set.append(user_like_random_fea)
        self._update_cand_items(user_like_random_fea, acc_rej=True)
        self._updata_reachable_feature()  # self.reachable_feature = []
        self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']
        self.cur_conver_step += 1

        print('=== init user prefer feature: {}'.format(self.cur_node_set))
        self._update_feature_entropy()  # update entropy
        print('reset_reachable_feature num: {}'.format(len(self.reachable_feature)))

        # Sort reachable features according to the entropy of features
        reach_fea_score = self._feature_score()
        max_ind_list = []
        for k in range(self.cand_num):
            max_score = max(reach_fea_score)
            max_ind = reach_fea_score.index(max_score)
            reach_fea_score[max_ind] = 0
            if max_ind in max_ind_list:
                break
            max_ind_list.append(max_ind)

        max_fea_id = [self.reachable_feature[i] for i in max_ind_list]
        [self.reachable_feature.remove(v) for v in max_fea_id]
        [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]

        return self._get_state(), self._get_cand(), self._get_action_space()

    def _get_cand(self):
        if self.random_sample_feature:
            cand_feature = self._map_to_all_id(
                random.sample(self.reachable_feature, min(len(self.reachable_feature), self.cand_num)), 'feature')
        else:
            cand_feature = self._map_to_all_id(self.reachable_feature[:self.cand_num], 'feature')

        if self.random_sample_item:
            cand_item = self._map_to_all_id(
                random.sample(self.cand_items, min(len(self.cand_items), self.cand_item_num)), 'item')
        else:
            cand_item = self._map_to_all_id(self.cand_items[:self.cand_item_num], 'item')
        cand = cand_feature + cand_item
        return cand

    def _get_action_space(self):
        action_space = [self._map_to_all_id(self.reachable_feature, 'feature'), self._map_to_all_id(self.cand_items, 'item')]
        return action_space

    def _get_state(self):
        if self.data_name in ['YELP_STAR']:
            self_cand_items = self.cand_items[:5000]
            set_cand_items = set(self_cand_items)
        else:
            self_cand_items = self.cand_items
        user = [self.user_id]
        cur_node = [x + self.user_length + self.item_length for x in self.cur_node_set]
        cand_items = [x + self.user_length for x in self_cand_items]
        reachable_feature = [x + self.user_length + self.item_length for x in self.reachable_feature]
        neighbors = cur_node + user + cand_items + reachable_feature + self.time_node

        idx = dict(enumerate(neighbors))
        idx = {v: k for k, v in idx.items()}

        i = []

        #  1.
        for item in self_cand_items:
            item_idx = item + self.user_length
            for fea in self.item_feature_pair[item]:
                fea_idx = fea + self.user_length + self.item_length
                i.append([idx[item_idx], idx[fea_idx]])
                i.append([idx[fea_idx], idx[item_idx]])

        user_idx = len(cur_node)
        cand_item_score = self.sigmoid(self.cand_item_score)

        # 2.1
        for fea in cur_node:
            i.append([idx[item_idx], idx[fea]])
            i.append([idx[fea], idx[item_idx]])
        # 2.2
        for item, score in zip(self.cand_items, cand_item_score):
            if self.data_name in ['YELP_STAR']:
                if item not in set_cand_items:
                    continue
            item_idx = item + self.user_length
            i.append([user_idx, idx[item_idx]])
            i.append([idx[item_idx], user_idx])
        # 2.3
        for time in self.time_node:
            i.append([user_idx, idx[time]])
            i.append([idx[time], user_idx])

        i = torch.LongTensor(i)     # len x 2
        edge_index = i.t()          # 2 * edgeSize
        neighbors = torch.LongTensor(neighbors)

        state = {'cur_node': cur_node,      # acc_fea
                 'neighbors': neighbors,    # full_id
                 'adj': edge_index,         # 2 * edgeSize
                 'user_length': self.user_length,
                 'item_length': self.item_length,
                 'feature_length': self.feature_length,
                 'data_name': self.data_name,
                 }
        return state

    def step(self, action, sorted_actions, embed=None):
        if embed is not None:
            self.ui_embeds = embed[:self.user_length + self.item_length]
            self.feature_emb = embed[self.user_length + self.item_length:]

        done = 0
        print('---------------step:{}-------------'.format(self.cur_conver_step))

        if self.cur_conver_step == self.max_turn:
            reward = self.reward_dict['until_T']
            self.conver_his[self.cur_conver_step - 1] = self.history_dict['until_T']
            print('--> Maximum number of turns reached !')
            done = 1
        elif action >= self.user_length + self.item_length:  # ask feature
            asked_feature = self._map_to_old_id(action)
            print('-->action: ask features {}, max entropy feature {}'.format(asked_feature,
                                                                              self.reachable_feature[:self.cand_num]))
            reward, done, acc_rej = self._ask_update(
                asked_feature)  # update user's profile:  user_acc_feature & user_rej_feature
            self._update_cand_items(asked_feature, acc_rej)  # update cand_items
        else:  # recommend items

            # ===================== rec update=========
            recom_items = []
            for act in sorted_actions:
                if act < self.user_length + self.item_length:
                    recom_items.append(self._map_to_old_id(act))
                    if len(recom_items) == self.rec_num:
                        break
            reward, done = self._recommend_update(recom_items)
            # ========================================
            if reward > 0:
                print('-->Recommend successfully!')
            else:
                print('-->Recommend fail !')

        self._updata_reachable_feature()  # update user's profile: reachable_feature

        print('reachable_feature num: {}'.format(len(self.reachable_feature)))
        print('cand_item num: {}'.format(len(self.cand_items)))

        self._update_feature_entropy()
        if len(self.reachable_feature) != 0:  # if reachable_feature == 0 :cand_item= 1
            reach_fea_score = self._feature_score()  # compute feature score

            max_ind_list = []
            for k in range(self.cand_num):
                max_score = max(reach_fea_score)
                max_ind = reach_fea_score.index(max_score)
                reach_fea_score[max_ind] = 0
                if max_ind in max_ind_list:
                    break
                max_ind_list.append(max_ind)
            max_fea_id = [self.reachable_feature[i] for i in max_ind_list]
            [self.reachable_feature.remove(v) for v in max_fea_id]
            [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]

        self.cur_conver_step += 1
        return self._get_state(), self._get_cand(), self._get_action_space(), reward, done


    def _updata_reachable_feature(self):
        next_reachable_feature = []
        reachable_item_feature_pair = {}
        for cand in self.cand_items:
            fea_belong_items = list(self.kg.G['item'][cand]['belong_to'])  # A-I
            next_reachable_feature.extend(fea_belong_items)  # [ fea..., fea..., fea... ]
            reachable_item_feature_pair[cand] = list(set(fea_belong_items) - set(self.user_rej_feature))
            next_reachable_feature = list(set(next_reachable_feature))
        self.reachable_feature = list(
            set(next_reachable_feature) - set(self.user_acc_feature) - set(self.user_rej_feature))
        self.item_feature_pair = reachable_item_feature_pair


    def _feature_score(self):
        reach_fea_score = []
        for feature_id in self.reachable_feature:
            '''
            score = self.attr_ent[feature_id]
            reach_fea_score.append(score)
            '''
            feature_embed = self.feature_emb[feature_id]
            score = 0
            # 1.  ua^T
            score += np.inner(np.array(self.user_embed), feature_embed)
            prefer_embed = self.feature_emb[self.user_acc_feature, :]
            unprefer_embed = self.feature_emb[self.user_rej_feature, :]
            # rej_embed in user_acc_feature
            rej_item_embed = self.ui_embeds[self.rej_items, :]

            # 2. score += ap_{acc}^T
            for i in range(len(self.user_acc_feature)):
                score += np.inner(prefer_embed[i], feature_embed)
            # 3. score -= ap_{rej}^T
            if i in range(len(self.user_rej_feature)):
                score -= self.sigmoid([np.inner(unprefer_embed[i], feature_embed)])[0]
            # 4. score -= ai_{rej}^T
            for i in range(len(self.rej_items)):
                score -= self.sigmoid([np.inner(rej_item_embed[i], feature_embed)])[0]

            reach_fea_score.append(score)

        return reach_fea_score


    def _item_score(self):
        cand_item_score = []
        for item_id in self.cand_items:
            item_embed = self.ui_embeds[self.user_length + item_id]
            score = 0
            # 1.  uv^T
            score += np.inner(np.array(self.user_embed), item_embed)
            # perfer_embed in user_acc_feature
            prefer_embed = self.feature_emb[self.user_acc_feature, :]  # np.array (x*64)
            # rej_embed in user_acc_feature
            rej_item_embed = self.ui_embeds[self.rej_items, :]
            # 2. for p in user_acc_feature  =>  score += vp^T
            for i in range(len(self.user_acc_feature)):
                score += np.inner(prefer_embed[i], item_embed)
            # 3. for i in rej_item  =>  score -= vi^T
            for i in range(len(self.rej_items)):
                score -= self.sigmoid([np.inner(rej_item_embed[i], item_embed)])[0]
                # score -= np.inner(unprefer_embed[i], item_embed)
            cand_item_score.append(score)
        return cand_item_score


    def _ask_update(self, asked_feature):
        '''
        :return: reward, acc_feature, rej_feature
        '''
        done = 0
        feature_groundtrue = self.kg.G['item'][self.target_item]['belong_to']

        if asked_feature in feature_groundtrue:
            acc_rej = True
            self.user_acc_feature.append(asked_feature)
            self.cur_node_set.append(asked_feature)
            reward = self.reward_dict['ask_suc']
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']  # update conver_his
        else:
            acc_rej = False
            self.user_rej_feature.append(asked_feature)
            reward = self.reward_dict['ask_fail']
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_fail']  # update conver_his

        if self.cand_items == []:  # candidate items is empty
            done = 1
            reward = self.reward_dict['cand_none']

        return reward, done, acc_rej

    def _update_cand_items(self, asked_feature, acc_rej):
        if acc_rej:  # accept feature
            print('=== ask acc: update cand_items')
            feature_items = self.kg.G['feature'][asked_feature]['belong_to']
            self.cand_items = set(self.cand_items) & set(feature_items)  # itersection
            self.cand_items = list(self.cand_items)

        else:  # reject feature
            print('=== ask rej: update cand_items')

            feature_items = self.kg.G['feature'][asked_feature]['belong_to']
            self.cand_items = set(self.cand_items) - set(feature_items)
            self.cand_items = list(self.cand_items)

        # select topk candidate items to recommend
        cand_item_score = self._item_score()
        item_score_tuple = list(zip(self.cand_items, cand_item_score))
        sort_tuple = sorted(item_score_tuple, key=lambda x: x[1], reverse=True)
        self.cand_items, self.cand_item_score = zip(*sort_tuple)


    def _recommend_update(self, recom_items):
        print('-->action: recommend items')
        print(set(recom_items) - set(self.cand_items[: self.rec_num]))
        self.cand_items = list(self.cand_items)
        self.cand_item_score = list(self.cand_item_score)
        if self.target_item in recom_items:
            reward = self.reward_dict['rec_suc']
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_scu']  # update state vector: conver_his
            tmp_score = []
            for item in recom_items:
                idx = self.cand_items.index(item)
                tmp_score.append(self.cand_item_score[idx])
            self.cand_items = recom_items
            self.cand_item_score = tmp_score
            done = recom_items.index(self.target_item) + 1
        else:
            reward = self.reward_dict['rec_fail']
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_fail']  # update state vector: conver_his
            if len(self.cand_items) > self.rec_num:
                for item in recom_items:
                    del self.item_feature_pair[item]
                    idx = self.cand_items.index(item)
                    self.cand_items.pop(idx)
                    self.cand_item_score.pop(idx)
                full_id = self._map_to_all_id(recom_items, 'item')
                self.rej_items.extend(full_id)
            done = 0
        return reward, done

    def _update_feature_entropy(self):
        if self.ent_way == 'entropy':
            cand_items_fea_list = []
            for item_id in self.cand_items:
                cand_items_fea_list.append(list(self.kg.G['item'][item_id]['belong_to']))
            cand_items_fea_list = list(_flatten(cand_items_fea_list))
            self.attr_count_dict = dict(Counter(cand_items_fea_list))
            self.attr_ent = [0] * self.attr_state_num  # reset attr_ent
            real_ask_able = list(set(self.reachable_feature) & set(self.attr_count_dict.keys()))
            for fea_id in real_ask_able:
                p1 = float(self.attr_count_dict[fea_id]) / len(self.cand_items)
                p2 = 1.0 - p1
                if p1 == 1:
                    self.attr_ent[fea_id] = 0
                else:
                    ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                    self.attr_ent[fea_id] = ent
        elif self.ent_way == 'weight_entropy':
            cand_items_fea_list = []
            self.attr_count_dict = {}
            # cand_item_score = self._item_score()
            cand_item_score_sig = self.sigmoid(self.cand_item_score)  # sigmoid(score)
            for score_ind, item_id in enumerate(self.cand_items):
                cand_items_fea_list = list(self.kg.G['item'][item_id]['belong_to'])
                for fea_id in cand_items_fea_list:
                    if self.attr_count_dict.get(fea_id) == None:
                        self.attr_count_dict[fea_id] = 0
                    self.attr_count_dict[fea_id] += cand_item_score_sig[score_ind]

            self.attr_ent = [0] * self.attr_state_num  # reset attr_ent
            real_ask_able = list(set(self.reachable_feature) & set(self.attr_count_dict.keys()))
            sum_score_sig = sum(cand_item_score_sig)

            for fea_id in real_ask_able:
                p1 = float(self.attr_count_dict[fea_id]) / (sum_score_sig + 0.0001)
                p2 = 1.0 - p1
                if p1 == 1 or p1 <= 0:
                    self.attr_ent[fea_id] = 0
                else:
                    ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                    self.attr_ent[fea_id] = ent

    def sigmoid(self, x_list):
        x_np = np.array(x_list)
        s = 1 / (1 + np.exp(-x_np))
        return s.tolist()

    def _map_to_all_id(self, x_list, old_type):
        if old_type == 'item':
            return [x + self.user_length for x in x_list]
        elif old_type == 'feature':
            return [x + self.user_length + self.item_length for x in x_list]
        elif old_type == 'time':
            return [x + self.user_length + self.item_length + self.feature_length for x in x_list]
        else:
            return x_list

    def _map_to_old_id(self, x):
        if x >= self.user_length + self.item_length:
            x -= (self.user_length + self.item_length)
        elif x >= self.user_length:
            x -= self.user_length
        return x

