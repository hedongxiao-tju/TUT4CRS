import os
import json
import random

import numpy as np
from easydict import EasyDict as edict


class MovieDataset(object):
    def __init__(self, data_dir,args):
        self.args = args
        self.noise = args.noise
        self.noise_rate = args.noise_rate

        self.data_dir = data_dir + '/Graph_generate_data'
        self.load_entities()
        self.load_relations()

    def get_relation(self):
        #Entities
        USER = 'user'
        ITEM = 'item'
        FEATURE = 'feature'

        #Relations
        INTERACT = 'interact'
        # FRIEND = 'friends'
        # LIKE = 'like'
        BELONG_TO = 'belong_to'

        # relation_name = [INTERACT, FRIEND, LIKE, BELONG_TO]
        relation_name = [INTERACT,BELONG_TO]

        myelp_relation = {
            USER: {
                INTERACT: ITEM,
                # FRIEND: USER,
                # LIKE: FEATURE,
            },
            ITEM: {
                BELONG_TO: FEATURE,
                INTERACT: USER
            },
            FEATURE: {
                # LIKE: USER,
                BELONG_TO: ITEM
            }
        }

        myelp_relation_link_entity_type = {
            INTERACT:  [USER, ITEM],
            # FRIEND:  [USER, USER],
            # LIKE:  [USER, FEATURE],
            BELONG_TO:  [ITEM, FEATURE]
        }
        return myelp_relation, relation_name, myelp_relation_link_entity_type

    def load_entities(self):
        entity_files = edict(
            user='user_dict.json',
            item='item_dict.json',
            feature='tag_map.json',
        )
        for entity_name in entity_files:
            with open(os.path.join(self.data_dir,entity_files[entity_name]), encoding='utf-8') as f:
                mydict = json.load(f)
            if entity_name == 'feature':
                entity_id = list(mydict.values())
            else:   # user item
                entity_id = list(map(int, list(mydict.keys())))
            setattr(self, entity_name, edict(id=entity_id, value_len=max(entity_id)+1))
            print('Load', entity_name, 'of size', len(entity_id))
            print(entity_name, 'of max id is', max(entity_id))

    def load_relations(self):
        """
        relation: head entity---> tail entity
        --
        """
        Book_relations = edict(
            interact=('user_item_train.json', self.user, self.item), #(filename, head_entity, tail_entity)
            # friends=('user_dict.json', self.user, self.user),
            # like=('user_dict.json', self.user, self.feature),
            belong_to=('item_dict.json', self.item, self.feature),
        )

        for name in Book_relations:  #interaction\friends\like\belong_to
            #  Save tail_entity
            relation = edict(
                data=[],
            )
            knowledge = [list([]) for i in range(Book_relations[name][1].value_len)]
            # load relation files
            with open(os.path.join(self.data_dir, Book_relations[name][0]), encoding='utf-8') as f:
                mydict = json.load(f)
            if name in ['interact']:
                for key, value in mydict.items():
                    head_id = int(key)
                    tail_ids = [review['item'] for review in value]
                    knowledge[head_id] = tail_ids

            elif name in ['friends', 'like']:
                for key in mydict.keys():
                    head_str = key
                    head_id = int(key)
                    tail_ids = mydict[head_str][name]
                    knowledge[head_id] = tail_ids
            elif name in ['belong_to']:   #item_dict.json
                for key in mydict.keys():
                    head_str = key
                    head_id = int(key)
                    tail_ids = mydict[head_str]['feature_index']
                    knowledge[head_id] = tail_ids
            relation.data = knowledge

            setattr(self, name, relation)
            tuple_num = 0
            for i in knowledge:
                tuple_num += len(i)
            print('Load', name, 'of size', tuple_num)






