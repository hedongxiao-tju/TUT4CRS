import pickle
from datetime import datetime

import json
import numpy as np
import random
import torch
import os
import sys
# from knowledge_graph import KnowledgeGraph
# from data_process import LastFmDataset
# from KG_data_generate.lastfm_small_data_process import LastFmSmallDataset
# from KG_data_generate.lastfm_knowledge_graph import KnowledgeGraph
#Dataset names
YELP_STAR = 'YELP_STAR'
MOVIE = 'MOVIE'


DATA_DIR = {
    YELP_STAR: './data/yelp_star',
    MOVIE: './data/movie-20M_dict',
}
TMP_DIR = {
    YELP_STAR: './tmp/yelp_star',
    MOVIE:'./tmp/movie'
}


def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var
def save_dataset(dataset, dataset_obj):
    dataset_file = TMP_DIR[dataset] + '/dataset.pkl'
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset_obj, f)

def load_dataset(dataset):
    dataset_file = TMP_DIR[dataset] + '/dataset.pkl'
    dataset_obj = pickle.load(open(dataset_file, 'rb'))
    return dataset_obj

def save_kg(dataset, kg):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    pickle.dump(kg, open(kg_file, 'wb'))

def load_kg(dataset):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    kg = pickle.load(open(kg_file, 'rb'))
    return kg

def save_graph(dataset, graph):
    graph_file = TMP_DIR[dataset] + '/graph.pkl'
    pickle.dump(graph, open(graph_file, 'wb'))

def load_graph(dataset):
    graph_file = TMP_DIR[dataset] + '/graph.pkl'
    graph = pickle.load(open(graph_file, 'rb'))
    return graph

def load_embed(dataset, embed, epoch):
    if embed:
        path = TMP_DIR[dataset] + '/embeds/' + '{}-ndarray.pkl'.format(embed)
    else:
        return None
    with open(path, 'rb') as f:
        embeds = pickle.load(f)
        print('{} Embedding load successfully!'.format(embed))
        return embeds

def load_timetrain_embed(dataset,args):
    print('loading time emb...')
    path = TMP_DIR[dataset] + f'/Time-model-embeds/new-iter-{args.time_emb_file}.pkl'
    print(f'load time aware embedding path is : {path}')
    with open(path, 'rb') as f:
        embeds = pickle.load(f)
    return embeds

def load_timetrain_embed_processData(dataset,time_emb_file):
    print('loading time emb...')
    path = TMP_DIR[dataset] + f'/Time-model-embeds/new-iter-{time_emb_file}.pkl'
    with open(path, 'rb') as f:
        embeds = pickle.load(f)
    return embeds

def load_rl_agent(dataset, filename, epoch_user):
    model_file = TMP_DIR[dataset] + '/RL-agent/' + filename + '-epoch-{}.pkl'.format(epoch_user)
    model_dict = torch.load(model_file, map_location='cuda')
    print('RL policy model load at {}'.format(model_file))
    return model_dict

def save_rl_agent(dataset, model, filename, epoch_user):
    model_file = TMP_DIR[dataset] + '/RL-agent/' + filename + '-epoch-{}.pkl'.format(epoch_user)
    if not os.path.isdir(TMP_DIR[dataset] + '/RL-agent/'):
        os.makedirs(TMP_DIR[dataset] + '/RL-agent/')
    torch.save(model, model_file)
    print('RL policy model saved at {}'.format(model_file))


def save_rl_mtric(dataset, filename, epoch, SR, spend_time, mode='train'):
    pass
    # PATH = TMP_DIR[dataset] + '/RL-log-merge/' + filename + '.txt'
    # if not os.path.isdir(TMP_DIR[dataset] + '/RL-log-merge/'):
    #     os.makedirs(TMP_DIR[dataset] + '/RL-log-merge/')
    # if mode == 'train':
    #     with open(PATH, 'a') as f:
    #         f.write('===========Train===============\n')
    #         f.write('Starting {} user epochs\n'.format(epoch))
    #         f.write('training SR@5: {}\n'.format(SR[0]))
    #         f.write('training SR@10: {}\n'.format(SR[1]))
    #         f.write('training SR@15: {}\n'.format(SR[2]))
    #         f.write('training Avg@T: {}\n'.format(SR[3]))
    #         f.write('training hDCG: {}\n'.format(SR[4]))
    #         f.write('Spending time: {}\n'.format(spend_time))
    #         f.write('================================\n')
    #         # f.write('1000 loss: {}\n'.format(loss_1000))
    # elif mode == 'test':
    #     with open(PATH, 'a') as f:
    #         f.write('===========Test===============\n')
    #         f.write('Testing {} user tuples\n'.format(epoch))
    #         f.write('Testing SR@5: {}\n'.format(SR[0]))
    #         f.write('Testing SR@10: {}\n'.format(SR[1]))
    #         f.write('Testing SR@15: {}\n'.format(SR[2]))
    #         f.write('Testing Avg@T: {}\n'.format(SR[3]))
    #         f.write('Testing hDCG: {}\n'.format(SR[4]))
    #         f.write('Testing time: {}\n'.format(spend_time))
    #         f.write('================================\n')
    #         # f.write('1000 loss: {}\n'.format(loss_1000))

def save_rl_model_log(dataset, filename, epoch, epoch_loss, train_len):
    pass
    # PATH = TMP_DIR[dataset] + '/RL-log-merge/' + filename + '.txt'
    # if not os.path.isdir(TMP_DIR[dataset] + '/RL-log-merge/'):
    #     os.makedirs(TMP_DIR[dataset] + '/RL-log-merge/')
    # with open(PATH, 'a') as f:
    #     f.write('Starting {} epoch\n'.format(epoch))
    #     f.write('training loss : {}\n'.format(epoch_loss / train_len))
    #     # f.write('1000 loss: {}\n'.format(loss_1000))

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def set_cuda(args):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    devices_id = [int(device_id) for device_id in args.gpu.split()]
    device = (
        torch.device("cuda:{}".format(str(devices_id[0])))
        if use_cuda
        else torch.device("cpu")
    )
    return device, devices_id


def date2array(date_string, date_name='MOVIE'):
    # year_start  year_num
    Date_Info_Dict = {
        'MOVIE': {
            'year_start': 1996,
            'year_num': 20,
            'month_num': 12,
            'day_num': 31,
            'hour_num': 24
        },
        'YELP_STAR': {
            'year_start': 2004,
            'year_num': 15,
            'month_num': 12,
            'day_num': 31,
            'hour_num': 0
        }
    }

    if date_name in ('YELP_STAR',):
        date_object = datetime.strptime(date_string, "%Y-%m-%d").timetuple()
        # （tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, tm_wday, tm_yday, tm_isdst）
        year = date_object.tm_year
        month = date_object.tm_mon
        day = date_object.tm_mday
        weekday = date_object.tm_wday
        # hour = date_object.tm_hour
        return [year - Date_Info_Dict[date_name]['year_start'],
                month - 1 + Date_Info_Dict[date_name]['year_num'],
                day - 1 + Date_Info_Dict[date_name]['year_num'] + Date_Info_Dict[date_name]['month_num'],
                weekday + Date_Info_Dict[date_name]['year_num'] + Date_Info_Dict[date_name]['month_num'] + Date_Info_Dict[date_name]['day_num']]
    else:
        date_object = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S").timetuple()
        # （tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, tm_wday, tm_yday, tm_isdst）
        year = date_object.tm_year
        month = date_object.tm_mon
        day = date_object.tm_mday
        hour = date_object.tm_hour
        weekday = date_object.tm_wday
        return [year - Date_Info_Dict[date_name]['year_start'],
                month - 1 + Date_Info_Dict[date_name]['year_num'],
                day - 1 + Date_Info_Dict[date_name]['year_num'] + Date_Info_Dict[date_name]['month_num'],
                hour + Date_Info_Dict[date_name]['year_num'] + Date_Info_Dict[date_name]['month_num'] + Date_Info_Dict[date_name]['day_num'],
                weekday + Date_Info_Dict[date_name]['year_num'] + Date_Info_Dict[date_name]['month_num'] + Date_Info_Dict[date_name]['day_num'] + Date_Info_Dict[date_name]['hour_num']
                ]

def loadPKL(file):
    print('load pkl...')
    with open(file,'rb') as f:
        dict = pickle.load(f)
    return dict

def savePKL(file, obj):
    print('save pkl...')
    with open(file,'wb') as f:
        pickle.dump(obj,f)

def loadJson(file):
    with open(file) as f:
        dict = json.load(f)
    return dict

def saveJson(file, obj):
    with open(file,'w') as f:
        json.dump(obj,f,indent=4)

