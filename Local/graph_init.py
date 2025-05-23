
import argparse

from Graph_generate.movie_data_process import MovieDataset
from Graph_generate.movie_graph import MovieGraph
from utils import *
from Graph_generate.yelp_data_process import YelpDataset
from Graph_generate.yelp_graph import YelpGraph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default=YELP_STAR, choices=[MOVIE, YELP_STAR],
                        help='One of {LAST_FM, LAST_FM_STAR, YELP, YELP_STAR}.')
    args = parser.parse_args()
    DatasetDict = {
        MOVIE: MovieDataset,
        YELP_STAR: YelpDataset,
    }
    GraphDict = {
        MOVIE: MovieGraph,
        YELP_STAR: YelpGraph,
    }

    # Create 'data_name' instance for data_name.
    print('Load', args.data_name, 'from file...')
    print(TMP_DIR[args.data_name])
    if not os.path.isdir(TMP_DIR[args.data_name]):
        os.makedirs(TMP_DIR[args.data_name])
    dataset = DatasetDict[args.data_name](DATA_DIR[args.data_name])
    save_dataset(args.data_name, dataset)
    print('Save', args.data_name, 'dataset successfully!')

    # Generate graph instance for 'data_name'
    print('Create', args.data_name, 'graph from data_name...')
    dataset = load_dataset(args.data_name)
    kg = GraphDict[args.data_name](dataset)
    save_kg(args.data_name, kg)
    print('Save', args.data_name, 'graph successfully!')


def construct(kg):
    users = kg.G['user'].keys()
    items = kg.G['item'].keys()
    features = kg.G['feature'].keys()
    num_node = len(users) + len(items) + len(features)
    graph = np.zeros((num_node, num_node))
    for i in range(num_node):
        for j in range(num_node):
            if i < len(users) and j < len(users)+len(items):
                graph[i][j] = 1
                graph[j][i] = 1
            elif i >= len(users) and i < len(users)+len(items):
                if j-len(users)-len(items) in kg.G['item'][i-len(users)]['belong_to']:
                    graph[i][j] = 1
                    graph[j][i] = 1
            else:
                pass
    print(graph)
    return graph


if __name__ == '__main__':
    main()

