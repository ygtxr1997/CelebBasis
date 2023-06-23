import os
import time
from tqdm import tqdm
import pickle


def gen_pickle_abs(in_root: str = '/gavin/datasets/web/ori',
                   out_path: str = '/gavin/datasets/web/256x256.pickle',
                   ):
    dataset_list = os.listdir(in_root)
    dataset_list.sort()

    print('Generating pickle list...(saved to %s)' % out_path)
    list_to = []
    for img_name in tqdm(dataset_list):
        list_to.append(os.path.join(in_root, img_name))
    pickle.dump(list_to, open(out_path, 'wb+'))

    print('Checking pickle list...')
    with open(out_path, 'rb') as f_from:
        p_from = pickle.load(f_from)
    print('pickle type:', type(p_from), ', total len:', len(p_from))
    print('pickle[0]:', p_from[0])
    if len(p_from) >= 2:
        print('pickle[1]:', p_from[1])
        print('pickle[-2]:', p_from[-2])
        print('pickle[-1]:', p_from[-1])
    print('-' * 50)
