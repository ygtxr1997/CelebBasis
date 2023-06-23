import os
import numpy as np


def read_txt(txt_path: str, split_code: str = '\t'):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        row = line.strip().split(split_code)
        row = [float(x) for x in row]
        data.append(row)
    data_np = np.array(data)
    print(f'[Read txt] from {txt_path}', data_np.shape)
    return data_np


if __name__ == "__main__":
    read_txt("exp_id-txt_sota.txt")
