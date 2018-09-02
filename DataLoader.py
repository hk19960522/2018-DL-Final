import os
import random
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

from config import Config


class PersonData:
    def __init__(self, content):
        self.id = (content[-1], content[0])  # type and ID
        self.state = content[6: 9]  # 3 0/1 state
        self.bound_box = [content[1: 3], content[3: 5]]
        self.bound_box = [list(map(int, sublist)) for sublist in self.bound_box]
        self.position = [(self.bound_box[0][0]+self.bound_box[1][0])/2, (self.bound_box[0][1]+self.bound_box[1][1])/2]
        self.frame = int(content[5])  # appear frame num
        self.type = content[-1]

    def make_data(self):
        pData = [self.frame, self.position, self.type]
        return pData

    def same_person(self, other):
        return self.id == other.id

    def to_feature(self):
        return self.position + get_one_hot(type)

def load_data(path):
    raw_data = []
    if os.path.isfile(path):
        print('File is exist.')
        contents = open(path, 'r').read().replace('\n', ' ').split(' ')
        contents.pop(-1)
        if len(contents) % 10 is not 0:
            print('Error: File Format is wrong. Length of file is not match.')
            exit(0)
        # slice data
        for idx in range(0, int(len(contents)/10)):
            pData = PersonData(contents[idx*10: idx*10+10])
            raw_data.append(pData)
    else:
        print('Error: File %s is not exist.'.format(path))
        exit(0)
    return raw_data


def get_one_hot(label):
    if label == '"Pedestrian"':
        one_hot = [1.0, 0.0, 0.0]
    elif label == '"Biker"':
        one_hot = [0.0, 1.0, 0.0]
    elif label == '"Skater"':
        one_hot = [0.0, 0.0, 1.0]
    else:
        one_hot = [0.0, 0.0, 0.0]
    return one_hot


def get_data_loader(path, config):
    def to_row(ss):
        if len(ss) < config.input_frame + config.input_frame or len(ss[0]) < config.pedestrian_num:
            return None
        first_frame = random.sample(ss[0], config.pedestrian_num)
        ret = []
        for fp in first_frame:
            person_time_series = []
            for people in ss:
                if len(people) < config.pedestrian_num:
                    return None
                find_people = [p for p in people if p.same_person(fp)]
                if len(find_people) == 0:
                    return None
                person_time_series.append(find_people[0].to_feature())
            ret.append(person_time_series)
        return np.array(ret)

    raw_data = load_data(path)

    # make time dict
    time_dict = {}
    for pData in raw_data:
        if pData.frame in time_dict:
            time_dict[pData.frame].append(pData)
        else:
            time_dict[pData.frame] = [pData]

    sample_train, sample_target = [], []
    for t, pDatas in time_dict.items():
        series = []
        for i in range(config.input_frame + config.target_frame):
            key = t + i * config.sample_rate
            if key in time_dict:
                series.append(time_dict[key])
        series = to_row(series)
        if series is not None:
            sample_train.append(series[:, :config.input_frame])
            sample_target.append(series[:, -config.target_frame:])

    sample_train = torch.tensor(sample_train)
    sample_target = torch.tensor(sample_target)
    # print('train:', sample_train.size())
    # print('target:', sample_target.size())
    dataset = TensorDataset(sample_train, sample_target)
    return DataLoader(dataset, batch_size=config.batch_size)


if __name__ == '__main__':
    dl = get_data_loader('test.txt', Config())
