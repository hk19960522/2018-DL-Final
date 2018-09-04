import os
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class PersonData:
    def __init__(self, content):
        self.id = content[0]  # type and ID
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
        return [self.position[0] / 1920.0, self.position[1] / 1080.0] + get_one_hot(self.type)


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
        for i in range(0, int(len(contents)/10)):
            pData = PersonData(contents[i*10: i*10+10])
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


def get_data_loader(path, train_frame, target_frame, pedestrian_num, sample_rate):
    def to_row(ss):
        if len(ss) < train_frame + target_frame or len(ss[0]) < pedestrian_num:
            return None
        first_frame = random.sample(ss[0], pedestrian_num)
        ret = []
        for fp in first_frame:
            person_time_series = []
            for people in ss:
                if len(people) < pedestrian_num:
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
        # if pData.type not in ['"Pedestrian"', '"Biker"', '"Skater"']:
        #     continue  # uncomment this to filter type
        if pData.frame in time_dict:
            time_dict[pData.frame].append(pData)
        else:
            time_dict[pData.frame] = [pData]

    sample_train, sample_target = [], []
    sample_frame = []
    for t, pDatas in time_dict.items():
        series = []
        for i in range(train_frame + target_frame):
            key = t + i * sample_rate
            if key in time_dict:
                series.append(time_dict[key])
        series = to_row(series)
        if series is not None:
            sample_train.append(series[:, :train_frame])
            sample_target.append(series[:, -target_frame:])
            sample_frame.append(t + (train_frame-1) * sample_rate)

    sample_train = torch.tensor(sample_train)
    sample_target = torch.tensor(sample_target)
    # print('train:', sample_train.size())
    # print('target:', sample_target.size())
    print('File loaded.')
    return sample_train, sample_target, sample_frame


def get_cuhk_data(dir, train_frame, target_frame, pedestrian_num):    # load data
    def get_frames(time_dict, interval):
        frame = []
        first = int(min(time_dict))
        for i in range(0, interval * 20, 20):
            if str(i + first) in time_dict:
                frame.append(i + first)
        return frame if len(frame) is interval else None

    id_dict = {}
    for filename in os.listdir(dir):
        time_dict = {}
        with open(os.path.join(dir, filename)) as f:
            lines = f.read().replace('\n', ' ').split(' ')[:-1]
            if len(lines) % 3 is not 0:
                print('ignore file %s' % filename)
                continue
            for i in range(0, len(lines), 3):
                t = lines[i+2]
                x, y = int(lines[i]), int(lines[i+1])
                time_dict[t] = [x, y]
        fname = str(int(filename.split('.')[0]))
        id_dict[fname] = time_dict

    train, test, sample_frame = [], [], []
    for fname in id_dict:
        frames = get_frames(id_dict[fname], train_frame + target_frame)
        if frames is None:
            continue
        data = []
        for time_dict in id_dict.values():
            if all([str(f) in time_dict for f in frames]):
                data.append([time_dict[str(ff)] for ff in frames])
            if len(data) is pedestrian_num:
                break
        if len(data) is not pedestrian_num:
            continue
        # print(len(data), end=' ')
        data = torch.FloatTensor(data)
        train.append(data[:, :train_frame])
        test.append(data[:, -target_frame:])
        sample_frame.append(frames[train_frame-1])
    train = torch.stack(train)
    test = torch.stack(test)
    print('Data loaded.')
    return train, test, sample_frame


if __name__ == '__main__':
    d1, d2, d3 = get_cuhk_data('./dataset/Annotation/', 5, 5, 20)
    print(d1.size(), d2.size(), len(d3))
    torch.save((d1, d2, d3), './dataset/AnnotationDataset.pkl')
