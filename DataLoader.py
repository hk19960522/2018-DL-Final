import os
import torch


class PersonData:
    def __init__(self, content):
        self.id = (content[-1], content[0])  # type and ID
        self.state = content[6: 9]  # 3 0/1 state
        self.bound_box = [content[1: 3], content[3: 5]]
        self.bound_box = [list(map(int, sublist)) for sublist in self.bound_box]
        self.position = [(self.bound_box[0][0]+self.bound_box[1][0])/2, (self.bound_box[0][1]+self.bound_box[1][1])/2]
        self.frame = int(content[5])  # appear frame num
        self.type = content[-1]

    def get_id(self):
        return self.id

    def get_state(self):
        return self.state

    def get_pos(self):
        return self.position

    def get_bound_box(self):
        return self.bound_box

    def get_frame(self):
        return self.frame

    def get_type(self):
        return self.type

def make_data(d):
    pData = [d.get_frame(), d.get_pos(), d.get_type()]
    return pData

def load_data(path):
    data_dict = {}
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
            if pData.get_id() in data_dict:
                data_dict[pData.get_id()].append(make_data(pData))
            else:
                data_dict[pData.get_id()] = [make_data(pData)]
        # sort data by frame
        for item in data_dict:
            data_dict[item] = sorted(data_dict[item], key=lambda l: l[0])
            #data_dict[item].append(0)  # index of item
    else:
        print('Error: File %s is not exist.'.format(path))
        exit(0)
    return data_dict

def get_one_hot(label):
    if label == '"Pedestrian"':
        one_hot = [1, 0, 0]
    elif label == '"Biker"':
        one_hot = [0, 1, 0]
    elif label == '"Skater"':
        one_hot = [0, 0, 1]
    else :
        one_hot = [0, 0, 0]
    return one_hot


class DataLoader:
    def __init__(self, path):
        self.data_dict = load_data(path)

    #  need to call this function from 0 to the end, or it will not work
    def get_frame_data(self, frame):
        data = {}
        for key, value in self.data_dict.items():
            idx = value[-1]
            if idx < len(value)-1 and value[idx][0] == frame:
                data[key] = value[idx]
                value[-1] += 1
        return data

    def get_frame_data(self, start, end, step):
        data = {}
        for key, value in self.data_dict.items():
            row_data = []
            in_range = False
            for (idx, pData) in enumerate(value):
                if pData[0] == start:
                    in_range = True
                    # print(key)
                    for cnt in range(0, end-start+1):
                        # print('Hi', value[idx + cnt])
                        if idx + cnt >= len(value) or (start + cnt) != value[idx + cnt][0]:
                            # print('Error')
                            in_range = False
                            break

                        if cnt % step == 0:
                            new_data = value[idx + cnt][1] + get_one_hot(value[idx + cnt][-1])
                            row_data.append(new_data)
                    break
            if in_range:
                data[key] = row_data
        return data

    def batchify(self, batch_size, input_frame, target_frame):
        data = self.get_frame_data(0, input_frame + target_frame, 2)
        input_list, target_list = [], []
        count = 0
        for v in data.values():
            count += 1
            input_list.append(v[:input_frame])
            target_list.append(v[-target_frame:])
            if count % batch_size == 0:
                yield torch.tensor(input_list), torch.tensor(input_list)
                input_list, target_list = [], []


if __name__ == '__main__':
    ''' 
    f = open('result', 'w')
    
    for item in data_dict:
        f.write(str(item) + ": \n")
        data_dict[item] = sorted(data_dict[item], key=lambda s: s[0])
        for d in data_dict[item]:
            f.write(str(d)+'\n')
    
    d = get_frame_data(0)
    for key, value in d.items():
        print(key)
        print(value, '\n')
    '''
    dl = DataLoader('test.txt')
    print(dl.get_frame_data(0, 4, 2))
