import os

class PersonData():
    def __init__(self, content):
        self.id = (content[-1], content[0])  # type and ID
        self.state = content[6: 9]  # 3 0/1 state
        self.bound_box = [content[1: 3], content[3: 5]]
        self.bound_box = [list(map(int, sublist)) for sublist in self.bound_box]
        self.position = [(self.bound_box[0][0]+self.bound_box[1][0])/2, (self.bound_box[0][1]+self.bound_box[1][1])/2]
        self.frame = int(content[5])  # appear frame num

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


data_dict = {}

def make_data(d):
    pData = [d.get_frame(), d.get_pos(), d.get_bound_box()]
    return pData


def load_data(path):
    if os.path.isfile(path):
        print('File is exist.')
        contents = open(path, 'r').read().replace('\n', ' ').split(' ')
        contents.pop(-1)
        if len(contents) % 10 is not 0:
            print('Error: File Format is wrong. Length of file is not match.')
            exit(0)

        for idx in range(0, int(len(contents)/10)):
            pData = PersonData(contents[idx*10: idx*10+10])
            if pData.get_id() in data_dict:
                data_dict[pData.get_id()].append(make_data(pData))
            else:
                data_dict[pData.get_id()] = [make_data(pData)]
    else:
        print('Error: File is not exist.')
        exit(0)


load_data('test.txt')


f = open('result', 'w')

for item in data_dict:
    f.write(str(item) + ": \n")
    #print(item, ": ")
    data_dict[item] = sorted(data_dict[item], key=lambda s: s[0])
    for d in data_dict[item]:
        f.write(str(d)+'\n')
        #print(d)
    #print(data_dict[item])
    

