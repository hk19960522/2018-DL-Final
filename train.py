import torch

import model
from config import Config
from DataLoader import DataLoader


def batchify(data):
    for v in data.values():
        input_, target_ = v[:-1], v[-1:]
        yield torch.tensor(input_), torch.tensor(target_)


class CIDNN_training:
    def __init__(self):
        # prepare data
        self.config = Config()
        self.data = self.load_data('test.txt')
        self.batch_fn = batchify


    def load_data(self, path=None):
        DataLoader.load_data(path)
        return DataLoader.get_frame_data(0, 4, 2)

    def main(self):
        # prepare model
        # train loop
        for input_traces, target_traces in self.batch_fn(self.data):
            print(input_traces.size())
            print(target_traces.size())



if __name__ == '__main__':
    cidnn = CIDNN_training()
    print(cidnn.data)
    cidnn.main()
# TODO: main function
