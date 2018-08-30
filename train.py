import torch
import torch.optim as optim
import torch.nn.functional as F

from model import *
from config import Config
from DataLoader import DataLoader


class CIDNN_Training:
    def __init__(self):
        # prepare data
        self.config = Config()
        self.dataloader = DataLoader(path='test.txt')
        self.motionEncoder = MotionEncoder()
        self.locationEncoder = LocationEncoder(self.config.pedestrian_num,
                                               self.config.input_size,
                                               self.config.hidden_size)
        self.crowdInteraction = CrowdInteraction(self.config.pedestrian_num,
                                                 self.config.hidden_size,
                                                 self.config.hidden_size)
        self.displaceDecoder = DisplacementPrediction()  # TODO: module initialization
        self.optim_ME = optim.Adam(self.motionEncoder.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.optim_LE = optim.Adam(self.locationEncoder.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.optim_DD = optim.Adam(self.displaceDecoder.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

    def main_step(self, input_traces, target_traces):
        # TODO: main compute step
        return F.mse_loss(input_traces, target_traces)

    def train(self, input_, target_):
        self.motionEncoder.zero_grad()
        self.locationEncoder.zero_grad()
        self.crowdInteraction.zero_grad()
        self.displaceDecoder.zero_grad()

        loss = self.main_step(input_, target_)
        loss.backward()

        self.optim_ME.step()
        self.optim_LE.step()
        self.optim_DD.step()

        return loss

    def main(self):
        # train loop
        for input_traces, target_traces in self.dataloader.batchify(self.config.batch_size,
                                                                    self.config.input_frame,
                                                                    self.config.target_frame):
            print(input_traces.size())
            print(target_traces.size())
            self.train(input_traces, target_traces)


if __name__ == '__main__':
    cidnn = CIDNN_Training()
    print(cidnn.dataloader.data_dict)
    cidnn.main()
# TODO: main function
