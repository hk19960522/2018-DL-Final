import torch
import torch.optim as optim
import torch.nn.functional as F

from pprint import pprint
from model import *
from config import Config
from DataLoader import get_data_loader


class CIDNN_Training:
    def __init__(self):
        # prepare data
        self.config = Config()
        self.dataloader = get_data_loader('test.txt', config=self.config)
        self.motionEncoder = MotionEncoder(self.config.pedestrian_num,
                                           self.config.n_layers,
                                           self.config.input_size,
                                           self.config.hidden_size)
        self.locationEncoder = LocationEncoder(self.config.pedestrian_num,
                                               self.config.input_size,
                                               self.config.hidden_size,
                                               self.config.batch_size)
        self.crowdInteraction = CrowdInteraction(self.config.pedestrian_num,
                                                 self.config.hidden_size)
        self.displaceDecoder = DisplacementPrediction(self.config.pedestrian_num,
                                                      self.config.hidden_size,
                                                      self.config.target_size)  # TODO: module initialization
        self.optim_ME = optim.Adam(self.motionEncoder.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.optim_LE = optim.Adam(self.locationEncoder.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.optim_DD = optim.Adam(self.displaceDecoder.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

    def main_step(self, input_traces, target_traces):
        # TODO: main compute step
        # (batch, pedestrain_num, frame, feature)
        batch_size = input_traces.size(0)
        target = input_traces[:, :, self.config.input_frame-1]
        hidden = self.motionEncoder.init_hidden(batch_size)
        prediction_traces = []

        # observation
        for frame in range(0, self.config.input_frame):
            _, hidden = self.motionEncoder(input_traces[:, :, frame], hidden)

        for frame in range(0, self.config.target_frame):

            location = self.locationEncoder(target)
            motion, hidden = self.motionEncoder(target, hidden)
            # crowd = self.crowdInteraction(location, motion)
            displacement = self.displaceDecoder(torch.bmm(location, motion))

            displacement = torch.cat((displacement, target[:, :, 2:]), dim=2)
            target = target + displacement
            prediction_traces.append(target)

        prediction_traces = torch.stack(prediction_traces, 2)

        MSE_loss = ((target_traces - prediction_traces) ** 2).sum() / self.config.pedestrian_num
        # MSE_loss = ((target_traces - prediction_traces) ** 2).sum(3).sqrt().maen()
        return MSE_loss, prediction_traces

    def train(self, input_, target_):
        self.motionEncoder.zero_grad()
        self.locationEncoder.zero_grad()
        self.crowdInteraction.zero_grad()
        self.displaceDecoder.zero_grad()

        loss, traces = self.main_step(input_, target_)
        print(loss)
        loss.backward()

        self.optim_ME.step()
        self.optim_LE.step()
        self.optim_DD.step()

        return loss

    def test(self, input_, target_):
        return self.main_step(input_, target_)

    def main(self):
        # train loop
        for epoch in range(0, self.config.n_epochs):
            for input_traces, target_traces in self.dataloader:
                input_traces = target_traces.float()
                target_traces = target_traces.float()
                input_traces.requires_grad = True
                target_traces.requires_grad = True
                self.train(input_traces, target_traces)


if __name__ == '__main__':
    cidnn = CIDNN_Training()
    cidnn.main()
# TODO: main function
