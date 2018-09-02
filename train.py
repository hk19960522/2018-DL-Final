import argparse
import os

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from pprint import pprint
from model import *
from config import Config
from DataLoader import get_data_loader


class CIDNN_Training:
    def __init__(self, config):
        # prepare data
        self.config = config
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
                                                      self.config.target_size)
        self.optim_ME = optim.Adam(self.motionEncoder.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.optim_LE = optim.Adam(self.locationEncoder.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.optim_DD = optim.Adam(self.displaceDecoder.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        self.train_loss_series = []

        self.motionEncoder = self.motionEncoder.cuda()
        self.locationEncoder = self.locationEncoder.cuda()
        self.displaceDecoder = self.displaceDecoder.cuda()

    def main_step(self, input_traces, target_traces):
        # (batch, pedestrain_num, frame, feature)
        batch_size = input_traces.size(0)
        total_size = 1

        for i in target_traces.size():
            total_size *= i
        total_size /= target_traces.size(-1)
        total_size *= 2
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

        # print((target_traces - prediction_traces))
        MSE_loss = ((target_traces[:, :, :, :2] - prediction_traces[:, :, :, :2]) ** 2).sum()
        MSE_loss = (MSE_loss / total_size).sqrt()
        # MSE_loss = ((target_traces - prediction_traces) ** 2).sum(3).sqrt().mean()
        # MSE_loss = nn.MSELoss(target_traces, prediction_traces)
        self.train_loss_series.append(MSE_loss.item())
        return MSE_loss, prediction_traces

    def train(self, input_, target_):
        self.motionEncoder.zero_grad()
        self.locationEncoder.zero_grad()
        self.crowdInteraction.zero_grad()
        self.displaceDecoder.zero_grad()

        loss, traces = self.main_step(input_, target_)
        # print(loss)
        loss.backward()

        self.optim_ME.step()
        self.optim_LE.step()
        self.optim_DD.step()

        return loss.item()

    def test(self, input_, target_):
        ret, _ = self.main_step(input_, target_)
        return ret

    def main(self):
        start_epoch = 0
        if config.resume:
            arg = self.load()
            if arg is not None and 'epoch' in args:
                start_epoch = arg['epoch']
        # train loop
        for epoch in range(start_epoch, start_epoch + self.config.n_epochs):
            try:
                epoch_loss = []
                for input_traces, target_traces in self.dataloader:
                    input_traces = input_traces.float().cuda()
                    target_traces = target_traces.float().cuda()
                    input_traces.requires_grad = True
                    target_traces.requires_grad = True
                    loss = self.train(input_traces, target_traces)
                    epoch_loss.append(loss)

                avg = np.average(epoch_loss)
                print('Epoch %d Average Loss : %f' % (epoch, avg))
                self.train_loss_series.append(avg)
            except KeyboardInterrupt:
                print('KeyboardInterrupt')
                self.save(epoch=epoch)
                exit(0)

    def save(self, dir_path='./weights/', **args):
        print(args)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        args['motionEncoder'] = self.motionEncoder.state_dict()
        args['locationEncoder'] = self.locationEncoder.state_dict()
        args['displaceDecoder'] = self.displaceDecoder.state_dict()
        args['train_loss_series'] = self.train_loss_series
        torch.save(args, dir_path + 'model.pkl')

    def load(self, dir_path='./weights/'):
        dir_path += 'model.pkl'
        if not os.path.exists(dir_path):
            return None
        args = torch.load(dir_path)
        self.motionEncoder.load_state_dict(args['motionEncoder'])
        self.locationEncoder.load_state_dict(args['locationEncoder'])
        self.displaceDecoder.load_state_dict(args['displaceDecoder'])
        self.train_loss_series = args['train_loss_series']
        print('model loaded from %s' % dir_path, args)
        return args


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser(description='CIDNN training')
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--epochs', '-e', default=10000)
    args = parser.parse_args()

    config = Config()
    config.n_epochs = args.epochs
    config.resume = args.resume
    cidnn = CIDNN_Training(config)
    cidnn.main()

