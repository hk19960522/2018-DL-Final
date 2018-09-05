import argparse
import os

import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from DataLoader import get_data_loader
from draw import *
from model import *


class CIDNN_Training:
    def __init__(self, config):
        # prepare data
        self.config = config
        self.sample_input, self.sample_target, self.sample_frame = \
            torch.load(config.coord_filename)
        data = TensorDataset(self.sample_input, self.sample_target)
        self.dataloader = DataLoader(data, batch_size=self.config.batch_size)
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
        self.optim_ME = optim.Adam(self.motionEncoder.parameters(),
                                   lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.optim_LE = optim.Adam(self.locationEncoder.parameters(),
                                   lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.optim_DD = optim.Adam(self.displaceDecoder.parameters(),
                                   lr=self.config.lr, weight_decay=self.config.weight_decay)

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
            if target.size(2) > displacement.size(2):
                displacement = torch.cat((displacement,
                                          torch.zeros(displacement.size(0),
                                                      displacement.size(1),
                                                      target.size(2) - displacement.size(2)).cuda()),
                                         dim=2)
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

    def test(self):
        assert self.config.test
        self.load()
        inp_list, tar_list, pre_list = [], [], []
        for inp, tar in self.dataloader:
            # cat = torch.zeros(inp.size(0), inp.size(1), inp.size(2), 3).cuda()
            inp = inp.float().cuda()
            tar = tar.float().cuda()
            _, prediction = self.main_step(inp, tar)
            pre_list.append(prediction.data)
        predict = torch.cat(pre_list).cpu()

        show_CUHK(self.config.video_filename,
                  self.sample_input, self.sample_target, predict, self.sample_frame)

    def main(self):
        assert not self.config.test
        start_epoch = 0
        if self.config.resume:
            arg = self.load()
            if arg is not None and 'epoch' in args:
                start_epoch = arg['epoch']
        # train loop
        min_loss = 1.0
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
                if min_loss > avg:
                    self.save(epoch=epoch)
                    min_loss = avg
            except KeyboardInterrupt:
                print('KeyboardInterrupt')
                # self.save(dir_path='./weights/interrupt/', epoch=epoch)
                exit(0)

    def save(self, dir_path='./weights/', **save_dict):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        save_dict['motionEncoder'] = self.motionEncoder.state_dict()
        save_dict['locationEncoder'] = self.locationEncoder.state_dict()
        save_dict['displaceDecoder'] = self.displaceDecoder.state_dict()
        save_dict['train_loss_series'] = self.train_loss_series
        torch.save(save_dict, dir_path + self.config.name + '.pkl')
        print('save model to %s'.format(dir_path))

    def load(self, dir_path='./weights/'):
        dir_path += self.config.name + '.pkl'
        if not os.path.exists(dir_path):
            print('File %s not found.'.format(dir_path))
            exit(0)
        save_dict = torch.load(dir_path)
        self.motionEncoder.load_state_dict(save_dict['motionEncoder'])
        self.locationEncoder.load_state_dict(save_dict['locationEncoder'])
        self.displaceDecoder.load_state_dict(save_dict['displaceDecoder'])
        self.train_loss_series = save_dict['train_loss_series']
        print('model loaded from %s' % dir_path)
        return args


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser(description='CIDNN training')
    parser.add_argument('--name', default='model')
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--n_epochs', '-e', default=10000)
    parser.add_argument('--test', '-t', action='store_true')

    parser.add_argument('--pedestrian_num', default=20)
    parser.add_argument('--hidden_size', default=256)
    parser.add_argument('--sample_rate', default=20, help='frame sampling stride')
    parser.add_argument('--coord_filename', default='./dataset/AnnotationDataset.pkl')
    parser.add_argument('--video_filename', default='./dataset/Frame/')
    parser.add_argument('--input_frame', default=5)
    parser.add_argument('--input_size', default=2)
    parser.add_argument('--n_layers', default=1)
    parser.add_argument('--target_frame', default=5)
    parser.add_argument('--target_size', default=2)
    parser.add_argument('--lr', default=2e-3)
    parser.add_argument('--weight_decay', default=5e-3)
    parser.add_argument('--batch_size', default=1000)

    args = parser.parse_args()
    cidnn = CIDNN_Training(args)
    if args.test:
        cidnn.test()
    else:
        cidnn.main()

