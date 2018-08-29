import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        pass

    def forward(self, x):
        pass


class MotionEncoder(nn.Module):
    def __init__(self):
        super(MotionEncoder, self).__init__()
        pass

    def forward(self, *input):
        pass


class LocationEncoder(nn.Module):
    def __init__(self, pedestrian_num, input_size, hidden_size):
        super(LocationEncoder, self).__init__()
        self.pedestrian_num = pedestrian_num
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, self.hidden_size)
        pass

    def forward(self, data):
        outputs = []
        for idx in range(0, self.pedestrian_num):
            output = nn.ReLU(self.fc1(data[idx]))
            output = nn.ReLU(self.fc2(output))
            output = self.fc3(output)
            outputs = torch.cat([outputs, output.unsqueeze(0)], dim=0)  # unsqueeze to add a dimension
        return outputs


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        pass

    def forward(self, x):
        pass


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        pass

    def forward(self, x):
        pass
# TODO: all
