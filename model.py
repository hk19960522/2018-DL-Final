import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        pass

    def forward(self, x):
        pass


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
