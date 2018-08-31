import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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

    def forward(self, x):
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
        outputs = self.get_hidden_output(data)
        outputs = self.get_spatial_affinity(outputs)
        return outputs

    def get_hidden_output(self, data):
        outputs = torch.Tensor([])
        for idx in range(0, self.pedestrian_num):
            print(data[idx])
            output = F.relu(self.fc1(data[idx]))
            output = F.relu(self.fc2(output))
            output = self.fc3(output)
            outputs = torch.cat([outputs, output.unsqueeze(0)], dim=0)  # unsqueeze to add a dimension
        return outputs

    def get_spatial_affinity(self, data):
        #output = torch.Tensor([])
        output = torch.zeros(self.pedestrian_num, self.pedestrian_num)
        for i in range(0, self.pedestrian_num):
            row_data = torch.Tensor([])
            for j in range(0, i+1):
                row_data = torch.cat([row_data, torch.dot(data[i], data[j]).unsqueeze(0)], dim=0)
            #output = torch.cat([output, row_data.unsqueeze(0)], dim=0)
            output[i, 0:i+1] = row_data
        '''
        outputs will be like this :
        <h1, h1>
        <h2, h1>, <h2, h2>
        <h3, h1>, <h3, h2>, <h3, h3>
        ......
        '''
        return self.softmax(output)

    def softmax(self, data):
        output = torch.zeros(self.pedestrian_num, self.pedestrian_num)
        for i in range(0, self.pedestrian_num):
            count = 0
            for j in range(0, self.pedestrian_num):
                count += math.exp(data[max(i, j)][min(i, j)].item())
            for j in range(0, self.pedestrian_num):
                output[i][j] = math.exp(data[max(i, j)][min(i, j)].item()) / count
        return output


class CrowdInteraction(nn.Module):
    def __init__(self, pedestrian_num, hidden_size):
        super(CrowdInteraction, self).__init__()
        self.pedestrian_num = pedestrian_num
        self.hidden_size = hidden_size

    def forward(self, location_data, motion_data):
        output = torch.zeros(self.pedestrian_num, self.hidden_size)
        for i in range(0, self.pedestrian_num):
            for j in range(0, self.pedestrian_num):
                output[i] += torch.mul(motion_data[j], location_data[i][j].item())
        return output


class DisplacementPrediction(nn.Module):
    def __init__(self, pedestrian_num, input_size, output_size):
        super(DisplacementPrediction, self).__init__()
        self.pedestrian_num = pedestrian_num
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, data):
        output = torch.zeros(self.pedestrian_num, self.output_size)
        for idx in range(0, self.pedestrian_num):
            output[idx] = self.fc1(data[idx])
        return output

def test():
    input = torch.Tensor([[[1, 2], [2, 3], [3, 4]],
                          [[2, 4], [4, 8], [8, 16]],
                          [[1, 3], [3, 5], [5, 7]]])
    lstm_fake = torch.Tensor([[1, 2, 3, 4, 5, 6],
                              [10, 20, 30, 40, 50, 60],
                              [100, 200, 300, 400, 500, 600]])
    location_net = LocationEncoder(3, 2, 128)
    crowd_net = CrowdInteraction(3, 6)
    prediction_net = DisplacementPrediction(3, 6, 2)
    for i in range(0, 1):
        input_data = input[:,i:i+1].view(3, -1)
        #print(location_net.get_hidden_output(input_data).size())
        out = location_net.forward(input_data)
        print(out)
        out = crowd_net.forward(out, lstm_fake)
        print(out)
        out = prediction_net.forward(out)
        print(out)


#test()

# TODO : LSTM Module