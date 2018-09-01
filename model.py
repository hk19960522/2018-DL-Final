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
    def __init__(self, pedestrian_num, layer_num, input_size, hidden_size):
        super(MotionEncoder, self).__init__()
        self.pedestrian_num = pedestrian_num
        self.layer_num = layer_num
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.layer_num)
        pass

    def forward(self, data, hidden):
        next_hidden_list = []
        output_list = []

        print('data: \n', data)
        for idx in range(0, self.pedestrian_num):
            input_data = data[idx].unsqueeze(0).unsqueeze(0)
            print(idx, ': ')
            print(data[idx])
            output_data, next_hidden = self.lstm(input_data, (hidden[idx][0], hidden[idx][1]))

            next_hidden_list.append(next_hidden)
            output_list.append(output_data)

        output = torch.stack(output_list, 0)
        return output, next_hidden_list

    def init_hidden(self, batch_size): # batch_size is frame length ( maybe )
        hidden = [[torch.zeros(self.layer_num, batch_size, self.hidden_size)
                   for _ in range(0, 2)]
                  for _ in range(0, self.pedestrian_num)]
        return hidden


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
    input = torch.Tensor(torch.randn(1000, 20, 5, 5))
    motion_out = torch.Tensor(torch.randn(1000, 20, 2))  # should be batch_size, ped_num, displacement
    location_net = LocationEncoder(20, 5, 128)
    crowd_net = CrowdInteraction(20, 128)
    prediction_net = DisplacementPrediction(20, 5, 2)
    for i in range(0, 1):
        input_data = input[:,i:i+1].view(1000, -1)
        #print(location_net.get_hidden_output(input_data).size())
        out = location_net(input_data)
        print(out)
        out = crowd_net(out, motion_out)
        print(out)
        out = prediction_net(out)
        print(out)

    print('lstm test:')
    lstm = MotionEncoder(3, 2, 2, 5)

    hidden = lstm.init_hidden(1)
    #print(hidden[0][0], '\n', hidden[0][1])

    for i in range(0, 2): #frame
        print(input[:, i])
        print(hidden)
        _, hidden = lstm(input[:, i], hidden)


test()

# TODO : LSTM Module