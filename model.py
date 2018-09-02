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

        #print('data: \n', data)
        for idx in range(0, self.pedestrian_num):
            input_data = data[:, idx].unsqueeze(0)
            #print(idx, ': ')
            #print(data[idx])
            output_data, next_hidden = self.lstm(input_data, (hidden[idx][0], hidden[idx][1]))

            next_hidden_list.append(next_hidden)
            #print('lstm : ', output_data.size())
            output_list.append(output_data.squeeze(0))

        output = torch.stack(output_list, 1)
        return output, next_hidden_list

    def init_hidden(self, batch_size): # batch_size is frame length ( maybe )
        hidden = [[torch.zeros(self.layer_num, batch_size, self.hidden_size)
                   for _ in range(0, 2)]
                  for _ in range(0, self.pedestrian_num)]
        return hidden


class LocationEncoder(nn.Module):
    def __init__(self, pedestrian_num, input_size, hidden_size, batch_size):
        super(LocationEncoder, self).__init__()
        self.pedestrian_num = pedestrian_num
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, self.hidden_size)
        self.soft = nn.Softmax(dim=1)
        pass

    def forward(self, data):
        outputs = self.get_hidden_output(data)
        output = self.Attention(outputs, outputs)
        return output

    def get_hidden_output(self, data):
        output_list = []
        for idx in range(0, self.pedestrian_num):
            #print(data[idx])
            output = F.relu(self.fc1(data[:, idx]))
            output = F.relu(self.fc2(output))
            output = self.fc3(output)

            output_list.append(output)

        outputs = torch.stack(output_list, 1)
        return outputs

    def Attention(self, input_data, target_data):
        Attn = torch.bmm(target_data, input_data.transpose(1, 2))

        inner_Attn = Attn

        Attn_size = Attn.size()
        Attn = Attn - Attn.max(2)[0].unsqueeze(2).expand(Attn_size)
        exp_Attn = torch.exp(Attn)

        # batch-based softmax
        Attn = exp_Attn / exp_Attn.sum(2).unsqueeze(2).expand(Attn_size)
        return Attn

    def get_spatial_affinity(self, data):
        #output = torch.Tensor([])
        #print(data.size())
        output = torch.zeros(self.batch_size, self.pedestrian_num, self.pedestrian_num)

        for batch in range(0, self.batch_size):
            for i in range(0, self.pedestrian_num):
                row_data = torch.Tensor([])
                for j in range(0, i+1):
                    row_data = torch.cat([row_data, torch.dot(data[batch][i], data[batch][j]).unsqueeze(0)], dim=0)
                #output = torch.cat([output, row_data.unsqueeze(0)], dim=0)
                output[batch, i, 0:i+1] = row_data
            for i in range(0, self.pedestrian_num):
                col_data = output[batch, :, i].view(1, -1)
                output[batch, i, :] = col_data
            #print(output[batch])
            output[batch] = self.soft(output[batch])
        '''
        outputs will be like this :
        <h1, h1>, <h2, h1>, <h3, h1> ...
        <h2, h1>, <h2, h2>, <h3, h2> ...
        <h3, h1>, <h3, h2>, <h3, h3> ...
        ......
        '''
        return output


    def softmax(self, data):
        output = torch.zeros(self.batch_size, self.pedestrian_num, self.pedestrian_num)
        exp_data = torch.exp(data)
        for batch in range(0, self.batch_size):
            for i in range(0, self.pedestrian_num):
                count = 0
                for j in range(0, self.pedestrian_num):
                    count += exp_data[batch][max(i, j)][min(i, j)].item()
                for j in range(0, self.pedestrian_num):
                    output[batch][i][j] = exp_data[batch][max(i, j)][min(i, j)].item() / count
        return output


class CrowdInteraction(nn.Module):
    def __init__(self, pedestrian_num, hidden_size):
        super(CrowdInteraction, self).__init__()
        self.pedestrian_num = pedestrian_num
        self.hidden_size = hidden_size

    def forward(self, location_data, motion_data):
        '''
        output = torch.zeros(self.batch_size, self.pedestrian_num, self.hidden_size)
        for batch in range(0, self.batch_size):
            for i in range(0, self.pedestrian_num):
                for j in range(0, self.pedestrian_num):
                    output[batch][i] += torch.mul(motion_data[batch][j], location_data[batch][i][j].item())
        '''
        return torch.bmm(location_data, motion_data)


class DisplacementPrediction(nn.Module):
    def __init__(self, pedestrian_num, input_size, output_size):
        super(DisplacementPrediction, self).__init__()
        self.pedestrian_num = pedestrian_num
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, data):
        output_list = []
        for idx in range(0, self.pedestrian_num):
            output_list.append(self.fc1(data[:, idx]))

        output = torch.stack(output_list, 1)
        #print(output.size())
        return output


def test():
    batch = 1000
    frame = 5
    person = 20
    input = torch.Tensor(torch.randn(batch, person, frame, 5))
    motion_out = torch.Tensor(torch.randn(batch, person, 2))  # should be batch_size, ped_num, displacement
    location_net = LocationEncoder(person, 5, 128, batch)
    crowd_net = CrowdInteraction(person, 128, batch)
    prediction_net = DisplacementPrediction(person, 128, 2)
    lstm = MotionEncoder(person, 2, 5, 128)

    hidden = lstm.init_hidden(batch)

    #observation frame
    for f in range(0, frame):
        input_data = input[:, :, f, :]
        #print(input_data.size())
        _, hidden = lstm(input_data, hidden)

    for f in range(0, frame):
        print(f)
        input_data = input[:, :, f, :]
        location_out = location_net(input_data)

        lstm_out, hidden = lstm(input_data, hidden)
        crowd_out = crowd_net(location_out, lstm_out)

        prediction = prediction_net(crowd_out)
        #print(prediction)

    print('Done.')
    '''
    for i in range(0, 1):
        input_data = input[:, i:i+1].view(1000, -1)
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
    '''


#test()

# TODO : LSTM Module
