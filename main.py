'''
Author : 蔡宗翰
Date : 2021/3/21(日)
DSAI HW1 - Electricity Forecasting
Describtion : 
    因政府公開資料有編碼差異，因此在parameter中有多增加一個encoding的部分。
    Code 修改自 https://wizardforcel.gitbooks.io/learn-dl-with-pytorch-liaoxingyu/content/5.3.html
    若直接執行，將會有training 階段， epoch 為1000 ，若無GPU慎用亦或自行更改epoch。
'''
#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


def readFile(filepath, encoding='big5'):
    data_csv = pd.read_csv(filepath,  encoding=encoding, parse_dates=['日期'])
    data_csv = data_csv[['日期', '備轉容量(MW)']]
    last_date = data_csv.tail(1)['日期']

    data_csv = data_csv.set_index('日期')

    dataset = data_csv.values.astype('float32')

    max_value = np.max(dataset)
    min_value = np.min(dataset)
    scalar = max_value - min_value
    dataset = list(map(lambda x: x / scalar, dataset))
    data_X, data_Y = create_dataset(dataset)
    return data_X, data_Y, scalar, last_date


def splitData(data_X, data_Y):
    train_size = int(len(data_X) * 0.7)
    test_size = len(data_X) - train_size
    train_X = data_X[:train_size]
    train_Y = data_Y[:train_size]
    test_X = data_X[train_size:]
    test_Y = data_Y[train_size:]

    train_X = train_X.reshape(-1, 1, 1)
    train_Y = train_Y.reshape(-1, 1, 1)
    test_X = test_X.reshape(-1, 1, 1)
    test_Y = test_Y.reshape(-1, 1, 1)

    train_x = torch.from_numpy(train_X)
    train_y = torch.from_numpy(train_Y)
    test_x = torch.from_numpy(test_X)
    test_y = torch.from_numpy(test_Y)
    return (train_x, train_y), (test_x, test_y)


class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=3):
        super(lstm_reg, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(hidden_size, output_size)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.rnn(x)
        s, b, h = x.shape
        x = x.view(s*b, h)
        x = self.reg(x)
        x = x.view(s, b, -1)
        # x = self.sigmoid(x)
        return x


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))


def train(net, criterion, optimizer, train_data, test_data, epochs=1000):
    min_loss = 1.0
    train_loss = None
    test_loss = None
    loss_record = []
    train_x, train_y = train_data
    test_x, test_y = test_data
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=60)
    for e in range(epochs):
        net.train()
        var_x, var_y = Variable(train_x).to(
            device), Variable(train_y).to(device)
        out = net(var_x)
        optimizer.zero_grad()
        loss = criterion(out, var_y)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        loss_record.append(train_loss)
        net.eval()
        with torch.no_grad():
            var_test_x, var_test_y = Variable(test_x).to(
                device), Variable(test_y).to(device)
            test_out = net(var_test_x)
            loss = criterion(test_out, var_test_y)
            test_loss = loss.item()
        scheduler.step(test_loss)
        if test_loss < min_loss:
            torch.save(net, './best_model.pt')
            print('Save model : loss :', test_loss)
            min_loss = test_loss
        if (e + 1) % 100 == 0:
            print('Epoch: {}, Train Loss: {:.5f}, Test Loss: {:.5f}'.format(
                e + 1, train_loss, test_loss))
    plt.plot(loss_record)
    plt.show()

def predict(output_filename,data_X, scalar, last_date):
    model = torch.load('./best_model.pt').to(device)
    model.eval()

    data_X = data_X.reshape(-1, 1, 1)
    data_X = torch.from_numpy(data_X)
    val = Variable(data_X).to(device)
    future = {}
    for i in range(8):
        pred = model(val)
        date = str((last_date + np.timedelta64(i+1, 'D')).values)[2:12]
        future[date] = int(pred[-1].cpu().data.numpy().squeeze()*scalar)

        data_X = np.append(data_X, pred[-1].cpu().data.numpy())
        data_X = data_X.reshape(-1, 1, 1).astype('float32')
        tensor_X = torch.from_numpy(data_X)
        val = Variable(tensor_X).to(device)

    df = pd.DataFrame(future, index=['value'])
    df.to_csv(output_filename)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', default='file.csv',
                        help='input training data file name')
    parser.add_argument('--encoding', default='big5',
                        help='input cvs encoding.')
    parser.add_argument('--output', default='submission.csv',
                        help='output file name.')
    args = parser.parse_args()
    net = lstm_reg(1, 70).to(device)
    criterion = RMSELoss
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    data_X, data_Y, scalar, last_date = readFile(args.training, args.encoding)
    train_data, test_data = splitData(data_X, data_Y)
   
    train(net, criterion, optimizer, train_data, test_data)
    predict(args.output, data_X, scalar, last_date)
