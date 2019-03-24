# coding: utf-8
import math
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from chainer import Function, training, utils, Variable, Link, Chain, initializers

def conv3x3(in_planes, out_planes, stride=1):
    return L.Convolution2D(in_planes, out_planes, ksize=3, stride=stride,pad=1,nobias=True,initialW=initializers.LeCunNormal)


class BasicBlock(chainer.Chain):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        #self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1 = L.Convolution2D(inplanes, planes, ksize=3, stride=stride,pad=1,nobias=True,initialW=initializers.LeCunNormal)
        self.bn1 = L.BatchNormalization(planes)
        #self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        self.conv2 = L.Convolution2D(planes, planes, ksize=3, stride=1,pad=1,nobias=True,initialW=initializers.LeCunNormal)
        self.bn2 = L.BatchNormalization(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        print(chainer.backends.cuda.get_device_from_array(x.array))
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(chainer.Chain):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.avgpool = nn.AvgPool2d(2)
        self.fc = L.Linear(512 * block.expansion, num_classes)
        self.bnfc = L.BatchNormalization(num_classes)
        

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = chainer.Sequential(
                L.Convolution2D(self.inplanes, planes * block.expansion,
                          ksize=1, stride=stride, nobias=True, initialW=initializers.LeCunNormal),
                L.BatchNormalization(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return chainer.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.average_pooling_2d(x, 2)
        x = F.reshape(x, (x.shape[0],-1))
        #x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bnfc(x)
        return x


class GRU(chainer.Chain):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, every_frame=True):
        #  chainer.links.GRU(in_size, out_size, init=None, inner_init=None, bias_init=0)
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.every_frame = every_frame
        # stateless GRU NOTE it was using batch_first
        #self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.gru = L.NStepBiGRU(num_layers, input_size, hidden_size,dropout=0)
        self.fc = L.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        '''
            in chainer : h-> (2*layers, batch , hidden)
                         x-> a list with x[i]-> (seq_len, input_size)
                         y-> a list with y[i]-> (seq_len, hidden)

            in pytorch : h-> same
                         x-> (batch, seq_len, input_size)
                         y-> (batch, seq_len, hidden)
        '''
        # need to check if i have to use cupy or numpy
        #h0 = chainer.Variable(np.zeros(self.num_layers*2, x.size(0), self.hidden_size)) 
        # CHAINER gru takes input as a list of chainer variables, i.e. x should be a list and x[i] should be
        # (seq_len,input size)
        x = [r[:] for r in x]
        print('before gru', chainer.backends.cuda.get_device_from_array(x.array))
        _, out = self.gru(None, x) # if none then chainer automatically does the correct thing
        if self.every_frame:
            out = self.fc(out)  # predictions based on every time step
        else:
            out = self.fc(out[:, -1, :])  # predictions based on last time-step
        out = F.stack(out)
        return out


class Lipreading(chainer.Chain):
    def __init__(self, mode, inputDim=256, hiddenDim=512, nClasses=500, frameLen=29, every_frame=True):
        super(Lipreading, self).__init__()
        self.mode = mode
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.frameLen = frameLen
        self.every_frame = every_frame
        self.nLayers = 2
        # frontend3D
        self.frontend3D = chainer.Sequential(
                L.Convolution3D(1, 64, ksize=(5, 7, 7), stride=(1, 2, 2), pad=(2, 3, 3), nobias=True),
                L.BatchNormalization(64),
                F.relu, # chainer doesnt support maxpool3d
                L.Convolution3D(64, 64, ksize=(1, 3, 3), stride=(1, 2, 2), pad=(0, 1, 1), nobias=True)
            )
        # resnet
        self.resnet34 = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=self.inputDim)
        # backend_conv
        self.backend_conv1 = chainer.Sequential(
                L.Convolution1D(self.inputDim, 2*self.inputDim, 5, 2, 0, nobias=True),
                L.BatchNormalization(2*self.inputDim),
                F.relu,
                L.Convolution1D(2*self.inputDim, 2*self.inputDim, ksize=2,stride=2,nobias=True),
                #nn.MaxPool1d(2, 2),
                L.Convolution1D(2*self.inputDim, 4*self.inputDim, ksize=5, stride=2, pad=0, nobias=True),
                L.BatchNormalization(4*self.inputDim),
                F.relu
            )
        self.backend_conv2 = chainer.Sequential(
                L.Linear(4*self.inputDim, self.inputDim),
                L.BatchNormalization(self.inputDim),
                F.relu,
                L.Linear(self.inputDim, self.nClasses)
            )
        # backend_gru
        self.gru = GRU(self.inputDim, self.hiddenDim, self.nLayers, self.nClasses, self.every_frame)
        # initialize
        #self._initialize_weights()

    def forward(self, x):
        print(type(x))
        print('before frontend3d', chainer.backends.cuda.get_device_from_array(x.array))
        x = self.frontend3D(x)
        print(x.shape)
        x = F.transpose(x,axes=(0,2,1,3,4))
        # x = x.transpose(1, 2)
        # x = x.contiguous()
        #x = x.view(-1, 64, x.size(3), x.size(4))
        x = F.reshape(x, (-1, 64, x.shape[3], x.shape[4]))
        x = self.resnet34(x)
        if self.mode == 'temporalConv':
            x = F.reshape(x, (-1, self.frameLen, self.inputDim))
            # x = x.transpose(1, 2)
            x = F.transpose(x, axes=(0,2,1))
            x = self.backend_conv1(x)
            # x = torch.mean(x, 2)
            x = F.mean(x,2)
            x = self.backend_conv2(x)
        elif self.mode == 'backendGRU' or self.mode == 'finetuneGRU':
            # x = x.view(-1, self.frameLen, self.inputDim)
            x = F.reshape(x,(-1, self.frameLen, self.inputDim))
            x = self.gru(x)
        else:
            raise Exception('No model is selected')
        return x



def lipreading(mode, inputDim=256, hiddenDim=512, nClasses=500, frameLen=29, every_frame=True):
    model = Lipreading(mode, inputDim=inputDim, hiddenDim=hiddenDim, nClasses=nClasses, frameLen=frameLen, every_frame=every_frame)
    return model
