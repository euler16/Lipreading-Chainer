# coding: utf-8
import os
import time
import random
import logging
import argparse
import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F

from chainer import serializers
from chainer.dataset import concat_examples
from chainer.backends import cuda
from chainer import Function, training, utils, Variable, Link, Chain, initializers, optimizers

from utils import *
from model import *
from dataset import *
from lr_scheduler import *
from cvtransforms import *


SEED = 1
np.random.seed(SEED)

GPU = 1
device = cuda.get_device(GPU)
device.use()

def data_loader(args):
    dsets = {x: MyDataset(x, args.dataset) for x in ['train', 'val', 'test']}
    dset_loaders = {x: chainer.iterators.MultithreadIterator(dsets[x], batch_size=args.batch_size, repeat=False, shuffle=True, n_threads=args.workers) for x in ['train', 'val', 'test']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
    print('\nStatistics: train: {}, val: {}, test: {}'.format(dset_sizes['train'], dset_sizes['val'], dset_sizes['test']))
    return dset_loaders, dset_sizes


def reload_model(model, logger, path=""):
    if not bool(path):
        logger.info('train from scratch')
        return model
    else:
        serializers.load_npz(path, model)
        logger.info('*** model has been successfully loaded! ***')
        return model


def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def train_test(model, dset_loaders, epoch, phase, optimizer, args, logger, use_gpu, save_path):
    if phase == 'val' or phase == 'test':
        # model.eval()
        chainer.global_config.train = False
    if phase == 'train':
        chainer.global_config.train = True
        logger.info('-' * 10)
        logger.info('Epoch {}/{}'.format(epoch, args.epochs - 1))
        # logger.info('Current Learning rate: {}'.format(showLR(optimizer)))

    running_loss, running_all, accuracy = 0., 0., 0.
    model.to_gpu()
    print('checking the model', chainer.backends.cuda.get_device_from_array(model.array))
    for batch_idx, sample in enumerate(dset_loaders[phase]):
        inputs, targets = concat_examples(sample)
        if phase == 'train':
            batch_img = RandomCrop(inputs, (88, 88))
            batch_img = ColorNormalize(batch_img)
            batch_img = HorizontalFlip(batch_img)
        elif phase == 'val' or phase == 'test':
            batch_img = CenterCrop(inputs, (88, 88))
            batch_img = ColorNormalize(batch_img)
        else:
            raise Exception('the dataset doesn\'t exist')

        batch_img = np.reshape(batch_img, (batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], batch_img.shape[3], 1))
        inputs = Variable(batch_img, requires_grad=False)
        inputs.to_gpu()
        #inputs = inputs.float().permute(0, 4, 1, 2, 3)
        inputs = F.transpose(inputs, axes=(0,4,1,2,3))
        targets = Variable(targets)
        targets.to_gpu()
        print('inputs', chainer.backends.cuda.get_device_from_array(inputs.array))
        print('target', chainer.backends.cuda.get_device_from_array(targets.array))
        if phase == 'train':
            outputs = model(inputs)
            if args.every_frame:
                outputs = F.mean(outputs, 1)
            # preds = F.argmax(F.softmax(outputs,axis=1), 1)
            # loss = criterion(outputs, targets)
            loss = F.softmax_cross_entropy(outputs, targets)
            model.cleargrads()
            loss.backward()

            optimizer.update()

        if phase == 'val' or phase == 'test':
            with chainer.no_backprop_mode():
                outputs = model(inputs)
                if args.every_frame:
                    outputs = F.mean(outputs, 1)
                preds = F.argmax(F.softmax(outputs,axis=1), 1)
                loss = F.softmax_cross_entropy(outputs, targets)

        # stastics
        running_loss += loss.array[0] * inputs.shape[0]
        running_all += len(inputs)
        accuracy = F.accuracy(outputs, targets)
        accuracy.to_cpu()
        if batch_idx == 0:
            since = time.time()
        elif batch_idx % args.interval == 0 or (batch_idx == len(dset_loaders[phase])-1):
            print('Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss: {:.4f}\tAcc:{:.4f}\tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                running_all,
                len(dset_loaders[phase].dataset),
                100. * batch_idx / (len(dset_loaders[phase])-1),
                running_loss / running_all,
                accuracy,
                time.time()-since,
                (time.time()-since)*(len(dset_loaders[phase])-1) / batch_idx - (time.time()-since)))
    print(logger.info('{} Epoch:\t{:2}\tLoss: {:.4f}\tAcc:{:.4f}'.format(
        phase,
        epoch,
        running_loss / len(dset_loaders[phase].dataset),
        accuracy)+'\n'))
    if phase == 'train':
        #torch.save(model.state_dict(), save_path+'/'+args.mode+'_'+str(epoch+1)+'.pt')
        serializers.save_npz(os.path.join(save_path,args.mode+'_'+str(epoch)+'.pt'), model)
        return model


def test_adam(args, use_gpu):
    if args.every_frame and args.mode != 'temporalConv':
        save_path = './' + args.mode + '_every_frame'
    elif not args.every_frame and args.mode != 'temporalConv':
        save_path = './' + args.mode + '_last_frame'
    elif args.mode == 'temporalConv':
        save_path = './' + args.mode
    else:
        raise Exception('No model is found!')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    # logging info
    filename = save_path+'/'+args.mode+'_'+str(args.lr)+'.txt'
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename, mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    model = lipreading(mode=args.mode, inputDim=256, hiddenDim=512, nClasses=args.nClasses, frameLen=29, every_frame=args.every_frame)
    # reload model
    model.to_gpu()
    model = reload_model(model, logger, args.path)
    model.to_gpu()
    
    if args.mode == 'temporalConv' or args.mode == 'finetuneGRU':
        # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.)
        optimizer = optimizers.Adam(alpha=args.lr)
        optimizer.setup(model)
    elif args.mode == 'backendGRU':
        for param in model.params():
            param.requires_grad = False
        for param in model.gru.params():
            param.requires_grad = True
        optimizer = optimizers.Adam(alpha=args.lr)
        optimizer.setup(model)
    else:
        raise Exception('No model is found!')

    dset_loaders, dset_sizes = data_loader(args)
    scheduler = AdjustLR(optimizer, [args.lr], sleep_epochs=5, half=5, verbose=1)
    if args.test:
        train_test(model, dset_loaders, 0, 'val', optimizer, args, logger, use_gpu, save_path)
        train_test(model, dset_loaders, 0, 'test', optimizer, args, logger, use_gpu, save_path)
        return
    for epoch in range(args.epochs):
        scheduler.step(epoch)
        model = train_test(model, dset_loaders, epoch, 'train', optimizer, args, logger, use_gpu, save_path)
        train_test(model, dset_loaders, epoch, 'val', optimizer, args, logger, use_gpu, save_path)


def main():
    # Settings
    args = parse_args()

    use_gpu = True
    test_adam(args, use_gpu)


if __name__ == '__main__':
    main()
