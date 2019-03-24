import numpy as np
import glob
import time
import cv2

from chainer import dataset


def load_file(filename):
    cap = np.load(filename)
    arrays = np.stack([cv2.cvtColor(cap[_], cv2.COLOR_RGB2GRAY)
                      for _ in range(29)], axis=0)
    arrays = arrays / 255.
    return arrays


class MyDataset(dataset.DatasetMixin):
    def __init__(self, folds, path):
        self.folds = folds
        self.path = path
        with open('../label_sorted.txt') as f:
            self.data_dir = f.read().splitlines()
        self.data_files = glob.glob(self.path+'*/'+self.folds+'/*.npy')
        self.list = {}
        for i, x in enumerate(self.data_files):
            target = x.split('/')[-3]
            for j, elem in enumerate(self.data_dir):
                if elem == target:
                    self.list[i] = [x]
                    self.list[i].append(j)

        print('Load {} part'.format(self.folds))
    
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        inputs = load_file(self.list[idx][0])
        labels = self.list[idx][1]
        return inputs, labels
    
    def get_example(self, idx):
        inputs = load_file(self.list[idx][0])
        labels = self.list[idx][1]
        return inputs, labels