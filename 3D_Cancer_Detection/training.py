import argparse
import datetime
import os
import sys

import numpy as np

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from dsets import LunaDataset
from util.logconf import logging
from model import LunaModel

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX=0
METRICS_PRED_NDX=1
METRICS_LOSS_NDX=2
METRICS_SIZE = 3

class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

      #  self.use_cuda = torch.cuda.is_available()
       # self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.use_mps1 = torch.backends.mps.is_available()
        self.use_mps2 = torch.backends.mps.is_built()
        self.device = torch.device("mps" if self.use_mps1 and self.use_mps2 else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()
        

    def initModel(self):
        model = LunaModel()
        if self.use_mps1 and self.use_mps2:
            log.info("Using Apple's M1 chip as a GPU device.")
            #    model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)


    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

    

if __name__ == '__main__':
    LunaTrainingApp().main()
