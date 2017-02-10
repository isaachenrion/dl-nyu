from __future__ import print_function
import pickle
import numpy as np
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
from models import Vanilla
from tf_logger import TFLogger

WORKING_DIR = ""
DATA_DIR = os.path.join(WORKING_DIR, 'data')
LABELED = os.path.join(DATA_DIR, 'train_labeled.p')
UNLABELED = os.path.join(DATA_DIR, 'train_unlabeled_.p')
VALIDATION = os.path.join(DATA_DIR, 'validation.p')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

class Experiment:
    def __init__(self, args):
        self.args = args
        self.do_admin()
        self.load_data()
        self.build_model()
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum)
        self.global_step = 0
        self.write_settings()

    def do_admin(self):
        # make the log directory
        dt = datetime.datetime.now()
        unique_id = '{}-{}___{:02d}-{:02d}-{:02d}'.format(dt.strftime("%b"), dt.day, dt.hour, dt.minute, dt.second)
        self.experiment_dir = os.path.join(WORKING_DIR, 'experiments', unique_id)
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        self.experiment_log = os.path.join(self.experiment_dir, "log.txt")

    def superprint(self, text):
        print(text)
        with open(self.experiment_log, 'a') as f:
            f.write('{}\n'.format(text))

    def load_data(self):
        # load data
        print('loading data!')
        trainset_labeled = pickle.load(open(LABELED, "rb"))
        trainset_unlabeled = pickle.load(open(UNLABELED, "rb"))
        validation = pickle.load(open(VALIDATION, "rb"))

        self.train_labeled_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=args.batch_size, shuffle=True, **kwargs)
        self.train_unlabeled_loader = torch.utils.data.DataLoader(trainset_unlabeled, batch_size=args.batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(validation, batch_size=args.batch_size, shuffle=True, **kwargs)

    def build_model(self):
        # build model
        self.model = Vanilla()
        if args.cuda:
            self.model.cuda()

    def write_settings(self):
        # write settings and get loggers
        settings_log = os.path.join(self.experiment_dir, "settings.txt")
        with open(settings_log, 'w') as f:
            f.write('{}\n'.format(self.model.children))
            for arg, value in vars(self.args).items():
                f.write("{}: {}\n".format(arg, value))
        self.train_tf_logger = TFLogger(os.path.join(self.experiment_dir, 'train'))
        self.eval_tf_logger = TFLogger(os.path.join(self.experiment_dir, 'eval'))

    def train(self, epoch):
        self.model.train()
        for batch_idx, ((data, target), (data_u, target_u)) in enumerate(zip(self.train_labeled_loader, self.train_unlabeled_loader)):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            data_u = Variable(data_u)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            output_u = self.model(data_u)
            output_u_ = output_u.detach()
            _, pred = torch.max(output_u_, 1)
            loss_u = F.nll_loss(output_u, pred.view(-1))
            #import ipdb; ipdb.set_trace()
            #loss_u = torch.mean(loss_u)
            #import ipdb; ipdb.set_trace()
            loss_u.backward()
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                self.superprint('Train Epoch: {} [{:4}/{} ({:2.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_labeled_loader.dataset),
                    100. * batch_idx / len(self.train_labeled_loader), loss.data[0]))
                pred = output.data.max(1)[1] # get the index of the max log-probability
                correct = pred.eq(target.data).cpu().sum() / self.args.batch_size
                self.train_tf_logger.log(step=self.global_step, loss=loss.data[0], accuracy=correct)
            self.global_step += 1

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(self.test_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            test_loss += F.nll_loss(output, target).data[0]
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

        test_loss = test_loss
        test_loss /= len(self.test_loader) # loss function already averages over batch size
        accuracy = correct/len(self.test_loader.dataset)
        self.superprint('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * accuracy))
        self.eval_tf_logger.log(step=self.global_step, loss=test_loss, accuracy=accuracy)
        self.global_step += 1

    def run(self):
        for epoch in range(1, self.args.epochs + 1):
            self.train(epoch)
            self.test(epoch)

experiment = Experiment(args)
experiment.run()
