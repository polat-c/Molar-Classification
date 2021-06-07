import torch
import torch.optim as optim
import os
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
# import torch.sigmoid as sigmoid
import numpy as np
import sklearn.metrics as sk_metrics
import yaml
import argparse
import pickle

from utils import cuda, eval_str, progress_bar
import models
import datasets

# Vizualization
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np


class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    images_train=[],
                    images_eval=[], )

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


class Solver(object):
    def __init__(self, args):

        self.config_file = args.config_file
        with open(self.config_file) as file:
            self.configs = yaml.load(file, Loader=yaml.FullLoader)

        ## model arguments
        self.model, self.in_channels, self.cifar, self.size, self.n_classes, self.pretrained = self.configs['model'].values()

        ## dataset arguments
        dset_args = argparse.Namespace()
        dset_name = self.configs['dataset']['name']
        if dset_name == 'cifar10':
            dset_name, dset_args.dset_dir, dset_args.batch_size, dset_args.test_portion, dset_args.num_workers, dset_args.train = self.configs['dataset'].values()
        elif dset_name == 'OPG':
            dset_name, dset_args.dset_dir, dset_args.csv_file, dset_args.considered_class, dset_args.ROI_size ,dset_args.batch_size, dset_args.test_portion, dset_args.num_workers = self.configs['dataset'].values()
        self.batch_size = dset_args.batch_size

        ## optimizer arguments
        self.learning_rate = eval(self.configs['optimizer']['learning_rate'])
        self.weight_decay = eval(self.configs['optimizer']['weight_decay'])

        ## general arguments
        self.epochs = self.configs['general']['epochs']
        self.gather_step = self.configs['general']['gather_step']
        self.eval_step = self.configs['general']['eval_step']
        self.use_cuda = self.configs['general']['cuda'] and torch.cuda.is_available()
        self.exp_name = self.configs['general']['exp_name']
        self.ckpt_name = eval_str(self.configs['general']['ckpt_name'])
        self.weights = eval_str(self.configs['general']['weights']) ## if not none, give the path of the trained feature extractor weights

        self.global_iter = 0
        self.epoch_counter = 0
        self.best_valid_loss = np.inf


        # DEVICE
        self.device = 'cuda' if (torch.cuda.is_available() and self.use_cuda) else 'cpu'


        # DATASET
        if dset_name == 'cifar10':
            self.train_loader, self.valid_loader = datasets.returnCIFAR10data(dset_args)
        elif dset_name == 'OPG':
            self.train_loader, self.valid_loader = datasets.returnOPGdata(dset_args)
            test_dataset = datasets.DentalImagingDataset(root_dir=dset_args.dset_dir, csv_file=dset_args.csv_file[:-21]+'test_annotations.csv',
                                                         transform=None, considered_class = dset_args.considered_class, ROI_size = dset_args.ROI_size)
            self.test_loader = DataLoader(test_dataset, batch_size=dset_args.batch_size, shuffle=False, num_workers=dset_args.num_workers, pin_memory=True, drop_last=False)

        # MODEL
        net = eval('models.' + self.model)
        self.net = cuda(net(self.in_channels, self.cifar, self.size, self.n_classes, self.pretrained), uses_cuda=self.use_cuda)


        # OPTIMIZER
        if not self.pretrained:
            ### CIFAR OPTIM/SCHEDULER ###
            self.optim = optim.SGD(self.net.parameters(), lr=self.learning_rate,
                                  momentum=0.9, weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=200)
            ##############################
            #### OPG OPTIM/SCHEDULER ####
            # self.optim = optim.Adam(self.net.parameters(), lr=self.learning_rate,
            #                         weight_decay=self.weight_decay,
            #                         betas=(0.9, 0.999))
            # self.scheduler = ReduceLROnPlateau(optimizer=self.optim,
            #                                    factor=0.8,
            #                                    patience=2)
            ##############################

        else:
            for param in self.net.features.parameters():
                param.requires_grad = False
            self.optim = optim.Adam(self.net.l1.parameters(), lr=self.learning_rate,
                                    weight_decay=self.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=len(self.train_loader),
                                                                        eta_min=0,
                                                                        last_epoch=-1)

        # WEIGHTS (SIMCLR)
        if self.weights is not None:
            if self.use_cuda:
                m_weights = torch.load(self.weights)
            else:
                m_weights = torch.load(self.weights, map_location=torch.device('cpu'))
            for i in range(4):
                m_weights.popitem()
            self.net.load_state_dict(m_weights, strict=False)
            for param in self.net.features.parameters():
                param.requires_grad = False
            #self.optim = optim.Adam(self.net.l1.parameters(), lr=self.learning_rate,
            #                        weight_decay=self.weight_decay)
            #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=len(self.train_loader),
            #                                                            eta_min=0,
            #                                                            last_epoch=-1)
            self.params_to_optimize = list(net.l1.parameters()) + list(net.l2.parameters())
            self.optim = optim.SGD(self.params_to_optimize, lr=self.learning_rate,
                                   momentum=0.9, weight_decay=5e-4)
            #self.optim = optim.SGD(self.net.parameters(), lr=self.learning_rate,
            #                       momentum=0.9, weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=200)

        # Load/prepare checkpoints
        self.ckpt_dir = os.path.join('./checkpoints', self.exp_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        # Save Output
        self.output_dir = os.path.join('./outputs', self.exp_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # Tensorboard
        self.gather = DataGather()
        self.writer = SummaryWriter(logdir="./logdir/" + self.exp_name)
        self.net.set_writer(self.writer)

    def train(self):
        self.net.train()

        total = self.batch_size * len(self.train_loader)
        print('------------------------------------------------')
        print(f'Training for total of {self.batch_size} x {len(self.train_loader)} = {total} images!')
        print('------------------------------------------------')

        #pbar = tqdm(total=self.epochs)
        #pbar.update(self.epoch_counter)

        criterion = nn.CrossEntropyLoss()

        while self.epoch_counter < self.epochs:
            #pbar.update(1)
            #pbar2 = tqdm(total=len(self.train_loader))
            #pbar2.update(0)
            self.epoch_counter += 1

            ###########################
            train_loss = 0
            correct = 0
            total = 0
            ###########################

            for idx, (x, target) in enumerate(self.train_loader):
                self.global_iter += 1
                #pbar2.update(1)
                if bool(torch.isnan(x).any()): ## check if there is incompatibilities
                    continue

                #x = Variable(cuda(x, self.use_cuda))
                #target = Variable(cuda(target, self.use_cuda))
                x = x.to(self.device)
                target = target.to(self.device)

                #out = self.net(x.float())
                out = self.net(x)

                #loss = models.criterion(out, target)
                loss = criterion(out, target)

                # Grad descent and backward pass
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                ###########################
                train_loss += loss.item()
                _, predicted = out.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                progress_bar(idx, len(self.train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (train_loss / (idx + 1), 100. * correct / total, correct, total))
                ###########################

            # Record training losses
            if self.epoch_counter % self.gather_step == 0:# Record training loss

                self.writer.add_scalar("1/Loss - Cross Entropy [Training]",
                                        loss,
                                        self.epoch_counter)

            # Evaluation
            if self.epoch_counter % self.eval_step == 0:
                #print('--------------------------------------------------')
                #print('EVALUTAION EPOCH {ep}'.format(ep=self.epoch_counter))
                # Accuracy Full Test_set
                metrics_test = self.evaluate_predictor() # print statements in .evaluate_predictor()

                if metrics_test['loss'] < self.best_valid_loss:
                    self.best_valid_loss = metrics_test['loss']
                    #self.save_checkpoint('best.pt')
                    self.save_checkpoint('best', self.epoch_counter, self.best_valid_loss)
                    #pbar.write('Saved best checkpoint(epoch:{})'.format(self.epoch_counter))
                    print('Saved best checkpoint(epoch:{})'.format(self.epoch_counter))

                # self.save_checkpoint('last.pt')
                self.save_checkpoint('last', self.epoch_counter, self.best_valid_loss)
                #pbar.write('Saved last checkpoint(epoch:{})'.format(self.epoch_counter))
                print('Saved last checkpoint(epoch:{})'.format(self.epoch_counter))

                # Record evaluation loss
                self.writer.add_scalar("2/Loss - Cross Entropy [Evaluation]",
                                        metrics_test['loss'],
                                        self.epoch_counter)
                # Record evaluation accuracy
                self.writer.add_scalar("2/Loss - Accuracy [Evaluation]",
                                        metrics_test['acc'],
                                        self.epoch_counter)
                # Record evaluation balanced accuracy
                self.writer.add_scalar("2/Loss - Bal. Accuracy [Evaluation]",
                                        metrics_test['bal_acc'],
                                        self.epoch_counter)
                # Record evaluation F1 Score
                self.writer.add_scalar("2/Loss - F1-Score [Evaluation]",
                                        metrics_test['f1_scr'],
                                        self.epoch_counter)
                # Record evaluation recall
                self.writer.add_scalar("2/Loss - Recall [Evaluation]",
                                        metrics_test['recall'],
                                        self.epoch_counter)
                # Record evaluation precision
                self.writer.add_scalar("2/Loss - Precision [Evaluation]",
                                        metrics_test['precision'],
                                        self.epoch_counter)
                #print('---> Current cross entropy loss: {valid_loss}'.format(valid_loss=metrics_test['loss']))
                #print('---> Current accuracy: {acc}'.format(acc=metrics_test['acc']))
                #print('---> Current f1-score: {f1}'.format(f1=metrics_test['f1_scr']))
                print('---> Current balanced accuracy: {acc}'.format(acc=metrics_test['bal_acc']))
                #print('--------------------------------------------------')

            torch.cuda.empty_cache()

        #pbar.write("[Training Finished]")
        print('[Training Finished]')
        #pbar.close()

    def evaluate_predictor(self):

        with torch.no_grad():
            self.net.eval()
            counter = 0

            out_test_labels = np.empty([0,0])
            target_test_labels = np.empty([0, 0])

            valid_loss = 0.0

            criterion = nn.CrossEntropyLoss()

            ########################
            test_loss = 0
            correct = 0
            total = 0
            ########################

            for i, (x_test, labels_test) in enumerate(self.valid_loader):

                if bool(torch.isnan(x_test).any()): ## check if there is incompatibilities
                    continue

                #x_test = Variable(cuda(x_test, self.use_cuda))
                #labels_test = Variable(cuda(labels_test, self.use_cuda))
                x_test = x_test.to(self.device)
                labels_test = labels_test.to(self.device)

                #out_test = self.net(x_test.float())
                out_test = self.net(x_test)
                if len(out_test.shape) == 1:
                    continue

                #loss_test = models.criterion(out_test, labels_test)
                loss_test = criterion(out_test, labels_test)
                valid_loss += loss_test.item()
                counter += 1

                _, out_test_label = torch.max(out_test.data, 1)
                out_test_label = out_test_label.cpu().numpy()
                out_test_labels = np.append(out_test_labels, out_test_label)

                target_test_label = labels_test.cpu().numpy()
                target_test_labels = np.append(target_test_labels, target_test_label)

                ########################
                test_loss += loss_test.item()
                _, predicted = out_test.max(1)
                total += labels_test.size(0)
                correct += predicted.eq(labels_test).sum().item()

                progress_bar(i, len(self.valid_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (i + 1), 100. * correct / total, correct, total))
                ########################

            valid_loss /= counter

            # Sklearn metrics
            metrics_test = self.report_metrics_binary(target_test_labels, out_test_labels)
            #print(out_test_labels)
            #print(target_test_labels)
            metrics_test['loss'] = valid_loss

        self.net.train()
        return metrics_test

    def report_metrics_binary(self,target,out_label):
        metrics = {}
        metrics['acc'] = sk_metrics.accuracy_score(target, out_label)
        metrics['bal_acc'] = sk_metrics.balanced_accuracy_score(target, out_label)
        metrics['f1_scr'] = sk_metrics.f1_score(target, out_label, average='micro')
        metrics['recall'] = sk_metrics.recall_score(target, out_label,average='micro')
        metrics['precision'] = sk_metrics.precision_score(target, out_label, average='micro')

        return metrics

    # def save_checkpoint(self, filename, silent=True):
    #     model_states = {'net': self.net.state_dict(), }
    #     optim_states = {'optim': self.optim.state_dict(), }
    #     states = {'iter': self.global_iter,
    #               'model_states': model_states,
    #               'optim_states': optim_states}
    #
    #     file_path = os.path.join(self.ckpt_dir, filename)
    #     with open(file_path, mode='wb+') as f:
    #         torch.save(states, f)
    #     if not silent:
    #         print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))
    #
    # def load_checkpoint(self, filename):
    #     file_path = os.path.join(self.ckpt_dir, filename)
    #     if os.path.isfile(file_path):
    #         checkpoint = torch.load(file_path)
    #         self.global_iter = checkpoint['iter']
    #         self.net.load_state_dict(checkpoint['model_states']['net'])
    #         self.optim.load_state_dict(checkpoint['optim_states']['optim'])
    #         print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(file_path))

    def test(self):

        outputs = []

        with torch.no_grad():
            self.net.eval()
            counter = 0

            out_test_labels = np.empty([0,0])
            target_test_labels = np.empty([0, 0])

            valid_loss = 0.0

            criterion = nn.CrossEntropyLoss()

            ########################
            test_loss = 0
            correct = 0
            total = 0
            ########################

            for i, (x_test, labels_test) in enumerate(self.test_loader):

                if bool(torch.isnan(x_test).any()): ## check if there is incompatibilities
                    continue

                #x_test = Variable(cuda(x_test, self.use_cuda))
                #labels_test = Variable(cuda(labels_test, self.use_cuda))
                x_test = x_test.to(self.device)
                labels_test = labels_test.to(self.device)

                #out_test = self.net(x_test.float())
                out_test = self.net(x_test)
                if len(out_test.shape) == 1:
                    continue

                #loss_test = models.criterion(out_test, labels_test)
                loss_test = criterion(out_test, labels_test)
                valid_loss += loss_test.item()
                counter += 1

                _, out_test_label = torch.max(out_test.data, 1)
                out_test_label = out_test_label.cpu().numpy()
                out_test_labels = np.append(out_test_labels, out_test_label)

                target_test_label = labels_test.cpu().numpy()
                target_test_labels = np.append(target_test_labels, target_test_label)

                ########################
                test_loss += loss_test.item()
                _, predicted = out_test.max(1)
                total += labels_test.size(0)
                correct += predicted.eq(labels_test).sum().item()

                progress_bar(i, len(self.valid_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (i + 1), 100. * correct / total, correct, total))
                ########################
                outputs.append(out_test)

            valid_loss /= counter

            # Sklearn metrics
            metrics_test = self.report_metrics_binary(target_test_labels, out_test_labels)
            #print(out_test_labels)
            #print(target_test_labels)
            metrics_test['loss'] = valid_loss

        self.net.train()

        with open('./outputs/'+self.exp_name+'/outputs.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(outputs, f, pickle.HIGHEST_PROTOCOL)

        with open('./outputs/'+self.exp_name+'/metrics.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(metrics_test, f, pickle.HIGHEST_PROTOCOL)

        #return metrics_test

    def save_checkpoint(self, filename, epoch, valid_loss, silent=True):
        model_states = {'net': self.net.state_dict(), }
        optim_states = {'optim': self.optim.state_dict(), }
        states = {'valid_loss': valid_loss,
                  'epoch': epoch,
                  'model_states': model_states,
                  'optim_states': optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (epoch {})".format(file_path, epoch))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):

            if self.use_cuda:
                checkpoint = torch.load(file_path)
            else:
                checkpoint = torch.load(file_path, map_location=torch.device('cpu'))

            self.best_valid_loss = checkpoint['valid_loss']
            self.epoch_counter = checkpoint['epoch']

            self.net.load_state_dict(checkpoint['model_states']['net'])
            # try:
            #     self.net.load_state_dict(checkpoint['model_states']['net'])
            # except:
            #     weights = checkpoint['model_states']['net']
            #     for i in range(3): # removing last 3 layers used to train SimCLR
            #         weights.popitem()
            #     self.net.load_state_dict(weights, strict=False)

            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (epoch {})'".format(file_path, self.epoch_counter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))