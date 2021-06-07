import argparse
import torch
import torch.optim as optim
import os
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import yaml
import sklearn.metrics as sk_metrics

import models
from utils import cuda
from dataset import returnCIFAR10data, returnOPGdata

# Vizualization
from tensorboardX import SummaryWriter



# datagather class for tensorboard
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



class SimCLR_Solver(object):

    def __init__(self, args):
        self.model = args.model
        #self.pretrain = args.pretrain # we may use this to test vanilla resnet

        self.use_cuda = args.cuda and torch.cuda.is_available()

        self.dataset = args.dataset # 'cifar10' by default
        self.config_file = args.config_file # 'config.yaml' by default
        self.train_mode = args.train_mode

        self.global_iter = 0
        self.epoch_counter = 0
        self.best_valid_loss = np.inf

        # configs
        with open(self.config_file) as file:
            self.configs = yaml.load(file, Loader=yaml.FullLoader)

        self.batch_size = self.configs['batch_size']
        self.test_portion = self.configs['test_portion']
        self.num_workers = self.configs['num_workers']
        self.input_shape = eval(self.configs['input_shape'])

        self.gather_step = self.configs['gather_step']
        self.eval_step = self.configs['eval_step']

        if not self.train_mode:
            self.test_portion = 0

        dset_args = argparse.Namespace()
        dset_args.dset_dir = args.dset_dir
        dset_args.batch_size = self.batch_size
        dset_args.test_portion = self.test_portion
        dset_args.num_workers = self.num_workers
        dset_args.train_mode = self.train_mode
        dset_args.input_shape = self.input_shape

        dset_args.csv_file = args.csv_file
        dset_args.labeled = args.labeled
        dset_args.window_size = args.window_size
        ## args to use in classifier
        dset_args.test_portion_clf = 1
        dset_args.use_test_to_train = False

        self.dset_args = dset_args

        # dataset
        if self.dataset == 'cifar10':
            self.train_loader, self.valid_loader = returnCIFAR10data(dset_args)
        elif self.dataset == 'OPG':
            self.train_loader, self.valid_loader = returnOPGdata(dset_args)

        # model
        net = eval('models.' + self.model)
        self.net = cuda(net(**self.configs['model']), self.use_cuda)

        # optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), eval(self.configs['learning_rate']),
                                          weight_decay=eval(self.configs['weight_decay']))

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader), eta_min=0,
                                                               last_epoch=-1)

        # Load/prepare checkpoints
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.exp_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        # Save Output
        self.output_dir = os.path.join(args.output_dir, args.exp_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # Tensorboard
        self.gather = DataGather()
        self.writer = SummaryWriter(logdir="./logdir/" + args.exp_name)
        self.net.set_writer(self.writer)

    # training process
    def train(self):
        self.net.train()

        pbar = tqdm(total=self.configs['epochs'])
        pbar.update(self.epoch_counter)

        while self.epoch_counter < self.configs['epochs']:
            self.epoch_counter += 1
            pbar.update(1)

            pbar2 = tqdm(total=len(self.train_loader))

            for (xis, xjs), _ in self.train_loader: # _ are the corresponding classes
                self.global_iter += 1
                pbar2.update(1)

                xis = Variable(cuda(xis, self.use_cuda)).float()
                xjs = Variable(cuda(xjs, self.use_cuda)).float()

                ris, zis = self.net(xis)
                rjs, zjs = self.net(xjs)

                zis = F.normalize(zis, dim=1)
                zjs = F.normalize(zjs, dim=1)

                batch_size = xis.shape[0]  # redefine batch_size, so last batch is compatible with loss function
                                           # You can simply set drop_last = True in the dataloader object

                loss = models.SimCLR_criterion(zis, zjs, use_cuda=self.use_cuda, batch_size=batch_size,
                                               temperature=self.configs['loss']['temperature']) # computes loss function considering similarity

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                torch.cuda.empty_cache()


            # Record training losses
            if self.epoch_counter % self.gather_step == 0:

                self.writer.add_scalar("1/Loss - NTXentLoss [Training]",
                                               loss,
                                               self.epoch_counter)

            # Evaluation and Save Checkpoints
            if self.epoch_counter % self.eval_step == 0:
                valid_loss = self.evaluate()
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    self.save_checkpoint('best.pth', self.epoch_counter, self.best_valid_loss)
                    pbar.write('Saved checkpoint (best.pth) (epoch:{})'.format(self.epoch_counter))
                self.save_checkpoint('last.pth', self.epoch_counter, valid_loss)           # uncomment if you want to save on every eval_step
                pbar.write('Saved checkpoint (last.pth) (epoch:{})'.format(self.epoch_counter))

                self.writer.add_scalar("1/Loss - NTXentLoss [Validation]",
                                       valid_loss,
                                       self.epoch_counter)


        pbar.write("[Training Finished]")
        pbar.close()

    def evaluate(self):

        with torch.no_grad():
            self.net.eval()
            counter = 0

            valid_loss = 0.0
            for (xis, xjs), _ in self.valid_loader:
                xis = Variable(cuda(xis, self.use_cuda)).float()
                xjs = Variable(cuda(xjs, self.use_cuda)).float()

                ris, zis = self.net(xis)
                rjs, zjs = self.net(xjs)

                zis = F.normalize(zis, dim=1)
                zjs = F.normalize(zjs, dim=1)

                batch_size = zis.shape[0] # redefine batch_size, so last batch is compatible with loss function

                loss = models.SimCLR_criterion(zis, zjs, batch_size=batch_size,
                                               temperature=self.configs['loss']['temperature'])
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        self.net.train()
        return valid_loss

    def save_checkpoint(self, filename, epoch, valid_loss, silent=True):
        model_states = {'net': self.net.state_dict(), }
        optim_states = {'optim': self.optimizer.state_dict(), }
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
            checkpoint = torch.load(file_path)
            if filename == 'best.pth':
                self.best_valid_loss = checkpoint['valid_loss']
            else:
                bst_pth = os.path.join(self.ckpt_dir, 'best.pth')
                if os.path.isfile(bst_pth):
                    best_ckpt = torch.load(bst_pth)
                    self.best_valid_loss = best_ckpt['valid_loss']
            self.epoch_counter = checkpoint['epoch']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optimizer.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (epoch {})'".format(file_path, self.epoch_counter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))


#################################################################################
#################################################################################
#################################################################################


class classification_Solver(object):

    def __init__(self, args):
        self.model = args.model

        self.use_cuda = args.cuda and torch.cuda.is_available()

        self.dataset = args.dataset  # 'cifar10' by default
        if self.dataset == 'cifar10':
            self.cifar = True
        else:
            self.cifar = False
        self.train_mode = args.train_mode
        self.weights = args.weights # path to the weights folder

        self.batch_size = args.batch_size
        self.test_portion = args.test_portion
        self.num_workers = args.num_workers
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay

        self.input_shape = args.input_shape
        self.in_channels = args.in_channels
        self.n_classes = args.n_classes
        self.pretrained = args.pretrained

        self.gather_step = args.gather_step
        self.eval_step = args.eval_step

        self.global_iter = 0
        self.epoch_counter = 0
        self.best_valid_loss = np.inf

        if not self.train_mode:
            self.test_portion = 0

        dset_args = argparse.Namespace()
        dset_args.dset_dir = args.dset_dir
        dset_args.batch_size = self.batch_size
        dset_args.test_portion = self.test_portion
        dset_args.num_workers = self.num_workers
        dset_args.train_mode = self.train_mode
        dset_args.input_shape = self.input_shape

        dset_args.csv_file = args.csv_file
        dset_args.labeled = args.labeled
        dset_args.window_size = args.window_size
        dset_args.considered_class = args.considered_class
        dset_args.test_portion_clf = args.test_portion_clf
        dset_args.use_test_to_train = args.use_test_to_train

        self.dset_args = dset_args

        # dataset
        if self.dataset == 'cifar10':
            self.train_loader, self.valid_loader = returnCIFAR10data(dset_args)
        elif self.dataset == 'OPG':
            self.train_loader, self.valid_loader = returnOPGdata(dset_args)

        # model
        net = eval('models.' + self.model)
        self.net = cuda(net(in_channels=self.in_channels, cifar=self.cifar,
                            n_classes=self.n_classes, pretrained=self.pretrained), self.use_cuda)

        # optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.learning_rate,
                                            weight_decay=self.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader),
                                                                    eta_min=0,
                                                                    last_epoch=-1)

        # Load/prepare checkpoints
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.exp_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.weights is not None:
            self.load_checkpoint(self.weights)
            ## freezing weights ##
            for param in self.net.features.parameters():
                param.requires_grad = False
            ######################
            # reset optimizer after loading pretrained weights
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                                self.learning_rate,
                                                weight_decay=self.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                        T_max=len(self.train_loader),
                                                                        eta_min=0,
                                                                        last_epoch=-1)
            self.epoch_counter = 0
            self.best_valid_loss = np.inf
        elif self.ckpt_name is not None:
            self.load_checkpoint(os.path.join(self.ckpt_dir, self.ckpt_name))

        # Save Output
        self.output_dir = os.path.join(args.output_dir, args.exp_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # Tensorboard
        self.gather = DataGather()
        self.writer = SummaryWriter(logdir="./classification_logdir/" + args.exp_name)
        self.net.set_writer(self.writer)

    # training process
    def train(self):
        self.net.train()

        pbar = tqdm(total=self.epochs)
        pbar.update(self.epoch_counter)

        while self.epoch_counter < self.epochs:
            self.epoch_counter += 1
            pbar.update(1)

            #pbar2 = tqdm(total=len(self.train_loader))
            #pbar2.update(0)

            for (xis, xjs), labels in self.train_loader: # _ are the corresponding classes
                self.global_iter += 1
                #pbar2.update(1)

                x = Variable(cuda(xis, self.use_cuda)).float()
                labels = Variable(cuda(labels, self.use_cuda))
                out = self.net(x)

                loss = models.classification_criterion(out, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            # Record training losses
            if self.epoch_counter % self.gather_step == 0:

                self.writer.add_scalar("1/Loss - Cross Entropy [Training]",
                                               loss,
                                               self.epoch_counter)

            # Evaluation and Save Checkpoints
            if self.epoch_counter % self.eval_step == 0:
                print('--------------------------------------------------')
                print('EVALUTAION EPOCH {ep}'.format(ep=self.epoch_counter))

                valid_loss, metrics_test = self.evaluate()
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    self.save_checkpoint('best.pth', self.epoch_counter, self.best_valid_loss)
                    pbar.write('Saved checkpoint(epoch:{})'.format(self.epoch_counter))
                self.save_checkpoint('last.pth', self.epoch_counter, valid_loss)           # uncomment if you want to save on every eval_step
                pbar.write('Saved checkpoint(epoch:{})'.format(self.epoch_counter))

                self.writer.add_scalar("2/Loss - Cross Entropy [Validation]",
                                       valid_loss,
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

                print('---> Current cross entropy loss: {valid_loss}'.format(valid_loss=valid_loss))
                print('---> Current accuracy: {acc}'.format(acc=metrics_test['acc']))
                print('---> Current f1-score: {f1}'.format(f1=metrics_test['f1_scr']))
                print('--------------------------------------------------')

            torch.cuda.empty_cache()

        pbar.write("[Training Finished]")
        pbar.close()

    def evaluate(self):

        with torch.no_grad():
            self.net.eval()
            counter = 0

            out_test_labels = np.empty([0,0])
            target_test_labels = np.empty([0, 0])

            valid_loss = 0.0
            for (xis, xjs), labels in self.valid_loader:
                x = Variable(cuda(xis, self.use_cuda)).float()
                labels = Variable(cuda(labels, self.use_cuda))
                out = self.net(x)

                loss = models.classification_criterion(out, labels)
                valid_loss += loss.item()
                counter += 1

                _, out_test_label = torch.max(out.data, 1)
                out_test_label = out_test_label.cpu().numpy()
                out_test_labels = np.append(out_test_labels, out_test_label)

                target_test_label = labels.cpu().numpy()
                target_test_labels = np.append(target_test_labels, target_test_label)

            valid_loss /= counter

            # Sklearn metrics
            metrics_test = self.report_metrics_binary(target_test_labels, out_test_labels)

        self.net.train()
        return valid_loss, metrics_test

    def report_metrics_binary(self, target, out_label):
        metrics = {}
        metrics['acc'] = sk_metrics.accuracy_score(target, out_label)
        metrics['bal_acc'] = sk_metrics.balanced_accuracy_score(target, out_label)
        metrics['f1_scr'] = sk_metrics.f1_score(target, out_label, average='micro')
        metrics['recall'] = sk_metrics.recall_score(target, out_label, average='micro')
        metrics['precision'] = sk_metrics.precision_score(target, out_label, average='micro')

        return metrics

    def save_checkpoint(self, filename, epoch, valid_loss, silent=True):
        model_states = {'net': self.net.state_dict(), }
        optim_states = {'optim': self.optimizer.state_dict(), }
        states = {'valid_loss': valid_loss,
                  'epoch': epoch,
                  'model_states': model_states,
                  'optim_states': optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (epoch {})".format(file_path, epoch))

    def load_checkpoint(self, file_path):
        if os.path.isfile(file_path):

            if self.use_cuda:
                checkpoint = torch.load(file_path)
            else:
                checkpoint = torch.load(file_path, map_location=torch.device('cpu'))

            self.best_valid_loss = checkpoint['valid_loss']
            self.epoch_counter = checkpoint['epoch']

            try:
                self.net.load_state_dict(checkpoint['model_states']['net'])
            except:
                weights = checkpoint['model_states']['net']
                for i in range(3): # removing last 3 layers used to train SimCLR
                    weights.popitem()
                self.net.load_state_dict(weights, strict=False)

            self.optimizer.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (epoch {})'".format(file_path, self.epoch_counter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
