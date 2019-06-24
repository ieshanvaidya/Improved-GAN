import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import time
import copy
import logging 

class GaussianNoiseLayer(nn.Module):
    '''
    Adds gaussian noise to the input
    
    Adapted from: https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/nn.py
    '''
    def __init__(self, device, sigma = 0.1, deterministic = False, use_last_noise = False):
        super(GaussianNoiseLayer, self).__init__()
        self.sigma = sigma
        self.deterministic = deterministic
        self.use_last_noise = use_last_noise

    def forward(self, input):
        if self.deterministic or self.sigma == 0:
            return input
        else:
            if not self.use_last_noise:
                self.noise = torch.normal(torch.zeros(input.shape), std = self.sigma).to(device)
        return input + self.noise

class LinearWeightNorm(torch.nn.Module):
    '''
    Adapted from: https://github.com/Sleepychord/ImprovedGAN-pytorch/blob/master/functional.py
    '''
    def __init__(self, in_features, out_features, bias=True, weight_scale=None, weight_init_stdv=0.1):
        super(LinearWeightNorm, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.randn(out_features, in_features) * weight_init_stdv)
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        if weight_scale is not None:
            assert type(weight_scale) == int
            self.weight_scale = Parameter(torch.ones(out_features, 1) * weight_scale)
        else:
            self.weight_scale = 1
            
    def forward(self, x):
        W = self.weight * self.weight_scale / torch.sqrt(torch.sum(self.weight ** 2, dim = 1, keepdim = True))
        return F.linear(x, W, self.bias)
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', weight_scale=' + str(self.weight_scale) + ')'

class Generator(nn.Module):
    def __init__(self, device, ngpu, nz, image_size):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(
            nn.Linear(nz, 500),
            nn.BatchNorm1d(500, eps = 1e-6, momentum = 0.5),
            nn.Softplus(),
            nn.Linear(500, 500),
            nn.BatchNorm1d(500, eps = 1e-6, momentum = 0.5),
            nn.Softplus(),
            LinearWeightNorm(500, image_size ** 2, weight_scale = 1),
            nn.Softplus()
        )
        
        self.main.apply(self.init_weights)
        
    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
        
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, device, ngpu, image_size):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        nlabels = 10
        
        self.get_features = nn.Sequential(
            LinearWeightNorm(image_size ** 2, 1000),
            nn.ReLU(),
            GaussianNoiseLayer(device = device, sigma = 0.3),
            LinearWeightNorm(1000, 500),
            nn.ReLU(),
            GaussianNoiseLayer(device = device, sigma = 0.5),
            LinearWeightNorm(500, 250),
            nn.ReLU(),
            GaussianNoiseLayer(device = device, sigma = 0.5),
            LinearWeightNorm(250, 250),
            nn.ReLU(),
            GaussianNoiseLayer(device = device, sigma = 0.5),
            LinearWeightNorm(250, 250),
            nn.ReLU()
        )
        
        self.get_logits = LinearWeightNorm(250, nlabels, weight_scale = 1)
        
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            features = nn.parallel.data_parallel(self.get_features, input, range(self.ngpu))
            logits = nn.parallel.data_parallel(self.get_logits, features, range(self.ngpu))
        else:
            features = self.get_features(input)
            logits = self.get_logits(features)
    
        return features, logits

class ImprovedGAN:
    '''
    Base class for Improved-GAN model
    '''
    def __init__(self, device, args):
        self.device = device
        self.generator = Generator(device, args.ngpu, args.nz, args.image_size).to(self.device)
        self.discriminator = Discriminator(device, args.ngpu, args.image_size).to(self.device)
        
        try:
            os.makedirs(args.savedir)
        except OSError:
            pass
        
        if args.resume:
            self.generator.load_state_dict(torch.load(os.path.join(args.savedir, 'generator.pth')))
            self.discriminator.load_state_dict(torch.load(os.path.join(args.savedir, 'discriminator.pth')))
        
        # Logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        file_handler = logging.FileHandler(os.path.join(args.savedir, 'train.log'))
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Optimizers
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr = args.lr, betas = (args.beta1, 0.999))
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr = args.lr, betas = (args.beta1, 0.999))
        
        # Data
        self.train_dataset = dset.MNIST(root = args.dataroot, download = True,
                                        transform = transforms.Compose([
                                        transforms.ToTensor()
                                        ]))
        self.test_dataset = dset.MNIST(root = args.dataroot, train = False, download = True,
                                       transform = transforms.Compose([
                                       transforms.ToTensor()
                                       ]))
        
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size = args.batch_size, num_workers = args.workers)
        
        self.nlabels = 10
        self.best_accuracy = 0
        self.args = args
        
    def get_semisupervised_dataset(self):
        '''
        Creates supervised-unsupervised split
        '''
        class_count = [0] * self.nlabels

        supervised_data = []
        supervised_labels = []
        unsupervised_data = []
        unsupervised_labels = []
        
        l = len(self.train_dataset)

        perm = list(range(l))
        random.shuffle(perm)

        for i in range(l):
            datum, label = self.train_dataset[perm[i]]
            if class_count[label] < self.args.nexamples:
                supervised_data.append(datum.numpy())
                supervised_labels.append(label)
                class_count[label] += 1
            else:
                unsupervised_data.append(datum.numpy())
                unsupervised_labels.append(label)

        supervised_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(np.array(supervised_data)), torch.LongTensor(np.array(supervised_labels)))
        unsupervised_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(np.array(unsupervised_data)), torch.LongTensor(np.array(unsupervised_labels)))
        return(supervised_dataset, unsupervised_dataset)
       
    def train_generator(self, real_input):
        '''
        Trains generator with feature matching loss
        '''
        self.noise = torch.randn((self.args.batch_size, self.args.nz), device = self.device)
        fake_input = self.generator(self.noise)
        
        fake_features, fake_logits = self.discriminator(fake_input.view((self.args.batch_size, -1)))
        real_features, real_logits = self.discriminator(real_input.view((self.args.batch_size, -1)))

        fake_mean = torch.mean(fake_features, dim = 0)
        real_mean = torch.mean(real_features, dim = 0)

        loss = torch.mean((fake_mean - real_mean) ** 2)

        self.generator.zero_grad()
        self.discriminator.zero_grad()
        loss.backward()
        self.gen_optimizer.step()
        
        return loss.item()
    
    def train_discriminator(self, supervised_input, supervised_labels, unsupervised_input):
        noise = torch.randn((self.args.batch_size, self.args.nz), device = self.device)
        fake_input = self.generator(noise).detach()
        
        _, supervised_logits = self.discriminator(supervised_input.view((self.args.batch_size, -1)))
        _, unsupervised_logits = self.discriminator(unsupervised_input.view((self.args.batch_size, -1)))
        _, fake_logits = self.discriminator(fake_input.view((self.args.batch_size, -1)))

        supervised_prob = torch.logsumexp(supervised_logits, 1)
        unsupervised_prob = torch.logsumexp(unsupervised_logits, 1)
        fake_prob = torch.logsumexp(fake_logits, 1)
        
        supervised_loss = -torch.mean(torch.gather(supervised_logits, 1, supervised_labels.unsqueeze(1))) + torch.mean(supervised_prob)
        
        unsupervised_loss = 0.5 * (-torch.mean(unsupervised_prob) + torch.mean(F.softplus(unsupervised_prob)) +
                                    torch.mean(F.softplus(fake_prob)))

        # Weighted loss (equal weight)
        loss = supervised_loss + 1.0 * unsupervised_loss
        
        accuracy = torch.mean((supervised_logits.max(1)[1] == supervised_labels).float())
        
        self.discriminator.zero_grad()
        loss.backward()
        self.dis_optimizer.step()
        
        return supervised_loss.item(), unsupervised_loss.item(), accuracy.item()
    
    def train(self):
        supervised_dataset, unsupervised_dataset = self.get_semisupervised_dataset()
        
        for epoch in range(self.args.epochs):
            self.generator.train()
            self.discriminator.train()
            
            unsupervised_dataloader = torch.utils.data.DataLoader(unsupervised_dataset, batch_size = self.args.batch_size, shuffle = True, drop_last = True, num_workers = self.args.workers)
            unsupervised_dataiter = iter(torch.utils.data.DataLoader(unsupervised_dataset, batch_size = self.args.batch_size, shuffle = True, drop_last = True, num_workers = self.args.workers))
            supervised_dataiter = iter(torch.utils.data.DataLoader(supervised_dataset, batch_size = self.args.batch_size, shuffle = True, drop_last = True, num_workers = self.args.workers))
            
            supervised_loss, unsupervised_loss, generator_loss = 0, 0, 0
            accuracy = 0
            batch = 0
            
            for unsupervised_input, _ in unsupervised_dataloader:
                batch += 1
                unsupervised_input = unsupervised_input.to(self.device)
                
                unsupervised_iterinput, _ = unsupervised_dataiter.next()
                unsupervised_iterinput = unsupervised_iterinput.to(self.device)
                
                try:
                    supervised_input, supervised_labels = supervised_dataiter.next()
                except StopIteration as e:
                    supervised_dataiter = iter(torch.utils.data.DataLoader(supervised_dataset, batch_size = self.args.batch_size, shuffle = True, drop_last = True, num_workers = self.args.workers))
                    supervised_input, supervised_labels = supervised_dataiter.next()
                
                supervised_input, supervised_labels = supervised_input.to(self.device), supervised_labels.to(self.device)
                    
                sl, ul, acc = self.train_discriminator(supervised_input, supervised_labels, unsupervised_input)
                supervised_loss += sl
                unsupervised_loss += ul
                accuracy += acc
                
                gl = self.train_generator(unsupervised_iterinput)
                if epoch > 1 and gl > 1:
                    gl = self.train_generator(unsupervised_iterinput)
                
                generator_loss += gl
            
            supervised_loss /= batch
            unsupervised_loss /= batch
            generator_loss /= batch
            accuracy /= batch
            
            eval_accuracy = self.evaluate()
            
            if eval_accuracy > self.best_accuracy:
                self.best_accuracy = eval_accuracy
                torch.save(self.generator.state_dict(), os.path.join(self.args.savedir, 'generator.pth'))
                torch.save(self.discriminator.state_dict(), os.path.join(self.args.savedir, 'discriminator.pth'))
                
            self.logger.info(f'Epoch: {epoch + 1}, Supervised Loss: {supervised_loss}, Unsupervised Loss: {unsupervised_loss}, Generator Loss: {generator_loss}, Train Accuracy: {accuracy}, Test Accuracy: {eval_accuracy}')
                    
    def evaluate(self):
        self.generator.eval()
        self.discriminator.eval()
        accuracy = 0
        batch = 0
        
        for test_input, test_labels in self.test_dataloader:
            batch += 1
            test_input = test_input.to(self.device)
            test_labels = test_labels.to(self.device)
            features, logits = self.discriminator(test_input.view((test_input.shape[0], -1)))
            acc = torch.mean((logits.max(1)[1] == test_labels).float())
            accuracy += acc
        
        return accuracy.item() / batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type = str, default = 'data', help = 'path to dataset')
    parser.add_argument('--savedir', type = str, default = 'log', help = 'path to save models, logs')
    parser.add_argument('--workers', type = int, help = 'number of data loading workers', default = 2)
    parser.add_argument('--nexamples', type = int, help = 'number of supervised examples per class', default = 10)
    parser.add_argument('--batch_size', type = int, default = 64, help = 'input batch size')
    parser.add_argument('--image_size', type = int, default = 28, help='the height / width of the input image to network')
    parser.add_argument('--nz', type = int, default = 100, help = 'size of the latent z vector (noise) that is fed to the generator')
    parser.add_argument('--epochs', type = int, default = 10, help = 'number of epochs to train for')
    parser.add_argument('--lr', type = float, default = 0.003, help = 'learning rate, default=0.003')
    parser.add_argument('--beta1', type = float, default = 0.5, help = 'beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action = 'store_true', help = 'enables cuda')
    parser.add_argument('--ngpu', type = int, default = 1, help = 'number of GPUs to use')
    parser.add_argument('--manual_seed', type = int, help = 'manual seed')
    parser.add_argument('--resume', action = 'store_true', help = 'resume from previous checkpoint')

    args = parser.parse_args()

    assert args.batch_size <= args.nexamples * 10, 'not enough examples to create a batch, increase nexamples or decrease batch_size'

    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 100000)

    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    cudnn.benchmark = True

    device = torch.device("cuda:0" if args.cuda else "cpu")

    model = ImprovedGAN(device, args)
    model.train()