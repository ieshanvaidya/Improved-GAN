import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import time
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | mnist')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--nexamples', type=int, help='number of supervised examples per class', default=50)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nfeat', type=int, default=100)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output logs, images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


def get_semisupervised_data(raw_dataset, nexamples, nlabels):
    class_count = [0] * nlabels

    supervised_data = []
    supervised_labels = []
    unsupervised_data = []
    unsupervised_labels = []

    perm = list(range(len(raw_dataset)))
    random.shuffle(perm)

    for i in range(raw_dataset.__len__()):
        datum, label = raw_dataset[perm[i]]
        if class_count[label] < nexamples:
            supervised_data.append(datum.numpy())
            supervised_labels.append(label)
            class_count[label] += 1
        else:
            unsupervised_data.append(datum.numpy())
            unsupervised_labels.append(label)

    supervised_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(np.array(supervised_data)), torch.LongTensor(np.array(supervised_labels)))
    unsupervised_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(np.array(unsupervised_data)), torch.LongTensor(np.array(unsupervised_labels)))
    return(supervised_dataset, unsupervised_dataset)

if opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    test_dataset = dset.CIFAR10(root=opt.dataroot, train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.Resize(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))

    nlabels=10
    nc=3

elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(opt.imageSize),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,)),
                       ]))
    test_dataset = dset.MNIST(root=opt.dataroot, train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.Resize(opt.imageSize),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,)),
                              ]))
    nlabels=10
    nc=1

assert dataset

supervised_dataset, unsupervised_dataset = get_semisupervised_data(dataset, opt.nexamples, nlabels)

supervised_dataloader = torch.utils.data.DataLoader(supervised_dataset, batch_size=opt.batchSize,
                                                    shuffle=True, num_workers=int(opt.workers))
unsupervised_dataiter = iter(torch.utils.data.DataLoader(unsupervised_dataset, batch_size=opt.batchSize,
                                                         shuffle=True, num_workers=int(opt.workers)))
val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nfeat = int(opt.nfeat)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
)

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.get_features = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, nfeat, 4, 1, 0, bias=False),
            Flatten()
        )
        
        #Feature size of 64 | If adjusting, change
        self.get_logits = nn.Sequential(nn.Linear(nfeat, nlabels))

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            features = nn.parallel.data_parallel(self.get_features, input, range(self.ngpu))
            logits = nn.parallel.data_parallel(self.get_logits, features, range(self.ngpu))
        else:
            features = self.get_features(input)
            logits = self.get_logits(features)
    
        #return features.view(-1, 1).squeeze(1), logits.view(-1, 1).squeeze(1)
        return features, logits

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

netD_criterion_unsupervised = nn.BCELoss()
netD_criterion_supervised = nn.CrossEntropyLoss()
netG_criterion = nn.MSELoss()

real_label = 1
fake_label = 0
top_k = 5

# setup optimizer
netD_optimizer = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
netG_optimizer = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

since = time.time()

best_model_wts = copy.deepcopy(netD.state_dict())
best_acc = 0.0
f = open(f"{opt.outf}/log.txt", "w+")

dataloaders = {'train': supervised_dataloader, 'val': val_dataloader}

###################
### CHECK AHEAD ###
###################

for epoch in range(opt.niter):
    f.write('-' * 10 + '\n')
    f.write(f'Epoch {epoch + 1} of {opt.niter}\n')
    f.write('-' * 10 + '\n')
    f.flush()

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            netG.train()  # Set model to training mode
            netD.train()
        else:
            netG.eval()   # Set model to evaluate mode
            netD.eval()

        netG_loss = 0.0
        netD_loss = 0.0
        n_correct_top_1 = 0
        n_correct_top_k = 0
        n_samples = 0

        for supervised_inputs, supervised_labels in dataloaders[phase]:
            supervised_inputs = supervised_inputs.to(device)
            supervised_labels = supervised_labels.to(device)


            n_samples += opt.batchSize
            netD.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):   
                supervised_features, supervised_logits = netD(supervised_inputs)
                softmax_output = torch.nn.functional.log_softmax(supervised_logits, 1)

                if phase == 'train':
                    # Unsupervised data needed only for train
                    try:
                        unsupervised_inputs, _ = unsupervised_dataiter.next()
                    except StopIteration as e:
                        unsupervised_dataiter = get_unsupervised_dataiter()
                        unsupervised_inputs, _ = unsupervised_dataiter.next()
                    
                    unsupervised_inputs = unsupervised_inputs.to(device)
                    #########################################################
                    # (1) netD: Compute loss for unsupervised real data
                    #########################################################
                    # train with real data
                    labels = torch.full((opt.batchSize,), real_label, device=device)
                    real_features, real_logits = netD(unsupervised_inputs)
                    netD_error_unsupervised_real = netD_criterion_unsupervised(torch.sigmoid(torch.logsumexp(real_logits, 1)), labels)
                    netD_error_unsupervised_real.backward()
                    
                    #########################################################
                    # (2) netD: Compute loss for generated fake data
                    #########################################################
                    # train with fake data
                    noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
                    fake_inputs = netG(noise)
                    labels.fill_(fake_label)
                    fake_features, fake_logits = netD(fake_inputs.detach())
                    netD_error_unsupervised_fake = netD_criterion_unsupervised(torch.sigmoid(torch.logsumexp(fake_logits, 1)), labels)
                    netD_error_unsupervised_fake.backward()
                    
                    #########################################################
                    # (3) netD: Compute loss for supervised real data
                    #########################################################
                    netD_error_supervised = netD_criterion_supervised(softmax_output, supervised_labels)
                    netD_error_supervised.backward()
                    netD_error = netD_error_supervised + netD_error_unsupervised_real + netD_error_unsupervised_fake

                    netD_optimizer.step()
                    
                    #########################################################
                    # (4) netG: Compute feature matching loss
                    #########################################################
                    netG.zero_grad()
                    fake_features, _ = netD(fake_inputs)
                    real_features, _ = netD(unsupervised_inputs)
                    netG_error = netG_criterion(fake_features.mean(), real_features.mean())
                    netG_error.backward()

                    netG_optimizer.step()

            # statistics
            netG_loss += netG_error.item()
            netD_loss += netD_error.item()
            
            # Top 1 accuracy
            pred_top_1 = torch.topk(supervised_logits, k=1, dim=1)[1]
            n_correct_top_1 += pred_top_1.eq(supervised_labels.view_as(pred_top_1)).int().sum().item()

            # Top k accuracy
            pred_top_k = torch.topk(supervised_logits, k=top_k, dim=1)[1]
            target_top_k = supervised_labels.view(-1, 1).expand(-1, top_k)
            n_correct_top_k += pred_top_k.eq(target_top_k).int().sum().item()
            
            # Log every 100 batches
            #if n_samples % (opt.batchSize * 10) == 0:
        f.write(f"Phase: {phase}, Generator Loss: {netG_loss / n_samples:.4f}, Discriminator Loss: {netD_loss / n_samples:.4f}, " +
                f"Top 1: {n_correct_top_1 / n_samples:.4f}, Top {top_k}: {n_correct_top_k / n_samples:.4f}\n")
        f.flush()

        epoch_acc = n_correct_top_k / n_samples

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(netD.state_dict())
            torch.save(netG.state_dict(), f'{opt.outf}/netG_best.pth')
            torch.save(netD.state_dict(), f'{opt.outf}/netD_best.pth')

time_elapsed = time.time() - since

f.write('\n' + '='*50 + '\n')
f.write(f'Training complete in {time_elapsed // 60:.0f} minutes and {time_elapsed % 60:.0f} seconds\n')
f.write(f'Best Validation Accuracy: {best_acc:.4f}\n')
f.write('='*50)
f.flush()
f.close()