# Improved-GAN

PyTorch implementation of the paper [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf) for MNIST.

Based on openai code along with implementation of Sleepychord in [PyTorch](https://github.com/Sleepychord/ImprovedGAN-pytorch).

## Usage
```
usage: main.py [-h] [--dataroot DATAROOT] [--savedir SAVEDIR] [--workers WORKERS]
               [--nexamples NEXAMPLES] [--batch_size BATCHSIZE] [--image_size IMAGESIZE] [--nz NZ]
               [--epochs EPOCHS] [--lr LR] [--beta1 BETA1]
               [--cuda] [--ngpu NGPU] [--manual_seed MANUALSEED] [--resume]

optional arguments:
  -h, --help                  show this help message and exit
  --dataroot DATAROOT         path to dataset, default=data
  --savedir SAVEDIR           path for saving models and logs, default=log
  --workers WORKERS           number of data loading workers, default=2
  --nexamples NEXAMPLES       number of examples per class to use as supervised data, default=10
  --batch_size BATCHSIZE       input batch size, default=64
  --image_size IMAGESIZE       the height / width of the input image to network, default=28
  --nz NZ                     size of the latent z vector, default=100
  --epochs EPOCHS             number of epochs to train for, default=10
  --lr LR                     learning rate, default=0.003
  --beta1 BETA1               beta1 for adam. default=0.5
  --cuda                      enables cuda
  --ngpu NGPU                 number of GPUs to use
  --manual_seed MANUALSEED    seed for random number generator
  --resume                    resume from last checkpoint, requires generator.pth and discriminator.pth in savedir
```