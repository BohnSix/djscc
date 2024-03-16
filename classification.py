from __future__ import print_function
import os
import argparse
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from tiny_imagenet import TinyImageNet
from resnet import *
from wideresnet import *
import logging
# from bpg_ldpc import LDPCTransmitter, BPGEncoder, BPGDecoder
from torch_impl import JSCC, Calculate_filters

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2'

"""

nohup python classification.py --lr 0.1 --wd 2e-4 --epochs 100 --data CIFAR-10 --arch ResNet18 --aug --seed 1 > ./logs/output.log 2>&1 & 
tail -f ./logs/output.log

"""

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--data', default='Tiny-ImageNet', choices=['Tiny-ImageNet', 'CIFAR-10', 'CIFAR-100', 'ImageNet'], help='data')
parser.add_argument('--arch', default='ResNet18', choices=['ResNet18', 'WideResNet34'], help='model')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4, #5e-4
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--aug', action='store_true', default=False,
                    help='data augumentation')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./checkpoint/baseline/',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=5, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()

# settings
model_dir = args.model_dir + time.strftime('%Y-%m-%d-%H-%M-%S-', time.localtime()) + args.arch + '-Standard-' + args.data + '-aug-' + str(args.aug)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=os.path.join(model_dir, 'train.log'))
logger.info(args)


use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}
# BPG-LDPC
# SNR=10, bw=0.500000, k=3072, n=4608, m=16, PSNR=38.58, SSIM=0.97
# bw = 1 / 2
# esno_db = 10
# k, n, m = 3072, 4608, 16
# b = 256
# max_bytes = b * 32 * 32 * 3 * bw * math.log2(m) * k / n / 8
# ldpctransmitter = LDPCTransmitter(k, n, m, esno_db, 'AWGN')
# bpgencoder = BPGEncoder()
# bpgdecoder = BPGDecoder()

# JSCC
SNR = 20
CHANNEL_TYPE = "awgn"
COMPRESSION_RATIO = 0.04
EPOCHS = 1000
NUM_WORKERS = 4
LEARNING_RATE = 0.001
CHANNEL_SNR_TRAIN = 10
TRAIN_IMAGE_NUM = 50000
TEST_IMAGE_NUM = 10000
TRAIN_BS = 64
TEST_BS = 4096
K = Calculate_filters(COMPRESSION_RATIO)
net = JSCC(K, snr_db=SNR).cuda()
net.load_state_dict(torch.load("/media/bohnsix/djscc/checkpoints/jscc_model_17"))




class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean).to("cuda"))
        self.register_buffer('std', torch.Tensor(std).to("cuda"))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std






# setup data loader
if args.data == 'ImageNet':
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    trainset = datasets.ImageFolder('/data/ZNY/data/ImageNet/train', transform_train)
    testset = datasets.ImageFolder("/data/ZNY/data/ImageNet/val", transform_test)
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    class_number = 1000
    size = 224
elif args.data == 'Tiny-ImageNet':
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    trainset = TinyImageNet('../data/tiny-imagenet-200', train=True, transform=transform_train)
    testset = TinyImageNet('../data/tiny-imagenet-200', train=False, transform=transform_test)
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    class_number = 200
    size = 64
elif args.data == 'CIFAR-10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    class_number = 10
    size = 32
elif args.data == 'CIFAR-100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])
    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
    norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    class_number = 100
    size = 32


train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)






def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # BPG-LDPC
        # image = imBatchtoImage(data)
        # src_bits = bpgencoder.encode(image.numpy(), max_bytes)
        # rcv_bits = ldpctransmitter.send(src_bits)
        # decoded_image = bpgdecoder.decode(rcv_bits.numpy(), image.shape)
        # data = decoded_image

        #JSCC
        decoded_img, chn_out = net(data.cuda())
        data = decoded_img

        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(data), target)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            # BPG-LDPC
            # image = imBatchtoImage(data)
            # src_bits = bpgencoder.encode(image.numpy(), max_bytes)
            # rcv_bits = ldpctransmitter.send(src_bits)
            # decoded_image = bpgdecoder.decode(rcv_bits.numpy(), image.shape)
            # data = decoded_image

            # JSCC
            decoded_img, chn_out = net(data.cuda())
            data = decoded_img

            data, target = data.cuda(), target.cuda()
            output = model(data)
            train_loss += F.cross_entropy(output, target, reduction='mean').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    logger.info('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # BPG-LDPC
            # image = imBatchtoImage(data)
            # src_bits = bpgencoder.encode(image.numpy(), max_bytes)
            # rcv_bits = ldpctransmitter.send(src_bits)
            # decoded_image = bpgdecoder.decode(rcv_bits.numpy(), image.shape)
            # data = decoded_image

            # JSCC
            decoded_img, chn_out = net(data.cuda())
            data = decoded_img

            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='mean').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    logger.info('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy






def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def imBatchtoImage(batch_images):
    '''
    turns b, 32, 32, 3 images into single sqrt(b) * 32, sqrt(b) * 32, 3 image.
    '''
    batch, h, w, c = batch_images.shape
    b = int(batch ** 0.5)

    divisor = b
    while batch % divisor != 0:
        divisor -= 1

    image = batch_images.reshape(-1, batch//divisor, h, w, c)
    image = image.transpose(0, 2, 1, 3, 4)
    image = image.reshape(-1, batch//divisor*w, c)

    return torch.round(image)


def main():
    if args.arch == 'ResNet18':
        model = ResNet18(size, class_number).cuda()
    elif args.arch == 'WideResNet34':
        model = WideResNet(image_size=size, depth=34, num_classes=class_number, widen_factor=10).cuda()

    if args.aug:
        model = nn.Sequential(norm_layer, model).cuda()
    model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        start_time = time.time()
        train(args, model, train_loader, optimizer, epoch)
        print('using time:', time.time() - start_time)
        logger.info('using time: {}'.format(time.time() - start_time))

        _, _ = eval_train(model, train_loader)
        _, _ = eval_test(model, test_loader)

        logger.info('================================================================')

        torch.save(optimizer.state_dict(),
                   os.path.join(model_dir, 'opt-last.tar'))
        torch.save(model.module.state_dict(),
                   os.path.join(model_dir, 'model-last.pth'))

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.module.state_dict(),
                       os.path.join(model_dir, 'model-{}.pth'.format(epoch)))

if __name__ == '__main__':
    main()

