from __future__ import print_function
import argparse
import torch

import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from dataloaders.VQADataset import VQADataset
import time
from models.combined_model import returnmodel

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--parallel', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--num-workers',  default=8,
                    help='enables CUDA training')

parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
args.parallel = args.parallel and args.cuda

torch.manual_seed(args.seed)
if args.cuda:
    print("Cuda is available")
    torch.cuda.manual_seed(args.seed)

opt = {'dir': '/home-3/pmahaja2@jhu.edu/scratch/vqa2018_data', 'images': 'Images', 'nans': 2000, 'sampleans': True,
       'maxlength': 26, 'minwcount': 0, 'nlp': 'mcb', 'pad': 'left'}
if args.cuda:
    kwargs = {'num_workers': int(args.num_workers), 'pin_memory': False}
else:
    kwargs = {'num_workers': int(args.num_workers), 'pin_memory': True}

train_dataset = VQADataset("train", opt)
train_loader = train_dataset.data_loader(shuffle=True, batch_size=args.batch_size, **kwargs)

test_dataset = VQADataset("val", opt)
test_loader = test_dataset.data_loader(shuffle=False, batch_size=args.test_batch_size, **kwargs)


model = returnmodel(args.cuda, args.parallel)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    begin = time.time()
    model.train()
    for batch_idx, data in enumerate(train_loader):
        if args.cuda:
            question, image, target = data['question'].cuda(), data['image'].float().cuda(), data['answer'].cuda()
        else:
            question, image, target = data['question'], data['image'].float(), data['answer']

        question, image, target = Variable(question), Variable(image), Variable(target)
        optimizer.zero_grad()
        output = model(question, image)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime : {:.2f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0], time.time() - begin))
            begin = time.time()
def test(loss):
    begin = time.time()

    model.eval()
    test_loss = 0
    correct = 0
    for data in test_loader:
       
        if args.cuda:
            question, image, target = data['question'].cuda(), data['image'].float().cuda(), data['answer'].cuda()
        else:
            question, image, target = data['question'], data['image'].float(), data['answer']

        question, image, target = Variable(question), Variable(image), Variable(target)
        output = model(question, image)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), time.time() - begin))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
