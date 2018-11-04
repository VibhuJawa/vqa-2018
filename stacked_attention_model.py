from __future__ import print_function
import argparse
import torch

import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from dataloaders.VQADataset1 import VQADataset1
from dataloaders.featureset import FeaturesDataset

import time
#from models.combined_model import returnmodel
#from models.attention_model import returnmodel
from models.stacked_attention_model import returnmodel

import numpy as np
import os
from utils import utils, logger
import sys

model_details = "stacked_attention_concat_model"

from tensorboardX import SummaryWriter

tensorboard_writer = SummaryWriter('logs/stacked_attention_concat_model'.format(model_details),comment="Stacked Attention Model")

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--logdir', default="logs", type=str, help='log directory')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--parallel', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--num-workers', default=8,
                    help='enables CUDA training')

parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
args.parallel = args.parallel and args.cuda

torch.manual_seed(args.seed)
if args.cuda:
    print("Cuda is available")
    torch.cuda.manual_seed(args.seed)
if args.cuda:
    kwargs = {'num_workers': int(args.num_workers), 'pin_memory': False}
else:
    kwargs = {'num_workers': int(args.num_workers), 'pin_memory': True}

opt = {'dir': '/home-3/pmahaja2@jhu.edu/scratch/vqa2018_data', 'images': 'Images',
       'nans': 2000, 'sampleans': True,
       'maxlength': 26, 'minwcount': 0,
       'nlp': 'mcb', 'pad': 'left'}

################################################
# Create Features Dataset
################################################

time1 = time.time()

img_features = {
    'mscocoa': {
        'train2014': FeaturesDataset('mscocoa', 'train2014', opt),
        'val2014': FeaturesDataset('mscocoa', 'val2014', opt)
    },
    'abstract_v002': {
        'train2015': FeaturesDataset('abstract_v002', 'train2015', opt),
        'val2015': FeaturesDataset('abstract_v002', 'val2015', opt),
        'scene_img_abstract_v002_train2017': FeaturesDataset('abstract_v002', 'scene_img_abstract_v002_train2017', opt),
        'scene_img_abstract_v002_val2017': FeaturesDataset('abstract_v002', 'scene_img_abstract_v002_val2017', opt)
    }
}

print("Time to load image features : ", time.time() - time1)

################################################
# Create Dataset
################################################
time1 = time.time()

train_dataset = VQADataset1("train", img_features, opt)
train_loader = train_dataset.data_loader(shuffle=True, batch_size=args.batch_size, **kwargs)

test_dataset = VQADataset1("val", img_features, opt)
test_loader = test_dataset.data_loader(shuffle=False, batch_size=args.test_batch_size, **kwargs)

print("Time to load Dataset : ", time.time() - time1)

################################################
# Create Model and Optimizer
################################################


model = returnmodel(args.cuda, args.parallel)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=10e-5, momentum=args.momentum)

################################################
# Count model parameters
################################################

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

################################################
# Create log directory
################################################
exp_logger = None
logdir = os.path.join(opt['dir'], args.logdir)

if os.path.isdir(logdir):
    pass
else:
    os.system('mkdir -p ' + os.path.join(opt['dir'], args.logdir))

if exp_logger is None:
    #  Set loggers
    exp_name = os.path.basename(logdir)  # add timestamp
    exp_logger = logger.Experiment(exp_name)
    exp_logger.add_meters('train', logger.make_meters())
    exp_logger.add_meters('test', logger.make_meters())
    exp_logger.add_meters('val', logger.make_meters())
    exp_logger.info['model_params'] = params
    print('Model has {} parameters'.format(exp_logger.info['model_params']))


def train(epoch, logger):
    begin = time.time()
    model.train()
    meters = logger.reset_meters('train')
    start = time.time()

    for batch_idx, data in enumerate(train_loader):
        batch_size = data['question'].size(0)

        # Measures the data loading time
        meters['data_time'].update(time.time() - begin, n=batch_size)

        if args.cuda:
            question, image, target = data['question'].cuda(), data['image'].float().cuda(), data['answer'].cuda(
                async=True)
        else:
            question, image, target = data['question'], data['image'].float(), data['answer']

        question, image, target = Variable(question), Variable(image), Variable(target)

        # Compute output and loss
        output = model(question, image)
        if args.cuda:
            torch.cuda.synchronize()
        loss = F.nll_loss(output, target)

        # Log the loss
        meters['loss'].update(loss.item(), n=batch_size)

        # Measure accuracy
        acc1, acc5 = utils.accuracy(output.data, target.data, topk=(1, 5))
        meters['acc1'].update(acc1[0], n=batch_size)
        meters['acc5'].update(acc5[0], n=batch_size)
        #print(loss)
        tensorboard_writer.add_scalar("train loss",loss.item(), epoch*len(train_loader)+batch_idx)
        tensorboard_writer.add_scalar("train top1 acc",acc1[0], epoch*len(train_loader)+batch_idx)
        tensorboard_writer.add_scalar("train top 5 acc ",acc5[0], epoch*len(train_loader)+batch_idx)


        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5)

        if args.cuda:
            torch.cuda.synchronize()
       # optimizer.step()
        #if args.cuda:
         #   torch.cuda.synchronize()

        meters['batch_time'].update(time.time() - begin, n=batch_size)
        begin = time.time()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print("Time since last print : {}".format(time.time() - start))
            start = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {acc1.val:.3f} ({acc1.avg:.3f})\t'
                  'Acc@5 {acc5.val:.3f} ({acc5.avg:.3f})'.format(
                epoch, batch_idx, len(train_loader),
                batch_time=meters['batch_time'], data_time=meters['data_time'],
                loss=meters['loss'], acc1=meters['acc1'], acc5=meters['acc5']))
        sys.stdout.flush()
    logger.log_meters('train', n=epoch)


def test(logger, epoch):
    model.eval()
    test_loss = 0
    meters = logger.reset_meters('val')
    begin = time.time()
    for batch_idx, data in enumerate(test_loader):
        batch_size = data['answer'].size(0)

        if args.cuda:
            question, image, target = data['question'].cuda(async=True), data['image'].float().cuda(async=True), data[
                'answer'].cuda(async=True)
        else:
            question, image, target = data['question'], data['image'].float(), data['answer']

        question, image, target = Variable(question, volatile=True), Variable(image, volatile=True), Variable(target,
                                                                                                              volatile=True)

        # Compute output and loss
        output = model(question, image)
        loss = F.nll_loss(output, target).data[0]
        test_loss += loss.item()  # sum up batch loss

        meters['loss'].update(loss.item(), n=batch_size)

        acc1, acc5 = utils.accuracy(output.data, target.data, topk=(1, 5))
        meters['acc1'].update(acc1[0], n=batch_size)
        meters['acc5'].update(acc5[0], n=batch_size)

        tensorboard_writer.add_scalar("Test loss", loss.item(), epoch * len(test_loader) + batch_idx)
        tensorboard_writer.add_scalar("Test top1 acc", acc1[0], epoch * len(test_loader) + batch_idx)
        tensorboard_writer.add_scalar("Test top 5 acc ", acc5[0], epoch * len(test_loader) + batch_idx)

        meters['batch_time'].update(time.time() - begin, n=batch_size)

    test_loss /= len(test_loader.dataset)
    print('\n Test set: Average loss: {:.4f}, Acc@1 {acc1.avg:.3f} Acc@5 {acc5.avg:.3f}'
          .format(test_loss, acc1=meters['acc1'], acc5=meters['acc5']))
    logger.log_meters('val', n=epoch)
    
    return meters['acc1'].avg


best_acc = 0
for epoch in range(1, args.epochs + 1):
    train(epoch, exp_logger)
    test_acc = test(exp_logger, epoch)
    is_best = test_acc > best_acc
    if is_best:
        print("Saving model with {} accuracy".format(test_acc))
    best_acc = max(test_acc, best_acc)
    if is_best:
        torch.save(model.state_dict(), os.path.join(opt['dir'], 'best_model_'+str(best_acc) +'.pt'))

    #utils.save_checkpoint({
    #    'epoch': epoch,
    #    'best_acc1': best_acc,
    #    'exp_logger': exp_logger
    #},
    #    model.state_dict(),
    #    optimizer.state_dict(),
    #    logdir,
    #    True,
    #    True,
    #    is_best)
