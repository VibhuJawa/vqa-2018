from __future__ import print_function
import argparse
import torch
import sys
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from dataloaders.VQADataset import VQADataset
import time
from models.stacked_attention_model import returnmodel

from tensorboardX import SummaryWriter

import numpy as np
import os
from utils import utils, logger

model_details = "stacked_attention_concat_model"

tensorboard_writer = SummaryWriter('logs/custom_attention_concat_model_2'.format(model_details),comment="Stacked Attention Model")


# Training settings
parser = argparse.ArgumentParser(description='Visual Question Answering')
parser.add_argument('--logdir', default="logs", type=str, help='log directory')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--parallel', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--num-workers',  default=8,
                    help='enables CUDA training')

parser.add_argument('--cuda', action='store_true', default=False,
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


opt = {'dir': 'data/', 'images': 'Images', 'nans': 2000, 'sampleans': True,
       'maxlength': 26, 'minwcount': 0, 'nlp': 'mcb', 'pad': 'left'}

################################################
# Create Dataset
################################################
#TODO change back to train
train_dataset = VQADataset("dummydata_", opt)

train_loader = train_dataset.data_loader(shuffle=True, batch_size=args.batch_size, **kwargs)

#TODO change back to val
test_dataset = VQADataset("dummydata_", opt)
test_loader = test_dataset.data_loader(shuffle=False, batch_size=args.test_batch_size, **kwargs)

################################################
# Create Model and Optimizer
################################################


model = returnmodel(args.cuda, args.parallel)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

################################################
# Count model parameters
################################################

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

################################################
# Create log directory
################################################
exp_logger = None
logdir = os.path.join(opt['dir'],args.logdir)

if os.path.isdir(logdir):
    pass
else:
    os.system('mkdir -p ' + os.path.join(opt['dir'],args.logdir))

if exp_logger is None:
    # Set loggers
    exp_name = os.path.basename(logdir) # add timestamp
    exp_logger = logger.Experiment(exp_name )
    exp_logger.add_meters('train', logger.make_meters())
    exp_logger.add_meters('test', logger.make_meters())
    exp_logger.add_meters('val', logger.make_meters())
    exp_logger.info['model_params'] = params
    print('Model has {} parameters'.format(exp_logger.info['model_params']))



def train(epoch, logger,tensorboard_writer):
    begin = time.time()
    model.train()
    meters = logger.reset_meters('train')
    start = time.time()
    for batch_idx, data in enumerate(train_loader):
        batch_size = data['question'].size(0)

        # Measures the data loading time
        meters['data_time'].update(time.time() - begin, n=batch_size)

        if args.cuda:
            question, image, target = data['question'].cuda(), data['image'].float().cuda(), data['answer'].cuda()
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

        tensorboard_writer.add_scalar("train_loss",loss.item(), epoch*len(train_loader)+batch_idx)
        tensorboard_writer.add_scalar("train_top1_acc",acc1[0], epoch*len(train_loader)+batch_idx)
        tensorboard_writer.add_scalar("train-top_5_acc ",acc5[0], epoch*len(train_loader)+batch_idx)


        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        if args.cuda:
            torch.cuda.synchronize()
        optimizer.step()
        if args.cuda:
            torch.cuda.synchronize()

        meters['batch_time'].update(time.time() - begin, n=batch_size)

        begin = time.time()
        #optimizer.step()
        
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
            question, image, target = data['question'].cuda(), data['image'].float().cuda(), data['answer'].cuda()
        else:
            question, image, target = data['question'], data['image'].float(), data['answer']

        question, image, target = Variable(question, volatile=True), Variable(image, volatile=True), Variable(target, volatile=True)

        # Compute output and loss
        output = model(question, image)
        loss = F.nll_loss(output, target).data[0]
        test_loss += loss  # sum up batch loss

        meters['loss'].update(loss, n=batch_size)

        acc1, acc5 = utils.accuracy(output.data, target.data, topk=(1, 5))
        meters['acc1'].update(acc1[0], n=batch_size)
        meters['acc5'].update(acc5[0], n=batch_size)

        tensorboard_writer.add_scalar("Test_loss",loss.item(), epoch*len(test_loader)+batch_idx)
        tensorboard_writer.add_scalar("Test_top1_acc",acc1[0], epoch*len(test_loader)+batch_idx)
        tensorboard_writer.add_scalar("Test_top5_acc ",acc5[0], epoch*len(test_loader)+batch_idx)


        meters['batch_time'].update(time.time() - begin, n=batch_size)


    test_loss /= len(test_loader.dataset)
    print('\n Test set: Average loss: {:.4f}, Acc@1 {acc1.avg:.3f} Acc@5 {acc5.avg:.3f}'
          .format(test_loss, acc1=meters['acc1'], acc5=meters['acc5']))
    logger.log_meters('val', n=epoch)

    return meters['acc1'].avg

best_acc = 0
for epoch in range(1, args.epochs + 1):
    print(epoch)

    train(epoch, exp_logger,tensorboard_writer)
    test_acc = test(exp_logger, epoch)
    is_best = test_acc  > best_acc
    if is_best:
         print("Saving model with {} accuracy".format(test_acc))
    if is_best:
        torch.save(model.state_dict(), os.path.join(opt['dir'], 'custom_attaention_best_model_' + str(best_acc) + '.pt'))

    if args.cuda:
         torch.cuda.synchronize()
    #
    # best_acc = max(test_acc, best_acc)
    # print("Best accuracy so far :", best_acc)
    # if is_best:
    #     torch.save(model.state_dict(), os.path.join(opt['dir'], 'best_model_'+str(best_acc) +'.pt'))
    # #utils.save_checkpoint({
    # #    'epoch': epoch,
    # #    'best_acc1': best_acc,
    # #    'exp_logger': exp_logger
    # #},
    # #    model.state_dict(),
    # #    optimizer.state_dict(),
    # #    logdir,
    # #    True,
    # #    True,
    #     #iTrue)
    # #    is_best)
tensorboard_writer.close()