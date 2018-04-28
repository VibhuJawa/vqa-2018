from __future__ import print_function
import argparse
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim.scripts.glove2word2vec import glove2word2vec
import pickle

from gensim.models.keyedvectors import KeyedVectors
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
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
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")


# pre process model_dir
model_dir = "data/processed/nans,2000_maxlength,26_minwcount,0_nlp,mcb_pad,left_trainsplit,train"
# aid_to_ans.pickle
aid_to_ans = pickle.load(open(os.path.join(model_dir,"aid_to_ans.pickle"),"rb"))
# ans_to_aid.pickle
ans_to_aid = pickle.load(open(os.path.join(model_dir,"ans_to_aid.pickle"),"rb"))
# testdevset.pickle
testdevset = pickle.load(open(os.path.join(model_dir,"testdevset.pickle"),"rb"))
# testset.pickle
testset = pickle.load(open(os.path.join(model_dir,"testset.pickle"),"rb"))
# trainset.pickle
trainset = pickle.load(open(os.path.join(model_dir,"trainset.pickle"),"rb"))
# valset.pickle
valset = pickle.load(open(os.path.join(model_dir,"valset.pickle"),"rb"))
# wid_to_word.pickle
wid_to_word = pickle.load(open(os.path.join(model_dir,"wid_to_word.pickle"),"rb"))
# word_to_wid.pickle
word_to_wid = pickle.load(open(os.path.join(model_dir,"word_to_wid.pickle"),"rb"))


len_answers=len(word_to_wid)

# change directory

glove_dir = "/Users/jawa/Desktop/mark_ra/falconet/models/resources/"
glove_dim = 200
glove_file = "glove.twitter.27B.{}d.txt".format(glove_dim)
glove2word2vec(glove_input_file=glove_dir+glove_file, word2vec_output_file="resources/gensim_glove_vectors.txt")
glove_model = KeyedVectors.load_word2vec_format("resources/gensim_glove_vectors.txt", binary=False)


class VisualQuestionsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, ds, ans_to_aid, aid_to_ans, wid_to_word, word_to_wid, image_root_dir=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.question_answer_ds = ds
        self.ans_to_aid = ans_to_aid
        self.aid_to_ans = aid_to_ans
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.image_root_dir = image_root_dir

    def __len__(self):
        return len(self.question_answer_ds)

    def __getitem__(self, idx):

        # self.image_root_dir
        image_feat = None
        question = self.question_answer_ds[idx]['question_wids']

        # question_vec = [glove_model[wid_to_word[x]] for x in question if x!=0 ]
        question_vec = []
        for x in question:
            if x == 0:
                question_vec.append(glove_dim * [0])
            else:
                word = wid_to_word[x]
                if word.isdigit():
                    word = p.number_to_words(word)
                if word in glove_model:
                    question_vec.append(glove_model[word])
                else:
                    question_vec.append(glove_dim * [0])

        answer_id = self.question_answer_ds[idx]['answer_aid']
        y = np.zeros(len_answers)
        y[answer_id] = 1
        question_vec = np.asarray(question_vec)

        return question_vec, y

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_dataset = VisualQuestionsDataset(trainset,ans_to_aid,aid_to_ans,wid_to_word,word_to_wid)
train_loader = torch.utils.data.DataLoader(train_dataset,  batch_size=32, shuffle=True,num_workers=4)

val_dataset = torch.utils.data.DataLoader(valset,  batch_size=32, shuffle=True,num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size, shuffle=True, **kwargs)



class VQA_BASIC_WITHOUT_IMAGE_MODEL(nn.Module):
    def __init__(self, input_size,hidden_size =2048,n_layers=2,len_answers=2000):
        super().__init__()
        self.rnn =  nn.LSTM(input_size = input_size, hidden_size = hidden_size , num_layers =n_layers)
        self.linear = nn.Linear(hidden_size,len_answers)
        self.sofmax = nn.Softmax()
    def forward(self,x,input=None):
        x,  final_state = self.rnn(x)
        # picking the last elemnt from the sequence as output to the fc
        x = x[-1,:,:]
        x = self.linear(x)
        x = self.sofmax(x)
        return x

model = VQA_BASIC_WITHOUT_IMAGE_MODEL(glove_dim).to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        break

def val():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    print("Epoch 1",{epoch})
    val()
