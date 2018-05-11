from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F

import sys
from PIL import Image

import pickle
from scipy import misc
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from torch.autograd import Variable
from dataloaders.VQADataset import VQADataset
import time
from models.combined_model import returnmodel
from models.extraction_model import get_pretrained_model
import numpy as np
import os
from utils import utils, logger

# Get vocablary related files 

# pre process model_dir
model_dir = "data/processed/nans,2000_maxlength,26_minwcount,0_nlp,mcb_pad,left_trainsplit,train"

# aid_to_ans.pickle
aid_to_ans = pickle.load(open(os.path.join(model_dir,"aid_to_ans.pickle"),"rb"))
# ans_to_aid.pickle
ans_to_aid = pickle.load(open(os.path.join(model_dir,"ans_to_aid.pickle"),"rb"))
# testdevset.pickle
# wid_to_word.pickle
wid_to_word = pickle.load(open(os.path.join(model_dir,"wid_to_word.pickle"),"rb"))
# word_to_wid.pickle
word_to_wid = pickle.load(open(os.path.join(model_dir,"word_to_wid.pickle"),"rb"))

# Load resnet model

img_name = "data/walking.jpeg"
resent_model = get_pretrained_model(data_parallel=False,cuda=False)

# Load our model

model = returnmodel(False,False)


# original saved file with DataParallel
state_dict = torch.load('40.46902011103839.model',map_location='cpu')
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)



print(state_dict['module.combined_models.0.embeddings.weight'][10][0],
state_dict['module.combined_models.0.embeddings.weight'][11][2],
state_dict['module.combined_models.0.embeddings.weight'][12][3],
state_dict['module.combined_models.0.embeddings.weight'][13][4],
     )


model

# Load Image

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform=transforms.Compose([transforms.Resize(448),transforms.CenterCrop(448),transforms.ToTensor(),normalize])
I = Image.open(img_name).convert('RGB')
img_ten = transform(I)
img_ten = img_ten.unsqueeze(0)
# img_ten = img_ten.permute(0,3,1,2);
img_feat = resent_model(Variable(img_ten.float()));

# Question Pre Processing

q = "What is the man doing in the street"
s = q.lower()
s = s.split()
s_id = np.asarray([0]*5+[word_to_wid[x] for x in s])
s_t = torch.from_numpy(s_id)
s_t = s_t.unsqueeze(0)



# Forward Pass Model

model.eval()
answer = model(Variable(s_t),img_feat)
top_val,indices = torch.topk(answer,5,dim=1)
indices = indices.data.cpu().numpy()
top_val = top_val.data.cpu().numpy()
indices = indices.flatten()
top_val = top_val.flatten()

indices
print("Image Name: {}".format(img_name))
for i,sol_id in enumerate(indices):
    print("Soltion: {} -- Prob: {}".format(aid_to_ans[sol_id],np.exp(top_val[i])))


