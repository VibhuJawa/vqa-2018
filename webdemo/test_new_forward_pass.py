from __future__ import print_function
import torch

from PIL import Image

import pickle
from torchvision import transforms

from torch.autograd import Variable
from models.combined_model import returnmodel
from models.extraction_model import get_pretrained_model
import numpy as np
import os
import string

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

resent_model = get_pretrained_model(data_parallel=False,cuda=False)
model = returnmodel(False,False)


# Load our model



# original saved file with DataParallel
state_dict = torch.load('40.46902011103839.model', map_location='cpu')
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)


def run_forward(img_name, q):

    # Load Image
    # img_name = "data/walking.jpeg"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    transform=transforms.Compose([transforms.Resize(448),transforms.CenterCrop(448),transforms.ToTensor(),normalize])
    try:
        I = Image.open("static/images/" + img_name).convert('RGB')
    except:
        return "Image doesn't exist"
    img_ten = transform(I)
    img_ten = img_ten.unsqueeze(0)
    # img_ten = img_ten.permute(0,3,1,2);
    img_feat = resent_model(Variable(img_ten.float()))

    # Question Pre Processing

    # q = "What is the man doing in the street"
    s = q.lower()
    # print("ASDASDSAD", s)

    punc = string.punctuation
    thestring = s
    s = list(thestring)
    s = ''.join([o for o in s if not o in punc]).split()
    # print("PQRPQR", s)

    s_id = []
    for x in s:
        if x not in word_to_wid:
            s_id.append(word_to_wid['UNK'])
        else:
            s_id.append(word_to_wid[x])

    s_t = torch.from_numpy(np.asarray(s_id))
    s_t = s_t.unsqueeze(0)

    # Forward Pass Model

    model.eval()
    answer = model(Variable(s_t),img_feat)
    top_val,indices = torch.topk(answer,5,dim=1)
    indices = indices.data.cpu().numpy()
    top_val = top_val.data.cpu().numpy()
    indices = indices.flatten()
    top_val = top_val.flatten()

    probabilties = []
    for i,sol_id in enumerate(indices):
        probabilties.append((aid_to_ans[sol_id], str(round(np.exp(top_val[i]) * 100, 2))))

    return probabilties


