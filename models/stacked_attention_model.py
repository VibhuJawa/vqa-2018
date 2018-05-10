import torch.optim as optim
import numpy as np
################################################
# Create Model and Optimizer
################################################


import torch
from torch import nn
import torch.nn.functional as F


class StackedAttentionModel(nn.Module):
    def __init__(self, num_classes=2000, question_vocab_size=13995, embedding_size=256,
                 hidden_size=512, img_maps=2048, k=400):
        super().__init__()
        self.img_maps = img_maps
        # self.qmodel = qmodel
        # self.imgmodel = imgmodel

        self.embeddings = nn.Embedding(question_vocab_size, embedding_size, padding_idx=0)
        self.rnn = nn.GRU(embedding_size, img_maps // 2, batch_first=True, bidirectional=True)

        self.img_weight_k1 = nn.Linear(img_maps, k)
        self.que_weights_k1 = nn.Linear(img_maps, k, bias=False)

        self.img_weight_k2 = nn.Linear(img_maps, k)
        self.que_weights_k2 = nn.Linear(img_maps, k, bias=False)

        self.energy_k1 = nn.Linear(k, 1)
        self.energy_k2 = nn.Linear(k, 1)

        self.v = nn.Parameter(torch.FloatTensor(1, hidden_size + embedding_size))

        self.fc1 = nn.Linear(img_maps, num_classes)

    def forward(self, question, image, hidden=None):
        max_len = question.size(1)  # Maximum length of question
        batch_size = image.size(0)  # Current Batch size, if you need this you're doing something wrong
        #         print(max_len, batch_size)
        # b * 2048 * 14 * 14 ==> b * 2048 * 196

        image_features = image.view(batch_size, self.img_maps, -1).permute(0, 2, 1)
        embedded_questions = self.embeddings(question)
        rnn_encoding, last_hidden = self.rnn(embedded_questions)
        question_embedding = torch.cat([last_hidden[0], last_hidden[1]], dim=1)

        #########################################
        # Step 1
        ########################################

        downsampled_questions1 = self.img_weight_k1(question_embedding)
        downsampled_image1 = self.que_weights_k1(image_features)
        hA1 = F.tanh(downsampled_image1 + downsampled_questions1.unsqueeze(dim=1))
        alpha1 = F.softmax(self.energy_k1(hA1), dim=1)
        context1 = image_features * alpha1
        context_sum1 = context1.sum(dim=1)

        visual_question_embedding = context_sum1 + question_embedding

        #########################################
        # Step 2
        ########################################

        downsampled_questions2 = self.img_weight_k2(visual_question_embedding)
        downsampled_image2 = self.que_weights_k2(image_features)
        hA2 = F.tanh(downsampled_image2 + downsampled_questions2.unsqueeze(dim=1))
        alpha2 = F.softmax(self.energy_k2(hA2), dim=1)
        context2 = image_features * alpha2
        context_sum2 = context2.sum(dim=1)
        visual_question_embedding_final = context_sum2 + visual_question_embedding

        ###########################
        # Done with last encoding
        ##############################
        return F.log_softmax(self.fc1(visual_question_embedding_final), dim=1)


def returnmodel(cuda=True, data_parallel=True, num_classes=2000, question_vocab_size=13995, embedding_size=256,
                hidden_size=512, img_maps=2048, k=400):
    data_parallel = cuda and data_parallel

    model = StackedAttentionModel(num_classes, question_vocab_size, embedding_size, hidden_size, img_maps, k)

    if data_parallel:
        model = nn.DataParallel(model).cuda()

    if cuda and not data_parallel:
        model.cuda()

    return model
