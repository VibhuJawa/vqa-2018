import torch
from torch import nn
import torch.nn.functional as F


class QuestionModel(nn.Module):
    def __init__(self, embedding_size=300, hidden_size=256, question_vocab_size=13395, linear_size=2048):
        super().__init__()
        self.embeddings = nn.Embedding(question_vocab_size, embedding_size, padding_idx=0)
        self.rnn = nn.LSTM(embedding_size, hidden_size, bidirectional=True)
        self.linear1 = torch.nn.Linear(hidden_size * 2, (linear_size - hidden_size * 2) // 2)
        self.linear2 = torch.nn.Linear((linear_size - hidden_size * 2) // 2, linear_size)

    def forward(self, x, initial_states=None):
        # x stores integers and has shape [length, batch_size]
        x = self.embeddings(x)  # TODO
        # x now stores floats and has shape [length, batch_size, embedding_size]
        self.rnn.flatten_parameters()
        x, final_states = self.rnn(x, initial_states)  # TODO
        # x = torch.cat([x[-1,0] ,x[0][1]], dim=1)
        self.rnn.flatten_parameters()
        x = x[:,-1,:]
        #x = torch.cat([final_states[0][0], final_states[0][1]], dim=1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x


class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def forward(self, x, initial_states=None):
        x = self.avgpool(x)
        div = x.size(3) + x.size(2)
        x = x.sum(3)
        x = x.sum(2)
        x = x.view(x.size(0), -1)
        x = x.div(div)
        return x


class MergedModel(nn.Module):
    def __init__(self, qmodel, imgmodel, concat_size=2048, num_classes=2000):
        super().__init__()
        # self.qmodel = qmodel
        # self.imgmodel = imgmodel
        self.combined_models = nn.ModuleList([qmodel,imgmodel])
        self.avgpool = nn.AvgPool2d((2, 1))
        self.fc1 = nn.Linear(concat_size, num_classes)

    def forward(self, question, image):
        # q_out = self.qmodel(question)
        # img_out = self.imgmodel(image)
        q_out = self.combined_models[0](question)
        img_out = self.combined_models[1](image)
        x = torch.stack([img_out, q_out], dim=1)
        x = self.avgpool(x)
        # TODO check this 
        x = x.squeeze(dim = 1)
        x = F.log_softmax(self.fc1(x), dim=1)

        return x


def returnmodel(cuda=True, data_parallel=True, embedding_size=300, hidden_size=256, question_vocab_size=13395, linear_size=2048, num_classes=2000):
    data_parallel = cuda and data_parallel

    questionmodel = QuestionModel(embedding_size, hidden_size, question_vocab_size, linear_size)
    imagemodel = ImageModel()
    model = MergedModel(questionmodel, imagemodel, linear_size, num_classes)

    if data_parallel:
        model = nn.DataParallel(model).cuda()

    if cuda and not data_parallel:
        model.cuda()

    return model
