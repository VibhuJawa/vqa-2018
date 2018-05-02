################################################
# Create Model and Optimizer
################################################


import torch
from torch import nn
import torch.nn.functional as F


class AttentionModel(nn.Module):
    def __init__(self, num_classes=2000, question_vocab_size=13995, embedding_size=256, hidden_size=512):
        super().__init__()
        self.embeddings = nn.Embedding(question_vocab_size, embedding_size, padding_idx=0)
        self.rnn = nn.GRU(embedding_size + hidden_size, hidden_size, batch_first=False)
        self.maxpool = nn.MaxPool2d((1, 2048), stride=1)

        self.context2embed = nn.Linear(hidden_size, embedding_size)
        self.img2hidden = nn.Linear(2048, hidden_size)
        self.img2h0 = nn.Linear(196, hidden_size)

        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, num_classes)
        self.v = nn.Parameter(torch.FloatTensor(1, hidden_size + embedding_size))

    def forward(self, question, image, hidden=None):
        max_len = question.size(1)  # Maximum length of question
        batch_size = image.size(0)  # Current Batch size, if you need this you're doing something wrong
        #         print(max_len, batch_size)
        #         print("Image : ", image.shape)
        #         print("Question : ", question.shape)
        # b * 2048 * 14 * 14 ==> b * 2048 * 196
        image_copy = image.view(batch_size, 2048, -1).permute(0, 2, 1)
        hidden = self.img2h0(self.maxpool(image_copy).squeeze(dim=2)).unsqueeze(dim=0)

        #         hidden = None
        #         print("Hidden ",hidden.shape)
        img_hidden = self.img2hidden(image_copy).permute(0, 2, 1)
        #         print("Image hidden shape : ", img_hidden.shape)

        embedded_questions = self.embeddings(question)
        #         print("Question shape", embedded_questions.shape)

        for i in range(max_len):
            curr_word = embedded_questions[:, i, :].unsqueeze(dim=0)
            #             print("Dim of curr word ", curr_word.shape)
            energy = torch.bmm(hidden.permute(1, 0, 2), img_hidden)
            #             print("Energy ", energy.shape)
            alpha = F.softmax(energy, dim=2)
            #             print("alpha ",alpha.shape)
            context = img_hidden * alpha
            context_sum = context.sum(dim=2)
            #             print("Context sum", context_sum.shape)
            concat_word_img = torch.cat([curr_word, context_sum.unsqueeze(dim=0)], dim=2)
            embedding_with_attention = self.v * concat_word_img
            out, hidden = self.rnn(embedding_with_attention, hidden)
        # print(out.shape)
        x = out.squeeze(dim=0)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def returnmodel(cuda=True, data_parallel=True, num_classes=2000, question_vocab_size=13995, embedding_size=256,
                hidden_size=512):
    data_parallel = cuda and data_parallel

    model = AttentionModel(num_classes, question_vocab_size, embedding_size, hidden_size)

    if data_parallel:
        model = nn.DataParallel(model).cuda()

    if cuda and not data_parallel:
        model.cuda()

    return model
