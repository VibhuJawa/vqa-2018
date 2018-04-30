import os
import pickle
import torch
# import ..utils as data
import copy
import numpy as np
import h5py
from dataloaders.AbstractDataset import AbstractVQADataset
from utils.dataloader1 import DataLoader


class VQADataset1(AbstractVQADataset):
    def __init__(self, data_split, image_features, opt):
        super(VQADataset1, self).__init__(data_split, opt)

        '''
        Do we need to sample answers as per probability distribution?
        '''
        if 'train' not in self.data_split:  # means self.data_split is 'val' or 'test'
            self.opt['sampleans'] = False
        assert 'sampleans' in self.opt, \
            "opt['vqa'] does not have 'samplingans' " \
            "entry. Set it to True or False."

        self.hdf5_file_dict = {}
        self.img_feat_dict = image_features
        if self.data_split == 'test':
            path_testdevset = os.path.join(self.subdir_processed, 'testdevset.pickle')
            with open(path_testdevset, 'rb') as handle:
                self.testdevset_vqa = pickle.load(handle)
            self.is_qid_testdev = {}
            for i in range(len(self.testdevset_vqa)):
                qid = self.testdevset_vqa[i]['question_id']
                self.is_qid_testdev[qid] = True

    def _raw(self):
        raise NotImplementedError

    def _interim(self):
        raise NotImplementedError

    def _processed(self):
        raise NotImplementedError

    def __getitem__(self, index):
        item = {}
        # TODO: better handle cascade of dict items
        item_vqa = self.dataset[index]

        # Process Question (word token)
        #         item['question_id'] = item_vqa['question_id']
        length = len(item_vqa['question_wids']) - item_vqa['question_length']

        #         item['question'] = torch.LongTensor(item_vqa['question_wids'][length:])
        item['question'] = torch.LongTensor(item_vqa['question_wids'])

        if self.data_split == 'test':
            if item['question_id'] in self.is_qid_testdev:
                item['is_testdev'] = True
            else:
                item['is_testdev'] = False
        else:
            ## Process Answer if exists
            if self.opt['sampleans']:
                proba = item_vqa['answers_count']
                proba = proba / np.sum(proba)
                item['answer'] = int(np.random.choice(item_vqa['answers_aid'], p=proba))
            else:
                item['answer'] = item_vqa['answer_aid']
        imgurl = item_vqa['image_name']
        parent_dir = os.path.dirname(imgurl)
        file_name = os.path.basename(imgurl)
        dataset = os.path.dirname(parent_dir)
        split = os.path.basename(parent_dir).replace("extracted_","")
        
        item_img = self.img_feat_dict[dataset][split].get_by_name(file_name.strip())



        item['image'] = item_img['visual']

        item['word_count'] = item_vqa['question_length']
        return item

    def __len__(self):
        return len(self.dataset)

    def num_classes(self):
        return len(self.aid_to_ans)

    def vocab_words(self):
        return list(self.wid_to_word.values())

    def vocab_answers(self):
        return self.aid_to_ans

    def data_loader(self, batch_size=10, num_workers=4, shuffle=False, pin_memory=False):
        return DataLoader(self,
                          batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory)

    def get_img_url(self, image):
        parts = image.split("_")
        directory = 'mscocoa'
        if 'COCO' not in parts[0]:
            directory = 'abstract_v002'
        subdirectory = parts[1]
        return os.path.join(self.opt['dir'], self.opt['images'], directory, subdirectory, "_extracted", image)
