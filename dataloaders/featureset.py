import os
import h5py
import torch
import torch.utils.data as data


class FeaturesDataset(data.Dataset):
    def __init__(self, dataset, data_split, opt):
        self.data_split = data_split
        self.opt = opt
        self.dir_extract = os.path.join(self.opt['dir'], self.opt['images'], dataset,
                                                                   'extracted_'+data_split)
        self.path_hdf5 = os.path.join(self.dir_extract, 'set.hdf5')
        assert os.path.isfile(self.path_hdf5), \
            'File not found in {}, you must extract the features first with extract.py'.format(self.path_hdf5)
        self.hdf5_file = h5py.File(self.path_hdf5, 'r')  # , driver='mpio', comm=MPI.COMM_WORLD)
        self.dataset_features = self.hdf5_file['att']
        self.index_to_name, self.name_to_index = self._load_dicts()

    def _load_dicts(self):
        self.path_fname = os.path.join(self.dir_extract, 'set.txt')
        with open(self.path_fname, 'r') as handle:
            self.index_to_name = handle.readlines()
        self.index_to_name = [name[:-1] for name in self.index_to_name]  # remove char '\n'
        self.name_to_index = {name: index for index, name in enumerate(self.index_to_name)}
        return self.index_to_name, self.name_to_index

    def __getitem__(self, index):
        item = {}
        item['name'] = self.index_to_name[index]
        item['visual'] = self.get_features(index)
        # item = torch.Tensor(self.get_features(index))
        return item

    def get_features(self, index):
        return torch.LongTensor(self.dataset_features[index])


    def get_by_name(self, image_name):
        index = self.name_to_index[image_name]
        return self[index]

    def __len__(self):
        return self.dataset_features.shape[0]
