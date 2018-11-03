import argparse
import os
import time
import h5py

import torch
import torch.nn.parallel
from torch.autograd import Variable

import torchvision.transforms as transforms
from models.extraction_model import get_pretrained_model

from utils.logger import AvgMeter
from utils.ImagesFolder import ImagesFolder
from torch.utils.data.dataloader import DataLoader

parser = argparse.ArgumentParser(description='Extract')

parser.add_argument('--dataset', default='mscocoa',
                    choices=['mscocoa', 'abstract_v002'],
                    help='dataset type: mscoco (default) | abstract_v002')

# TODO change it default folder on MARCC ~/scratch/vqa2018-data/Images/
parser.add_argument('--vqa_data', default='data/Images/',
                    help='dir dataset to download or/and load images')

parser.add_argument('--data_split', default='train2014', required=True, type=str,
                    help='Options: (default) train | val | test')

parser.add_argument('--arch', '-a', default='resnet152', help='Architecture to use')

parser.add_argument('--cuda', '-c', default=True, help="Use CUDA?")

parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')

parser.add_argument('--batch_size', '-b', default=80, type=int,
                    help='mini-batch size (default: 80)')

# TODO uncomment this
# parser.add_argument('--mode', default='both', type=str,
#                     help='Options: att | noatt | (default) both')

# TODO uncomment this
parser.add_argument('--size', default=448, type=int,
                    help='Image size (448 for noatt := avg pooling to get 224) (default:448)')


def main():
    global args
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and args.cuda

    print("Using pre-trained model '{}'".format(args.arch))

    # if not args.cuda:
    #     model = get_pretrained_model(args.arch, cuda=args.cuda, data_parallel=False)
    # else:
    #     model = get_pretrained_model(args.arch, cuda=args.cuda, data_parallel=True)


    dataset_folder = os.path.join(args.vqa_data, args.dataset)
    split_folder = os.path.join(dataset_folder, args.data_split)
    print(split_folder)
    assert os.path.exists(split_folder)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    dataset = ImagesFolder(split_folder, transform=transforms.Compose([
        transforms.Resize(args.size),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        normalize]))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers)

    extract_name = 'extracted_{}'.format(args.data_split)
    dir_extract = os.path.join(dataset_folder, extract_name)
    path_file = os.path.join(dir_extract, "set")
    os.system('mkdir -p ' + dir_extract)


    data_parallel = False
    if args.cuda:
        data_parallel = True

    model = get_pretrained_model(args.arch, args.cuda, data_parallel)
    #print(type(model))
    extract(data_loader, model, path_file)


def extract(data_loader, model, path_file):
    path_hdf5 = path_file + '.hdf5'
    path_txt = path_file + '.txt'
    hdf5_file = h5py.File(path_hdf5, 'w')

    # estimate output shapes
    output = model(torch.ones(1, 3, args.size, args.size))
    print(output.shape)
    print(type(output))
    nb_images = len(data_loader.dataset)
    shape_att = (nb_images, output.size(1), output.size(2), output.size(3))
    print('Warning: shape_att={}'.format(shape_att))
    hdf5_att = hdf5_file.create_dataset('att', shape_att,
                                            dtype='f')  # , compression='gzip')

    model.eval()

    batch_time = AvgMeter()
    data_time = AvgMeter()
    begin = time.time()
    end = time.time()

    idx = 0
    print("Started Extraction")
    for i, input in enumerate(data_loader):
        input_var = input['visual']
        output_att = model(input_var)


        

        batch_size = output_att.size(0)
        hdf5_att[idx:idx + batch_size] = output_att.data.cpu().numpy()
        idx += batch_size

        batch_time.update(time.time() - end)
        end = time.time()

        print('Extract: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                i, len(data_loader),
                batch_time=batch_time,
                data_time=data_time, ))

    hdf5_file.close()

    # Saving image names in the same order than extraction
    with open(path_txt, 'w') as handle:
        for name in data_loader.dataset.imgs:
            handle.write(name + '\n')

    end = time.time() - begin
    print('Finished in {}m and {}s'.format(int(end / 60), int(end % 60)))


if __name__ == '__main__':
    main()
