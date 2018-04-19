#!/usr/bin/env python

import os
import argparse


def download_vqa(folder):
    # Download VQA Questions (mscocoa)
    os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip -P %szip/' % (args.folder))
    os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip -P %szip/' % (args.folder))
    os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Test_mscoco.zip -P %szip/' % (args.folder))

    # Download VQA Questions (binary abstract)
    os.system(
        'wget http://visualqa.org/data/abstract_v002/vqa/Questions_Binary_Train2017_abstract_v002.zip -P %szip/' % (
        args.folder))
    os.system('wget http://visualqa.org/data/abstract_v002/vqa/Questions_Binary_Val2017_abstract_v002.zip -P %szip/' % (
    args.folder))

    # Download VQA Questions (abstract)
    os.system(
        'wget http://visualqa.org/data/abstract_v002/vqa/Questions_Train_abstract_v002.zip -P %szip/' % (args.folder))
    os.system(
        'wget http://visualqa.org/data/abstract_v002/vqa/Questions_Val_abstract_v002.zip -P %szip/' % (args.folder))
    os.system(
        'wget http://visualqa.org/data/abstract_v002/vqa/Questions_Test_abstract_v002.zip -P %szip/' % (args.folder))

    # Download the VQA Annotations (mscocoa)
    os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P %szip/' % (args.folder))
    os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P %szip/' % (args.folder))

    # Download the VQA Annotations (binary abstract)
    os.system(
        'wget http://visualqa.org/data/abstract_v002/vqa/Annotations_Binary_Train2017_abstract_v002.zip -P %szip/' % (
        args.folder))
    os.system(
        'wget http://visualqa.org/data/abstract_v002/vqa/Annotations_Binary_Val2017_abstract_v002.zip -P %szip/' % (
        args.folder))

    # Download the VQA Annotations (abstract)
    os.system(
        'wget http://visualqa.org/data/abstract_v002/vqa/Annotations_Train_abstract_v002.zip -P %szip/' % (args.folder))
    os.system(
        'wget http://visualqa.org/data/abstract_v002/vqa/Annotations_Val_abstract_v002.zip -P %szip/' % (args.folder))

    # Download VQA Images (mscocoa)
    os.system('wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip -P %szip/' % (args.folder))
    os.system('wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip -P %szip/' % (args.folder))
    os.system('wget http://msvocds.blob.core.windows.net/coco2015/test2015.zip -P %szip/' % (args.folder))

    # Download VQA Images (binary abstract) (images)
    os.system(
        'wget http://visualqa.org/data/abstract_v002/scene_img/scene_img_abstract_v002_binary_train2017.zip -P %szip/' % (
        args.folder))
    os.system(
        'wget http://visualqa.org/data/abstract_v002/scene_img/scene_img_abstract_v002_binary_val2017.zip -P %szip/' % (
        args.folder))

    # Download VQA Images (abstract scenes)
    os.system(
        'wget http://visualqa.org/data/abstract_v002/scene_img/scene_img_abstract_v002_train2015.zip -P %szip/' % (
        args.folder))
    os.system('wget http://visualqa.org/data/abstract_v002/scene_img/scene_img_abstract_v002_val2015.zip -P %szip/' % (
    args.folder))
    os.system('wget http://visualqa.org/data/abstract_v002/scene_img/scene_img_abstract_v002_test2015.zip -P %szip/' % (
    args.folder))

    # Unzip the files

    os.system('mkdir %sQuestions' % (folder))
    os.system('unzip %szip/v2_Questions_Train_mscoco.zip -d %sQuestions/' % (folder, folder))
    os.system('unzip %szip/v2_Questions_Val_mscoco.zip -d %sQuestions/' % (folder, folder))
    os.system('unzip %szip/v2_Questions_Test_mscoco.zip -d %sQuestions/' % (folder, folder))
    os.system('unzip %szip/Questions_Binary_Train2017_abstract_v002.zip -d %sQuestions/' % (folder, folder))
    os.system('unzip %szip/Questions_Binary_Val2017_abstract_v002.zip -d %sQuestions/' % (folder, folder))
    os.system('unzip %szip/Questions_Train_abstract_v002.zip -d %sQuestions/' % (folder, folder))
    os.system('unzip %szip/Questions_Val_abstract_v002.zip -d %sQuestions/' % (folder, folder))
    os.system('unzip %szip/Questions_Test_abstract_v002.zip -d %sQuestions/' % (folder, folder))

    os.system('mkdir %sAnnotations' % (folder))
    os.system('unzip %szip/v2_Annotations_Train_mscoco.zip -d %sAnnotations/' % (folder, folder))
    os.system('unzip %szip/v2_Annotations_Val_mscoco.zip -d %sAnnotations/' % (folder, folder))
    os.system('unzip %szip/Annotations_Binary_Train2017_abstract_v002.zip -d %sAnnotations/' % (folder, folder))
    os.system('unzip %szip/Annotations_Binary_Val2017_abstract_v002.zip -d %sAnnotations/' % (folder, folder))
    os.system('unzip %szip/Annotations_Train_abstract_v002.zip -d %sAnnotations/' % (folder, folder))
    os.system('unzip %szip/Annotations_Val_abstract_v002.zip -d %sAnnotations/' % (folder, folder))

    os.system('mkdir -p %sImages/mscocoa/test2015' % (folder))
    os.system('mkdir %sImages/mscocoa/test2014' % (folder))
    os.system('mkdir %sImages/mscocoa/val2014' % (folder))
    os.system('unzip  %szip/train2014.zip -d %sImages/mscocoa/' % (folder, folder))
    os.system('unzip %szip/val2014.zip -d %sImages/mscocoa/' % (folder, folder))
    os.system('unzip %szip/test2015.zip -d %sImages/mscocoa/' % (folder, folder))

    os.system('mkdir -p %s/Images/abstract_v002/train2015' % (folder))
    os.system('mkdir %s/Images/abstract_v002/val2015' % (folder))
    os.system('mkdir %s/Images/abstract_v002/test2015' % (folder))

    os.system('unzip %szip/scene_img_abstract_v002_binary_train2017.zip -d %sImages/abstract_v002/' % (folder, folder))
    os.system('unzip %szip/scene_img_abstract_v002_binary_val2017.zip -d %sImages/abstract_v002/' % (folder, folder))

    os.system(
        'unzip %szip/scene_img_abstract_v002_train2015.zip -d %sImages/abstract_v002/train2015/' % (folder, folder))
    os.system('unzip %szip/scene_img_abstract_v002_val2015.zip -d %sImages/abstract_v002/val2015/' % (folder, folder))
    os.system('unzip %szip/scene_img_abstract_v002_test2015.zip -d %sImages/abstract_v002/test2015/' % (folder, folder))

    os.system('ln -s %sAnnotations/ .' % (folder))
    os.system('ln -s %sQuestions/ .' % (folder))
    os.system('ln -s %sImages/ .' % (folder))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Set up the dataset')

    parser.add_argument('--folder', required=False, default="",
                        help='The path to the data directory. (default : current) if you want to install in data/')

    args = parser.parse_args()
    download_vqa(args.folder)
    print(("Extracted everything. You might want to delete %szip folder" % (args.folder)))
