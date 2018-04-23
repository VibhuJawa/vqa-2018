#!/usr/bin/env python

import json
import os
import argparse
from collections import Counter


def get_subtype(split='train'):
    if split in ['train', 'val']:
        return split + '2014'
    else:
        return 'test2015'


def get_image_name_old(subtype='train2014', image_id='1', format='%s/COCO_%s_%012d.jpg'):
    return format % (subtype, subtype, image_id)


def get_image_name(subtype='train2014', image_id='1', format='COCO_%s_%012d.jpg'):
    return format % (subtype, image_id)


def interim(questions, split='train', annotations=[]):
    print('Interim', split)
    data = []
    for i in range(len(questions)):
        row = {}
        row['question_id'] = questions[i]['question_id']
        row['image_name'] = get_image_name(get_subtype(split), questions[i]['image_id'])
        row['question'] = questions[i]['question']
        if split in ['train', 'val', 'trainval']:
            row['answer'] = annotations[i]['multiple_choice_answer']
            answers = []
            for ans in annotations[i]['answers']:
                answers.append(ans['answer'])
            row['answers_occurence'] = Counter(answers).most_common()
        data.append(row)
    return data


def extract_annotations(data_dir = './'):
    path_train_qa = os.path.join(data_dir, 'interim', 'train_questions_annotations.json')
    path_val_qa = os.path.join(data_dir, 'interim', 'val_questions_annotations.json')
    path_trainval_qa = os.path.join(data_dir, 'interim', 'trainval_questions_annotations.json')
    path_test_q = os.path.join(data_dir, 'interim', 'test_questions.json')
    path_testdev_q = os.path.join(data_dir, 'interim', 'testdev_questions.json')

    print('Loading annotations and questions...')
    annotations_train_1 = json.load(
        open(os.path.join(data_dir, 'annotations', 'v2_mscoco_train2014_annotations.json'), 'r'))
    annotations_train_2 = json.load(
        open(os.path.join(data_dir, 'annotations', 'abstract_v002_train2015_annotations.json'), 'r'))
    annotations_train_3 = json.load(
        open(os.path.join(data_dir, 'annotations', 'abstract_v002_train2017_annotations.json'), 'r'))

    questions_train_1 = json.load(
        open(os.path.join(data_dir, 'Questions', 'v2_OpenEnded_mscoco_train2014_questions.json'), 'r'))
    questions_train_2 = json.load(
        open(os.path.join(data_dir, 'Questions', 'OpenEnded_abstract_v002_train2015_questions.json'), 'r'))
    questions_train_3 = json.load(
        open(os.path.join(data_dir, 'Questions', 'OpenEnded_abstract_v002_train2017_questions.json'), 'r'))

    annotations_val_1 = json.load(open(os.path.join(data_dir, 'annotations', 'v2_mscoco_val2014_annotations.json'), 'r'))
    annotations_val_2 = json.load(
        open(os.path.join(data_dir, 'annotations', 'abstract_v002_val2015_annotations.json'), 'r'))
    annotations_val_3 = json.load(
        open(os.path.join(data_dir, 'annotations', 'abstract_v002_val2017_annotations.json'), 'r'))

    questions_val_1 = json.load(
        open(os.path.join(data_dir, 'Questions', 'v2_OpenEnded_mscoco_val2014_questions.json'), 'r'))
    questions_val_2 = json.load(
        open(os.path.join(data_dir, 'Questions', 'OpenEnded_abstract_v002_val2015_questions.json'), 'r'))
    questions_val_3 = json.load(
        open(os.path.join(data_dir, 'Questions', 'OpenEnded_abstract_v002_val2017_questions.json'), 'r'))

    questions_test = json.load(
        open(os.path.join(data_dir, 'Questions', 'v2_OpenEnded_mscoco_test2015_questions.json'), 'r'))

    question_test_dev = json.load(
        open(os.path.join(data_dir, 'Questions', 'v2_OpenEnded_mscoco_test-dev2015_questions.json'), 'r'))

    val_merge_1 = interim(questions=questions_val_1['questions'], annotations=annotations_val_1['annotations'],
                          split='val')
    val_merge_2 = interim(questions=questions_val_2['questions'], annotations=annotations_val_2['annotations'],
                          split='val')
    val_merge_3 = interim(questions=questions_val_3['questions'], annotations=annotations_val_3['annotations'],
                          split='val')

    train_merge_1 = interim(questions=questions_train_1['questions'], annotations=annotations_train_1['annotations'],
                            split='train')
    train_merge_2 = interim(questions=questions_train_2['questions'], annotations=annotations_train_2['annotations'],
                            split='train')
    train_merge_3 = interim(questions=questions_train_3['questions'], annotations=annotations_train_3['annotations'],
                            split='train')

    testset = interim(questions=questions_test['questions'], split='test')
    test_dev_set = interim(questions=question_test_dev['questions'],split='test_dev')

    trainset = train_merge_1 + train_merge_2 + train_merge_3
    valset = val_merge_1 + val_merge_2 + val_merge_3
    trainval = trainset + valset


    if not os.path.exists(os.path.join(data_dir, 'interim')):
        os.makedirs(os.path.join(data_dir, 'interim'))

    json.dump(trainset, open(path_train_qa, 'w'))
    json.dump(valset, open(path_val_qa, 'w'))
    json.dump(testset,open(path_test_q,'w'))
    json.dump(trainval,open(path_trainval_qa,'w'))
    json.dump(test_dev_set,open(path_testdev_q,'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data Dir')

    parser.add_argument('--folder', required=False, default='./',
                        help='The path to the data directory. (default : current) if you want to install in data/')

    args = parser.parse_args()
    extract_annotations(args.folder)
