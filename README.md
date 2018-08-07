# Visual Question Answering: May I Have Your Attention Please? 

## Project Overview

[VQA](http://visualqa.org) is a new dataset containing openended questions about images. 
These questions require an understanding of vision,language and commonsense knowledge to answer. 

We experiment with various methods and reproduce them in Py-Torch. We try to analyse the different models presented and also our own model which uses attention. 

In totality we experiment with 7 different models, in a constrained environment of training upto only 10 epochs. 

We explain the models we tried, and the choices we made while creating our own model. 

We additionaly release an extremely lightweight yet modular code which can be used to perform more experiments with no time setting up the framework.

## Project Report
[Project Report](VQA.pdf)

## [Demo Link](http://34.239.173.78:5000/process_vqa?filename=fullsizeoutput_2.jpeg&question=who+is+in+the+photo)


# Repo Setup Instructions:


## Data processing:
### Data download
Run [download.py](data/download.py) with --folder as the desired folder to download the vqa data set from the website.

### Extract Image Features 
Run [extract_image_features.py](extract_image_features.py) to download the image process them using resnet-152 (or custom) and dump them into h5py files to make processing easier.

Arguments to be provided:
* vqa_data: 
* data_split
* arch 
* cuda
* num_workers
* batch_size

### Extract Question/Answer Features 

#### Interim Extract 
Run [extract_annotations_interim.py](extract_annotations_interim.py)

Creates a interim json for train,test,val and creates a mapping between image name and location extracted for downstream processing.

#### Final Processing
Run [extract_annotations_processed.py](extract_annotations_processed.py)

Arguments to be provided:
* dir (directory of the data)
* train_split (train/val/test)
* nan (Number of top answers)
* maxlength (Max lenfth of words in the question)
* nlp (Tokenization method) (options: nlp/mcb/naive)
* pad (Padding space) (default: left/right)

## Model Training:
Run the following files for training the model.
* stacked attention model [stacked_attention_model.py](stacked_attention_model.py)
* Concat Image+Qustion embedding model [std_combined_model.py](std_combined_model.py)
* Custom attention model [custom_attention_model.py](custom_attention_model.py)

## Cool Implimetnation Detail to save computation:
Here the custom change has been to pytorches dataloader to ensure that we can do padding based on the maximum of each batch even with pre-fetching with num_workers>2 in pytorch 0.3.1 . 

This essentially helps us make sure that each question length processed by an lstm is the max of that batch and not the global max hence saving a lot of computation time. 

Main Tweak: https://github.com/VibhuJawa/vqa-2018/blob/master/utils/dataloader1.py#L137

## Combatibily:
This code was written for pytorch 0.3.1 and will break in pytorch 0.4.1 because of changes in syntax, please update changes on your own.


## Contributors:
* [Vibhu Jawa](http://github.com/vibhujawa)
* [Praateek Mahajan](http://prtk.in)
* [Iskandar Atakhodjaev](https://github.com/atah1991)
