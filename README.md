# Getting started
## Implemented models
1. LeNet5
2. AlexNet
3. VGGNet (configuration D)
4. GoogLeNet
5. ResNet-34 (convolutional)

## Preparing system:

1. Install Git:  
If you on Linux, you have already installed it.  
Оn Windows:  
https://git-scm.com/download/win
2. Install Python:  
If you on Linux, you have already installed it.  
On Windows:  
https://www.python.org/downloads/
3. Clone repository: `git clone https://github.com/axelBaher/ecg-classification.git`
4. Setup virtual environment and install packages into it:   
`python setup.py`  
If script doesn't work for whatever reason, just run this command:  
`pip install -r requirements.txt`  
In this way, all the packages will be installed in your main (system) Python path.
5. Go to folder with scripts:  
`cd main`
6. Get necessary db and generate data:  
`python prep.py`

## Train
To start training, you need to run this command, in the figure brackets you need to type model, which will be trained:
`python train.py --config {model_name}`  
There are five models to choose (type exactly, as it will be written below):  
LeNet5, AlexNet, VGGNetD, GoogLeNet, ResNet34
In the `config/training/{model_name}` you can find configuration, that will be used in training.

## Interence
To start inference, you need to run this commmand:  
`python inference.py -name {model_name} -epoch {number_of_training_epoch} -b_size {batch_size} -val_split {validation_split} [-loss {loss_function}] [-opt {optimizer}]`  
You need to input model name and parameters for program to find pretrained weights.
## Pipeline
To start pipeline, you need to run this command:  
`python pipeline.py`
In the `config/pipeline.json` you can configure, which models will be trained and tested and with which parameters.
