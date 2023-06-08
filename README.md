# Getting started
## Implemented models
1. LeNet5
2. AlexNet
3. VGGNet (configuration D)
4. GoogLeNet
5. ResNet-34 (convolutional)

## Preparing system:

1. Install Git:  
Оn Linux:  
`sudo apt install git` or `sudo dnf install git-all`  
Оn Windows:  
https://git-scm.com/download/win
2. Install Python:  
On Linux:  
`sudo apt install python3`  
On Windows:  
https://www.python.org/downloads/
3. Clone repository: `git clone https://github.com/axelBaher/ecg-classification.git`
4. Setup virtual environment and install packages into it:   
`python setup.py`  
If script doesn't work for whatever reason, just run this command:  
`pip install -r requirements.txt`
In this way, all the packages will be installed in your main (system) Python path.
5. Get necessary db and generate data:  
`python prep.py`

## Train
To start training, you need to run this command, in the figure brackets you need to type model, which will be trained:
`python train.py --config {model_name}`  
There are five models to choose (type exactly, as it will be written below):  
LeNet5  
AlexNet  
VGGNetD  
GoogLeNet  
ResNet34  

[//]: # (## Testing:)

[//]: # (W.I.P)
