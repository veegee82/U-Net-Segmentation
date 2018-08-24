## U-Net Segmentation

"U-Net: Convolutional Networks for Biomedical Image Segmentation"

Paper: https://arxiv.org/pdf/1505.04597.pdf

#### Abstract
There is large consent that successful training of deep networks
requires many thousand annotated training samples. In this paper,
we present a network and training strategy that relies on the strong
use of data augmentation to use the available annotated samples more
efficiently. The architecture consists of a contracting path to capture
context and a symmetric expanding path that enables precise localization.
We show that such a network can be trained end-to-end from very
few images and outperforms the prior best method (a sliding-window
convolutional network) on the ISBI challenge for segmentation of neuronal
structures in electron microscopic stacks. Using the same network
trained on transmitted light microscopy images (phase contrast
and DIC) we won the ISBI cell tracking challenge 2015 in these categories
by a large margin. Moreover, the network is fast. Segmentation
of a 512x512 image takes less than a second on a recent GPU. The full
implementation (based on Caffe) and the trained networks are available
at http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net
.


## Results
* Comming soon

## Installation tf_base package
1. Clone the repository
```
$ git clone https://github.com/Shumway82/tf_base.git
```
2. Go to folder
```
$ cd tf_base
```
3. Install with pip3
``` 
$ pip3 install -e .
```

## Install U-Net-Segmentation package

1. Clone the repository
```
$ https://github.com/Shumway82/U-Net-Segmentation.git
```
2. Go to folder
```
$ cd Binary-Classification
```
3. Install with pip3
```
$ pip3 install -e .
```

## Usage-Example

1. Training
```
$ python pipeline_trainer.py --dataset "../Data/" --loss "dice"
```

2. Inferencing
```
$ python pipeline_inferencer.py --dataset "../Data/" --model_dir "Models_dice" 
```
