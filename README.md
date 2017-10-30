# CapsNet-Keras

A Keras implementation of CapsNet in Hinton's paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)


## Requirements
- [Keras](https://github.com/fchollet/keras) 

## Usage

### Training
**Step 1.**
Install Keras:

`$ pip install keras`

**Step 2.** 
Clone this repository with ``git``.

```
$ git clone https://github.com/xifengguo/CapsNet-Keras.git
$ cd CapsNet-Keras
```

**Step 3.** 
Training:
```
$ python CapsNet.py
```

## Results
 
Accuracy during training:   

   Epoch     |   1   |   2  |  3   |  4   |   5   |   6  |  7   |  8  |  9   |  10  | 11
   :---------|:------:|:---:|:----:|:----:|:------:|:---:|:----:|:---:|:----:|:---: |:---:  
   train_acc |  94.4 | 98.9 | 99.2 | 99.5 |  99.6 | 99.7 | 99.8 | 99.9| 99.9 | 99.9 | 99.95
   vali_acc  |  98.5 | 99.1 | 99.2 | 99.3 |  99.36| 99.36| 99.43| 99.34|99.42 |99.41| 99.44
   
Every epoch consumes more than 300s on a single GTX 1070 GPU.   
Maybe there're some problems in my implementation. Contributions are welcome.

## TODO:
- Add reconstruction part, which should be easy. 
- Optimize the code implementation and comments

## Other Implementations

- [CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow.git): 
Very good implementation. I referred to this repository in my code.

- [CapsNet-PyTorch](https://github.com/nishnik/CapsNet-PyTorch.git)

- [capsnet.pytorch](https://github.com/andreaazzini/capsnet.pytorch.git)