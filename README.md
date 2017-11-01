# CapsNet-Keras

A Keras implementation of CapsNet in Hinton's paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

**Recent updates:**
- Accelerate the code by optimizing Primary Capsule. Now it takes 110s per epoch.

**TODO**
- I will check the logic carefully in routing algorithm.
- The model has 8M parameters, while the paper said it should be 11M.
I'll figure out what's the problem.

**Contribution**
- Your contribution to the repo is welcome. Open an issue or contact me with 
`guoxifeng1990@163.com` or WeChat (微信号) `wenlong-guo`.

## Requirements
- [Keras](https://github.com/fchollet/keras) 
- matplotlib

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
Training without routing by setting num_routing=0.   

`$ python CapsNet.py --num_routing 0`

Other parameters include `batch_size, epochs, lam_recon, shift_fraction, save_dir` can 
passed to the function in the same way. Please refer to CapsNet.py

## Results

Main result by launching `python CapsNet.py`:

   Epoch     |   1   |   5  |  10  |  20    
   :---------|:------:|:---:|:----:|:----:
   train_acc |  83.4 | 98.2 | 98.7 | 99.1 
   vali_acc  |  97.8 | 98.9 | 99.0 | 99.2 
  
Losses and accuracies:   
![](result/log.png)


**Results without routing algorithm**   
by launching `python CapsNet.py --num_routing 0`   
The prior b is fixed to 0. But, interestingly, the model still gets good results, even better than using routing.  

   Epoch     |   1   |   2  |  5  |  10  |  20   
   :---------|:------:|:---:|:----:|:----:|:------:
   train_acc |  90.3 | 98.2 | 99.0 | 99.4 |  99.7 
   vali_acc  |  98.8 | 99.1 | 99.36| 99.48|  99.53
   

Every epoch consumes about `110s` on a single GTX 1070 GPU.   

## Other Implementations
- TensorFlow:
  - [CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow.git): 
Very good implementation. I referred to this repository in my code.
  - [CapsNet-tensorflow](https://github.com/InnerPeace-Wu/CapsNet-tensorflow)

- PyTorch:
  - [CapsNet-PyTorch](https://github.com/nishnik/CapsNet-PyTorch.git)
  - [capsnet.pytorch](https://github.com/andreaazzini/capsnet.pytorch.git)
  - [CapsNet](https://github.com/leftthomas/CapsNet)
  
- MXNet:
  - [CapsNet_Mxnet](https://github.com/AaronLeong/CapsNet_Mxnet)
  
- Chainer:
  - [dynamic_routing_between_capsules](https://github.com/soskek/dynamic_routing_between_capsules)