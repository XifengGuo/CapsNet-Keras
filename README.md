# CapsNet-Keras

A Keras implementation of CapsNet in Hinton's paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

Recent updates:

- Accelerate the code by optimizing Primary Capsule. Now it takes 110s per epoch.

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
Training without reconstruction network by setting lam_recon=0.   

`$ python CapsNet.py --lam_recon 0.`

Other parameters include `batch_size, epochs, lam_recon, num_routing, shift_fraction, save_dir` can 
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


## TODO: 
- Optimize the code implementation and comments. 
The paper says the CapsNet has 11M parameters, but my model only has 8M. 
There may be something wrong.

## Other Implementations

- [CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow.git): 
Very good implementation. I referred to this repository in my code.

- [CapsNet-PyTorch](https://github.com/nishnik/CapsNet-PyTorch.git)

- [capsnet.pytorch](https://github.com/andreaazzini/capsnet.pytorch.git)