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
Training without reconstruction network by setting lam_recon=0.   

`$ python CapsNet.py --lam_recon 0.`

Other parameters include `batch_size, epochs, lam_recon, num_routing, shift_fraction, save_dir` can 
passed to the function in the same way. Please refer to CapsNet.py

## Results

Accuracy with data augmentation (shift at most 2 pixels in each direction):     

   Epoch     |   1   |   5  |  10  |  15  |  20   
   :---------|:------:|:---:|:----:|:----:|:------:
   train_acc |  91.0 | 99.0 | 99.4 | 99.6 |  99.7 
   vali_acc  |  98.7 | 99.24| 99.31| 99.39|  99.52
   
   
Accuracy without data augmentation:   

   Epoch     |   1   |   2  |   5   |  10  | 11
   :---------|:------:|:---:|:------:|:---: |:---:  
   train_acc |  94.8 | 98.9 |  99.6 | 99.9 | 99.95
   vali_acc  |  98.7 | 99.0 | 99.39 | 99.38| 99.45

Every epoch consumes about 260s on a single GTX 1070 GPU.   
Maybe there're some problems in my implementation. Contributions are welcome.

## TODO: 
- Optimize the code implementation and comments. 
The paper says the CapsNet has 11M parameters, but my model only has 8M. 
There may be something wrong.

## Other Implementations

- [CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow.git): 
Very good implementation. I referred to this repository in my code.

- [CapsNet-PyTorch](https://github.com/nishnik/CapsNet-PyTorch.git)

- [capsnet.pytorch](https://github.com/andreaazzini/capsnet.pytorch.git)