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
## Results

Accuracy during training with reconstruction coefficient lam_recon=0.0005:   

   Epoch     |   1   |   2  |  3   |  4   |   5   |   6  |  7   |  8  |  9   |  10  | 11
   :---------|:------:|:---:|:----:|:----:|:------:|:---:|:----:|:---:|:----:|:---: |:---:  
   train_acc |  94.8 | 98.9 | 99.3 | 99.4 |  99.6 | 99.7 | 99.8 | 99.8| 99.9 | 99.9 | 99.95
   vali_acc  |  98.7 | 99.0 | 99.2 | 99.2 |  99.39| 99.30| 99.38| 99.35|99.46 |99.38| 99.45

Accuracy during training without reconstruction:   

   Epoch     |   1   |   2  |  3   |  4   |   5   |   6  |  7   |  8  |  9   |  10  | 11
   :---------|:------:|:---:|:----:|:----:|:------:|:---:|:----:|:---:|:----:|:---: |:---:  
   train_acc |  94.4 | 98.9 | 99.2 | 99.5 |  99.6 | 99.7 | 99.8 | 99.9| 99.9 | 99.9 | 99.95
   vali_acc  |  98.5 | 99.1 | 99.2 | 99.3 |  99.36| 99.36| 99.43| 99.34|99.42 |99.41| 99.44
   
Every epoch consumes more than 300s on a single GTX 1070 GPU.   
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