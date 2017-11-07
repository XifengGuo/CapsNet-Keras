# CapsNet by tf.keras
[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/XifengGuo/CapsNet-Keras/blob/master/LICENSE)

A tf.keras implementation of CapsNet in the paper:   
[Sara Sabour, Nicholas Frosst, Geoffrey E Hinton. Dynamic Routing Between Capsules. NIPS 2017](https://arxiv.org/abs/1710.09829)   
This repo will show how to use keras in TensorFlow v1.4.

**Differences with the paper:**   
- We use an additional learning rate decay with `decay factor = 0.9` and `step = 1 epoch`,    
while the paper didn't use (probably?)   
- We only report the test errors after `30 epochs` training (still under-fitting).   
In the paper, I suppose they trained for `1250 epochs` according to Figure A.1?
- We use MSE (mean squared error) as the reconstruction loss and 
the coefficient for the loss is `lam_recon=0.0005*784=0.392`.   
This should be **equivalent** with using SSE (sum squared error) and `lam_recon=0.0005` as in the paper.


**Contacts**
- Your contributions to the repo are always welcome. 
Open an issue or contact me with E-mail `guoxifeng1990@163.com` or WeChat `wenlong-guo`.


## Usage

**Step 1.
Install [TensorFlow v1.4](https://github.com/tensorflow/tensorflow).**
```
pip install tensorflow-gpu
```

**Step 2. Clone this repository to local.**
```
git clone -b tf1.4-keras https://github.com/XifengGuo/CapsNet-Keras.git
cd CapsNet-Keras
```

**Step 3. Train a CapsNet on MNIST**  

Training with the default setting:
```
$ python capsulenet.py
```
Training with one routing iteration (default 3).   
```
$ python capsulenet.py --num_routing 1
```

Other parameters include `batch_size, epochs, lam_recon, shift_fraction, save_dir` can be
passed to the function in the same way. Please refer to `capsulenet.py`

**Step 4. Test a pre-trained CapsNet model**

Suppose you have trained a model using the above command, then the trained model will be
saved to `result/trained_model.h5`. Now just launch the following command to get test results.
```
$ python capsulenet.py --is_training 0 --weights result/trained_model.h5
```
It will output the testing accuracy and show the reconstructed images.

## Results

**Test Errors**   

CapsNet classification test **error** on MNIST. Average and standard deviation results are
reported by 3 trials. The results can be reproduced by launching the following commands.   
 ```
 python capsulenet.py --num_routing 1 --lam_recon 0.0    #CapsNet-v1   
 python capsulenet.py --num_routing 1 --lam_recon 0.392  #CapsNet-v2
 python capsulenet.py --num_routing 3 --lam_recon 0.0    #CapsNet-v3 
 python capsulenet.py --num_routing 3 --lam_recon 0.392  #CapsNet-v4
```
   Method     |   Routing   |   Reconstruction  |  MNIST (%)  |  *Paper*    
   :---------|:------:|:---:|:----:|:----:
   Baseline |  -- | -- | --             | *0.39* 
   CapsNet-v1 |  1 | no | 0.39 (0.024)  | *0.34 (0.032)* 
   CapsNet-v2  |  1 | yes | 0.37 (0.022)| *0.29 (0.011)*
   CapsNet-v3 |  3 | no | 0.40 (0.016)  | *0.35 (0.036)*
   CapsNet-v4  |  3 | yes| 0.34 (0.009) | *0.25 (0.005)*
   
Losses and accuracies:   
![](result/log.png)


**Training Speed**  

About `110s / epoch` on a single GTX 1070 GPU.   


**Reconstruction result**  

The result of CapsNet-v4 by launching   
```
python capsulenet.py --is_training 0 --weights result/trained_model.h5
```
Digits at top 5 rows are real images from MNIST and 
digits at bottom are corresponding reconstructed images.

![](real_and_recon.png)
