# Neural Architecture Optimization
This is the Code for the Paper [Neural Architecture Optimization](https://arxiv.org/abs/1808.07233).

Authors: [Renqian Luo](http://home.ustc.edu.cn/~lrq)\*, [Fei Tian](https://ustctf.github.io/)\*, [Tao Qin](https://www.microsoft.com/en-us/research/people/taoqin/), [En-Hong Chen](http://staff.ustc.edu.cn/~cheneh/), [Tie-Yan Liu](https://www.microsoft.com/en-us/research/people/tyliu/). *=equal contribution

## License
The codes and models in this repo are released under the GNU GPLv3 license.

## Citation
If you find this work helpful in your research, please use the following BibTex entry to cite our paper.
```
@inproceedings{NAO,
  title={Neural Architecture Optimization},
  author={Renqian Luo and Fei Tian and Tao Qin and En-Hong Chen and Tie-Yan Liu},
  booktitle={Advances in neural information processing systems},
  year={2018}
}

```

_This is not an official Microsoft product._


## Requirment and Dependency
Tensorflow >= 1.4.0

Pytorch == 0.3.1

## CIFAR-10

### With Weight Sharing
#### To Search Architectures 
To search the CNN architectures for CIFAR-10 with weight sharing, please refer to:

| Script | Data| GPU | Search Time |
| ------------- | ------------- | ------------- | ------------- |
| ./NAO-WS/cnn/train_search.sh | [`Google Drive`](https://drive.google.com/open?id=11BJbR_qvvRKtaCzCBH5gMCXRWJb6m3ct) [`Baidu Pan`](https://pan.baidu.com/s/1EMZZNzdyovOW93ghonkHOA) | 1 V100 | 7.5 hours | 

```
cd NAO-WS/cnn
bash train_search.sh
```

Once the search is done, the final pool of architectures will be in ```models/child/arch_pool```. You can choose top-5 architectures to run them using ```train_final.sh``` and pass in the arch by setting the ```fixed_arc``` argument.

To obtain the best architecture, we perform grid search on the hyper-parameters for the top-5 architectures discovered.

#### To Train Discovered Architectures
To train a fixed CNN architecture, for example, our best architecture discovered, please refer to:

| Script | GPU | Time | Model Checkpoint | Parameter Size | Error Rate |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| ./NAO-WS/cnn/train_final.sh | 1 P40 | 42 hours | [`Google Drive`](https://drive.google.com/open?id=1JpQelT03YsxHxraT7Iskg0J6wXsPiPXI) [`Baidu Pan`](https://pan.baidu.com/s/1q6UIRnjm1eFizsyaCA-m_Q)| 2.5M | 3.50 |

and run:

```
cd NAO-WS/cnn
bash train_final.sh
```

If you want to run it with cutout, add ```--child_cutout_size=16``` in the script.

#### To Directly Evaluate an Architecture
To directly evaluate an architecture, for example, our best architecture discovered, please download the checkpoint above, move all the files to ```NAO-WS/cnn/models``` folder and run:

```
cd NAO-WS/cnn
bash test_final.sh    #This should give you an accuracy of 96.50% (error rate of 3.50%) without cutout
```

### Without Weight Sharing
#### To Search Architectures
Please refer to details in ```./NAO/README.md```

#### To Train Discovered Architectures
Please download data at [`Google Drive`](https://drive.google.com/open?id=1XcC_cycn1Dog4s_Bki8TV9XZYc1Ast3u) [`Baidu Pan`](https://pan.baidu.com/s/1VS2_K3nAzWZh-JIwVmNyCg)

You can train the best architecture discovered (show in Fig. 1 in the Appendix of the paper) using:

| Dataset | Script | GPU | Time | Checkpoint| Error Rate (Test)|
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|CIFAR-10| ./NAO/cnn/train_cifar10_final.sh | 2 P40 | 5 days | [`Google Drive`](https://drive.google.com/open?id=1TPgAZB7ZXAxaYmTj8efriJ6IbmSgMJKX) [`Baidu Pan`](https://pan.baidu.com/s/1r8nQIRE7F4jBTEKKqyaZuA)| 2.10% |
|CIFAR-100| ./NAO/cnn/train_cifar100_final.sh | 2 P40 | 5 days | [`Google Drive`](https://drive.google.com/open?id=15eDukFiGoGmqLbZAES826eFem99V_2bI) [`Baidu Pan`](https://pan.baidu.com/s/1r8nQIRE7F4jBTEKKqyaZuA)| 14.80% |

by running:
```
cd NAO/cnn
bash train_cifar10_final.sh
bash train_cifar100_final.sh
```

#### To Directly Evaluate an Architecturethe 
To directly evaluate an architecture, for example, our best architecture discovered, please download the checkpoint above, move all the files to ```NAO/cnn/models/cifar10 or NAO/cnn/models/cifar100/ ```, and run:
```
cd NAO/cnn
bash test_cifar10.sh     #This should give you an accuracy of 97.94% (error rate of 2.06%)
bash test_cifar100.sh    #This should give you an accuracy of 85.20% (error rate of 14.81%)
```

## PTB
#### To Search Architectures 
To search the RNN architectures for PTB with weight sharing, please refer to:

| Script | GPU | Search Time |
| ------------- | ------------- | ------------- |
| ./NAO-WS/rnn/train_search.sh | 1 V100 | 8 hours | 

```
cd NAO-WS/rnn
bash train_search.sh
```

Once the search is done, the final pool of architectures will be in ```models/child/arch_pool```. You can choose top-10 architectures to run them using ```train_final.sh``` and pass in the arch by setting the ```arch``` argument.

#### To Train Discovered Architectures
To train a fixed RNN architecture, for example, our best architecture discovered, please refer to:

| Script | Model Checkpoint | GPU | Time | PPL (Test) | 
| ------------- | ------------- | ------------- | ------------- | ------------- |
| ./NAO-WS/rnn/train_final.sh | [`Google Drive`](https://drive.google.com/open?id=1yMOSDR_Aq2kLLP7c5q5eJw5v9OyGsd_P) [`Baidu Pan`](https://pan.baidu.com/s/1r8nQIRE7F4jBTEKKqyaZuA)|1 V100 | 4 days | 56.80 |

```
cd NAO-WS/rnn
bash train_final.sh   #This should give you a test ppl of 56.66 at the end of training
```
#### To Directly Evaluate an Architecture
To directly evaluate an architecture, for example, our best architecture discovered, please download the checkpoint above, move all the files to ```./NAO/rnn/models``` folder and run:

```
cd NAO-WS/cnn
bash test_final.sh    #This should give you a test ppl of 56.66
```

### Without Weight Sharing
#### To Search Architectures
Please refer to details in ```NAO/README.md```

#### To Train Discovered Architectures
You can train the best architecture discovered (showin in Fig. 2 in the Appendix of the paper) using:

| Dataset | Script | GPU | Time | Checkpoint| PPL (Test)|
| ------------- | ------------- | ------------- | ------------- | ------------- |------------- |
|PTB| ./NAO/rnn/train_ptb_final.sh | 1 V100 | 4 days | [`Google Drive`](https://drive.google.com/open?id=1o8Nq890szQwlMZDHwzcGhZ3BsnH_sGvT) [`Baidu Pan`](https://pan.baidu.com/s/1jnjkyLylX1LqiD9m98vRqw)| 56.02 |
|WikiText-2| ./NAO/rnn/train_wt2_final.sh | 1 V100 | 4 days | [`Google Drive`](https://drive.google.com/open?id=1N0BbsJPJo02pE2ILfAi_RtLPOJJwBxfu) [`Baidu Pan`](https://pan.baidu.com/s/1jnjkyLylX1LqiD9m98vRqw)| 67.10 |

#### To Directly Evaluate an Architecture
To directly evaluate an architecture, for example, our best architecture discovered, please download the checkpoint above, move all the files to ```NAO/rnn/models/ptb or NAO/rnn/models/wt2 ```, and run:
```
cd NAO/rnn
bash test_ptb.sh    #This should give you a test ppl of 56.02
bash test_wt2.sh    #This should give you a test ppl of 67.10
```

## Acknowledgements
We thank Hieu Pham for the discussion on some details of [`ENAS`](https://github.com/melodyguan/enas) implementation, and Hanxiao Liu for the code base of language modeling task in [`DARTS`](https://github.com/quark0/darts) . We furthermore thank the anonymous reviewers for their constructive comments.
