# Data Augmentation
This repository aims to show how small change in data augmentation and network structure makes difference in neural network. And the tests are all on CIFAR10 dataset.

## network structures

Again, the source file relies on an unpublished python library, so you can only look at the structure of the network. BTW, the FP net structure works quite good on SVHN dataset, feel free to try :) 

```
|- train_cifar10_allconv.py  # a slight modification from [1]
|- train_cifar10_batchnorm.py  # train batch normalization net
|- train_cifar10_fpnet.py      # train float point net
|- train_cifar10_fpwider.py    # train the wider version of float point net
|- train_cifar10_quick.py      # train cifar10 quick net [2]
```

| Structure | Data Augmentation | Result (Error Rate)|
| --- | --- | --- |
| cifar10 quick net | - | 23% |
| FP net | - | ~23% |
| FP net + Dropout | - | ~17% |
| FP net + Dropout | standardization | ~16% |
| FP net + Dropout + L2 penalization | - | ~18% |
| FP net + Dropout + L2 penalization | GCN + whiten | ~17% |
| FP net + Dropout | Flipped + GCN + whiten| ~13% |
| FP net + Dropout | Flipped + cropped + GCN + whiten | 12.154% |
| Batch normalization net | GCN + whiten | 11.17% |
| All convolutional net | GCN + whiten | 23% |
| FP wider net | GCN + whiten | ~17% |
| FP wider net | Flipped + GCN + whiten | ~14.4% |
| FP wider net + Dropout | GCN + whiten | ~14.3% |
| FP wider net + Dropout | Flipped + GCN + whiten | ~11.3% |

The idea of Dropout and the values comes from [here](https://github.com/nagadomi/kaggle-cifar10-torch7 ). The meaning of 'Flipped' and 'cropped' can be easily understood from the codes or many other articles.




## References

[1]. Springenberg, Jost Tobias, Alexey Dosovitskiy, Thomas Brox, and Martin Riedmiller. *Striving for simplicity: The all convolutional net.* arXiv preprint arXiv:1412.6806 (2014).

[2]. Jia, Yangqing, et al. *Caffe: Convolutional architecture for fast feature embedding.* Proceedings of the 22nd ACM international conference on Multimedia. ACM, 2014.




