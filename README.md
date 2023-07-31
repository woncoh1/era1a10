# ResNet and one-cycle policy
> Train a ResNet with CIFAR-10 using one-cycle policy
- Dataset: CIFAR-10
- Model architecture: ResNet
- Learning-rate scheduler: one-cycle policy

## Installation
If you want to use our models, dataloaders, training engine, and other utilities, please run the following command:
```console
git clone https://github.com/woncoh1/era1a10.git
```
And then import the modules in Python:
```python
from era1a10 import data, models, engine, utils
```

## Targets
Acheive all of the followings using modular code organization:
- Test accuracy > 90.0 %
- Number of parameters: unlimited
- Number of epochs: 24

## Results
- Best train accuracy = 97.97 %
- Best test accuracy = 93.26 %
- Number of parameters = 6,573,120
- Number of epochs = 24

## Sample images
![image](https://github.com/woncoh1/era1a10/assets/12987758/df971189-f26e-4a5e-a876-24b20f70acdc)

## Model summary
```
==========================================================================================
Layer (type (var_name))                  Output Shape              Param #
==========================================================================================
PageNet (PageNet)                        [512, 10]                 --
├─Sequential (conv0)                     [512, 64, 32, 32]         --
│    └─Conv2d (0)                        [512, 64, 32, 32]         1,728
│    └─BatchNorm2d (1)                   [512, 64, 32, 32]         128
│    └─Dropout (2)                       [512, 64, 32, 32]         --
│    └─ReLU (3)                          [512, 64, 32, 32]         --
├─SkipBlock (conv1)                      [512, 128, 16, 16]        --
│    └─Sequential (conv)                 [512, 128, 16, 16]        --
│    │    └─Conv2d (0)                   [512, 128, 32, 32]        73,728
│    │    └─MaxPool2d (1)                [512, 128, 16, 16]        --
│    │    └─BatchNorm2d (2)              [512, 128, 16, 16]        256
│    │    └─Dropout (3)                  [512, 128, 16, 16]        --
│    │    └─ReLU (4)                     [512, 128, 16, 16]        --
│    └─Sequential (res)                  [512, 128, 16, 16]        --
│    │    └─Conv2d (0)                   [512, 128, 16, 16]        147,456
│    │    └─BatchNorm2d (1)              [512, 128, 16, 16]        256
│    │    └─Dropout (2)                  [512, 128, 16, 16]        --
│    │    └─ReLU (3)                     [512, 128, 16, 16]        --
│    │    └─Conv2d (4)                   [512, 128, 16, 16]        147,456
│    │    └─BatchNorm2d (5)              [512, 128, 16, 16]        256
│    │    └─Dropout (6)                  [512, 128, 16, 16]        --
│    │    └─ReLU (7)                     [512, 128, 16, 16]        --
├─Sequential (conv2)                     [512, 256, 8, 8]          --
│    └─Conv2d (0)                        [512, 256, 16, 16]        294,912
│    └─MaxPool2d (1)                     [512, 256, 8, 8]          --
│    └─BatchNorm2d (2)                   [512, 256, 8, 8]          512
│    └─Dropout (3)                       [512, 256, 8, 8]          --
│    └─ReLU (4)                          [512, 256, 8, 8]          --
├─SkipBlock (conv3)                      [512, 512, 4, 4]          --
│    └─Sequential (conv)                 [512, 512, 4, 4]          --
│    │    └─Conv2d (0)                   [512, 512, 8, 8]          1,179,648
│    │    └─MaxPool2d (1)                [512, 512, 4, 4]          --
│    │    └─BatchNorm2d (2)              [512, 512, 4, 4]          1,024
│    │    └─Dropout (3)                  [512, 512, 4, 4]          --
│    │    └─ReLU (4)                     [512, 512, 4, 4]          --
│    └─Sequential (res)                  [512, 512, 4, 4]          --
│    │    └─Conv2d (0)                   [512, 512, 4, 4]          2,359,296
│    │    └─BatchNorm2d (1)              [512, 512, 4, 4]          1,024
│    │    └─Dropout (2)                  [512, 512, 4, 4]          --
│    │    └─ReLU (3)                     [512, 512, 4, 4]          --
│    │    └─Conv2d (4)                   [512, 512, 4, 4]          2,359,296
│    │    └─BatchNorm2d (5)              [512, 512, 4, 4]          1,024
│    │    └─Dropout (6)                  [512, 512, 4, 4]          --
│    │    └─ReLU (7)                     [512, 512, 4, 4]          --
├─Sequential (trans)                     [512, 10]                 --
│    └─MaxPool2d (0)                     [512, 512, 1, 1]          --
│    └─Conv2d (1)                        [512, 10, 1, 1]           5,120
│    └─Flatten (2)                       [512, 10]                 --
==========================================================================================
Total params: 6,573,120
Trainable params: 6,573,120
Non-trainable params: 0
Total mult-adds (G): 194.18
==========================================================================================
Input size (MB): 6.29
Forward/backward pass size (MB): 2382.41
Params size (MB): 26.29
Estimated Total Size (MB): 2414.99
==========================================================================================
```

## Training log
```
  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00304, Accuracy = 45.46%, Epoch = 1
Test : Loss = 0.00248, Accuracy = 56.61%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00189, Accuracy = 65.79%, Epoch = 2
Test : Loss = 0.00179, Accuracy = 69.52%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00147, Accuracy = 73.73%, Epoch = 3
Test : Loss = 0.00150, Accuracy = 74.20%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00124, Accuracy = 78.04%, Epoch = 4
Test : Loss = 0.00171, Accuracy = 70.91%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00114, Accuracy = 79.78%, Epoch = 5
Test : Loss = 0.00140, Accuracy = 78.20%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00096, Accuracy = 83.00%, Epoch = 6
Test : Loss = 0.00103, Accuracy = 82.67%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00082, Accuracy = 85.63%, Epoch = 7
Test : Loss = 0.00094, Accuracy = 84.46%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00071, Accuracy = 87.50%, Epoch = 8
Test : Loss = 0.00084, Accuracy = 86.08%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00066, Accuracy = 88.20%, Epoch = 9
Test : Loss = 0.00082, Accuracy = 86.58%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00059, Accuracy = 89.38%, Epoch = 10
Test : Loss = 0.00085, Accuracy = 86.12%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00053, Accuracy = 90.48%, Epoch = 11
Test : Loss = 0.00065, Accuracy = 89.18%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00048, Accuracy = 91.47%, Epoch = 12
Test : Loss = 0.00068, Accuracy = 88.69%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00043, Accuracy = 92.36%, Epoch = 13
Test : Loss = 0.00060, Accuracy = 90.00%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00039, Accuracy = 93.12%, Epoch = 14
Test : Loss = 0.00060, Accuracy = 90.28%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00037, Accuracy = 93.39%, Epoch = 15
Test : Loss = 0.00056, Accuracy = 91.01%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00033, Accuracy = 94.15%, Epoch = 16
Test : Loss = 0.00056, Accuracy = 90.92%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00028, Accuracy = 94.90%, Epoch = 17
Test : Loss = 0.00055, Accuracy = 91.16%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00026, Accuracy = 95.42%, Epoch = 18
Test : Loss = 0.00051, Accuracy = 91.91%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00023, Accuracy = 96.03%, Epoch = 19
Test : Loss = 0.00054, Accuracy = 91.40%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00020, Accuracy = 96.39%, Epoch = 20
Test : Loss = 0.00049, Accuracy = 92.24%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00018, Accuracy = 96.85%, Epoch = 21
Test : Loss = 0.00049, Accuracy = 92.30%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00016, Accuracy = 97.29%, Epoch = 22
Test : Loss = 0.00047, Accuracy = 92.83%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00014, Accuracy = 97.64%, Epoch = 23
Test : Loss = 0.00045, Accuracy = 93.13%

  0%|          | 0/98 [00:00<?, ?it/s]
Train: Loss = 0.00012, Accuracy = 97.97%, Epoch = 24
Test : Loss = 0.00044, Accuracy = 93.26%
```

## Learning curves
![image](https://github.com/woncoh1/era1a10/assets/12987758/e762d60e-d557-4dd2-8c98-3c09ad9169e0)

## Misclassified predictions
![image](https://github.com/woncoh1/era1a10/assets/12987758/6a22f5b4-d9b7-444c-ab66-ebd456d47c53)

## References
- https://github.com/davidcpage/cifar10-fast
- https://www.learnpytorch.io/05_pytorch_going_modular/

## TODO
- [ ] README generator: nbdev-style automatic generation of README.md from index.ipynb
- [ ] Plug-in modules: easily assemble different combinations of components
- [ ] Real-time LR viewer: print learning rate per batch or epoch
- [ ] Hyperparameter scanner: search for new hyperparameter candidates among function parameters
- [ ] [einops](https://github.com/arogozhnikov/einops)
