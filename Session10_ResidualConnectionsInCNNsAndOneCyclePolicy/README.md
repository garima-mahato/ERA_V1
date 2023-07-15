# Session 10 - Residual Connections in CNNs and One Cycle Policy!

## Assignment

|Name|Code Link |
|---|---|
|Execution Jupyter Notebook|[Open](https://github.com/garima-mahato/ERA_V1/blob/main/Session10_ResidualConnectionsInCNNsAndOneCyclePolicy/ERA_V1_Session10_ResidualConnectionsInCNNsAndOneCyclePolicy_with_Adam.ipynb)|
| Data Loader Code | [Open](https://github.com/garima-mahato/ERA_V1_API/blob/main/data_engine/data_loader.py) |
| Data Augmenter Code | [Open](https://github.com/garima-mahato/ERA_V1_API/blob/main/data_engine/data_augmenter.py) |
| Model Code | [Open](https://github.com/garima-mahato/ERA_V1_API/blob/main/models/custom_resnet.py) |
| All Code | [Open](https://github.com/garima-mahato/ERA_V1_API) |

#### Results: 
  - Best Test Accuracy - 90.31%
  - Test Accuracy - 90.31%
  - Total Parameters - 6,573,120
  - Number of Epochs - 24

### CIFAR10 Dataset

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session10_ResidualConnectionsInCNNsAndOneCyclePolicy/asset/cifar10)

### Model Architecture

```
Model Summary:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4          [-1, 128, 32, 32]          73,728
         MaxPool2d-5          [-1, 128, 16, 16]               0
       BatchNorm2d-6          [-1, 128, 16, 16]             256
              ReLU-7          [-1, 128, 16, 16]               0
            Conv2d-8          [-1, 128, 16, 16]         147,456
       BatchNorm2d-9          [-1, 128, 16, 16]             256
             ReLU-10          [-1, 128, 16, 16]               0
           Conv2d-11          [-1, 128, 16, 16]         147,456
      BatchNorm2d-12          [-1, 128, 16, 16]             256
             ReLU-13          [-1, 128, 16, 16]               0
         ResBlock-14          [-1, 128, 16, 16]               0
           Conv2d-15          [-1, 256, 16, 16]         294,912
        MaxPool2d-16            [-1, 256, 8, 8]               0
      BatchNorm2d-17            [-1, 256, 8, 8]             512
             ReLU-18            [-1, 256, 8, 8]               0
           Conv2d-19            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-20            [-1, 512, 4, 4]               0
      BatchNorm2d-21            [-1, 512, 4, 4]           1,024
             ReLU-22            [-1, 512, 4, 4]               0
           Conv2d-23            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-24            [-1, 512, 4, 4]           1,024
             ReLU-25            [-1, 512, 4, 4]               0
           Conv2d-26            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-27            [-1, 512, 4, 4]           1,024
             ReLU-28            [-1, 512, 4, 4]               0
         ResBlock-29            [-1, 512, 4, 4]               0
        MaxPool2d-30            [-1, 512, 1, 1]               0
           Conv2d-31             [-1, 10, 1, 1]           5,120
================================================================
Total params: 6,573,120
Trainable params: 6,573,120
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.75
Params size (MB): 25.07
Estimated Total Size (MB): 31.84
----------------------------------------------------------------

```

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session10_ResidualConnectionsInCNNsAndOneCyclePolicy/asset/cifar10_s10_custom_resnet_torchviz.png)

---

#### <b>LR Finder </b>

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session10_ResidualConnectionsInCNNsAndOneCyclePolicy/asset/lr_finder.JPG)

##### <b>Train/Test Logs</b>

```
EPOCH: 1
Loss=2.0574100017547607 Batch_id=97 Accuracy=27.34: 100%
98/98 [00:24<00:00, 5.40it/s]

Test set: Average loss: 0.0041, Accuracy: 4139/10000 (41.39%)

EPOCH: 2
Loss=1.9861373901367188 Batch_id=97 Accuracy=46.32: 100%
98/98 [00:25<00:00, 5.68it/s]

Test set: Average loss: 0.0041, Accuracy: 4235/10000 (42.35%)

EPOCH: 3
Loss=1.9027334451675415 Batch_id=97 Accuracy=53.87: 100%
98/98 [00:25<00:00, 5.52it/s]

Test set: Average loss: 0.0038, Accuracy: 5574/10000 (55.74%)

EPOCH: 4
Loss=1.7896006107330322 Batch_id=97 Accuracy=63.75: 100%
98/98 [00:23<00:00, 5.61it/s]

Test set: Average loss: 0.0037, Accuracy: 6249/10000 (62.49%)

EPOCH: 5
Loss=1.739487648010254 Batch_id=97 Accuracy=70.04: 100%
98/98 [00:24<00:00, 5.14it/s]

Test set: Average loss: 0.0036, Accuracy: 6684/10000 (66.84%)

EPOCH: 6
Loss=1.6807174682617188 Batch_id=97 Accuracy=76.09: 100%
98/98 [00:24<00:00, 4.99it/s]

Test set: Average loss: 0.0035, Accuracy: 7300/10000 (73.00%)

EPOCH: 7
Loss=1.6618409156799316 Batch_id=97 Accuracy=80.04: 100%
98/98 [00:24<00:00, 5.08it/s]

Test set: Average loss: 0.0034, Accuracy: 7648/10000 (76.48%)

EPOCH: 8
Loss=1.6446408033370972 Batch_id=97 Accuracy=82.35: 100%
98/98 [00:24<00:00, 5.49it/s]

Test set: Average loss: 0.0033, Accuracy: 8099/10000 (80.99%)

EPOCH: 9
Loss=1.628773808479309 Batch_id=97 Accuracy=84.43: 100%
98/98 [00:25<00:00, 5.62it/s]

Test set: Average loss: 0.0033, Accuracy: 8082/10000 (80.82%)

EPOCH: 10
Loss=1.638575792312622 Batch_id=97 Accuracy=85.76: 100%
98/98 [00:24<00:00, 5.60it/s]

Test set: Average loss: 0.0033, Accuracy: 8265/10000 (82.65%)

EPOCH: 11
Loss=1.6079195737838745 Batch_id=97 Accuracy=87.02: 100%
98/98 [00:25<00:00, 5.70it/s]

Test set: Average loss: 0.0033, Accuracy: 8223/10000 (82.23%)

EPOCH: 12
Loss=1.5898020267486572 Batch_id=97 Accuracy=88.03: 100%
98/98 [00:24<00:00, 5.59it/s]

Test set: Average loss: 0.0033, Accuracy: 8307/10000 (83.07%)

EPOCH: 13
Loss=1.5813566446304321 Batch_id=97 Accuracy=89.01: 100%
98/98 [00:24<00:00, 5.60it/s]

Test set: Average loss: 0.0033, Accuracy: 8418/10000 (84.18%)

EPOCH: 14
Loss=1.5644396543502808 Batch_id=97 Accuracy=89.84: 100%
98/98 [00:24<00:00, 5.55it/s]

Test set: Average loss: 0.0032, Accuracy: 8414/10000 (84.14%)

EPOCH: 15
Loss=1.5524119138717651 Batch_id=97 Accuracy=90.91: 100%
98/98 [00:24<00:00, 4.82it/s]

Test set: Average loss: 0.0032, Accuracy: 8650/10000 (86.50%)

EPOCH: 16
Loss=1.5569329261779785 Batch_id=97 Accuracy=91.59: 100%
98/98 [00:24<00:00, 4.46it/s]

Test set: Average loss: 0.0032, Accuracy: 8587/10000 (85.87%)

EPOCH: 17
Loss=1.539732575416565 Batch_id=97 Accuracy=92.61: 100%
98/98 [00:25<00:00, 5.02it/s]

Test set: Average loss: 0.0032, Accuracy: 8615/10000 (86.15%)

EPOCH: 18
Loss=1.552171230316162 Batch_id=97 Accuracy=92.98: 100%
98/98 [00:25<00:00, 5.49it/s]

Test set: Average loss: 0.0032, Accuracy: 8714/10000 (87.14%)

EPOCH: 19
Loss=1.5574381351470947 Batch_id=97 Accuracy=93.70: 100%
98/98 [00:25<00:00, 5.76it/s]

Test set: Average loss: 0.0032, Accuracy: 8781/10000 (87.81%)

EPOCH: 20
Loss=1.5199304819107056 Batch_id=97 Accuracy=94.36: 100%
98/98 [00:25<00:00, 5.66it/s]

Test set: Average loss: 0.0032, Accuracy: 8902/10000 (89.02%)

EPOCH: 21
Loss=1.5145732164382935 Batch_id=97 Accuracy=95.01: 100%
98/98 [00:25<00:00, 5.61it/s]

Test set: Average loss: 0.0031, Accuracy: 8946/10000 (89.46%)

EPOCH: 22
Loss=1.528245449066162 Batch_id=97 Accuracy=95.42: 100%
98/98 [00:25<00:00, 5.68it/s]

Test set: Average loss: 0.0031, Accuracy: 8965/10000 (89.65%)

EPOCH: 23
Loss=1.490099310874939 Batch_id=97 Accuracy=95.90: 100%
98/98 [00:23<00:00, 5.64it/s]

Test set: Average loss: 0.0031, Accuracy: 9020/10000 (90.20%)

EPOCH: 24
Loss=1.4918019771575928 Batch_id=97 Accuracy=96.07: 100%
98/98 [00:23<00:00, 5.20it/s]

Test set: Average loss: 0.0031, Accuracy: 9031/10000 (90.31%)

```

##### <b>Train/Test Visualization</b>

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session10_ResidualConnectionsInCNNsAndOneCyclePolicy/asset/train_test_diff_graphs.png)

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session10_ResidualConnectionsInCNNsAndOneCyclePolicy/asset/train_vs_test_acc_comparison_graph.png)

#### <b>Correctly Classified Images </b>

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session10_ResidualConnectionsInCNNsAndOneCyclePolicy/asset/correctly_classified_imgs.png)

<b>GRAD-CAM of correctly classified Images </b>

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session10_ResidualConnectionsInCNNsAndOneCyclePolicy/asset/gradcam_correct_0_bird.png)

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session10_ResidualConnectionsInCNNsAndOneCyclePolicy/asset/gradcam_correct_1_deer.png)

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session10_ResidualConnectionsInCNNsAndOneCyclePolicy/asset/gradcam_correct_2_airplane.png)

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session10_ResidualConnectionsInCNNsAndOneCyclePolicy/asset/gradcam_correct_3_bird.png)

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session10_ResidualConnectionsInCNNsAndOneCyclePolicy/asset/gradcam_correct_4_airplane.png)

#### <b>Mis-classified Images </b>

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session10_ResidualConnectionsInCNNsAndOneCyclePolicy/asset/misclassified_imgs.png)

<b>GRAD-CAM of mis-classified Images </b>

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session10_ResidualConnectionsInCNNsAndOneCyclePolicy/asset/gradcam_incorrect_0_bird.png)

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session10_ResidualConnectionsInCNNsAndOneCyclePolicy/asset/gradcam_incorrect_1_airplane.png)

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session10_ResidualConnectionsInCNNsAndOneCyclePolicy/asset/gradcam_incorrect_2_dog.png)

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session10_ResidualConnectionsInCNNsAndOneCyclePolicy/asset/gradcam_incorrect_3_cat.png)

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session10_ResidualConnectionsInCNNsAndOneCyclePolicy/asset/gradcam_incorrect_4_bird.png)