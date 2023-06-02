<b>[Back to Main Page](https://garima-mahato.github.io/ERA_V1/)</b>

# MNIST Classifier

|Code Link |
|---|
|[ Google Colab](https://githubtocolab.com/garima-mahato/ERA_V1/blob/main/Session5_IntroductionToPyTorch/images/S5.ipynb)|

### Data Description

![Sample Data](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session5_IntroductionToPyTorch/images/data_sample.png)

---
### Model Description

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
================================================================
Total params: 593,200
Trainable params: 593,200
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.94
----------------------------------------------------------------
```

<b>Model Visualization</b>

![Model Visualization](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session5_IntroductionToPyTorch/images/mnist_cnn_torchviz.png)

---
### Training & Testing

#### Logs

```
Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 1
Train: Loss=0.4035 Batch_id=117 Accuracy=43.25: 100%|██████████| 118/118 [00:34<00:00,  3.40it/s]
Test set: Average loss: 0.2886, Accuracy: 9154/10000 (91.54%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 2
Train: Loss=0.1115 Batch_id=117 Accuracy=92.75: 100%|██████████| 118/118 [00:27<00:00,  4.33it/s]
Test set: Average loss: 0.1006, Accuracy: 9697/10000 (96.97%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 3
Train: Loss=0.0739 Batch_id=117 Accuracy=95.98: 100%|██████████| 118/118 [00:27<00:00,  4.36it/s]
Test set: Average loss: 0.0644, Accuracy: 9814/10000 (98.14%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 4
Train: Loss=0.1233 Batch_id=117 Accuracy=96.94: 100%|██████████| 118/118 [00:27<00:00,  4.33it/s]
Test set: Average loss: 0.0518, Accuracy: 9836/10000 (98.36%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 5
Train: Loss=0.0437 Batch_id=117 Accuracy=97.28: 100%|██████████| 118/118 [00:27<00:00,  4.25it/s]
Test set: Average loss: 0.0442, Accuracy: 9850/10000 (98.50%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 6
Train: Loss=0.0982 Batch_id=117 Accuracy=97.68: 100%|██████████| 118/118 [00:27<00:00,  4.34it/s]
Test set: Average loss: 0.0438, Accuracy: 9849/10000 (98.49%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 7
Train: Loss=0.0239 Batch_id=117 Accuracy=98.05: 100%|██████████| 118/118 [00:27<00:00,  4.33it/s]
Test set: Average loss: 0.0346, Accuracy: 9890/10000 (98.90%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 8
Train: Loss=0.0843 Batch_id=117 Accuracy=98.20: 100%|██████████| 118/118 [00:27<00:00,  4.33it/s]
Test set: Average loss: 0.0346, Accuracy: 9886/10000 (98.86%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 9
Train: Loss=0.0354 Batch_id=117 Accuracy=98.30: 100%|██████████| 118/118 [00:27<00:00,  4.37it/s]
Test set: Average loss: 0.0311, Accuracy: 9901/10000 (99.01%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 10
Train: Loss=0.0389 Batch_id=117 Accuracy=98.39: 100%|██████████| 118/118 [00:27<00:00,  4.27it/s]
Test set: Average loss: 0.0325, Accuracy: 9895/10000 (98.95%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 11
Train: Loss=0.0608 Batch_id=117 Accuracy=98.51: 100%|██████████| 118/118 [00:27<00:00,  4.35it/s]
Test set: Average loss: 0.0296, Accuracy: 9899/10000 (98.99%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 12
Train: Loss=0.0337 Batch_id=117 Accuracy=98.65: 100%|██████████| 118/118 [00:27<00:00,  4.35it/s]
Test set: Average loss: 0.0263, Accuracy: 9915/10000 (99.15%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 13
Train: Loss=0.0673 Batch_id=117 Accuracy=98.66: 100%|██████████| 118/118 [00:27<00:00,  4.33it/s]
Test set: Average loss: 0.0271, Accuracy: 9911/10000 (99.11%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 14
Train: Loss=0.0157 Batch_id=117 Accuracy=98.66: 100%|██████████| 118/118 [00:27<00:00,  4.36it/s]
Test set: Average loss: 0.0247, Accuracy: 9912/10000 (99.12%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 15
Train: Loss=0.0438 Batch_id=117 Accuracy=98.81: 100%|██████████| 118/118 [00:27<00:00,  4.28it/s]
Test set: Average loss: 0.0253, Accuracy: 9920/10000 (99.20%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 16
Train: Loss=0.0872 Batch_id=117 Accuracy=99.02: 100%|██████████| 118/118 [00:26<00:00,  4.39it/s]
Test set: Average loss: 0.0220, Accuracy: 9927/10000 (99.27%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 17
Train: Loss=0.0199 Batch_id=117 Accuracy=99.08: 100%|██████████| 118/118 [00:26<00:00,  4.41it/s]
Test set: Average loss: 0.0214, Accuracy: 9925/10000 (99.25%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 18
Train: Loss=0.1040 Batch_id=117 Accuracy=99.10: 100%|██████████| 118/118 [00:27<00:00,  4.30it/s]
Test set: Average loss: 0.0211, Accuracy: 9924/10000 (99.24%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 19
Train: Loss=0.0311 Batch_id=117 Accuracy=99.06: 100%|██████████| 118/118 [00:26<00:00,  4.42it/s]
Test set: Average loss: 0.0209, Accuracy: 9926/10000 (99.26%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 20
Train: Loss=0.0400 Batch_id=117 Accuracy=99.12: 100%|██████████| 118/118 [00:27<00:00,  4.34it/s]
Test set: Average loss: 0.0203, Accuracy: 9930/10000 (99.30%)

Adjusting learning rate of group 0 to 1.0000e-03.
```


#### Visualization

![Train/Test Acc/Loss](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session5_IntroductionToPyTorch/images/train_test_loss_acc.png)

---
### Code Description

##### Folder structure

```
_____
     |___ model.py (contains model description)
     |___ utils.py (contains utility functions)
                                               |___ set_device (function to set device as with/without cuda based on availability)
                                               |___ view_data (function to view sample data)
                                               |___ vis_train_test_comp_graphs (function to visualize the train loss/accuracy and test loss/accuracy)
                                               |___ GetCorrectPredCount (function to get correct prediction count)
                                               |___ train (function to train the model)
                                               |___ test (function to test the trained model)

```
