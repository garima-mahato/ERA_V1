# BackPropagation And Architecture Basics
---

## Part 1) Backpropagation

[GitHub Link to Neural Network Training excel sheet](https://github.com/garima-mahato/ERA_V1/blob/main/Session6_BackpropagationAndArchitecturalBasics/Part1/ERA_V1_S6_Backpropagation.xlsx)

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session6_BackpropagationAndArchitecturalBasics/Part1/assets/Backpropagation.JPG)

## Major Steps in NN Training

Suppose we have the below network:

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session6_BackpropagationAndArchitecturalBasics/Part1/assets/nn.PNG)


### 1) Initialization of Neural Network
Randomly initializing weights [w1,....,w8] and Learning Rate lr.


### 2) For each iteration, below steps are performed:

#### A) Forward Propagation

For the above network, output will be calculated using the below formulae:

```
h1 = w1*i1 + w2*i2	

h2 = w3*i1 + w4*i2

a_h1 = σ(h1) = 1/(1+exp(-h1))	

a_h2 = σ(h2) = 1/(1+exp(-h2))	

o1 = w5*a_h1 + w6*a_h2	

o2 = w7*a_h1 + w8*a_h2

a_o1 = σ(o1) = 1/(1+exp(-o1))		

a_o2 = σ(o2) = 1/(1+exp(-o2))	
```

#### B) Backpropagation

**I) Using the generated output and target, error is calculated.**

For above network, L2 error is calculated as shown below:

```
E1 = ½*(t1-a_o1)²	

E2 = ½*(t2-a_o2)²	

E_total = E1 + E2		
```

**II) Each of the weight in the network is updated as follows:**

i) Error gradient with respect to the specific weight(∂Error/∂w<sub>i</sub>) is calculated. Negative error indicates the direction in which the error can be minimized by updatig the specific weight.

Example:- For w5
```
∂E_total/∂w5 = ∂(E1+E2)/∂w5 = ∂E1/∂w5 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂w5		

∂E1/∂a_o1 = -1*(t1-a_o1) = a_o1-t1			

∂a_o1/∂o1 = ∂(σ(o1))/∂o1 = σ(o1)*(1-σ(o1)) = a_o1*(1-a_o1)

∂o1/∂w5 = a_h1			
```
Thus, it becomes

```
∂E_total/∂w5 = (a_o1-t1) * a_o1*(1-a_o1) * a_h1					
```

ii) Weight is updated using:

w<sub>i</sub><sup>j+1</sup> = w<sub>i</sub><sup>j</sup> - ɳ * ∂Error/∂w<sub>i</sub><sup>j</sup>

where i is the weight index,
      
j is the iteration number


For above network, below are the error gradients with respect to weights calculated using chain formula.

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session6_BackpropagationAndArchitecturalBasics/Part1/assets/bp.PNG)


---
# Error Graphs for various learning rates

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session6_BackpropagationAndArchitecturalBasics/Part1/assets/err_lr_rel.JPG)


## Part 2) Architecture Basics

<b>Best Test Accuracy: 99.54 % </b>

#### Model Architecture

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
           Dropout-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 24, 24, 24]           3,456
              ReLU-6           [-1, 24, 24, 24]               0
       BatchNorm2d-7           [-1, 24, 24, 24]              48
           Dropout-8           [-1, 24, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             240
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 14, 10, 10]           1,260
             ReLU-12           [-1, 14, 10, 10]               0
      BatchNorm2d-13           [-1, 14, 10, 10]              28
          Dropout-14           [-1, 14, 10, 10]               0
           Conv2d-15             [-1, 16, 8, 8]           2,016
             ReLU-16             [-1, 16, 8, 8]               0
      BatchNorm2d-17             [-1, 16, 8, 8]              32
          Dropout-18             [-1, 16, 8, 8]               0
           Conv2d-19             [-1, 16, 6, 6]           2,304
             ReLU-20             [-1, 16, 6, 6]               0
      BatchNorm2d-21             [-1, 16, 6, 6]              32
          Dropout-22             [-1, 16, 6, 6]               0
        AvgPool2d-23             [-1, 16, 1, 1]               0
           Conv2d-24             [-1, 16, 1, 1]             256
           Conv2d-25             [-1, 10, 1, 1]             160
================================================================
Total params: 10,008
Trainable params: 10,008
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.90
Params size (MB): 0.04
Estimated Total Size (MB): 0.94
----------------------------------------------------------------
```

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session6_BackpropagationAndArchitecturalBasics/Part2/assets/mnist_cnn_torchviz.png)

#### Training and Testing

<b>Logs</b>

```
Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 1
Train: Loss=0.2235 Batch_id=937 Accuracy=86.25: 100%|██████████| 938/938 [00:38<00:00, 24.68it/s]
Test set: Average loss: 0.0596, Accuracy: 9808/10000 (98.08%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 2
Train: Loss=0.2968 Batch_id=937 Accuracy=96.49: 100%|██████████| 938/938 [00:29<00:00, 31.44it/s]
Test set: Average loss: 0.0442, Accuracy: 9866/10000 (98.66%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 3
Train: Loss=0.0210 Batch_id=937 Accuracy=97.22: 100%|██████████| 938/938 [00:32<00:00, 28.77it/s]
Test set: Average loss: 0.0372, Accuracy: 9890/10000 (98.90%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 4
Train: Loss=0.0923 Batch_id=937 Accuracy=97.52: 100%|██████████| 938/938 [00:30<00:00, 30.73it/s]
Test set: Average loss: 0.0335, Accuracy: 9892/10000 (98.92%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 5
Train: Loss=0.1863 Batch_id=937 Accuracy=97.77: 100%|██████████| 938/938 [00:30<00:00, 31.22it/s]
Test set: Average loss: 0.0228, Accuracy: 9922/10000 (99.22%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 6
Train: Loss=0.0508 Batch_id=937 Accuracy=98.00: 100%|██████████| 938/938 [00:30<00:00, 30.64it/s]
Test set: Average loss: 0.0244, Accuracy: 9915/10000 (99.15%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 7
Train: Loss=0.1397 Batch_id=937 Accuracy=98.05: 100%|██████████| 938/938 [00:30<00:00, 30.95it/s]
Test set: Average loss: 0.0240, Accuracy: 9923/10000 (99.23%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 8
Train: Loss=0.0259 Batch_id=937 Accuracy=98.11: 100%|██████████| 938/938 [00:30<00:00, 31.25it/s]
Test set: Average loss: 0.0249, Accuracy: 9926/10000 (99.26%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 9
Train: Loss=0.0902 Batch_id=937 Accuracy=98.18: 100%|██████████| 938/938 [00:31<00:00, 30.06it/s]
Test set: Average loss: 0.0274, Accuracy: 9914/10000 (99.14%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 10
Train: Loss=0.0933 Batch_id=937 Accuracy=98.26: 100%|██████████| 938/938 [00:29<00:00, 31.39it/s]
Test set: Average loss: 0.0205, Accuracy: 9929/10000 (99.29%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 11
Train: Loss=0.1048 Batch_id=937 Accuracy=98.27: 100%|██████████| 938/938 [00:30<00:00, 31.18it/s]
Test set: Average loss: 0.0197, Accuracy: 9938/10000 (99.38%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 12
Train: Loss=0.0059 Batch_id=937 Accuracy=98.31: 100%|██████████| 938/938 [00:30<00:00, 30.55it/s]
Test set: Average loss: 0.0197, Accuracy: 9937/10000 (99.37%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 13
Train: Loss=0.0127 Batch_id=937 Accuracy=98.47: 100%|██████████| 938/938 [00:29<00:00, 31.69it/s]
Test set: Average loss: 0.0175, Accuracy: 9941/10000 (99.41%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 14
Train: Loss=0.0031 Batch_id=937 Accuracy=98.38: 100%|██████████| 938/938 [00:30<00:00, 30.76it/s]
Test set: Average loss: 0.0190, Accuracy: 9937/10000 (99.37%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 15
Train: Loss=0.0125 Batch_id=937 Accuracy=98.43: 100%|██████████| 938/938 [00:31<00:00, 30.09it/s]
Test set: Average loss: 0.0222, Accuracy: 9931/10000 (99.31%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 16
Train: Loss=0.0251 Batch_id=937 Accuracy=98.72: 100%|██████████| 938/938 [00:30<00:00, 30.92it/s]
Test set: Average loss: 0.0152, Accuracy: 9951/10000 (99.51%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 17
Train: Loss=0.0006 Batch_id=937 Accuracy=98.79: 100%|██████████| 938/938 [00:30<00:00, 30.77it/s]
Test set: Average loss: 0.0149, Accuracy: 9952/10000 (99.52%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 18
Train: Loss=0.0255 Batch_id=937 Accuracy=98.84: 100%|██████████| 938/938 [00:30<00:00, 30.94it/s]
Test set: Average loss: 0.0152, Accuracy: 9950/10000 (99.50%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 19
Train: Loss=0.0191 Batch_id=937 Accuracy=98.79: 100%|██████████| 938/938 [00:30<00:00, 31.23it/s]
Test set: Average loss: 0.0148, Accuracy: 9949/10000 (99.49%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 20
Train: Loss=0.3020 Batch_id=937 Accuracy=98.81: 100%|██████████| 938/938 [00:30<00:00, 30.91it/s]
Test set: Average loss: 0.0151, Accuracy: 9954/10000 (99.54%)

Adjusting learning rate of group 0 to 1.0000e-03.
```


<b>Visualization</b>

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session6_BackpropagationAndArchitecturalBasics/Part2/assets/train_test_acc_loss.png)
