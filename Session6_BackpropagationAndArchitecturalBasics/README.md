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

