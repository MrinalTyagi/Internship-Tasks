# Tasks 

## Task 1 : Animate bivariate normal distribution.
#### A. Reference Image 
![440px-MultivariateNormal](https://user-images.githubusercontent.com/21031150/210636314-ce503b09-5e0b-4312-9a73-7ce495f977ee.png)

#### B. Final Image
<img width="368" alt="2" src="https://user-images.githubusercontent.com/21031150/210636397-2495be91-e2ac-42db-b266-67b4207417ff.png">

#### C. Libraries Used :
1. Jax
2. Matplotlib - Used 3D plots
3. Ipywidgets - Used for interactivity

## Task 2 : Implement from scratch a sampling method to draw samples from a multivariate Normal (MVN) distribution in JAX. </h2>

Used Box Muller Transformation and Cholesky Decomposition to perform the task

#### A. Libraries Used :
1. Jax

#### B. Result :
<img width="745" alt="3" src="https://user-images.githubusercontent.com/21031150/210636505-5b8e7255-cbad-432f-a9c2-00f3041894dc.png">

For cov_matrix see notebook 

#### C. References :
1. https://www.baeldung.com/cs/uniform-to-normal-distribution

2. https://www.youtube.com/watch?v=HFMrsXNuTSY


## Task 3 : Implement two hidden layers neural network classifier from scratch in JAX. </h2>
Created a class neural network with dynamic input sized hidden layers. 

#### A. Model Settings : 
1. Layers : 256, 128
2. Learning Rate : 3e-2
3. N_epochs : 100

#### B. Libraries Used :
1. Jax
2. Sklearn : For MNIST dataset and for classification report

#### C. Result : 
1. F1 score on test data : 0.81
2. Training Loss Value : 0.88

<img width="502" alt="4" src="https://user-images.githubusercontent.com/21031150/210636687-86bf9f4d-7485-4630-abb2-881fa06a5e6d.png">

<img width="502" alt="5" src="https://user-images.githubusercontent.com/21031150/210636854-cc2a171b-55bc-4229-ab2e-1fbceaa0ac06.png">


