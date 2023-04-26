# NeuralNetwork_DeepLearning
Projects developed in the context of the "Neural Network and Deep Learning" exam.

### Homework 1:
In this homework we implement and test simple neural network architectures useful to solve su- pervised learning problems. 
The homework is structured in two distinct sections which are related to two distinct tasks. 
In the regression task our network has to approximate an unknown function through a little dataset, while in the classification task the network has to solve an image classifica- tion problem, correctly classifying images from the fashion-MNIST dataset. 
In both cases, in order to find the best performing and general model, hyperparameters tuning and regularization methods are implemented. Finally, k-fold cross validation is exploited to better evaluate the final results.

### Homework 2:
In this homework we are required to solve different tasks related to unsupervised deep learning. 
All the tasks in the homework are carried out within the FashionMNIST dataset. 
The basic task is to correctly implement a convolutional autoencoder and report its performances. 
Moreover, we are asked to exploit hyperparameters tuning strategies, advanced optimizers and regularization methods to improve the autoencoder performance. 
After that we fine-tune our convolutional autoencoder using the FashionMNIST dataset in a supervised way and we compare the results with the one obtained in the classification task of the first homework. 
As a more advanced task, we implement and test a variational autoencoder. 
Finally, we explore the latent space structure of the autoencoders we implemented and generate new samples from them. 
Plots that shows the learning process and the optimization procedure are produced for both autoencoders.

### Homework 3:
In this homework we have to implement and test neural networks in order to tackle some deep reinforcement learning tasks. 
We implement a deep Q-learning agent in order to learn to control the Cartpole environment. 
Moreover, an agent that interact with the MsPacman-v0 Gym environment using directly the screen pixels is implemented. 
