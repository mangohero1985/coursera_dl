# coursera_dl (latex math viewer: https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima/related)
## neural network and deep learning
   * week1:
      * overall for neural network and deep learning.
      * interview to Geoffrey Hinton
   * week2:
      * Python basic
        * Avoid use for-loop or while loop in computation to some extend. Try best to make matrix computation.
        * Using numpy function to complete matrix/vector computation.
        * Matrix/vector operated by numeric would be "broadcasting"
      * 0-layer logistic regression(only a activation node) 
        * if just make a simple logistic regression, w and b can be initialized in "zero" vector. But for multi-layer neural network w can't be initialized as "zero". The reason would be introduced in next section.
       * Different learning rate would influence performance deeply.
   * week3: 
      * Implement a simple 1-layer NN
       * Compared with previous logistic regression, 1-layer NN can solve some non-linear separable problem.
       * W must be initialized by a random normal. It is because that assigning different weights to each node in hidden layer can trigger them to learn different decision boundary. If initializing zero or same weight to each weight, it would result to a balance where each node learns same decision boundary, and then the whole network would be same with only on node. 
            But, b can be initialized as "zero". Zero is related with training data.
        * The loss function is chosen as cross-entropy for logistic activation. Here, the square loss + logistic would become as a non-convex which has many local minima. 
        * Cost function = averaging(loss function). Cost function would involve in back-propagation. 
        * Learning rate, hidden layer size and number of nodes are hyper-parameter. Kinds of method can be used to choose them. Try them by your self.
   * week4:
      * Deep neural network: step by step
        * The difference from shallow neural network is just using loop to finish forward and back propagation.
        * Cultivate yourself for a good writing habit. 
            * Superscript $[l]$ denotes a quantity associated with the $l^{th}$ layer.
                Example: $a^{[L]}$ is the $L^{th}$ layer activation. $W^{[L]}$ and $b^{[L]}$ are the $L^{th}$ layer parameters.
            * Superscript $(i)$ denotes a quantity associated with the $i^{th}$ example.
                Example: $x^{(i)}$ is the $i^{th}$ training example.
            * Lower-script $i$ denotes the $i^{th}$ entry of a vector.
                Example: $a^{[l]}_i$ denotes the $i^{th}$ entry of the $l^{th}$ layer's activations).
        * Activation function choose:
            * Logistic regression: 
            	Just for last layer for binary classification. Almost no ones choose it as activation function in hidden lay.
            *  Tanh: 
            better than logistic because the median point locates in right zero, but logistic locates in 0.5. 
            * Relu: 
            most used in current days. High speed even it is not differential, but still works well. 
            * Leaky-relu: 
            just a transform for relu when w less than zero, multiply a small number to w.              
 
 		So many and explore by your self. Don't be lazy and do by yourself.
## Hyperparameter_tuning, regularization and optimization
    * week1
      * Initialization
        * Zeros initialization
          - As said before, the symmetry can't be broken. all of nodes would be trained in same weight. No any effect in using hidden layer.
        * Random initialization
        * Xavier initialization
      * Regularization
      * Gradient checking
    * week2
    * week3
## Lecture3
## Lecture4
## Lecture4
