# coursera_dl
## nerual network and deep learning
    * week1:
        1. overall for neural network and deep learning.
        2. interview to Geoffrey Hinton
    * week2:
        1. Python basic
            a. Avoid use for-loop or while loop in computation to some extend. Try best to make matrix compuation.
            b. Using numpy function to complete matrix/vector computation.
            c. Matrix/vector operated by numeric would be "broadcasting"
        2. 0-layer logistic regression(only a activation node)
            a. if just make a simple logistic regression, w and b can be intialized in "zero" vector. But for multi-layer neural network w can't be intialized as "zero". The reason would be introduced in next setion.
            b. Different learing rate would influence perfromance deeply.
    * week3: 
        1. Implement a simple 1-layer NN
            a. Compaired with previous logistic regresion, 1-layer NN can solve some non-linear separatable problem.
            b. W must be initialized by a random normal. It is because that assigning different weights to each node in hidden layer can trigger them to learn different decision boundary. If initializing zero or same weight to each weight, it would result to a balance where each node learns same decision boundary, and then the whole network would be same with only on node. 
            But, b can be initilaized as "zero". Zero is related with training data.
            c. The loss function is chosen as cross-entropy for logitstic activation. Here, the square loss + logtistic would become as a non-convex which has many local mimia. 
            d. Cost function = avaraging(loss function). Cost function would involve in back-propogation. 
            e. Learning rate, hidden layer size and number of nodes are hyperparameter. Kinds of method can be used to choose them. Try them by your self.
    * week4:
        1. Deep neural network: step by step
            a. The difference from shallow neural network is just using loop to finish forward and back propogation.
        2. Cultivate yourself for a good writing habit. 
            ** Superscript $[l]$ denotes a quantity associated with the $l^{th}$ layer.
                Example: $a^{[L]}$ is the $L^{th}$ layer activation. $W^{[L]}$ and $b^{[L]}$ are the $L^{th}$ layer parameters.
            ** Superscript $(i)$ denotes a quantity associated with the $i^{th}$ example.
                Example: $x^{(i)}$ is the $i^{th}$ training example.
            ** Lowerscript $i$ denotes the $i^{th}$ entry of a vector.
                Example: $a^{[l]}_i$ denotes the $i^{th}$ entry of the $l^{th}$ layer's activations).
        3. Activation method choose:
            ** Logistic regression: Just for last layer for binary classification. Almost no ones choose it as activation function in hindden lay.
            **  Tanh: better than logistic because the median point locates in right zero, but logistic locates in 0.5. 
            ** Relu: most used in current days. High speed even it is not differential, but still works well. 
            ** Leaky-relu: just a transform for relu when w less than zero, mutiply a small number to w.
            So many and explore by your self. Don't be lazy and do by yourself.
## Lecture2
## Lecture3
## Lecture4
## Lecture4
