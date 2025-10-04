# Model1
#### Target:
1. Decide on the initial data transforms
2. Looking into the data in order to decide on applying the maxpool, like after how many pixel a feature can be distingushible, so that the receptive field is decided as well
3. Building the basic model skeleton 
4. Not worrying much on the accuracies as of now

#### Result:
1. Parameters: 72.2k
2. Best Train Accuracy: 99.16
3. Best Test Accuracy: 99.07

#### Analysis:
1. Model parameters are high
2. Able to see some overfitting 
3. Since the model base is not yet set no worries on the accuracy
4. For a data set like MNIST the model parameters is more, hence can be reduced further.


# Model 2
#### Target:
1. Keeping the model structure as is and reducing the parameters,
    i.e., making the model lighter

#### Result:
1. Parameters: 7k
2. Best Train Accuracy: 99.32
3. Best Test Accuracy: 98.94

#### Analysis:
1. In the course of reducing the parameters the accuracy took the hit 
2. Model can be pushed by regularizing the data	

# Model 3
#### Target:
1. Adding the BatchNorm and Dropout

#### Result:
1. Parameters: 7.1k
2. Best Train Accuracy: 99.09
3. Best Test Accuracy: 99.30

#### Analysis:
1. There was a slight increase in accuracy which can be further improved 

# Model 4
#### Target:
1. Adding a layer instead of 2 maxpool layer, so that we get more layers to convl upon
2. Adding a GAP layer

#### Result:
1. Parameters: 7.5k
2. Best Train Accuracy: 98.90
3. Best Test Accuracy: 99.21

#### Analysis:
1. Accuracy of 99.4 was still not achived

# Model 5
#### Target:
1. Adding LR scheduler

#### Result:
1. Parameters: 7.7k
2. Best Train Accuracy: 99.13
3. Best Test Accuracy: 99.51

#### Analysis:
1. Achived the accuracy under parameter restriction
