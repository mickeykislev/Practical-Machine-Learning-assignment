### Synopsis

In this current paper, the author tries to predict the performance of
weight lifting according to certain parameters of fitness, as they were
measured by Velloso et. al. (2003). The author secceed to build a model,
with a 85% accurecity, for predicting the ability of the participent to
lift dumbbell.

### Loading the data

The data was saved locally and uploaded by its csv file.

    # Load packages
    requiredPackages <- c('tidyverse', 'R.utils', 'caret', 'rattle', 'rmarkdown');
    for(p in requiredPackages){
      if(!require(p,character.only = TRUE)) install.packages(p)
      library(p,character.only = TRUE)
    }
    rm(p, requiredPackages)


    # Save a copy locally 
    if (!file.exists("pml-training.csv")) {
      download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv","pml-training.csv")
    }

    if (!file.exists("pml-testing.csv")) {
      download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv","pml-testing.csv")
    }

    Source_training <- read.csv("pml-training.csv", 
                          stringsAsFactors = FALSE,
                          na.strings = "#DIV/0!", 
                          skipNul = TRUE);

    Source_prediction <- read.csv("pml-testing.csv", 
                         stringsAsFactors = FALSE,
                         na.strings = "#DIV/0!", 
                         skipNul = TRUE);

An empty field was defined as “\#DIV/0!” in the original file, so I used
the “na.strings” definition to convert them to NA fields. I also difine
a skiping of empty colomns by using the “skipNul” definition.

### The preparation of the data

Then I run several commands for cleaning of close to 0 variable, which
are not indicative, using the following script:

    # remove variables with Nearly Zero Variance
    near_zero <- nearZeroVar(Source_training)
    Source_training <- Source_training[, -near_zero]
    Source_prediction  <- Source_prediction[, -near_zero]
    dim(Source_training)

    ## [1] 19622    83

I Added to that a cleaning of the data from unfilled columns

    # Clean NA's fields
    NA_clean <- apply(Source_training,2,function(x) {sum(is.na(x))}) 
    Source_training <- Source_training[,which(NA_clean == 0)]
    NA_clean <- apply(Source_prediction,2,function(x) {sum(is.na(x))}) 
    Source_prediction <- Source_prediction[,which(NA_clean == 0)]
    dim(Source_training)

    ## [1] 19622    59

Finally, Ideleted the first 7 columns which contain technical
information of date and user name.

    # Delete the first 7 veriables which are not relevent
    Source_training <- Source_training[,-c(1:7)]
    Source_prediction <- Source_prediction[,-c(1:7)]
    dim(Source_training)

    ## [1] 19622    52

### Creating a model

Since the data contain such a large number of observation, It was
possible to split the data into three subsets of: train (60%), test
(20%) and validation (20%) sets:

    set.seed(685)
    inTrain <- createDataPartition(y=Source_training$classe,
                                   p=0.6, list=FALSE)
    training <- Source_training[inTrain,]
    testing <- Source_training[-inTrain,]

    # create the test and the validation sets
    inTest <- createDataPartition(y=testing$classe, p=0.5, list=FALSE)
    validation <- testing[inTest,]
    testing <- testing[-inTest,]

I, than, commanded the R consule to calculate for me the best variables
and the conditions, which anentually will become the model, for
predicting the participent to lift dumbbell performance (var. “Classe”)

    # Choosing the veriables
    modFit <- train(classe ~ .,
                    method="rpart",
                    trControl = trainControl(method = "repeatedcv", number = 5, repeats = 5),
                    data=training, 
                    tuneLength = 52)
    print(modFit$finalModel)

I used the CART model using the method “rpart”, using a cross-validation
several time using the “repeatedcv” method. the reasults graphically
looks as presented in fig. 1:

    # Check the conditions
    fancyRpartPlot(modFit$finalModel)

![fig. 1: the tree of possible results of dumbbell
lifting](treePlot-1.png)

### Checking accuracy

For checking the accuracy of the model, I used the test set, split by me
earlier:

    # Predict on the test case of the train
    testing$predict <- predict(modFit,newdata=testing)
    testing$classe <- factor(testing$classe)

    # compare results with the test of the trail
    confusionMatrix(testing$classe,testing$predict)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1073   26    4    9    4
    ##          B   54  625   43   13   24
    ##          C    2   39  589   28   26
    ##          D   16   30   34  533   30
    ##          E   13   27   47   37  597
    ## 
    ## Overall Statistics
    ##                                                
    ##                Accuracy : 0.871                
    ##                  95% CI : (0.8601, 0.8814)     
    ##     No Information Rate : 0.2952               
    ##     P-Value [Acc > NIR] : < 0.00000000000000022
    ##                                                
    ##                   Kappa : 0.8366               
    ##                                                
    ##  Mcnemar's Test P-Value : 0.0004602            
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D
    ## Sensitivity            0.9266   0.8367   0.8215   0.8597
    ## Specificity            0.9844   0.9578   0.9704   0.9667
    ## Pos Pred Value         0.9615   0.8235   0.8611   0.8289
    ## Neg Pred Value         0.9697   0.9614   0.9605   0.9735
    ## Prevalence             0.2952   0.1904   0.1828   0.1580
    ## Detection Rate         0.2735   0.1593   0.1501   0.1359
    ## Detection Prevalence   0.2845   0.1935   0.1744   0.1639
    ## Balanced Accuracy      0.9555   0.8972   0.8959   0.9132
    ##                      Class: E
    ## Sensitivity            0.8767
    ## Specificity            0.9618
    ## Pos Pred Value         0.8280
    ## Neg Pred Value         0.9738
    ## Prevalence             0.1736
    ## Detection Rate         0.1522
    ## Detection Prevalence   0.1838
    ## Balanced Accuracy      0.9192

The check showed 87% accuracy. I, than used the validation set, as a
duble check

    # Predict on the validation case of the train
    validation$predict <- predict(modFit,newdata=validation)
    validation$classe <- factor(validation$classe)

    # compare results with the test of the train
    confusionMatrix(validation$classe,validation$predict)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1059   24   11   12   10
    ##          B   45  626   48   22   18
    ##          C    7   32  597   23   25
    ##          D   28   26   54  508   27
    ##          E   13   27   40   35  606
    ## 
    ## Overall Statistics
    ##                                                
    ##                Accuracy : 0.8657               
    ##                  95% CI : (0.8546, 0.8762)     
    ##     No Information Rate : 0.2937               
    ##     P-Value [Acc > NIR] : < 0.00000000000000022
    ##                                                
    ##                   Kappa : 0.8299               
    ##                                                
    ##  Mcnemar's Test P-Value : 0.00007244           
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D
    ## Sensitivity            0.9193   0.8517   0.7960   0.8467
    ## Specificity            0.9794   0.9583   0.9726   0.9594
    ## Pos Pred Value         0.9489   0.8248   0.8728   0.7900
    ## Neg Pred Value         0.9669   0.9655   0.9528   0.9720
    ## Prevalence             0.2937   0.1874   0.1912   0.1529
    ## Detection Rate         0.2699   0.1596   0.1522   0.1295
    ## Detection Prevalence   0.2845   0.1935   0.1744   0.1639
    ## Balanced Accuracy      0.9494   0.9050   0.8843   0.9030
    ##                      Class: E
    ## Sensitivity            0.8834
    ## Specificity            0.9645
    ## Pos Pred Value         0.8405
    ## Neg Pred Value         0.9750
    ## Prevalence             0.1749
    ## Detection Rate         0.1545
    ## Detection Prevalence   0.1838
    ## Balanced Accuracy      0.9239

This set showed a resembling accuracy of 86.5%, which means that the
accuracy of the model is about ~86% and above to predict the performance
of a subject, lifting dumbbell.

### Forecast: performing a prediction

As we now have the model to predict, I used this model to predict the
outcome of 20 attempts to lift dumbbell, using a special dataset, which
was not part of the train set, and provided separately:

    # predict on test values
    predict(modFit,newdata=Source_prediction)

    ##  [1] B A B C A E D D A A C C B A E E A B B B
    ## Levels: A B C D E

These outcomes are the prediction of the system for 20 observation of
test dataset, provided separately.

### Reference:

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H.
Qualitative Activity Recognition of Weight Lifting Exercises.
Proceedings of 4th International Conference in Cooperation with SIGCHI
(Augmented Human ’13) . Stuttgart, Germany: ACM SIGCHI, 2013.
