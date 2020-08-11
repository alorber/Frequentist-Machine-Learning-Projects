# Project 3 - Cross Validation
###### Following section 7.10.2 of [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/).

The third project of *Frequentist Machine Learning* was to demonstrate the correct method of doing cross-validation.

The dataset created for the project had 50 samples in two equal-sized classes, with 5000 quantitative predictors (standard Gaussian) that are independent of the class label.

The true test error rate of any classifier should be 50%, which is seen when the correct method is used. However, when the incorrect method is used, the test error rate drops as low as 3%.

**The incorrect method for cross-validation:**
1. Screen the predictors: find a subset of “good” predictors that show fairly strong (univariate) correlation with the class labels.
2. Using just this subset of predictors, build a multivariate classifier.
3. Use cross-validation to estimate the prediction error of the final model.

**The correct method for cross-validation:**
1. Divide the samples into K cross-validation folds (groups) at random.
2. For each fold k = 1,2,...,K:

    (a) Find a subset of “good” predictors that show fairly strong (univariate) correlation with the class labels, using all of the samples except those in fold k.
    
    (b) Using just this subset of predictors, build a multivariate classifier, using all of the samples except those in fold k.
    
    (c) Use the classifier to predict the class labels for the samples in fold k.
3. Use the error accumulated over all K folds to produce the cross-validation estimate of prediction error. 
