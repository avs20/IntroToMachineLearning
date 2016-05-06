
1. Summarize for us the goal of this project and how machine learning is useful
in trying to accomplish it. As part of your answer, give some background on the
dataset and how it can be used to answer the project question. Were there any
 outliers in the data when you got it, and how did you handle those?  [relevant
 rubric items: “data exploration”, “outlier investigation”]

2. What features did you end up using in your POI identifier, and what
selection process did you use to pick them? Did you have to do any scaling?
 Why or why not? As part of the assignment, you should attempt to engineer
 your own feature that does not come ready-made in the dataset -- explain
 what feature you tried to make, and the rationale behind it. (You do not
 necessarily have to use it in the final analysis, only engineer and test it.)
  In your feature selection step, if you used an algorithm like a decision tree,
  please also give the feature importances of the features that you use, and if
  you used an automated feature selection function like SelectKBest, please
  report the feature scores and reasons for your choice of parameter values.
   [relevant rubric items: “create new features”, “properly scale features”,
    “intelligently select feature”]

**3. What algorithm did you end up using? What other one(s) did you try? How did
 model performance differ between algorithms?  [relevant rubric item: “pick an
 algorithm”]**

I did the training on the following algorithms

1. RandomForest
2. ***Logistic Regression ( WINNER )***
3. Support Vector Machine 

**4. What does it mean to tune the parameters of an algorithm, and what can
 happen if you don’t do this well?  How did you tune the parameters of your
  particular algorithm? (Some algorithms do not have parameters that you need
  to tune -- if this is the case for the one you picked, identify and briefly
   explain how you would have done it for the model that was not your final
   choice or a different model that does utilize parameter tuning, e.g. a
   decision tree classifier).  [relevant rubric item: “tune the algorithm”]**

Each algorithm has some parameters inherent to it that lets us decide how much
flexible we can be with the results or what type of loss do we want to
calculate from the available list of choices. Not choosing the correct parameters
 may lead us to not the optimum results. Also the parameters change for
different datasets, so we also tune the parameters according to the dataset.

The algorithm I chose finally is Logistic Regression and it has a list of parameters
to tune from penalty type, C and tolerance. *Penalty* tries the different types
of norm. *C* is used for the trade-off between smooth boundary or correct
classification. *Tolerance* is the stopping criteria for the algorithm.

While searching for different algorithms I tested RandomForest and SVM. They have
some different params like number of estimators and features to use, gamma,
kernel type ,etc..


**5. What is validation, and what’s a classic mistake you can make if you do it
wrong? How did you validate your analysis?  [relevant rubric item: “validation
strategy”]
Give at least 2 evaluation metrics and your average performance for each of
them.  Explain an interpretation of your metrics that says something
human-understandable about your algorithm’s performance. [relevant rubric item:
“usage of evaluation metrics”]**

Validation is the process of making sure that the data that we have used for
the algorithm doesn't leak into the data and the algorithm is not biased towards
it. For this purpose we take a portion of data and hide it from the algorithm
until the very end of our model being complete. The algorithm never sees this
data and we check the performance on this data. This way we can be sure that the
algorithm is not overfitting on the training set and the performance on the test
set is the one true performance.

We also make sure that the data in the training set and the testing set is
in same proportion with respect to classes available. for eg : if the training
set contains 70% of class-1 and 30% of class-2 then the same ratio should also
be maintained in the test set.

##### Usage of evaluation metrics
For this project I used the **precision, recall** metrics for evaluating the
performance of the algorithm. Simple accuracy score will not be a good metric
because it can be skewed if we have very high number of samples from one class
and very low for another as if even if the algorithm makes all the wrong
guesses then still the accuracy will be high.

**precision, recall** are the two metrics which overcome this problem by
considering the following data

* **true-positive** - the number of persons identified as POI which were POI
* **true-negative** - the number of persons identified as non-POI which were
non-POI
* **false-positive** - the number of persons identified as POI which were
non-POI
* **false-negative** - the number of persons identified as non-POI which were
POI

From the above defintions we want to increase true-positive, true-negative and
decrease false-positive and false-negative.

precision is p = (tp / (tp+fp))

recall is r = (tp/ (tp+fn)

High precision relates to a low false positive rate, and high recall relates to
a low false negative rate. In layman's terms we are identifying correct persons
as POI and rejecting who are not.

The target of the model was to acheive 0.3 accuracy on the both precision
and recall. The current model acheives 0.52 in precision and 0.37 in recall
scores.
This means that for every correct positive prediction we also did an incorrect
positive prediction. And we missed 63% of the POI and classified them as non-POI.
