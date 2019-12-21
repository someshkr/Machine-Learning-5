# Machine-Learning-5
Ensemble Learning and Decision Tree


If you aggregate the predictions of a group of predictors (such as classifiers and regressors), you will often get better predictions than the best individual predictor. A group of predictors is called an ensemble, thus, this technique is called Ensemble Learning, and an Ensemble Learning algorithm is called an Ensemble method.

Why ensembles?
	Lower error
	Less over-fitting
	Taste great

Voting is one of the simplest way of combining the predictions from multiple machine learning algorithms. Voting classifier isn’t an actual classifier but a wrapper for set of different ones that are trained and valuated in parallel in order to exploit the different peculiarities of each algorithm.

We can train data set using different algorithms and ensemble then to predict the final output. The final output on a prediction is taken by majority vote according to two different strategies :

1. Hard voting / Majority voting : 
Hard voting is the simplest case of majority voting. In this case, the class that received the highest number of votes Nc​(y​​t) will be chosen. Here we predict the class label y^ via majority voting of each classifier.
2. Soft voting : 
In this case, the probability vector for each predicted class (for all classifiers) are summed up &averaged. The winning class is the one corresponding to the highest value (only recommended if the classifiers are well calibrated).

One way to get a diverse set of classifiers is to use very different training algorithms,
as just discussed. Another approach is to use the same training algorithm for every
predictor, but to train them on different random subsets of the training set. When
sampling is performed with replacement, this method is called bagging (short for
bootstrap aggregating).
When sampling is performed without replacement, it is called pasting.

In other words, both bagging and pasting allow training instances to be sampled several
times across multiple predictors, but only bagging allows training instances to be
sampled several times for the same predictor.Once all predictors are trained, the ensemble can make a prediction for a new instance by simply aggregating the predictions of all predictors. The aggregation
function is typically the statistical mode (i.e., the most frequent prediction, just like a
hard voting classifier) for classification, or the average for regression.

Each Indivisual predictor has a higher bias than if it were trained on the original training set, but
aggregation reduces both bias and variance. Generally, the net result is that the
ensemble has a similar bias but a lower variance than a single predictor trained on the
original training set.

Out-of-Bag Evaluation:

With bagging, some instances may be sampled several times for any given predictor,
while others may not be sampled at all. By default a BaggingClassifier samples m
training instances with replacement (bootstrap=True), where m is the size of the
training set. This means that only about 63% of the training instances are sampled on
average for each predictor.6 The remaining 37% of the training instances that are not
sampled are called out-of-bag (oob) instances. Note that they are not the same 37%
for all predictors.
Since a predictor never sees the oob instances during training, it can be evaluated on
these instances, without the need for a separate validation set. You can evaluate the
ensemble itself by averaging out the oob evaluations of each predictor.


Random Patches and Random Subspaces:

The BaggingClassifier class supports sampling the features as well. This is controlled
by two hyperparameters: max_features and bootstrap_features. They work
the same way as max_samples and bootstrap, but for feature sampling instead of
instance sampling. Thus, each predictor will be trained on a random subset of the
input features.
This is particularly useful when you are dealing with high-dimensional inputs (such
as images). Sampling both training instances and features is called the Random
Patches method.Keeping all training instances (i.e., bootstrap=False and max_sam
ples=1.0) but sampling features (i.e., bootstrap_features=True and/or max_features smaller than 1.0) is called the Random Subspaces method.Sampling features results in even more predictor diversity, trading a bit more bias for a lower variance.
