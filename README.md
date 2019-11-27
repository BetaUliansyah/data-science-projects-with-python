# Data Science Projects with Python
By Barbora Stetinova
June 2019

Videos taken from https://www.packtpub.com/data/data-science-projects-with-python-e-learning uring Free Weekend access 20-23 September 2019 (with message: You are accessing this content as part of the Free Weekend)

Use pandas and Matplotlib to critically examine a dataset with summary statistics and graphs and extract meaningful insights.

## video1_1
Course Overview

Let’s begin the course with the content coverage.

## video1_2
Installation and Setup

Before you start this course, make sure you have installed the Anaconda environment as we will be using the Anaconda distribution of Python.

## video1_3
Lesson Overview
Let us begin with the first lesson and understand what we are going to cover in our learning journey.

## video1_4
Python and the Anaconda Package Management System
In this video, we will use the Python programming language. Python is a top language for data science and is one of the fastest growing programming languages. A commonly cited reason for Python's popularity is that it is easy to learn. If you have Python experience, that's great; however, if you have experience with other languages, such as C, Matlab, or R, you shouldn't have much trouble using Python. You should be familiar with the general constructs of computer programming to get the most out of this course. Examples of such constructs are for loops and if statements that guide the control flow of a program. No matter what language you have used, you are likely familiar with these constructs, which you will also find in Python. Here are the topics that we will cover now:

## video1_5
Different Types of Data Science Problems
Much of your time as a data scientist is likely to be spent wrangling data: figuring out how to get it, getting it, examining it, making sure it's correct and complete, and joining it with other types of data. pandas will facilitate this process for you. However, if you aspire to be a machine learning data scientist, you will need to master the art and science of predictive modeling. This means using a mathematical model, or idealized mathematical formulation, to learn the relationships within the data, in the hope of making accurate and useful predictions when new data comes in. Here are the topics that we will cover now:

## video1_6
Loading the Case Study Data with Jupyter and pandas

Now it's time to take a first look at the data we will use in our case study. We won’t do anything in this video other than ensure that we can load the data into a Jupyter Notebook correctly. Examining the data, and understanding the problem you will solve with it, will come later. Here are the topics that we will cover now:

## video1_7
Getting Familiar with Data and Performing Data Cleaning

Let us get a bit more familiar with data and performing data cleaning. Here are the topics that we will cover now:

To help clean the case study data, we introduce the concept of a logical mask, also known as a Boolean mask. A logical mask is a way to filter an array, or series, by some condition. For example, we can use the "is equal to" operator in Python, ==, to find all locations of an array that contain a certain value. Other comparisons, such as "greater than" (>), "less than" (<), "greater than or equal to" (>=), and "less than or equal to" (<=), can be used similarly. The output of such a comparison is an array or series of True/False values, also known as Boolean values. Here are the topics that we will cover now:

## video1_8
Boolean Masks

To help clean the case study data, we introduce the concept of a logical mask, also known as a Boolean mask. A logical mask is a way to filter an array, or series, by some condition. For example, we can use the "is equal to" operator in Python, ==, to find all locations of an array that contain a certain value. Other comparisons, such as "greater than" (>), "less than" (<), "greater than or equal to" (>=), and "less than or equal to" (<=), can be used similarly. The output of such a comparison is an array or series of True/False values, also known as Boolean values. Here are the topics that we will cover now:

## video1_9
Data Quality Assurance and Exploration

So far, we remedied two data quality issues just by asking basic questions or by looking at the .info() summary. Let's now look at the first few columns. Before we get to the historical bill payments, we have the credit limits of the accounts of LIMIT_BAL, and the demographic features SEX, EDUCATION, MARRIAGE, and AGE. Our business partner has reached out to us, to let us know that gender should not be used to predict credit-worthiness, as this is unethical by their standards. So, we keep this in mind for future reference. Now we explore the rest of these columns, making any corrections that are necessary. Here are the topics that we will cover now:

## video1_10
Deep Dive: Categorical Features

Machine learning algorithms only work with numbers. If your data contains text features, for example, these would require transformation to numbers in some way. We learned above that the data for our case study is, in fact, entirely numerical. However, it's worth thinking about how it got to be that way. Consider the EDUCATION feature.

## video1_11
Exploring the Financial History Features in the Dataset

We are ready to explore the rest of the features in the case study dataset. We will first practice loading a DataFrame from the CSV file we saved at the end of the last video.

## video1_12
Lesson Summary

Summarize your learning from this lesson.

## video2_1
Lesson Overview

Let us begin with the second lesson and understand what we are going to cover in our learning journey.

## video2_2
Exploring the Response Variable and Concluding the Initial Exploration

We have now looked through all the features to see whether any data is missing, as well as to generally examine them. The features are important because they constitute the inputs to our machine learning algorithm. On the other side of the model lies the output, which is a prediction of the response variable. For our problem, this is a binary flag indicating whether an account will default the next month, which would have been October for our historical dataset.

## video2_3
Introduction to Scikit-Learn

While pandas will save you a lot of time in loading, examining, and cleaning data, the machine learning algorithms that will enable you to do predictive modelling are in other packages. We consider scikit-learn to be the premier machine learning package for Python, outside of deep learning. While it's impossible for any one package to offer "everything," scikit-learn comes close in terms of accommodating a wide range of approaches for classification and regression, and unsupervised learning. Here are the topics that we will cover now:

## video2_4
Model Performance Metrics for Binary Classification

Before we start building predictive models in earnest, we would like to know how we can determine, once we've created a model, whether it is "good" in some sense of the word. As you may imagine, this question has received a lot of attention from researchers and practitioners. Consequently, there is a wide variety of model performance metrics to choose from. Here are the topics that we will cover now:

## video2_5
True Positive Rate, False Positive Rate, and Confusion Matrix

In binary classification, there are just two labels to consider: positive and negative. As a more descriptive way to look at model performance than the accuracy of prediction across all samples, we can also look at the accuracy of only those samples that have a positive label. The proportion of these that we successfully predict as positive, is called the true positive rate (TPR). If we say that P is the number of samples in the positive class in the testing data, and TP is the number of true positives, defined as the number of positive samples that were predicted to be positive by the model, then the TPR is as follows: Here are the topics that we will cover now:

## video2_6
Obtaining Predicted Probabilities from a Trained Logistic Regression Model

In the following video, we will get familiar with the predicted probabilities of logistic regression and how to obtain them from a scikit-learn model. Here are the topics that we will cover now:

## video2_7
Lesson Summary

Lesson Summary


## video3_1
Lesson Overview

Let us begin with the third lesson and understand what we are going to cover in our learning journey.

## video3_2
Examining the Relationships between Features and the Response

In order to make accurate predictions of the response variable, good features are necessary. We need features that are clearly linked to the response variable in some way. Thus far, we've examined the relationship between a couple features and the response variable, either by calculating a groupby/mean of the response variable, or by trying models directly, which is another way to make this kind of exploration.

## video3_3
Finer Points of the F-test: Equivalence to t-test for Two Classes and Cautions

When we use an F-test to look at the difference in means between just two groups, as we've done here for the binary classification problem of the case study, the test we are performing reduces to what's called a t-test. An F-test is extensible to three or more groups and so is useful for multiclass classification. A t-test just compares the means between two groups of samples, to see if the difference in those means is statistically significant. Here are the topics that we will cover now:

## video3_4
Univariate Feature Selection: What It Does and Doesn't Do

In the earlier videos, we have learned techniques for going through features one by one to see whether they have predictive power. This is a good first step, and if you already have features that are very predictive of the outcome variable, you may not need to spend much more time considering features before modeling. However, there are drawbacks to univariate feature selection. It does not consider the interactions between features. For example, what if the credit default rate is very high specifically for people with a certain education level and a certain range of credit limit? Here are the topics that we will cover now:

## video3_5
Generalized Linear Models (GLMs)

Logistic regression is part of a broader class of statistical models called Generalized Linear Models (GLMs). GLMs are connected to the fundamental concept of ordinary linear regression, which may have one feature (that is, the line of best fit, y = mx + b, for a single feature, x) or more than one in multiple linear regression. The mathematical connection between GLMs and linear regression is the link function. The link function of logistic regression is the logit function we just learned about. Here are the topics that we will cover now:

## video4_1
Lesson Overview

Let us begin with the fourth lesson and understand what we are going to cover in our learning journey.

## video4_2
Estimating the Coefficients and Intercepts of Logistic Regression

In the previous lesson, we learned that the coefficients of a logistic regression (each of which goes with a particular feature), and the intercept, are determined when the .fit method is called on a logistic regression model in scikit-learn using the training data. These numbers are called the parameters of the model, and the process of finding the best values for them is called parameter estimation. Once the parameters are found, the logistic regression model is essentially a finished product; therefore, with just these numbers, we can use the trained logistic regression in any environment where we can perform common mathematical functions. Here are the topics that we will cover now:

## video4_3
Assumptions of Logistic Regression

Since it is a classical statistical model, like the F-test and Pearson correlation we already examined, logistic regression makes certain assumptions about the data. While it's not necessary to follow every one of these assumptions in the strictest possible sense, it's good to be aware of them. That way, if a logistic regression model is not performing very well, you can try to investigate and figure out why, using your knowledge of the ideal situation in which a logistic regression would work well. You may find slightly different lists of the specific assumptions from different resources, however those that are listed here are widely accepted. Here are the topics that we will cover now:

## video4_4
How Many Features Should You Include?

This is not so much an assumption as it is guidance on model building. There is no clear-cut law that states how many features to include in a logistic regression model. However, a common rule of thumb is the "rule of 10," which states that for every 10 occurrences of the rarest outcome class, 1 feature may be added to the model. So, for example, in a binary logistic regression problem with 100 samples, if the class balance has 20% positive outcomes and 80% negative outcomes, then there are only 20 positive outcomes in total, and so only two features should be used in the model. A "rule of 20" has also been suggested, which would be a more stringent limit on the number of features to include (1 feature in our example). Here are the topics that we will cover now:

## video4_5
Lasso (L1) and Ridge (L2) Regularization

Before applying regularization to a logistic regression model, let's take a moment to understand what regularization is and how it works. The two ways of regularizing logistic regression models in scikit-learn are called lasso (also known as L1 regularization) and ridge (also known as L2 regularization). When instantiating the model object from the scikit-learn class, you can choose either penalty = 'l1’ or 'l2'. These are called "penalties" because the effect of regularization is to add a penalty, or a cost, for having larger values of the coefficients in a fitted logistic regression model. Here are the topics that we will cover now:

## video4_6
Cross Validation: Choosing the Regularization Parameter and Other Hyperparameters

By now, you should be interested in using regularization in order to decrease the overfitting we observed when we tried to model the synthetic data. The question is, how do we choose the regularization parameter, C? C is an example of a model hyperparameter. Hyperparameters are different from the parameters that are estimated when a model is trained, such as the coefficients and the intercept of a logistic regression. Rather than being estimated by an automated procedure like the parameters are, hyperparameters are input directly by the user as keyword arguments, typically when instantiating the model class. So, how do we know what values to choose? Here are the topics that we will cover now:

## video4_7
Reducing Overfitting on the Synthetic Data Classification Problem

Here, we will use the cross-validation procedure in order to find a good value for the hyperparameter C. We will do this by using only the training data, reserving the testing data for after model building is complete. This video will illustrate a general procedure that you will be able to use with many kinds of machine learning models, so it is worth the time spent here.

## video4_8
Options for Logistic Regression in Scikit-Learn

We have used and discussed most of the options that you may supply to scikit-learn when instantiating or tuning the hyperparameters of a Logistic Regression model class. In this video, we will list them all and give some general advice usage. Here are the topics that we will cover now:

## video4_9
Lesson Summary

Summarize your learning from this lesson.

## video5_1
Lesson Overview

Let us begin with the fifth lesson and understand what we are going to cover in our learning journey.


## video5_2
Decision Trees

Decision trees and the machine learning models that are based on them, random forests and gradient boosted trees, are fundamentally different types of models than generalized linear models, such as logistic regression. GLMs are rooted in the theories of classical statistics, which have a long history. The mathematics behind linear regression were originally developed at the beginning of the 19th century, by Legendre and Gauss. Because of this, the normal distribution is also called the Gaussian. Here are the topics that we will cover now:

## video5_3
Training Decision Trees: Node Impurity
So far, we have treated the decision tree training process as a black box. At this point, you should understand how a decision tree makes predictions using features, and the class fractions of training samples in the leaf nodes. Here are the topics that we will cover now:

## video5_4
Using Decision Trees: Advantages and Predicted Probabilities

The logistic regression has a linear decision boundary, which will be the straight line between the lightest blue and red patches in the background. The logistic regression decision boundary goes right through the middle of the data and doesn't provide a useful classifier. This shows the power of decision trees "out of the box", without the need for engineering non-linear or interaction features. Here are the topics that we will cover now:

## video5_5
Random Forests: Ensembles of Decision Trees

As we saw in the previous video, decision trees are prone to overfitting. This is one of the principle criticisms of their usage, even though they are highly interpretable. We were able to limit this overfitting, to an extent, however, by limiting the maximum depth to which the tree could be grown. Here are the topics that we will cover now:

## video5_6
Fitting a Random Forest

In this video, we will extend our efforts with decision trees, by using the random forest model with cross-validation on the training data from the case study. We will observe the effect of increasing the number of trees in the forest and examine the feature importance that can be calculated using a random forest model.

## video5_7
Lesson Summary

Summarize your learning from this lesson.

## video6_1
Lesson Overview

Let us begin with the sixth lesson and understand what we are going to cover in our learning journey.

## video6_2
Review of Modeling Results

In order to develop a binary classification model to meet the business requirements of our client, we have now tried several modeling approaches to varying degrees of success. In the end, we will pick the one that worked the best, to perform additional analyses on and present to our client. However, it is also good to present the client with findings from the various options that were explored. This shows that a thorough job was done.

## video6_3
Dealing with Missing Data: Imputation Strategies

Recall that in Lesson 1, Data Exploration and Cleaning, we encountered a sizable proportion of samples in the dataset (3,021/29,685 = 10.2%) where the value of the PAY_1 feature was missing. This is a problem that needs to be dealt with, because many machine learning algorithms, including the implementations of logistic regression and random forest in scikit-learn, cannot accept input for model training or testing that includes missing values.

## video6_4
Cleaning the Dataset

In this video, we will be cleaning our dataset to address the missing data entries. We will use the same approach as that in Lesson 1, Data Exploration and Cleaning.


## video6_5
Mode and Random Imputation of PAY_1

In this video, we will try some of the simpler imputation strategies available for PAY_1 and see their effects on cross-validation performance. The first steps will be to append the samples with missing values for PAY_1 to the end of the testing set we've been working with, that has non-missing PAY_1. We'll need to shuffle this when performing cross-validation so that the samples with missing PAY_1 don't all wind up in the same fold, which would create a situation where data in one of the folds was different that the others.

## video6_6
A Predictive Model for PAY_1

The most accurate, but also the most labor-intensive way to impute a feature with missing values is to create a predictive model for that feature.

## video6_7
Using the Imputation Model and Comparing it to Other Methods

We created the model for model-based imputation, so we may now use modelimputed values for PAY_1 in cross-validation with the credit account default model and see how the performance is, in comparison to the simpler imputation methods we already tried.

## video6_8
Financial Analysis

The model performance metrics we have calculated so far were based on abstract measures that could be applied to analyze any classification model: how accurate a model is or how skillful a model is at identifying true positives relative to false positives (ROC AUC), or the correctness of positive predictions (precision). These metrics are important for understanding the basic workings of a model and are widely used within the machine learning community, so it's important to thoroughly understand them. However, for the application of a model to business processes, clients will not always be able to use such model performance metrics to establish an understanding of exactly how they will use a model to guide business decisions, or how much value a model can be expected to create. To go the extra mile and make the connection of the mathematical world of predicted probabilities and thresholds, to the business world of costs and benefits, a financial analysis of some kind is usually required. Here are the topics that we will cover now:

## video6_9
Final Thoughts on Delivering the Predictive Model to the Client

We have now completed modeling activities and created a financial analysis to indicate to the client how they can use the model. While we have created the essential intellectual contributions that are the data scientists' responsibility, it is necessary to agree with the client on the form in which all these contributions will be delivered.

## video6_10
Lesson Summary

Summarize your learning from this lesson.