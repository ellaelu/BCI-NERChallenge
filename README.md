# BCI-NERChallenge

Our Team's attempt at the BCI Challenge @ NER 2015

**Authors**:
* Talal Alqadi
* Ella Lucas
* Maisha Maliha
* Marcus Xue
* Chufan Yang

**Contents** :


- [Introduction](#introduction)
    - [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Results](#results)
- [Discussion](#Discussion)
- [References ](#references)
- [Code](#code)

## Introduction

This project summarizes our efforts in training a model to classify whether a feedback was good or bad based on the P300 spelling task and dataset described in BCI NER 2015 Kaggle Competition. The goal of this challenge is to determine whether the item selected by the individual is the correct one by analyzing the EEG evoked responses recorded every 5 milliseconds from 56 passive EEG sensors following the extended 10-20 system. We participated in this challenge by preprocessing the data and comparing model performances. 


BCIs are where the future is headed as they are being used in neuroprostheses, the medical market, and the gaming industry, among many other applications. Our project’s main objective is to classify the BCI data from the BCI NER 2015 Kaggle Competition. We will be using different kinds of models to find the most optimal one. The P300 dataset from the challenge has two outcomes 0 or 1 (good or bad). A good outcome is when the selected item is similar to the expected item and a bad outcome is when they are different. Our task is to correctly classify these outputs with the highest level of accuracy. In this Supervised Learning task, we compare the performances of models like Logistic Regression, SVM, LDA, RF, Decision Nets, EEGNet and Elasticnet. The main goal of our experiment is to familiarize ourselves with the classification task associated with BCI data.


#### Dataset
Download Data from: https://www.kaggle.com/c/inria-bci-challenge/data

EEG evoked responses are recorded every 5 milliseconds at 56 passive Ag/AgCl EEG sensors following the extended 10-20 system to evaluate whether the item selection is correct or not. Twenty-six subjects are part of this study, with sixteen of them being the training dataset. All subjects underwent 5 sessions, consisting of 60 feedbacks each, and 100 feedbacks for the last session, totalling up to 340 sessions for each subject. Thus each raw session file contains 100,000+ rows, and 59 columns (56 EEG frequencies, Time, EOG, and FeedBack Label).

Move all of the downloaded content to the Data subfolder within this repo, extract the train.zip and test.zip to their respective train and test subfolders.

## Preproccessing:

Run preprocess.ipynb to create X_train and X_test to be used in models

or

Ingest the data from as they are the post processed data using Riemann spaces (XDawnCovariance and Tangent Space mapping): 
* /Data/X_train_final.npy or /Data/X_train_final(bs).npy for the baseline corrected version
* /Data/X_test_final.npy or /Data/X_test_final(bs).npy for the baseline corrected version


Train labels can be found in:
* /Data/TrainLabels.csv

True Test labels can be found in:
* /Data/true_labels.csv

See Models.ipynb for application of data.

![Image of Preprocessing](https://i.imgur.com/7VxUS4j.png)

#### Bandpassing
EEG signals are bandpass filtered by a fifth order Butterworth filter low and high 
passes at 1hz and 40 hz.

#### Epoching and Baseline Correction
Epoching events in a dataset like this allow us to extract all the necessary data. For this case, the best results were from epoching 1.3 seconds after the Feedback event. 

Baseline correction was applied by calculating the mean of the EEG frequencies 400 to 300 ms before the event occurs. This allows us to subtract the potential noise in the data. This pre-processing leads the 5*16 (sessions * training_subjects) training input files into a (16*340, 56,  260) array, and the 5*10 (sessions * testing_subjects) testing input files into a (10*340, 56, 260)  array.

The 2 arrays consist of all EEG recordings at the feedback event, and after it by 1.3 seconds. The epoched recordings were baseline corrected by subtracting the values by the mean of the EEG values 400 to 300 ms before the Feedback. The EOG, time, and Feedback columns were all dropped after each epoch. The previous arrays are further pre-processed through xDawnCovariance and Tangent Space functions.

#### xDawnCovariance and Tangent Space
xDawnCovariance consists of estimating a special form covariance matrix with tools from Riemannian Geometry to manipulate them combined with xDawn spatial filtering (Rivet). Spatial filtering on the data increases signal to noise ratio and reduces dimensionality. For this case a set of 5 spatial filters were built using the XDawnCovariance function provided by the Riemannian Library. The spatial filters are then estimated using the xDawn algorithm. Tangent Space mapping is then applied to project the matrices into Euclidean space from the Reimannian manifold (Barachant). 

The final training and test sets are of the following shapes respectively: (5440, 210) and (3400, 210). Adding metadata is an option for some of the models, which increases the shape of the data into being (5440, 212) and (3400, 212) by adding the session number and feedback number in relation to the first feedback presented to the subject. 

## Results

The analysis techniques we applied are Logistic Regression, SVM, Random Forest, Decision Tree, EEGNet, as well as Elastic Net. The highest AUC calculated by sklearn.metrics was achieved by Logistic Regression with AUC = 0.67 on the baseline corrected version of the rebalanced dataset. We also submitted the models to Kaggle in order to receive their scores. The highest public score was achieved by Random Forests using the baseline corrected dataset with metadata included with a score of 0.696. The highest private score was achieved by Elastic Net on the non-baseline corrected dataset with metadata included with a score of 0.66, placing it in about 30th place on the Kaggle leaderboards.

## Discussion

We have learned that different models could vary dramatically to the output on the same data due to their internal decision making mechanism.. We also learn that the most complicated model does not guarantee the most accurate result, and sometimes a simple model, such as Logistic Regression or Elastic Net, can deliver an efficient solution to our current problem given efficient preprocessing. Each model performed best in different cases and in different manipulations of the dataset. This reinforces a crucial idea in Machine Learning that highlights the importance of pre-processing and its subsequent integration with the model.

With better optimization and data gathering techniques, we believe helping locked-in syndrome patients is one of the major potential applications of ML in the future. Given our time constraints, we were unable to examine and test out many different types of pre-processing techniques. Given more time, we could’ve examined different epoching, bandpass, and baseline correction values and seen how the differences in those values affect each model's performance. We would also be excited to try other machine learning models and compose a more comprehensive result to determine which one works the best on the given data set.


# References

> Dzulkifli, Syahizul Amri, et al. “Improved Weighted Learning Support Vector Machines (SVM) 
for High Accuracy.” Improved Weighted Learning Support Vector Machines (SVM) for High Accuracy | Proceedings of the 2019 2nd International Conference on Computational Intelligence and Intelligent Systems, 1 Nov. 2019, dl.acm.org/doi/pdf/10.1145/3372422.3372432.

> Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer 
interfaces. Journal of Neural Engineering, 15(5), 056013. doi: 10.1088/1741-2552/aace8c

> Mishra, A. (2018, February 24). Retrieved from https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234

> Tolles, Juliana; Meurer, William J (2016). "Logistic Regression Relating Patient Characteristics to Outcomes". JAMA. 316 (5): 533–4. doi:10.1001/jama.2016.7653. ISSN 0098-7484. OCLC 6823603312. PMID 27483067.

> Kamiński, B.; Jakubczyk, M.; Szufel, P. (2017). "A framework for sensitivity analysis of 
decision trees". Central European Journal of Operations Research. 26 (1): 135–159. doi:10.1007/s10100-017-0479-6. PMC 5767274. PMID 29375266.

> Kevric, J., & Subasi, A. (2017). Comparison of signal decomposition methods in classification of EEG signals for motor-imagery BCI system. Biomedical Signal Processing and Control, 31, 398–406. doi: 10.1016/j.bspc.2016.09.007

> Wang, Y., Deng, Y., Li, Z., & Zhang, H. (n.d.). Cogs 189 Final Project Presentation. La Jolla.

> Zou, Hui, and Trevor Hastie. “Addendum: Regularization and Variable Selection via the Elastic 
Net.” Journal of the Royal Statistical Society: Series B (Statistical Methodology), vol. 67, no. 5, 2005, pp. 768–768., doi:10.1111/j.1467-9868.2005.00527.x.

> Hastie, Trevor, et al. The Elements of Statistical Learning: Data Mining, Inference, and 
Prediction. Springer, 2017.

> Rivet, B.; Souloumiac, A.; Attina, V.; Gibert, G., "xDAWN Algorithm to Enhance Evoked Potentials: Application to Brain–Computer Interface," IEEE Transactions on Biomedical Engineering, vol.56, no.8, pp.2035,2043, Aug. 2009

> A. Barachant, S. Bonnet, M. Congedo and C. Jutten, “Multiclass Brain-Computer Interface Classification by Riemannian Geometry,” in IEEE Transactions on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012.

