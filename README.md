# CTR-Prediction-V2.0

## Background

CTR prediction systems use information such user profile, product profile and context information to predict the probability for clicking or not. Although machine learning has been the mainstream method of CTR prediction for a long time, diversity and evolving phenomenon of interests obstacle the improvement of prediction accuracy. For instance, a customer may be not only interested in a number of products at the same time, but interest variate with time. Additionally, interest as high-level features is believed to be better to reflect the motivation of a customer rather than the ground behaviors. Therefore, we proposed to implement a novel deep recurrent neural network. Various technologies are applied to conquer problems described abrove. All experiments are done based on a real-world dataset collected from Amazon.com that contains over 142 million ad records.


***************************************************************************************************************


Experiment Dataset: http://jmcauley.ucsd.edu/data/amazon/

This project proposed to develop a advanced CTR prediction model based on CTR-Prediction-V1.x. 

Inspried by Alibaba's (DIN, DIEN, AMS) and MS'(DCN) works, we found servious ways to improve the prediction accuracy. 
1. Enlarge the length of input vector (such as images and unstractured features)
2. Using the fixed length input more efficient (such as noise canceling of attention machenism, extract interantion and time serial info)
