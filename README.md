# CTR-Prediction-V2.0
Experiment Dataset: http://jmcauley.ucsd.edu/data/amazon/

This project proposed to develop a advanced CTR prediction model based on CTR-Prediction-V1.x. 

Inspried by Alibaba's (DIN, DIEN, AMS) and MS'(DCN) works, we found servious ways to improve the prediction accuracy. 
1. Enlarge the length of input vector (such as images and unstractured features)
2. Using the fixed length input more efficient (such as noise canceling of attention machenism, extract interantion and time serial info)

In the first stage:
we proposed to implement the DIEN model as our experiment based model
