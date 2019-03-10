#coding by Willhelm
#20190309
import pandas as pd
import numpy as np
import os
# import tensorflow as tf

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

json_add ="C:\\Users\\willh\\Documents\\FYP2\\DataLundary\\RecordsTextOnly\\back\\TrainingSet.json"
# ————————————————————————————
#record strcuture: ID192403  review1689188
#[behavior[multi-dimension, numpy array stored data Frame](asin, brand, categories, unixReviewTime, price, overall),
#CandidateAd[1](asin, brand, categories, unixReviewTime, price), label[1] ]
# ————————————————————————————
df= pd.read_json(open(json_add).read(), lines=True)
dr = np.array(df).reshape((-1,1))
np.random.shuffle(dr)
dr = dr.tolist()
l = []
for record in dr:
    l.append(record[0])
behavior = [d['behavior'] for d in l]
candidateAd = [d['candidateAd'] for d in l]
label = [d['label'] for d in l]


#reviewerID, asin, categories, price, brand, overall, unixReviewTime

for behavior_for_one_comstomer in behavior:
    for one_behavior in behavior_for_one_comstomer:
        cate =one_behavior[2] #might can be mini batch for a user's behavior as a package
        brand = one_behavior[3]
        price = one_behavior[5]
        review_time = one_behavior[7]
        print(cate)
        print("   ")
        print(brand)
        print("   ")
        print(price)
        print("   ")
        print(review_time)
        print("   ")
        break
    break
# ————————————————————————————
#Embedding Layer start
# ————————————————————————————

# ————————————————————————————
#Embedding Layer end
# ————————————————————————————

# ————————————————————————————
#interest extractor layer start
# ————————————————————————————

# ————————————————————————————
#interest extractor layer end
# ————————————————————————————

# ————————————————————————————
#interest evolving layer start
# ————————————————————————————

# ————————————————————————————
#interest evolving layer end
# ————————————————————————————

# ————————————————————————————
#NN start
# ————————————————————————————

# ————————————————————————————
#NN end
# ————————————————————————————

# ————————————————————————————
#input
# ————————————————————————————

# dr_train = dr[:int(len(dr)*0.7)]
# dr_test = dr[int(len(dr)*0.7):]

# ————————————————————————————
#training start
# ————————————————————————————

# ————————————————————————————
#Evaluation start
# ————————————————————————————

