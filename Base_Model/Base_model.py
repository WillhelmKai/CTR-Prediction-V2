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
df= pd.read_json(open(json_add).read(), lines=True)
dr = np.array(df).reshape((-1,1))
np.random.shuffle(dr)
dr_train = dr[:int(len(dr)*0.7)]
dr_test = dr[int(len(dr)*0.7):]
# ————————————————————————————
#training start
# ————————————————————————————

# ————————————————————————————
#Evaluation start
# ————————————————————————————

