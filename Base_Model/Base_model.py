#coding by Willhelm
#20190309
import pandas as pd
import numpy as np
import os
# import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

json_add ="C:\\Users\\willh\\Documents\\FYP2\\DataLundary\\RecordsTextOnly\\back\\TrainingSet.json"
# ————————————————————————————
#record strcuture: ID192403  review1689188
#[behavior[multi-dimension, numpy array stored data Frame](asin, brand, categories, unixReviewTime, price, overall),
#CandidateAd[1](asin, brand, categories, unixReviewTime, price), label[1] ]
# ————————————————————————————
# ————————————————————————————
#input
# ————————————————————————————
def read_records(add):
    training_set = 0
    testing_set = 0
    dr= pd.read_json(open(add).read(), lines=True)
    dr = pd.DataFrame.as_matrix(dr)
    print(dr.shape())
    return training_set, testing_set

read_records(json_add)

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
#training start
# ————————————————————————————

# ————————————————————————————
#Evaluation start
# ————————————————————————————

