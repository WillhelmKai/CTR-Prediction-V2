#coding by Willhelm
#20190309
import tensorflow as tf
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

trainning_set_ad ="C:\\Users\\willh\\Documents\\FYP2\\DataLundary\\RecordsTextOnly\\back\\TrainingSet.json"
# ————————————————————————————
#target record strcuture: ID192403  review1689188
#[reviewerID[1],behavior[multi-dimension, numpy array stored data Frame](asin, brand, categories, unixReviewTime, price, overall),
#CandidateAd[1](asin, brand, categories, unixReviewTime, price), label[1] ]
# ————————————————————————————

# ————————————————————————————
#Embedding Layer
# ————————————————————————————

# ————————————————————————————
# ————————————————————————————

