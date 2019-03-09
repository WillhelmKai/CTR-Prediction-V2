#coding by Willhelm
#20190309
import tensorflow as tf
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

trainning_set_ad ="C:\\Users\\willh\\Documents\\FYP2\\DataLundary\\RecordsTextOnly\\back\\TrainingSet.json"
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
#NN
# ————————————————————————————

# ————————————————————————————
#NN
# ————————————————————————————
