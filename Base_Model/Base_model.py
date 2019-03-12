#coding by Willhelm
#20190309
import tensorflow as tf
from tensorflow.contrib import slim
import pandas as pd
import numpy as np
import os

filename = 'C:\\Users\\willh\\Documents\\FYP2\\DataLundary\\RecordsTextOnly\\TrainingSet.tfrecord'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# json_add ="C:\\Users\\willh\\Documents\\FYP2\\DataLundary\\RecordsTextOnly\\back\\TrainingSet.json"
# json_add = '/home/ubuntu/fyp2/LundaryBack/TrainingSet.json'

# ————————————————————————————
#record strcuture: ID192403  review1689188
#[behavior[multi-dimension, numpy array stored data Frame](asin, brand, categories, unixReviewTime, price, overall),
#CandidateAd[1](asin, brand, categories, unixReviewTime, price), label[1] ]
# ————————————————————————————

filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,
        features={
            'behavior_asin': tf.FixedLenFeature([], tf.string),
            'behavior_categories': tf.FixedLenFeature([], tf.string),
            'behavior_brand': tf.FixedLenFeature([], tf.string),
            'behavior_price': tf.FixedLenFeature([], tf.string),
            'behavior_review_time': tf.FixedLenFeature([], tf.string),
            # 'candidate_asin': tf.FixedLenFeature([], tf.string),
            'candidate_categories':tf.FixedLenFeature([], tf.string),
            'candidate_brand':tf.FixedLenFeature([], tf.string),
            'candidate_price':tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.float32)
        })
# features = tf.train.batch(features, batch_size = 1, capacity = 10 )
features = tf.train.shuffle_batch(features, batch_size=1, capacity=20, min_after_dequeue=10, num_threads=1)

ba_out = features['behavior_asin']

bc_out = tf.decode_raw(features['behavior_categories'], tf.uint8)
bc_out = tf.cast(bc_out, tf.float32)

bb_out = tf.decode_raw(features['behavior_brand'], tf.uint8)
bb_out = tf.cast(bb_out, tf.float32)

bp_out = tf.decode_raw(features['behavior_price'], tf.uint8)
bp_out = tf.cast(bp_out, tf.float32)

brt_out = tf.decode_raw(features['behavior_review_time'], tf.uint8)
brt_out = tf.cast(brt_out, tf.float32)

# ca_out = features['candidate_asin']

cc_out = tf.decode_raw(features['candidate_categories'], tf.uint8)
cc_out = tf.cast(cc_out, tf.float32)

cb_out = tf.decode_raw(features['candidate_brand'], tf.uint8)
cb_out = tf.cast(cb_out, tf.float32)

cp_out = tf.decode_raw(features['candidate_price'], tf.uint8)
cp_out = tf.cast(cp_out, tf.float32)

l_out = features['label']

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    print(sess.run(ba_out))
    # cc_val,cb_val,cp_val, l_val= sess.run([cc_out,cb_out,cp_out,l_out])
    # brt_val,bp_val,bb_val, bc_val= sess.run([brt_out,bp_out,bb_out,bc_out])
    # print(brt_val)
    # print("    ")
    # print(bp_val)
    # print("    ")
    # print(bb_val)
    # print("    ")
    # print(bc_val)
    # print("shape of cc "+str(cc_val.shape))
    # print("shape of bc "+str(bc_val.shape))
    # print("   ")
    # print("shape of cb "+str(cb_val.shape))
    # print("shape of bb "+str(bb_val.shape))


# df= pd.read_json(open(json_add).read(), lines=True)
# dr = np.array(df).reshape((-1,1))
# np.random.shuffle(dr)
# dr = dr.tolist()
# l = []
# for record in dr:
#     l.append(record[0])
# behavior = [d['behavior'] for d in l]
# candidateAd = [d['candidateAd'] for d in l]
# label = [d['label'] for d in l]

# #reviewerID, asin, categories[3526], price, brand[738], overall, unixReviewTime

# for behavior_for_one_comstomer in behavior:
#     for one_behavior in behavior_for_one_comstomer:
#         cate =one_behavior[2] #might can be mini batch for a user's behavior as a package
#         brand = one_behavior[3]
#         price = one_behavior[5]
#         review_time = one_behavior[7]
#         print(cate)
#         print(len(cate))
#         print("   ")
#         print(brand)
#         print(len(brand))
#         print("   ")
#         print(price)
#         print("   ")
#         print(review_time)
#         print("   ")
#         break
#     break

# ————————————————————————————
#Embedding Layer start
# ————————————————————————————
# categories_input = tf.placeholder(tf.float32, [None, 100])
# brand_input = tf.placeholder(tf.float32, [None, 100])
# price_input = tf.placeholder(tf.float32, [None, 100])
# review_time_input = tf.placeholder(tf.float32, [None, 100])

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

