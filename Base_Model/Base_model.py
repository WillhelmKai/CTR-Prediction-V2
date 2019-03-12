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
            'candidate_asin': tf.FixedLenFeature([], tf.string),
            'candidate_categories':tf.FixedLenFeature([], tf.string),
            'candidate_brand':tf.FixedLenFeature([], tf.string),
            'candidate_price':tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.float32)
        })
# features = tf.train.batch(features, batch_size = 1, capacity = 10 )
features = tf.train.shuffle_batch(features, batch_size=1, capacity=20, min_after_dequeue=10, num_threads=1)



bc_out = tf.cast(tf.decode_raw(features['behavior_categories'], tf.uint8), tf.float32)
bb_out = tf.cast(tf.decode_raw(features['behavior_brand'], tf.uint8), tf.float32)
bp_out = tf.cast(tf.decode_raw(features['behavior_price'], tf.uint8), tf.float32)
brt_out = tf.cast(tf.decode_raw(features['behavior_review_time'], tf.uint8), tf.float32)
cc_out = tf.cast(tf.decode_raw(features['candidate_categories'], tf.uint8), tf.float32)
cb_out = tf.cast(tf.decode_raw(features['candidate_brand'], tf.uint8), tf.float32)
cp_out = tf.cast(tf.decode_raw(features['candidate_price'], tf.uint8), tf.float32)

ba_out = features['behavior_asin']
ca_out = features['candidate_asin']
l_out = features['label']

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    print(sess.run(l_out))
    print("    ")
    print(sess.run(ca_out))
    print("    ")
    print(sess.run(ba_out))
    print("    ")
    cc_val,cb_val,cp_val, l_val= sess.run([cc_out,cb_out,cp_out,l_out])
    brt_val,bp_val,bb_val, bc_val= sess.run([brt_out,bp_out,bb_out,bc_out])
    print(brt_val)
    print("    ")
    print(bp_val)
    print("    ")
    print(bb_val)
    print("    ")
    print(bc_val)
    print("    ")
    print(cc_val)
    print("    ")
    print(cb_val)
    print("    ")
    print(cp_val)
    print("    ")
    print(l_val)


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

