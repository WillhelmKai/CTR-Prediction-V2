#coding by Willhelm
#20190309
import tensorflow as tf
from tensorflow.contrib import slim
import pandas as pd
import numpy as np
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def weight_variable(shape):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initial(shape))

def bias_variable(shape):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initial(shape))

filename = 'C:\\Users\\willh\\Documents\\FYP2\\DataLundary\\RecordsTextOnly\\TrainingSet.tfrecords'
# filename = '/home/ubuntu/fyp2/LundaryBack/TrainingSet.tfrecords'

# ———————————————————————————— 
#total 192403 records
#categories (1,738), brand (1,3526)
#record strcuture: ID192403  review1689188 with 
#[behavior[multi-dimension, numpy array stored data Frame](asin, brand, categories, unixReviewTime, price, overall),
#CandidateAd[1](asin, brand, categories, unixReviewTime, price), label[1] ]
# ————————————————————————————
epoch = 1
filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

# ————————————————————————————
#read and decode
# ————————————————————————————
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

# ————————————————————————————
#Embedding Layer start
# ————————————————————————————
ph_behavior_categories = tf.placeholder(tf.float32, [738])
ph_behavior_brand = tf.placeholder(tf.float32, [738])
ph_behavior_review_time = tf.placeholder(tf.float32, [1])
ph_behavior_price = tf.placeholder(tf.float32, [1])

W_bc=weight_variable([738, 500])
b_bc=bias_variable([500])
embeded_bc = tf.nn.tanh(tf.matmul(ph_behavior_categories, W_bc)+b_bc)

W_bb=weight_variable([3526, 500])
b_bb=bias_variable([500])
embeded_bc = tf.nn.tanh(tf.matmul(ph_behavior_brand, W_bb)+b_bb)

W_brt=weight_variable([1, 50])
b_brt=bias_variable([50])
embeded_bc = tf.nn.tanh(tf.matmul(ph_behavior_review_time, W_brt)+b_brt)

W_bp=weight_variable([1, 50])
b_bp=bias_variable([50])
embeded_bc = tf.nn.tanh(tf.matmul(ph_behavior_price, W_bp)+b_bp)

embeding_out = tf.concat([embeded_bc,embeded_bc,embeded_bc,embeded_bc], 1)
# ————————————————————————————
#Embedding Layer end
# ————————————————————————————
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    brt_val,bp_val,bb_val, bc_val= sess.run([brt_out,bp_out,bb_out,bc_out])
    cc_val,cb_val,cp_val, l_val= sess.run([cc_out,cb_out,cp_out,l_out])

    bc_val = tf.reshape(bc_val, [-1, cc_val.shape[1]])
    bb_val = tf.reshape(bb_val, [-1, cb_val.shape[1]])
    brt_val = tf.reshape(brt_val, [-1,1])
    bp_val = tf.reshape(bp_val, [-1,1])


    # for i in range(0,bc_val[0]):
    #      pass 
    print(sess.run(embeding_out, feed_dict={ph_behavior_categories:tf.reshape(bc_val[0],[1,-1]), 
    ph_behavior_brand:tf.reshape(bb_val[0],[1,-1]), 
    ph_behavior_review_time:tf.reshape(brt_val[0],[1,-1]),
    ph_behavior_price:tf.reshape(bp_val[0],[1,-1])
        }))


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

