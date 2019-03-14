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
ph_behavior_categories = tf.placeholder(tf.float32, [None,738])
ph_behavior_brand = tf.placeholder(tf.float32, [None,3526])
ph_behavior_review_time = tf.placeholder(tf.float32, [None,1])
ph_behavior_price = tf.placeholder(tf.float32, [None,1])

W_bc=weight_variable([738, 500]) #out [-1, 500]
b_bc=bias_variable([500])
embeded_bc = tf.nn.tanh(tf.matmul(ph_behavior_categories, W_bc)+b_bc) 

W_bb=weight_variable([3526, 500]) #out [-1, 500]
b_bb=bias_variable([500])
embeded_bb = tf.nn.tanh(tf.matmul(ph_behavior_brand, W_bb)+b_bb)

W_brt=weight_variable([1, 50]) #out [-1, 50]
b_brt=bias_variable([50])
embeded_brt = tf.nn.tanh(tf.matmul(ph_behavior_review_time, W_brt)+b_brt)

W_bp=weight_variable([1, 50]) #out [-1, 50]
b_bp=bias_variable([50])
embeded_bp = tf.nn.tanh(tf.matmul(ph_behavior_price, W_bp)+b_bp)

embedding_behavior_out = tf.concat([embeded_bc,embeded_bb,embeded_brt,embeded_bp], 1) #out [-1, 1100]
embedding_behavior_out = tf.reshape(embedding_behavior_out,[-1,1100,1])
# ————————————————————————————
#Embedding Layer end
# ————————————————————————————
# ————————————————————————————
#interest extractor layer start
# ————————————————————————————
cell = tf.nn.rnn_cell.GRUCell(num_units = 1)
init_state = cell.zero_state(batch_size=1100,dtype = tf.float32) #batch size intented to be, out can be [-1, 1100] multiply oneby one
first_GRU_outputs, final_state = tf.nn.dynamic_rnn(cell, embedding_behavior_out, initial_state=init_state, time_major=True)
#output [-1, 1100,1100]

W_temp=weight_variable([1100, 1050]) #out [-1, 50]
b_temp=bias_variable([1050])
first_GRU_outputs = tf.nn.tanh(tf.matmul(tf.reshape(first_GRU_outputs, [-1,1100]), W_temp)+b_temp)

    #[1100, -1] 
#down size  can be multiplied(one by one)by input [-1,1100] * [1100,-1]the output to ????
        # ————————————————————————————
        #Auxiliary Loss start
        # ————————————————————————————
        #target before sigmoid [-1, 1]
        # ————————————————————————————
        #Auxiliary Loss end
        # ————————————————————————————

# ————————————————————————————
#interest extractor layer end
# ————————————————————————————
# ————————————————————————————
#interest evolving layer start
# ————————————————————————————
#embedding candidate features
ph_candidate_categories = tf.placeholder(tf.float32, [None,738])
ph_candidate_brand = tf.placeholder(tf.float32, [None,3526])
ph_candidate_price = tf.placeholder(tf.float32, [None,1])

W_cc=weight_variable([738, 500]) #out [-1, 500]
b_cc=bias_variable([500])
embeded_cc = tf.nn.tanh(tf.matmul(ph_candidate_categories, W_cc)+b_cc) 

W_cb=weight_variable([3526, 500]) #out [-1, 500]
b_cb=bias_variable([500])
embeded_cb = tf.nn.tanh(tf.matmul(ph_candidate_brand, W_cb)+b_cb)

W_cp=weight_variable([1, 50]) #out [-1, 50]
b_cp=bias_variable([50])
embeded_cp = tf.nn.tanh(tf.matmul(ph_candidate_price, W_cp)+b_cp)

embedding_candidate_out = tf.concat([embeded_cc,embeded_cb,embeded_cp], 1)#out [-1, 1050] intened to be [-1,1100]

#attention machanism
W_attention = weight_variable([1050,1050])
attention_intermidiate_output = tf.matmul(first_GRU_outputs, tf.matmul(W_attention, tf.transpose(embedding_candidate_out)))
attention_output = tf.div(tf.exp(attention_intermidiate_output),tf.reduce_sum(tf.exp(attention_intermidiate_output)))

#second GRU
second_GRU_input = tf.reshape(tf.matmul(attention_output, embedding_candidate_out), [-1,1050,1]) #[deepth, 1100]
init_state_second = cell.zero_state(batch_size=1050,dtype = tf.float32) 
second_GRU_outputs, final_state_second = tf.nn.dynamic_rnn(cell, second_GRU_input, initial_state=init_state_second, time_major=True)
# ————————————————————————————
#interest evolving layer end
# ————————————————————————————
# ————————————————————————————
#NN start
# ————————————————————————————

# ————————————————————————————
#NN end
# ————————————————————————————


with tf.Session() as sess:
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    threats = tf.train.start_queue_runners(sess=sess, coord = coord)

#retrive data
    brt_val,bp_val,bb_val, bc_val= sess.run([brt_out,bp_out,bb_out,bc_out])
    cc_val,cb_val,cp_val, l_val= sess.run([cc_out,cb_out,cp_out,l_out])

#reformate as the time series behavior 
    bc_val = np.array(bc_val).reshape((-1, cc_val.shape[1])) # [-1, 738] deepth of behavior 
    bb_val = np.array(bb_val).reshape((-1, cb_val.shape[1])) # [-1, 3526]
    brt_val = np.array(brt_val).reshape((-1, 1))
    bp_val = np.array(bp_val).reshape((-1, 1))

    out = sess.run(second_GRU_outputs, feed_dict=
    {ph_behavior_categories:bc_val, ph_behavior_brand:bb_val, 
    ph_behavior_review_time:brt_val,ph_behavior_price:bp_val,
    ph_candidate_categories:cc_val, ph_candidate_brand:cb_val, 
    ph_candidate_price:cp_val
    })

    # out = sess.run(embedding_candidate_out, feed_dict=
    # {ph_candidate_categories:cc_val, ph_candidate_brand:cb_val, 
    # ph_candidate_price:cp_val})

    print(out)
    print("   ")
    print(out.shape)

    coord.request_stop()
    coord.join(threats)




# ————————————————————————————
#Evaluation start
# ————————————————————————————