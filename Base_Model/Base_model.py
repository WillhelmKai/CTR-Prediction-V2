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

filename = 'C:\\Users\\willh\\Documents\\FYP2\\DataLundary\\RecordsTextOnly\\TrainingSetTest.tfrecords'

# filename = 'C:\\Users\\willh\\Documents\\FYP2\\DataLundary\\RecordsTextOnly\\TrainingSet.tfrecords'

# filename = '/home/ubuntu/fyp2/LundaryBack/TrainingSet.tfrecords'

# ———————————————————————————— 
#total 192403 records
#categories (1,738), brand (1,3526)
#record strcuture: ID192403  review1689188 with 
#[behavior[multi-dimension, numpy array stored data Frame](asin, brand, categories, unixReviewTime, price, overall),
#CandidateAd[1](asin, brand, categories, unixReviewTime, price), label[1] ]
# ————————————————————————————
epoch = 1
iteration = 192403
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
            'candidate_review_time':tf.FixedLenFeature([1], tf.float32),
            'label': tf.FixedLenFeature([2], tf.float32)
        })
features = tf.train.shuffle_batch(features, batch_size=1, capacity=10, min_after_dequeue=5, num_threads=1)

bc_out = tf.cast(tf.decode_raw(features['behavior_categories'], tf.uint8), tf.float32)
bb_out = tf.cast(tf.decode_raw(features['behavior_brand'], tf.uint8), tf.float32)
bp_out = tf.cast(tf.decode_raw(features['behavior_price'], tf.uint8), tf.float32)
brt_out = tf.cast(tf.decode_raw(features['behavior_review_time'], tf.uint8), tf.float32)
cc_out = tf.cast(tf.decode_raw(features['candidate_categories'], tf.uint8), tf.float32)
cb_out = tf.cast(tf.decode_raw(features['candidate_brand'], tf.uint8), tf.float32)
cp_out = tf.cast(tf.decode_raw(features['candidate_price'], tf.uint8), tf.float32)


ba_out = features['behavior_asin']
ca_out = features['candidate_asin']
crt_out = features['candidate_review_time']
l_out = features['label']

# ————————————————————————————
#Embedding Layer start
# ————————————————————————————
ph_behavior_categories = tf.placeholder(tf.float32, [None,738])
ph_behavior_brand = tf.placeholder(tf.float32, [None,3526])
ph_behavior_review_time = tf.placeholder(tf.float32, [None,1])
ph_behavior_price = tf.placeholder(tf.float32, [None,1])

ph_candidate_categories = tf.placeholder(tf.float32, [None,738])
ph_candidate_brand = tf.placeholder(tf.float32, [None,3526])
ph_candidate_price = tf.placeholder(tf.float32, [None,1])
ph_candidate_review_time = tf.placeholder(tf.float32, [None,1])

ph_label = tf.placeholder(tf.float32, [None,2])

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
first_GRU_outputs, first_final_state = tf.nn.dynamic_rnn(cell, embedding_behavior_out, initial_state=init_state, time_major=True)
#output [-1, 1100,1100]

first_GRU_outputs = tf.reshape(first_GRU_outputs, [-1,1100])

# W_temp=weight_variable([1100, 1100]) #out [-1, 50]
# b_temp=bias_variable([1100])
# first_GRU_outputs = tf.nn.tanh(tf.matmul(tf.reshape(first_GRU_outputs, [-1,1100]), W_temp)+b_temp)

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
W_cc=weight_variable([738, 500]) #out [-1, 500]
b_cc=bias_variable([500])
embeded_cc = tf.nn.tanh(tf.matmul(ph_candidate_categories, W_cc)+b_cc) 

W_cb=weight_variable([3526, 500]) #out [-1, 500]
b_cb=bias_variable([500])
embeded_cb = tf.nn.tanh(tf.matmul(ph_candidate_brand, W_cb)+b_cb)

W_cp=weight_variable([1, 50]) #out [-1, 50]
b_cp=bias_variable([50])
embeded_cp = tf.nn.tanh(tf.matmul(ph_candidate_price, W_cp)+b_cp)

W_crt=weight_variable([1, 50]) #out [-1, 50]
b_crt=bias_variable([50])
embeded_ctr = tf.nn.tanh(tf.matmul(ph_candidate_review_time, W_crt)+b_crt)

embedding_candidate_out = tf.concat([embeded_cc,embeded_cb,embeded_cp,embeded_ctr], 1)#out [-1, 1050] intened to be [-1,1100]

#attention machanism
W_attention = weight_variable([1100,1100])
attention_intermidiate_output = tf.matmul(first_GRU_outputs, tf.matmul(W_attention, tf.transpose(embedding_candidate_out)))
attention_output = tf.div(tf.exp(attention_intermidiate_output),tf.reduce_sum(tf.exp(attention_intermidiate_output)))

#second GRU
second_GRU_input = tf.reshape(tf.matmul(attention_output, embedding_candidate_out), [-1,1100,1]) #[deepth, 1100]
init_state_second = cell.zero_state(batch_size=1100,dtype = tf.float32) 
second_GRU_outputs, final_state_second = tf.nn.dynamic_rnn(cell, second_GRU_input, initial_state=init_state_second, time_major=True)
#out size [length, 1100, 1]
# ————————————————————————————
#interest evolving layer end
# ————————————————————————————
# ————————————————————————————
#NN start
# ————————————————————————————
#flaten out put
W_NN_input = weight_variable([1100, 1])
b_NN_input = bias_variable([1])
NN_input_per= tf.nn.tanh(tf.matmul(tf.transpose(tf.reshape(second_GRU_outputs,[-1, 1100])), tf.reshape(second_GRU_outputs,[-1, 1100]))+b_NN_input)
NN_input = tf.concat([tf.transpose(tf.nn.tanh(tf.matmul(NN_input_per, W_NN_input)+b_NN_input))
    ,ph_candidate_categories,ph_candidate_brand,ph_candidate_price], 1)

W_fc_1 = weight_variable([5365, 200])
b_fc_1 = bias_variable([200])
h_fc_1 = tf.nn.tanh(tf.matmul(NN_input, W_fc_1)+b_fc_1)

W_fc_2 = weight_variable([200, 80])
b_fc_2 = bias_variable([80])
h_fc_2 = tf.nn.tanh(tf.matmul(h_fc_1, W_fc_2)+b_fc_2)

W_fc_3 = weight_variable([80, 2])
b_fc_3 = bias_variable([2])
final_result = tf.nn.softmax(tf.matmul(h_fc_2, W_fc_3)+b_fc_3)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=ph_label,logits=final_result))

precision=tf.metrics.precision(ph_label, final_result)
accuracy =tf.metrics.accuracy(ph_label, final_result)
AUC = tf.metrics.auc(ph_label, final_result)

train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)
# ————————————————————————————
#NN end
# ————————————————————————————
with tf.Session() as sess:
    precision_global = 0
    accuracy_global = 0
    AUC_global = 0
    
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    threats = tf.train.start_queue_runners(sess=sess, coord = coord)

#retrive data
    brt_val,bp_val,bb_val,bc_val= sess.run([brt_out,bp_out,bb_out,bc_out])
    cc_val,cb_val,cp_val,crt_val,l_val= sess.run([cc_out,cb_out,cp_out,crt_out,l_out])
# reformate as the time series behavior 
    bc_val = np.array(bc_val).reshape((-1, cc_val.shape[1])) # [-1, 738] deepth of behavior 
    bb_val = np.array(bb_val).reshape((-1, cb_val.shape[1])) # [-1, 3526]
    brt_val = np.array(brt_val).reshape((-1, 1))
    bp_val = np.array(bp_val).reshape((-1, 1))
    for i in range(epoch):
        print("Epoch No. "+str(i+1)+" started "+"\n")
        for j in range(iteration):
            global_step = i*iteration+j
            sess.run(train_step, feed_dict=
            {ph_behavior_categories:bc_val, ph_behavior_brand:bb_val, 
            ph_behavior_review_time:brt_val,ph_behavior_price:bp_val,
            ph_candidate_categories:cc_val, ph_candidate_brand:cb_val, 
            ph_candidate_review_time:crt_val,ph_candidate_price:cp_val,
            ph_label:l_val})

            loss_temp,precision_temp,accuracy_temp,AUC_temp= sess.run(
            [loss,precision,accuracy,AUC], feed_dict=
            {ph_behavior_categories:bc_val, ph_behavior_brand:bb_val, 
            ph_behavior_review_time:brt_val,ph_behavior_price:bp_val,
            ph_candidate_categories:cc_val, ph_candidate_brand:cb_val, 
            ph_candidate_review_time:crt_val,ph_candidate_price:cp_val,
            ph_label:l_val})

            precision_global =precision_global+precision_temp[0]
            accuracy_global =accuracy_global+accuracy_temp[0]
            AUC_global = AUC_global+AUC_temp[0]
            if (global_step%5000):
                print("Step: "+str(global_step)+"  Loss: "+str(loss_temp)+
                "  precision: "+str(precision_global/global_step)+
                "  accuracy: "+str(accuracy_global/global_step)+
                "  AUC: "+str(AUC_global/global_step))
    print("Training finished")
    coord.request_stop()
    coord.join(threats)
# ————————————————————————————
#Evaluation start
# ————————————————————————————