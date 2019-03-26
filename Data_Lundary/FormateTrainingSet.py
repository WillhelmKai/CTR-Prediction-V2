#coding by Willhelm
#20190315
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer 
from sklearn.utils import shuffle
pd.set_option('display.max_columns', 10000, 'display.max_rows', 10000)

def generate_false_candidate(behavior):
    result = 0 
    index = np.random.randint(0, len(df))#random number from item mate
    while(True):
        asin = df.ix[index, ['asin']]#retrive the asin from df
        behavior_asin = pd.DataFrame.from_records(behavior['asin'])
        if (behavior['asin'].isin(asin).any()):
            #if the item info in the history or consist with true candidate goes on
            index = np.random.randint(0, len(df))
        else:
            result =  df.ix[index]
            break
    return result 

def convert_to_numpy(input):
    result = [] 
    for item in input:
        result.append(item)
    result = np.array(result)
    return result

def toBytes(input):
    result = np.array(input).astype(np.uint8).tostring() 
    return result

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# ————————————————————————————
#target record strcuture: ID192403  review1689188
#[reviewerID[1],behavior[multi-dimension, numpy array stored data Frame](asin, brand, categories, unixReviewTime, price, overall),
#CandidateAd[1](asin, brand, categories, unixReviewTime, price), label[1] ]
# ————————————————————————————
#destination
# tfrecord_train = 'C:\\Users\\willh\\Documents\\FYP2\\DataLundary\\RecordsTextOnly\\TrainingSet.tfrecords'
# tfrecord_test = 'C:\\Users\\willh\\Documents\\FYP2\\DataLundary\\RecordsTextOnly\\TestingSet.tfrecords'
# text_json_add = 'C:\\Users\\willh\\Documents\\FYP2\\DataLundary\\RecordsTextOnly\\StrcuturedTextOnly.json'

tfrecord_train = '/home/ubuntu/fyp2/LundaryBack/TrainingSet.tfrecords'
tfrecord_test = '/home/ubuntu/fyp2/LundaryBack/TestingSet.tfrecords'
text_json_add = '/home/ubuntu/fyp2/LundaryBack/StrcuturedTextOnly.json'
data_str = open(text_json_add).read()
df = pd.read_json(data_str, lines= True)
# ————————————————————————————
# ————————————————————————————
# df = df[0:int(len(df)*0.01)]
# ————————————————————————————
# ————————————————————————————
# creat empty list with the length of df 
print("Binarization:")
mlb = MultiLabelBinarizer()
c = []
b = []
df.fillna(value = 'No_value', inplace=True)
for i in range(0,len(df)):
    b.append(df.ix[i,['brand']])
    c.append(df.ix[i,['categories']][0][0])
#encoding
b = np.array(mlb.fit_transform(b))
b = np.reshape(b, (len(df),1,-1))
df[['brand']] = pd.DataFrame.from_records(b)

#replace with np arrray
c = np.array(mlb.fit_transform(c))
c = np.reshape(c, (len(df),1,-1))
df[['categories']] = pd.DataFrame.from_records(c)

# ————————————————————————————
#form up dictionary for brand and categories
# ————————————————————————————
print("finished")

# select columns and remove ruplicate
print("forming reviewer dictionary")
reviewID_dic = df[['reviewerID']].drop_duplicates().reset_index(drop=True)
reviewID_dic = shuffle(reviewID_dic)
reviewID_train = reviewID_dic[: int(0.8*len(reviewID_dic))]
reviewID_test = reviewID_dic[int(0.8*len(reviewID_dic))+1:]
# ————————————————————————————
#make up training set
# ————————————————————————————
lower_boundary = 3  #if the num of user behavior lower than this num, record will be abandoned 
acc = 0
no_value = 0 
writer = tf.python_io.TFRecordWriter(tfrecord_train)
for reviewerID in reviewID_train.itertuples(index = False):
    #search all his or her historical behaviors sorted by time
    his_behavior = df.loc[df['reviewerID'] == reviewerID[0]].sort_values(by = ['unixReviewTime']).reset_index(drop=True)
    #prograssivly generate true and false able and record
    if (len(his_behavior) >= lower_boundary):
        # his_behavior.ix[0,len(his_behavior),['categories']] = his_behavior.ix[0,len(his_behavior),['categories']][0][0]
        for i in range(len(his_behavior)-1 ,len(his_behavior)): #select the behavior sequence number as the candidata ad
            behavior_asin = convert_to_numpy(his_behavior.loc[0:i-1]['asin'])
            behavior_categories = convert_to_numpy(his_behavior.loc[0:i-1]['categories']) #historical behavior categories
            behavior_brand = convert_to_numpy(his_behavior.loc[0:i-1]['brand'])
            behavior_price = convert_to_numpy(his_behavior.loc[0:i-1]['price'])
            behavior_review_time = convert_to_numpy(his_behavior.loc[0:i-1]['unixReviewTime'])

            try:
                candidate_true_asin = his_behavior.loc[i]['asin']
                candidate_true_categories = his_behavior.loc[i]['categories'] #candidate ad
                candidate_true_brand = his_behavior.loc[i]['brand']
                candidate_true_price = his_behavior.loc[i]['price']
                candidate_review_time = his_behavior.loc[i]['unixReviewTime']
                true_label = [1.0, 0.0]

                # print(str(behavior_asin))
                # break
                example = tf.train.Example(features = tf.train.Features(feature = {
                'behavior_asin':tf.train.Feature(bytes_list = tf.train.BytesList(value = [bytes(str(behavior_asin), encoding = "utf8")])),
                'behavior_categories':_bytes_feature(toBytes(behavior_categories)),
                'behavior_brand':_bytes_feature(toBytes(behavior_brand)),
                'behavior_price':_bytes_feature(toBytes(behavior_price)),
                'behavior_review_time':_bytes_feature(toBytes(behavior_review_time)),
                'candidate_asin': tf.train.Feature(bytes_list = tf.train.BytesList(value = [bytes(candidate_true_asin, encoding = "utf8")])),
                'candidate_categories':_bytes_feature(toBytes(candidate_true_categories)),
                'candidate_brand':_bytes_feature(toBytes(candidate_true_brand)),
                'candidate_price':_bytes_feature(toBytes(candidate_true_price)),
                'candidate_review_time':tf.train.Feature(float_list = tf.train.FloatList(value=[candidate_review_time])),
                'label':tf.train.Feature(float_list = tf.train.FloatList(value=true_label))
                }))

                serialized = example.SerializeToString()
                writer.write(serialized)


                candidate_false = generate_false_candidate(his_behavior.loc[0:i-1])
                candidate_true_asin = candidate_false['asin']
                candidate_true_categories = candidate_false['categories'] #candidate ad
                candidate_true_brand = candidate_false['brand']
                candidate_true_price = candidate_false['price']
                candidate_review_time = candidate_false['unixReviewTime']
                false_label = [0.0, 1.0]

                example = tf.train.Example(features = tf.train.Features(feature = {
                'behavior_asin':tf.train.Feature(bytes_list = tf.train.BytesList(value = [bytes(str(behavior_asin), encoding = "utf8")])),
                'behavior_categories':_bytes_feature(toBytes(behavior_categories)),
                'behavior_brand':_bytes_feature(toBytes(behavior_brand)),
                'behavior_price':_bytes_feature(toBytes(behavior_price)),
                'behavior_review_time':_bytes_feature(toBytes(behavior_review_time)),
                'candidate_asin': tf.train.Feature(bytes_list = tf.train.BytesList(value = [bytes(candidate_true_asin, encoding = "utf8")])),
                'candidate_categories':_bytes_feature(toBytes(candidate_true_categories)),
                'candidate_brand':_bytes_feature(toBytes(candidate_true_brand)),
                'candidate_price':_bytes_feature(toBytes(candidate_true_price)),
                'candidate_review_time':tf.train.Feature(float_list = tf.train.FloatList(value=[candidate_review_time])),
                'label':tf.train.Feature(float_list = tf.train.FloatList(value=false_label))
                }))

                serialized = example.SerializeToString()
                writer.write(serialized)

            except Exception as e:
                no_value = no_value+1
    acc = acc+1
    # break
    if (acc%50000):
        print('\r', str(acc/len(reviewID_dic)).ljust(10),end='')
writer.close()
print("Training generating done")
print("total record  "+str(acc))
print("no value record "+str(no_value))
print("   ")
# ————————————————————————————
# generate testing set
# ————————————————————————————
acc = 0
no_value = 0 
writer = tf.python_io.TFRecordWriter(tfrecord_test)
for reviewerID in reviewID_test.itertuples(index = False):
    #search all his or her historical behaviors sorted by time
    his_behavior = df.loc[df['reviewerID'] == reviewerID[0]].sort_values(by = ['unixReviewTime']).reset_index(drop=True)
    #prograssivly generate true and false able and record
    if (len(his_behavior) >= lower_boundary):
        # his_behavior.ix[0,len(his_behavior),['categories']] = his_behavior.ix[0,len(his_behavior),['categories']][0][0]
        for i in range(len(his_behavior)-1 ,len(his_behavior)): #select the behavior sequence number as the candidata ad
            behavior_asin = convert_to_numpy(his_behavior.loc[0:i-1]['asin'])
            behavior_categories = convert_to_numpy(his_behavior.loc[0:i-1]['categories']) #historical behavior categories
            behavior_brand = convert_to_numpy(his_behavior.loc[0:i-1]['brand'])
            behavior_price = convert_to_numpy(his_behavior.loc[0:i-1]['price'])
            behavior_review_time = convert_to_numpy(his_behavior.loc[0:i-1]['unixReviewTime'])

            try:
                candidate_true_asin = his_behavior.loc[i]['asin']
                candidate_true_categories = his_behavior.loc[i]['categories'] #candidate ad
                candidate_true_brand = his_behavior.loc[i]['brand']
                candidate_true_price = his_behavior.loc[i]['price']
                candidate_review_time = his_behavior.loc[i]['unixReviewTime']
                true_label = [1.0, 0.0]

                # print(str(behavior_asin))
                # break
                example = tf.train.Example(features = tf.train.Features(feature = {
                'behavior_asin':tf.train.Feature(bytes_list = tf.train.BytesList(value = [bytes(str(behavior_asin), encoding = "utf8")])),
                'behavior_categories':_bytes_feature(toBytes(behavior_categories)),
                'behavior_brand':_bytes_feature(toBytes(behavior_brand)),
                'behavior_price':_bytes_feature(toBytes(behavior_price)),
                'behavior_review_time':_bytes_feature(toBytes(behavior_review_time)),
                'candidate_asin': tf.train.Feature(bytes_list = tf.train.BytesList(value = [bytes(candidate_true_asin, encoding = "utf8")])),
                'candidate_categories':_bytes_feature(toBytes(candidate_true_categories)),
                'candidate_brand':_bytes_feature(toBytes(candidate_true_brand)),
                'candidate_price':_bytes_feature(toBytes(candidate_true_price)),
                'candidate_review_time':tf.train.Feature(float_list = tf.train.FloatList(value=[candidate_review_time])),
                'label':tf.train.Feature(float_list = tf.train.FloatList(value=true_label))
                }))

                serialized = example.SerializeToString()
                writer.write(serialized)


                candidate_false = generate_false_candidate(his_behavior.loc[0:i-1])
                candidate_true_asin = candidate_false['asin']
                candidate_true_categories = candidate_false['categories'] #candidate ad
                candidate_true_brand = candidate_false['brand']
                candidate_true_price = candidate_false['price']
                candidate_review_time = candidate_false['unixReviewTime']
                false_label = [0.0, 1.0]

                example = tf.train.Example(features = tf.train.Features(feature = {
                'behavior_asin':tf.train.Feature(bytes_list = tf.train.BytesList(value = [bytes(str(behavior_asin), encoding = "utf8")])),
                'behavior_categories':_bytes_feature(toBytes(behavior_categories)),
                'behavior_brand':_bytes_feature(toBytes(behavior_brand)),
                'behavior_price':_bytes_feature(toBytes(behavior_price)),
                'behavior_review_time':_bytes_feature(toBytes(behavior_review_time)),
                'candidate_asin': tf.train.Feature(bytes_list = tf.train.BytesList(value = [bytes(candidate_true_asin, encoding = "utf8")])),
                'candidate_categories':_bytes_feature(toBytes(candidate_true_categories)),
                'candidate_brand':_bytes_feature(toBytes(candidate_true_brand)),
                'candidate_price':_bytes_feature(toBytes(candidate_true_price)),
                'candidate_review_time':tf.train.Feature(float_list = tf.train.FloatList(value=[candidate_review_time])),
                'label':tf.train.Feature(float_list = tf.train.FloatList(value=false_label))
                }))

                serialized = example.SerializeToString()
                writer.write(serialized)

            except Exception as e:
                no_value = no_value+1
    acc = acc+1
    # break
    if (acc%50000):
        print('\r', str(acc/len(reviewID_dic)).ljust(10),end='')
writer.close()

print("Testing generating done")
print("total record  "+str(acc))
print("no value record "+str(no_value))
# ————————————————————————————
# the end
# ————————————————————————————