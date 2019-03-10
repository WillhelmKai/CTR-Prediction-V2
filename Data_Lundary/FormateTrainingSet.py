#coding by Willhelm
#20190315
import json
import gzip
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer 
pd.set_option('display.max_columns', 10000, 'display.max_rows', 10000)

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

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
# ————————————————————————————
#target record strcuture: ID192403  review1689188
#[reviewerID[1],behavior[multi-dimension, numpy array stored data Frame](asin, brand, categories, unixReviewTime, price, overall),
#CandidateAd[1](asin, brand, categories, unixReviewTime, price), label[1] ]
# ————————————————————————————
#destination
# traning_set_add = 'C:\\Users\\willh\\Documents\\FYP2\\DataLundary\\RecordsTextOnly\\TrainingSet.json'
tfrecord = 'C:\\Users\\willh\\Documents\\FYP2\\DataLundary\\RecordsTextOnly\\TrainingSet.tfrecords'

text_json_add = 'C:\\Users\\willh\\Documents\\FYP2\\DataLundary\\RecordsTextOnly\\StrcuturedTextOnly.json'

# traning_set_add = '/home/ubuntu/fyp2/LundaryBack/TrainingSet.json'
# text_json_add = '/home/ubuntu/fyp2/LundaryBack/StrcuturedTextOnly.json'

data_str = open(text_json_add).read()
df = pd.read_json(data_str, lines= True)
# ————————————————————————————
# ————————————————————————————
df = df[0:int(len(df)*0.01)]

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
# print("re: "+str(reviewID_dic.columns ))

# ————————————————————————————
#make up training set
# ————————————————————————————
lower_boundary = 3  #if the num of user behavior lower than this num, record will be abandoned 
acc = 0
df_result = [] #final datastructure to store 
writer = tf.python_io.TFRecordWriter(tfrecord)
for reviewerID in reviewID_dic.itertuples(index = False):
    #search all his or her historical behaviors sorted by time
    his_behavior = df.loc[df['reviewerID'] == reviewerID[0]].sort_values(by = ['unixReviewTime']).reset_index(drop=True)
    #prograssivly generate true and false able and record
    if (len(his_behavior) >= lower_boundary):
        # his_behavior.ix[0,len(his_behavior),['categories']] = his_behavior.ix[0,len(his_behavior),['categories']][0][0]
        for i in range(lower_boundary ,len(his_behavior)): #select the behavior sequence number as the candidata ad
            behavior = his_behavior.loc[0:i-1]['categories'] #historical behavior
            behavior_categories = convert_to_numpy(behavior)
            print("   shape of behavior ")
            print(behavior_categories.shape)
            print(behavior_categories)
            # behavior = behavior.to_records()
            print("  ")
            candidate_true = his_behavior.loc[i]['categories'] #candidate ad
            print(candidate_true)

            example = tf.train.Example(features = tf.train.Features(feature = {
            'candidate_categories':_float_feature(candidate_true),
            'behavior_categories':_float_feature(behavior_categories)
            }))
            # candidate_false = generate_false_candidate(his_behavior.loc[0:i-1])
            # # #generate true record
            # t = {'behavior':behavior,'candidateAd':candidate_true, 'label' : 1 }
            # df_result.append(t)
            # #generate false record
            # d = {'behavior':behavior,'candidateAd':candidate_false, 'label' : 0 }
            # df_result.append(d)
            writer.write(example.SerializeToString())


    else:#skip customers who have insufficient bahavior
        pass
    acc = acc+1
    break
    if (acc%5000):
        print('\r', str(acc/len(reviewID_dic)).ljust(10),end='')
df_result = pd.DataFrame.from_records(np.asarray(df_result))
writer.close()
# test = pd.read_json(out, lines= True)
# print("---------------")
# print(test.head().reset_index(drop=True))

# with open(traning_set_add, 'a') as f:
#     f.write(out)

# ————————————————————————————
# the end
# ————————————————————————————
