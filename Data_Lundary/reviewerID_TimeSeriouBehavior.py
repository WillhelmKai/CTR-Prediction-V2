#coding by Willhelm
#20190314
import json
import gzip
import pandas as pd
pd.set_option('display.max_columns', 10000, 'display.max_rows', 10000)
# ————————————————————————————
#fianal record strcuture:
#[UserID, asin, brand, categories, unixReviewTime, price, overall]
# ————————————————————————————
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

raw_mate_add = 'C:\\Users\\willh\\Documents\\FYP2\\AmazonRawData\\meta_Electronics.json.gz'
raw_review_add = "C:\\Users\\willh\\Documents\\FYP2\\AmazonRawData\\reviews_Electronics_5.json.gz"
destination = 'C:\\Users\\willh\\Documents\\FYP2\\DataLundary\\RecordsTextOnly\\StrcuturedTextOnly.json'
# ——————————————————————————————————————————————————————
# read zip file into pandas Datafram
#df_mate
#['asin', 'imUrl', 'description', 'categories', 'title', 'price','salesRank', 'related', 'brand']

#df_review
#Data format Index(['reviewerID', 'asin', 'reviewerName', 'helpful', 'reviewText',
#        'overall', 'summary', 'unixReviewTime', 'reviewTime']
# ——————————————————————————————————————————————————————
print("loading to dataframe")
df_mate = getDF(raw_mate_add) 
df_review = getDF(raw_review_add)
#remove redunant columns  and values
df_mate = df_mate[['asin','categories','price','brand']]
df_review=df_review[['reviewerID','asin','overall','unixReviewTime']]

#sort the review by time
df_review = df_review.sort_values(by=['unixReviewTime'], ascending = True)

#according to asin finding mate record and merge
result = pd.merge(df_review, df_mate, on='asin')
result.dropna()
print(result.columns)
print("  ")
print(result.head())

# ————————————————————————————
#result
#reviewerID, asin, categories, price, brand, overall, unixReviewTime
# ————————————————————————————

#sample out
#dump json
# ————————————————————————————
#the end
# ————————————————————————————