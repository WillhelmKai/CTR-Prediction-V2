#coding by Willhelm
#20190315
import json
import pandas as pd
#function convert brand into binary hash map

#function convert categories into binary hash map

# ————————————————————————————
#target record strcuture:
#[UserID[1],behavior[multi-dimension](asin, brand, categories, unixReviewTime, price, overall),
#CandidateAd[1](asin, brand, categories, unixReviewTime, price), label[1] ]
# ————————————————————————————
BrandDic = []
CateDic = []
TextOnlyAdd = 'C:\\Users\\willh\\Documents\\FYP2\\DataLundary\\RecordsTextOnly\\StrcuturedTextOnly.json'
#DataFrame = pd.DataFrame(json.loads(open(TextOnlyAdd,'r+').read()))

#form up dictionary for brand and categories
a = 0
with open(TextOnlyAdd) as f:
    line = f.readline()
    while line:
        print(line)
        a = a+1
        if (a > 2):
            break
    # d = json.loads(line)
    # print(d)
    f.close()

#for each user
    #archave all the behavior into a list
    #prograssivly generate true and false able and record
    #dump into json



# ————————————————————————————
# the end
# ————————————————————————————
