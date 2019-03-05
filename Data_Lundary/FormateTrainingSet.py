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

TextOnlyAdd = 'C:\\Users\\willh\\Documents\\FYP2\\DataLundary\\RecordsTextOnly\\StrcuturedTextOnly.json'
#DataFrame = pd.DataFrame(json.loads(open(TextOnlyAdd,'r+').read()))

#form up dictionary for brand and categories

#for each user
    #prograssivly generate true and false able and record
    #dump into json



# ————————————————————————————
# the end
# ————————————————————————————
