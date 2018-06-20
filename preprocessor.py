import sys;
reload(sys);
sys.setdefaultencoding("utf8")

import csv 
from string import punctuation
import json

with open("yelp.data", 'r') as f:
    reviews=[]
    for line in f: 
        review = json.loads(line) 
        reviews.append(review)
        
print len(reviews)

with open("yelp_dataset.csv", "w") as f:  
            rows = []
            for x in reviews:
                rows.append((x["funny"],x["user_id"],x["review_id"],x["text"],x["business_id"],x["stars"],x["date"],x["useful"],x["cool"]))
            #rows = sorted(rows, key=lambda row: row[1], reverse=True)
            #print(rows)
            writer=csv.writer(f, delimiter=',')          
            writer.writerows(rows)

