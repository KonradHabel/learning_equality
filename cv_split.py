import pandas as pd
import numpy as np
from collections import defaultdict


N_SPLITS = 10

#-----------------------------------------------------------------------------#
# Data                                                                        #
#-----------------------------------------------------------------------------#

topic_df = pd.read_csv('./data/kaggle/topics.csv')
content_df = pd.read_csv('./data/kaggle/content.csv')
correlation_df = pd.read_csv('./data/kaggle/correlations.csv')


split_data = correlation_df.merge(topic_df[['id', 'category', 'level','language', 'parent']],
                                  left_on="topic_id",
                                  right_on="id",
                                  how="left",
                                  ).drop(columns=['id'])

        

#-----------------------------------------------------------------------------#
# Minimize content intersection between folds                                 #
#-----------------------------------------------------------------------------#

print("Minimizing content intersections between folds:")

folds_per_language = dict()

languages = split_data["language"].unique()

for l in languages :
    
    print("Language:", l)
    
    df_tmp = split_data[split_data["language"]==l]

    content2topics = defaultdict(set)
    topic2content = dict()
    
    for i in range(len(df_tmp)):
        
        row = df_tmp.iloc[i]
        
        
        topic = row["topic_id"]
        
        content = row["content_ids"].split(" ")
        
        
        topic2content[topic] = set(content)
        
        for c in content:
            
            content2topics[c].add(topic)
        
    
    related_topics = defaultdict(set)
    
    for t , c in topic2content.items():
    
        
        for content in c:
            
            related_t = content2topics[content] 
    
            related_topics[t].update(related_t)
            
            
    related_topics_list = list(related_topics.values())
    
    
    folds = defaultdict(set)
    
    topics_used = set()
    
    fold_level = np.zeros(N_SPLITS)
    fold_id = np.arange(0, N_SPLITS)
    
    for rt in related_topics_list:
        
        
        it = np.full(N_SPLITS, 1000)
        
        for i in range(N_SPLITS):
            
            
            it[i] = len(folds[i].intersection(rt))
            
        min_value = it.min()
        
        
        select = it == min_value
        
        
        best_fold = fold_id[select][fold_level[select].argmin()]
        
        to_add = rt - topics_used
        
        
        folds[best_fold].update(to_add) 
        
        fold_level[best_fold] += len(to_add)
        
        topics_used.update(rt)
        
        
    folds_per_language[l] = folds
    
    
#-----------------------------------------------------------------------------#
# Data -> Folds                                                               #
#-----------------------------------------------------------------------------#

folds = defaultdict(set)


for l, folds_dict in folds_per_language.items():
    
    for fold, topics in folds_dict.items():
        
        folds[fold].update(topics)

    


topic2fold = dict()

for fold, topics in folds.items():
    
    for t in topics:
        topic2fold[t] = fold
    
    
split_data["fold"] = -1
split_data["fold"] = split_data["topic_id"].map(topic2fold)



content2fold = defaultdict(set)

for i in range(len(split_data)):
    
    row = split_data.iloc[i]
    
    
    content = row["content_ids"].split(" ")
    
    fold = row["fold"]

    for c in content:
        
        content2fold[c].add(fold)

#-----------------------------------------------------------------------------#
# Dataframe for correlations.csv                                              #
#-----------------------------------------------------------------------------#

print("\nPrepare: correlations.csv")

df_correlations = split_data[['topic_id', 
                              'content_ids',
                              'language',
                              'level',
                              'category',
                              'parent',
                              'fold']]

# To CSV and Pickle
df_correlations.to_csv("./data/correlations.csv", index=False)

#-----------------------------------------------------------------------------#
# Dataframe for topics.csv                                                    #
#-----------------------------------------------------------------------------#

print("\nPrepare: topics.csv")

topic_df["fold"] = -1   # -1 if topic without content
topic_df["fold"] = topic_df["id"].map(lambda x : topic2fold.get(x, -1))
topic_df.to_csv("./data/topics.csv", index=False)
    
#-----------------------------------------------------------------------------#
# Dataframe for content.csv                                                   #
#-----------------------------------------------------------------------------# 

print("\nPrepare: content.csv")

content_df["fold"] = content_df["id"].map(lambda x : " ".join([str(i) for i in list(content2fold[x])]))
content_df.to_csv('./data/content.csv', index=False)
    