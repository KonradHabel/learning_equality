import pandas as pd
from tqdm import tqdm
import pickle
import os
import re

class Topic:
    """
    Class comes official intial given notebook:
    https://www.kaggle.com/code/jamiealexandre/sample-notebook-data-exploration
    only the reverse order is modified
    """
    
    def __init__(self, topic_id):
        self.id = topic_id
        
    @property
    def parent(self):
        parent_id = topic_df.loc[self.id].parent
        if pd.isna(parent_id):
            return None
        else:
            return Topic(parent_id)

    @property
    def ancestors(self):
        ancestors = []
        parent = self.parent
        while parent is not None:
            ancestors.append(parent)
            parent = parent.parent
        return ancestors

    @property
    def siblings(self):
        if not self.parent:
            return []
        else:
            return [topic for topic in self.parent.children if topic != self]

    def get_breadcrumbs(self, separator=" >> ", include_self=True, include_root=True):
        ancestors = self.ancestors
        if include_self:
            ancestors = [self] + ancestors
        if not include_root:
            ancestors = ancestors[:-1]
        return separator.join([a.title for a in ancestors])

    @property
    def children(self):
        return [Topic(child_id) for child_id in topic_df[topic_df.parent == self.id].index]


    def __eq__(self, other):
        if not isinstance(other, Topic):
            return False
        return self.id == other.id

    def __getattr__(self, name):
        return topic_df.loc[self.id][name]

    def __str__(self):
        return self.title
    
    def __repr__(self):
        return f"<Topic(id={self.id}, title=\"{self.title}\")>"
    
    
def clean(x):
    x = str(x)
    if x != "" and len(x) > 1:
        x = x.strip().strip('\t').strip('\n')
    return x
    
def clean_and_cut(x):
    x = str(x)
    if x != "" and len(x) > 1:
        x = x.strip().strip('\t').strip('\n').replace("", "")
        
        x = re.sub(r'http\S+', '', x)
        
        x = " ".join(x.split(" ")[:32])[:256]
                 
    return x
    
  
#-----------------------------------------------------------------------------#
# Preprocess original csv without translation                                 #
#-----------------------------------------------------------------------------#  

target_folder = "./data/switch"

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

#-----------------------------------------------------------------------------#
# topics.csv                                                                  #
#-----------------------------------------------------------------------------#  

print("Create:", f'{target_folder}/topics_0.csv')

topic_df = pd.read_csv('./data/topics.csv').fillna({"title": "", "description": ""})

# delete line breaks
topic_df = topic_df.replace(to_replace= r'\r\n', value= ' ', regex=True)
topic_df = topic_df.replace(to_replace= r'\n', value= ' ', regex=True)

# cleanup columns we want to use
topic_df["title"] = topic_df["title"].map(clean)
topic_df["description"] = topic_df["description"].map(clean)

# just copy language column without changes for original csv
topic_df["language_t"] = topic_df["language"]

topic_df.to_csv(f'{target_folder}/topics_0.csv', index=False) 


topic2fold = dict(zip(topic_df["id"].values, topic_df["fold"].values))


#-----------------------------------------------------------------------------#
# content.csv                                                                 #
#-----------------------------------------------------------------------------# 

print("\nCreate:", f'{target_folder}/content_0.csv')

content_df = pd.read_csv('./data/content.csv', encoding="utf-8").fillna({"title": "", "description": "", "text": ""})

# delete line breaks
content_df = content_df.replace(to_replace= r'\r\n', value= ' ', regex=True)
content_df = content_df.replace(to_replace= r'\n', value= ' ', regex=True)

# cleanup columns we want to use
content_df["title"] = content_df["title"].map(clean)
content_df["description"] = content_df["description"].map(clean)
content_df["text"] = content_df["text"].map(clean_and_cut)

# just copy language column without changes for original csv
content_df["language_t"] = content_df["language"]

content_df.to_csv(f'{target_folder}/content_0.csv', index=False)

#-----------------------------------------------------------------------------#
# Create dict for fast topic2text lookup                                      #
#-----------------------------------------------------------------------------#      
    
correlations_df = pd.read_csv('./data/correlations.csv')

top = correlations_df["topic_id"].values

for i in range(4):
    
    print("\nCreate:", f"{target_folder}/topic2string_{i}.pkl")

    topic_df = pd.read_csv(f'{target_folder}/topics_{i}.csv').fillna({"title": "", "description": ""})
    
    if i > 0:
        topic_df["fold"] = -1   # -1 if topic without content
        topic_df["fold"] = topic_df["id"].map(lambda x : topic2fold.get(x, -1))
        topic_df.to_csv(f'{target_folder}/topics_{i}.csv', index=False)
    
    top = topic_df["id"].values

    topic2string = {}
    
    topic_df = topic_df.set_index("id")
    
    for t in tqdm(top):
        
        to = Topic(t)
        
        string = "{} # {}".format(to.get_breadcrumbs(separator=" # ", include_self=True), to.description)
        
        topic2string[t] = string
        
    
    with open(f"{target_folder}/topic2string_{i}.pkl", "wb") as f:
        pickle.dump(topic2string, f) 
    
    

    
    

  