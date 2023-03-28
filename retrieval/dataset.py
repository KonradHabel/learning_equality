import numpy as np
from torch.utils.data import Dataset
import random
import copy
import torch
from tqdm import tqdm
from collections import defaultdict
import time
import pickle
import pandas as pd

                  
class EqualDatasetTrain(Dataset):
    
    def __init__(self,
                 df_correlations,
                 fold,
                 tokenizer,
                 max_len,
                 shuffle_batch_size,
                 pool=(0,1,2,3),
                 init_pool=0,
                 train_on_all=False,
                 language="all",
                 debug=False):
        
        super().__init__()
        
        
        
        
        self.pool_id2language = {0: "org:    en, es, fr, pt",
                                 1: "switch: en<->fr, es<->pt",
                                 2: "switch: en<->es, fr<->pt",
                                 3: "switch: en<->pt, fr<->es"}

        self.df_correlations = df_correlations.set_index("topic_id")
 
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.shuffle_batch_size = shuffle_batch_size
        
        self.debug = debug
        
        self.content_pool = []
        self.topic2string_pool = []
        self.pair_pool = []
        self.topic2content_pool = []
        self.content2topic_pool = []
  
        
        for i in pool:
            
            df_content = pd.read_csv(f"./data/switch/content_{i}.csv").fillna({"title": "", "description": "", "text":""}).set_index("id")
            df_topic = pd.read_csv(f"./data/switch/topics_{i}.csv").fillna({"title": "", "description": ""}).set_index("id")
            
            
            if language != "all":
                df_topic = df_topic[df_topic["language_t"]==language].copy()
                    
            if train_on_all:
                topics = df_topic[df_topic["fold"] != -1].index 
            else:
                topics = df_topic[(df_topic["fold"] != fold) & (df_topic["fold"] != -1)].index
                
                
                
            pairs = []
            topic2content = {}
            content2topic = defaultdict(set)
    
            for t in topics:

                row = self.df_correlations.loc[t]
              
                content = row["content_ids"].split(" ")
                
                topic2content[t] = set(content)
                
                for c in content:
                    
                    pairs.append((t, c))
                    
                    content2topic[c].add(t)
            
            with open(f"./data/switch/topic2string_{i}.pkl", "rb") as f:
                topic2string = pickle.load(f)
     
   
            self.topic2string_pool.append(topic2string)
            self.content_pool.append(df_content)
            self.pair_pool.append(pairs)
            self.topic2content_pool.append(topic2content)
            self.content2topic_pool.append(content2topic)
            
        self.set_pool(init_pool)
    
        self.samples = self.pairs
        
        
        
        
    def set_pool(self, i):
        print(f"Set train pairs language to {self.pool_id2language[i]}")
        self.df_content = self.content_pool[i]
        self.topic2string = self.topic2string_pool[i]
        self.pairs = self.pair_pool[i]
        self.topic2content = self.topic2content_pool[i]
        self.content2topic = self.content2topic_pool[i]
        
    def __getitem__(self, index):
        
        
        topic, content = self.samples[index]
        
        content_row = self.df_content.loc[content]
        
        content_title = content_row["title"]
    
        content_description = content_row["description"]

        content_text = content_row["text"]
        if content_text:
            content_text = " ".join(content_text.split(" ")[:32])


        topic_text = self.topic2string[topic]
        
        content_text = "{} # {} # {}".format(content_title, content_description, content_text)
        
        
        inputs1 = self.tokenizer.encode_plus(
            topic_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
            return_tensors='pt'
        )
                

        inputs2 = self.tokenizer.encode_plus(
            content_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
            return_tensors='pt'
        )

        
        return inputs1['input_ids'], inputs1['attention_mask'], inputs2['input_ids'], inputs2['attention_mask']
    
    
    def smart_batching_collate(self, batch):

        input_ids_1 = [x[0] for x in batch]
        input_ids_1 = torch.cat(input_ids_1, dim=0)
        
        mask_1 = [x[1] for x in batch]
        mask_1 = torch.cat(mask_1, dim=0)
        
        
        input_ids_2 = [x[2] for x in batch]
        input_ids_2 = torch.cat(input_ids_2, dim=0)        
        
        mask_2 = [x[3] for x in batch]
        mask_2 = torch.cat(mask_2, dim=0)
        
        
        max_seq_length = torch.cat([mask_1, mask_2], dim=0).sum(-1).max().to(torch.long)

        input_ids_1 = input_ids_1[:, :max_seq_length]
        mask_1 = mask_1[:, :max_seq_length]
        
        input_ids_2 = input_ids_2[:, :max_seq_length]
        mask_2 = mask_2[:, :max_seq_length]
        

        return input_ids_1, mask_1, input_ids_2, mask_2

    def __len__(self):
        
        
        if self.debug:
            return len(self.samples[:self.debug])
        else:
            return len(self.samples)
    
    
    
    def shuffle(self, missing_list=None, wrong_dict=None, max_wrong=16):
        
        print("\nShuffle Batches:")
        pair_pool = copy.deepcopy(self.pairs)
        
        missing_pool = copy.deepcopy(missing_list)
        wrong_pool = copy.deepcopy(wrong_dict)
        
        # Shuffle pairs order
        random.shuffle(pair_pool)
        if missing_pool:
            random.shuffle(missing_pool)
        
        # Lookup if already used in epoch
        pairs_epoch = set()
        topic_batch = set()
        content_batch = set()


        # buckets
        batches = []
        current_batch = []

        # progressbar
        pbar = tqdm()
        
        # counter
        break_counter = 0
        oversample_missing = 0
        

        while True:
            
            pbar.update()
            
            
            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)
                                
                topic, content = pair
                
                if topic not in topic_batch and content not in content_batch and pair not in pairs_epoch and len(current_batch) < self.shuffle_batch_size:
                    
                    pairs_epoch.add(pair)
                    topic_batch.add(topic)
                    content_batch.update(self.topic2content[topic])
                    
                    current_batch.append(pair)
                    
                    # reset break counter
                    break_counter = 0

            
                    # Add other topic with the hard negative content for current topic
                    if wrong_pool is not None and len(current_batch) < self.shuffle_batch_size:
                        
                        wrong_pairs = copy.deepcopy(wrong_pool.get(topic, None))
                        
                        if wrong_pairs is not None and len(wrong_pairs) > 0:
                        
                            random.shuffle(wrong_pairs)
                            
                            for wrong_pair in wrong_pairs[:max_wrong]:
                            
                                topic_w, content_w = wrong_pair
                            
                                if topic_w not in topic_batch and content_w not in content_batch and wrong_pair not in pairs_epoch and len(current_batch) < self.shuffle_batch_size:
                                    
                                    topic_batch.add(topic_w)
                                    content_batch.update(self.topic2content[topic_w])
                                    
                                    current_batch.append(wrong_pair)
                                    
                                    pairs_epoch.add(wrong_pair)
                                    
                                    wrong_pool[topic].remove(wrong_pair)
                                            
                        
                        
                    
                # Oversample missing   
                if missing_pool is not None and len(missing_pool) > 0 and np.random.rand() < 0.5 and len(current_batch) < self.shuffle_batch_size:
                    
                    topic_m, content_m = missing_pool.pop(0)
                    
                    if topic_m not in topic_batch and content_m not in content_batch:
                        
                        topic_batch.add(topic_m)
                        content_batch.update(self.topic2content[topic_m])
                        current_batch.append((topic_m, content_m))
                        oversample_missing += 1
                           
                    else:
                        missing_pool.append((topic_m, content_m))
                        
                        
                        
                else:
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)
                    

                    break_counter += 1

                if break_counter >= 512:
                    break
                 
            else:
                
                break
            
            if len(current_batch) >= self.shuffle_batch_size:
                
                topic_batch = set()
                content_batch = set()
            
                batches.extend(current_batch)
                current_batch = []

            
        pbar.close()
        
        # wait before closing progress bar
        time.sleep(0.3)
        
        self.samples = batches
        

        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples))) 
        print("Break Counter:", break_counter)
        if missing_pool:
            print("Oversample missing: {} - Oversample missing left: {}".format(oversample_missing, len(missing_pool)))
        print("Pairs left:", len(pair_pool))
        print("First Element Topic: {} - Last Element Topic: {}".format(self.samples[0][0], self.samples[-1][0]))
    
        
        
class EqualDatasetEval(Dataset):
    
    def __init__(self,
                 mode,
                 typ,
                 fold,
                 tokenizer,
                 max_len,
                 pool=(0,1,2,3),
                 init_pool=0,
                 train_on_all=False,
                 language="all",
                 debug=False):
        
        super().__init__()
        
        self.mode = mode
        self.typ = typ
        self.debug = debug
        
        self.pool_id2language = {0: "org:    en, es, fr, pt",
                                 1: "switch: en<->fr, es<->pt",
                                 2: "switch: en<->es, fr<->pt",
                                 3: "switch: en<->pt, fr<->es"}
            
        
        
        if self.typ == "train":
            
            
            self.df_pool = []
            self.topic2string_pool = []
            
            for i in pool:
            
            
                if self.mode == "topic":
                    

                    with open(f"./data/switch/topic2string_{i}.pkl", "rb") as f:
                        topic2string = pickle.load(f)
                            
                    self.topic2string_pool.append(topic2string)
                        
                    df = pd.read_csv(f"./data/switch/topics_{i}.csv").fillna({"title": "", "description": ""}).set_index("id")
                    
                    if train_on_all:
                        
                        df = df[df["fold"] != -1] 
                    else:
                        df = df[(df["fold"] != fold) & (df["fold"] != -1)] 
                    
                else:
                    df = pd.read_csv(f"./data/switch/content_{i}.csv").fillna({"title": "", "description": "", "text":""}).set_index("id")
                    
                    
                if language != "all":
                    df = df[df["language_t"]==language].copy()
                    
                
                self.df_pool.append(df)
                
            self.set_pool(init_pool)
            
  
        else:
            if self.mode == "topic":

                with open("./data/switch/topic2string_0.pkl", "rb") as f:
                    topic2string = pickle.load(f)
                        
                self.topic2string = topic2string
                    
                df = pd.read_csv("./data/switch/topics_0.csv").fillna({"title": "", "description": ""}).set_index("id")
                  
                df = df[df["fold"] == fold]
                
            else:
                df = pd.read_csv("./data/switch/content_0.csv").fillna({"title": "", "description": "", "text":""}).set_index("id")
                
            if language != "all":
                df = df[df["language_t"]==language].copy()       
                      
            self.df = df
                
            self.ids = self.df.index    
         
      
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        
    def set_pool(self, i):
        print(f"Set {self.typ} {self.mode} to {self.pool_id2language[i]}")
        self.df = self.df_pool[i]
        self.ids = self.df.index
        if self.mode == "topic":
            self.topic2string = self.topic2string_pool[i]
            
        
    def __getitem__(self, index):
        
        
        text_id = self.ids[index]
        
        row = self.df.loc[text_id]
        
        language = row["language_t"]
        
        
        if self.mode == "topic":
            text_input = self.topic2string[text_id] 
        else:
            title = row["title"]
        
            description = row["description"]

            text = row["text"]
            if text:
                text = " ".join(text.split(" ")[:32])
   
            text_input = "{} # {} # {}".format(title, description, text)
        
        tok = self.tokenizer.encode_plus(text_input,
                                            None,
                                            add_special_tokens=True,
                                            max_length=self.max_len,
                                            padding="max_length",
                                            return_token_type_ids=True,
                                            truncation=True,
                                            return_tensors='pt')
        
        
        return tok['input_ids'], tok['attention_mask'], text_id, language

    def __len__(self):
        
        if self.debug:
            return len(self.ids[:self.debug])
        else:
            return len(self.ids)
            

    def smart_batching_collate(self, batch):
        
        input_ids = [x[0] for x in batch]
        input_ids = torch.cat(input_ids, dim=0)
        
        mask = [x[1] for x in batch]
        mask = torch.cat(mask, dim=0)
        
        max_seq_length = mask.sum(-1).max().to(torch.long)
        
        input_ids = input_ids[:, :max_seq_length]
        mask = mask[:, :max_seq_length]
        
        text_id =  [x[2] for x in batch]
        
        language = [x[3] for x in batch]
        
        return input_ids, mask, text_id, language




