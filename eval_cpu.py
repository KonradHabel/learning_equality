import os
import torch
import psutil

os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["OMP_SCHEDULE"]="STATIC"
os.environ["OMP_PROC_BIND"]="CLOSE"

# set to core-count of your CPU
os.environ["OMP_NUM_THREADS"]="8"   
torch.set_num_threads(psutil.cpu_count(logical=False))

import torch
import pandas as pd
import pickle
import numpy as np
from dataclasses import dataclass
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.quantization import quantize_dynamic

from retrieval.model import Net
from retrieval.evaluate import score_predictions, predict
from retrieval.dataset_inference import EqualDatasetEval, sort_input


@dataclass
class TrainingConfiguration:
    
    #--------------------------------------------------------------------------
    # Models:
    #--------------------------------------------------------------------------    
    # 'sentence-transformers/LaBSE'
    # 'microsoft/mdeberta-v3-base' 
    # 'sentence-transformers/stsb-xlm-r-multilingual'
    # 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    # 'sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens'
    #--------------------------------------------------------------------------
    
    transformer: str = 'sentence-transformers/LaBSE'
    pooling: str = "cls"                # "mean" | "cls" | "pooler"
    num_hidden_layers: int = 6          # None for no change | int if destilled
    proj = None
    
    # eval
    margin: float = 0.16
    batch_size: int = 8
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    normalize_features: bool = True
    verbose: bool = True

    # Data
    fold: int = 0                      # int | "all"
    max_len: int = 96
    language: str = "all"              # 'all' | 'en', es', 'pt', 'fr', ...
     
    checkpoint_folder = "./checkpoints_destilled/LaBSE"
    checkpoint = "weights_e39_0.6629.pth"
    
    num_workers: int = 0 if os.name == 'nt' else psutil.cpu_count(logical=False) 
    device: str = 'cpu' 
    
    
#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

config = TrainingConfiguration() 


if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\n{}[Model: {}]{}".format(20*"-", config.transformer, 20*"-"))

    model = Net(transformer_name=config.transformer,
                pretrained=False,
                gradient_checkpointing=config.gradient_checkpointing,
                num_hidden_layers=config.num_hidden_layers,
                pooling=config.pooling,
                projection=config.proj)
    

    print("Load:", f"{config.checkpoint_folder}/{config.checkpoint}")
    model_state_dict = torch.load(f"{config.checkpoint_folder}/{config.checkpoint}")  
    model.load_state_dict(model_state_dict, strict=True)  
    
    model.eval()
    
    model.transformer.embeddings = quantize_dynamic(model.transformer.embeddings, None, dtype=torch.float16)
    
    for i in range(config.num_hidden_layers):
        model.transformer.encoder.layer[i].attention = quantize_dynamic(model.transformer.encoder.layer[i].attention, None, dtype=torch.qint8)
        model.transformer.encoder.layer[i].intermediate = quantize_dynamic(model.transformer.encoder.layer[i].intermediate, None, dtype=torch.float16)
        model.transformer.encoder.layer[i].output = quantize_dynamic(model.transformer.encoder.layer[i].output, None, dtype=torch.float16)
    

    is1 = torch.zeros((1, config.max_len), dtype=torch.long)
    ie1 = torch.zeros((1, config.max_len), dtype=torch.float)

    with torch.jit.optimized_execution(False):
        model = torch.jit.trace(model, [is1, ie1])
    
    torch.jit.save(model, f"{config.checkpoint_folder}/model_traced.pth")

    
    #-----------------------------------------------------------------------------#
    # Tokenizer                                                                   #
    #-----------------------------------------------------------------------------#

    tokenizer = AutoTokenizer.from_pretrained(config.transformer)

    #-----------------------------------------------------------------------------#
    # Data                                                                        #
    #-----------------------------------------------------------------------------#
    df_correlations = pd.read_csv("./data/correlations.csv")

    gt_dict = dict()

    topics = df_correlations["topic_id"].values
    content = df_correlations["content_ids"].values

    for i in range(len(topics)):
        content_tmp = content[i].split(" ")
        topic_tmp = topics[i]
        gt_dict[topic_tmp] = content_tmp
        
    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#
    
    with open("./data/switch/topic2string_0.pkl", "rb") as f:
        topic2string = pickle.load(f)
               
    df_topics = pd.read_csv("./data/switch/topics_0.csv").fillna({"title": "", "description": ""}).set_index("id")
    
    
    if config.fold == "all":
        df_topics = df_topics[df_topics["fold"] != -1]
    else:
        df_topics = df_topics[df_topics["fold"] == config.fold]
    
    
    df_content = pd.read_csv("./data/switch/content_0.csv").fillna({"title": "", "description": "", "text":""}).set_index("id")
    df_content["text_cut"] = df_content["text"].map(lambda x : " ".join(x.split(" ")[:32]))
    df_content["input"] = df_content["title"] + " # " + df_content["description"] + " # " +  df_content["text_cut"]

    
    if config.language != "all":
        print(f"Only Eval language: {config.language}")
        df_topics = df_topics[df_topics["language"] == config.language]
        df_content = df_content[df_content["language"] == config.language]
    
    
    ids_t = df_topics.index.tolist()
    language_t = df_topics["language"].values.tolist()
    text_t = []
    for t in ids_t:
        text_t.append(topic2string[t])
    
    
    text_c = df_content["input"].values.tolist()
    ids_c = df_content.index.tolist()
    language_c = df_content["language"].values.tolist()
    
          
    
    sorted_topics = sort_input(text_t, ids_t, language_t, tokenizer, config.max_len)
    
    
    # Eval
    val_dataset_topic = EqualDatasetEval(sorted_topics,
                                         pad_token_id = tokenizer.pad_token_id,
                                         max_len=config.max_len)
    
    
    val_loader_topic = DataLoader(dataset=val_dataset_topic, 
                                  batch_size=config.batch_size, 
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  collate_fn=val_dataset_topic.smart_batching_collate
                                  )
    
    topic_preds = predict(config, model, val_loader_topic)
    
    topic_features, topic_ids, topic_language = topic_preds
    
    
    # reorder to df order
    topic_id_to_index = dict(zip(topic_ids, np.arange(len(topic_ids))))  
    
    
    reorder_t = []
    for idx in ids_t:
        reorder_t.append(topic_id_to_index[idx])    
    reorder_t = np.array(reorder_t)
    
    topic_ids = topic_ids[reorder_t]
    topic_language = topic_language[reorder_t]
    topic_features = topic_features[reorder_t]
    
    
    torch.save(topic_features, f"{config.checkpoint_folder}/topic_features.pt")
    torch.save(topic_ids, f"{config.checkpoint_folder}/topic_ids.pt")
    torch.save(topic_language, f"{config.checkpoint_folder}/topic_language.pt")  
    
    
    sorted_content = sort_input(text_c, ids_c, language_c, tokenizer, config.max_len)
    
  
    val_dataset_content = EqualDatasetEval(sorted_content,
                                           pad_token_id = tokenizer.pad_token_id,
                                           max_len=config.max_len)
    

    val_loader_content = DataLoader(dataset=val_dataset_content, 
                                    batch_size=config.batch_size, 
                                    shuffle=False,
                                    num_workers=config.num_workers,
                                    collate_fn=val_dataset_content.smart_batching_collate
                                    )
    
    content_preds = predict(config, model, val_loader_content) 
    
    
    content_features, content_ids, content_language = content_preds
    

    content_id_to_index = dict(zip(content_ids, np.arange(len(content_ids))))  
    
    reorder_c = []
    for idx in ids_c:
        reorder_c.append(content_id_to_index[idx])    
    reorder_c = np.array(reorder_c)
    
    content_ids = content_ids[reorder_c]
    content_language = content_language[reorder_c]
    content_features = content_features[reorder_c]

    torch.save(content_ids, f"{config.checkpoint_folder}/content_ids.pt")
    torch.save(content_language, f"{config.checkpoint_folder}/content_language.pt")
    torch.save(content_features, f"{config.checkpoint_folder}/content_features.pt")
   
    f2, precision, recall = score_predictions(topic_preds,
                                              content_preds,
                                              gt_dict,
                                              margin=0.16,
                                              device=config.device,
                                              use_cap=False)


    
