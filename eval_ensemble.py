import torch
import pandas as pd
import numpy as np
from collections import Counter
from dataclasses import dataclass
from retrieval.evaluate import f2_score, score_predictions

@dataclass
class Configuration:
    
    # Transformer
    transformers = ('sentence-transformers/LaBSE',
                    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    
    feautre_folders = ("./checkpoints/LaBSE",
                       "./checkpoints/paraphrase-multilingual-mpnet-base-v2")
    
    # Eval 
    margin: float = 0.16
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
      
    
#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 

if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Eval single model                                                           #
    #-----------------------------------------------------------------------------#
    
    df_correlations = pd.read_csv("./data/correlations.csv")

    gt_dict = dict()

    topics = df_correlations["topic_id"].values
    content = df_correlations["content_ids"].values

    for i in range(len(topics)):
        content_tmp = content[i].split(" ")
        topic_tmp = topics[i]
        gt_dict[topic_tmp] = content_tmp
        
    topic_ids_list = []
    topic_language_list = []
    topic_features_list = []
    
    content_ids_list = []
    content_language_list = []
    content_features_list = []
    
    for i, (transformer, folder) in enumerate(zip(config.transformers, config.feautre_folders)):
        
        print("\n{}[Model: {}]{}".format(20*"-", transformer, 20*"-"))
        
        
        topic_ids_list.append(torch.load(f"{folder}/topic_ids.pt"))
        topic_language_list.append(torch.load(f"{folder}/topic_language.pt"))  
        topic_features_list.append(torch.load(f"{folder}/topic_features.pt",
                                              map_location=torch.device(config.device)))
        
        content_ids_list.append(torch.load(f"{folder}/content_ids.pt"))
        content_language_list.append(torch.load(f"{folder}/content_language.pt"))
        content_features_list.append(torch.load(f"{folder}/content_features.pt",
                                                map_location=torch.device(config.device)))
        
    
        topic_preds = (topic_features_list[i], topic_ids_list[i], topic_language_list[i])
        content_preds = (content_features_list[i], content_ids_list[i], content_language_list[i])
        
        f2, precision, recall = score_predictions(topic_preds, content_preds, gt_dict, margin=config.margin, device=config.device)
        
        
       
    #-----------------------------------------------------------------------------#
    # Ensemble                                                                    #
    #-----------------------------------------------------------------------------# 
          
    topic_ids        = topic_ids_list[0]
    topic_language   = topic_language_list[0]
    
    content_ids      = content_ids_list[0]
    content_language = content_language_list[0]
    
    language_count   = Counter(topic_language).most_common()
    language_list    = [l for l, _ in language_count]
    
    language2sim = dict()
    
    for i in range(len(config.transformers)):
    
        topic_features = topic_features_list[i]
        content_features = content_features_list[i]
    
        print(f"Calculate Similiarity for: {config.transformers[i]}")
    
        for language in language_list:
    
            language_content_index = content_language==language
            language_topic_index = topic_language==language
    
            language_content_ids = content_ids[language_content_index]
            language_topic_ids = topic_ids[language_topic_index]
    
            language_content_features = content_features[language_content_index]
            language_topic_features = topic_features[language_topic_index] 
    
            if len(language_topic_features) > 0 and len(language_content_features) > 0:
    
                if language_topic_features.dim() == 1:
                    language_topic_features = language_topic_features.unsqueeze(0)
    
                if language_content_features.dim() == 1:
                    language_content_features = language_content_features.unsqueeze(0)
    
                with torch.no_grad():
                    sim = language_topic_features @ language_content_features.T  
    
                old_sim = language2sim.get(language, None)
    
                if old_sim is None:
                    language2sim[language] = sim   
                else:
                    language2sim[language] += sim
    
                                     
    num_models = len(config.transformers)
       
    topic_list = []
    content_list = []
    
    print(f"\nCalculate mean of similiarities of {num_models} models per language")
    
    print(f"\nSelect content per language using dynamic threshold of {config.margin}")
    
    scores = []
    precision_list = []
    recall_list = []
    
    print("\n{}[Ensemble]{}".format(30*"-", 30*"-"))
    
    for language in language_list: 
        
        scores_language = []
        precision_list_language = []
        recall_list_language = []
        selection_length = []
        
        language_content_ids = content_ids[content_language==language]
        language_topic_ids = topic_ids[topic_language==language]
    
        # Mean of Similiarities
        sim_matrix = language2sim[language] 
        sim_matrix /= num_models
        
        selection_length = []
        
        for i in range(len(language_topic_ids)):
            
            topic = language_topic_ids[i]
            
      
            sim = sim_matrix[i]
    
            th_tmp = sim.max() - config.margin * sim.max()
            p_select = (sim >= th_tmp).squeeze()
            c_choice = set(language_content_ids[p_select.cpu().numpy()].tolist())
            
            topic_list.append(topic)
            content_list.append(" ".join(list(c_choice)))
            selection_length.append(len(c_choice))
            
            
            gt = gt_dict[topic]
            f, precision, recall = f2_score(gt, c_choice)
            
            scores_language.append(f)
            precision_list_language.append(precision)
            recall_list_language.append(recall)
            
            scores.append(f)
            precision_list.append(precision)
            recall_list.append(recall)
            
            selection_length.append(len(c_choice))
            
        if len(scores_language) > 0:
            f2 = np.array(scores_language).mean() 
            precision = np.array(precision_list_language).mean()
            recall = np.array(recall_list_language).mean()
            
            if len(selection_length) > 0:
                selection_length = np.array(selection_length).mean().astype(np.int32)
            else:
                selection_length = 0
         
            print(f"{language.ljust(3)} Score: {f2:.5f} - Precision: {precision:.5f} - Recall: {recall:.3f} - Selected: {str(selection_length).rjust(3)} - ({sim_matrix.shape[0]}x{sim_matrix.shape[1]})")   

    f2 = np.array(scores).mean() 
    precision = np.array(precision_list).mean()
    recall = np.array(recall_list).mean()
    
    print("-"*80)
    print("Eval Score: {:.5f} - Precision: {:.5f} - Recall: {:.3f}".format(f2, precision, recall))
    print("-"*80)
            
            
            
            

    
        

 
















