import torch
import numpy as np
from tqdm import tqdm
import gc
from torch.cuda.amp import autocast
import torch.nn.functional as F
from collections import defaultdict, Counter

def predict(config, model, dataloader):
    
    model.eval()
    
    if config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    features_list = []
    ids_list = []
    language_list = []

    
    with torch.no_grad():
        
        for ids, mask, text_id, language in bar:
            
            ids_list.extend(text_id)
            language_list.extend(language)
            
            if config.device == "cuda":
        
                with autocast():
                    ids = ids.to(config.device)
                    mask = mask.to(config.device)
            
                    feature = model(ids, mask)
                    
                    if config.normalize_features:
                        feature = F.normalize(feature, dim=-1)
                
                # keep features on GPU for faster eval
                features_list.append(feature.to(torch.float16))
                
            else:
                feature = model(ids, mask)
                if config.normalize_features:
                    feature = F.normalize(feature, dim=-1)
                
                features_list.append(feature)
                   
        features = torch.cat(features_list, dim=0) 
        
    if config.verbose:
        bar.close()
        
    ids = np.array(ids_list)
    language = np.array(language_list) 
           
    return features, ids, language


def f2_score(gt, pd):

    gt = set(gt)
    pd = set(pd)

    if len(pd) == 0:
        precision = 0.0
    else:
        precision = len(gt.intersection(pd)) / len(pd)
        
        
    if len(gt) == 0:
        recall = 0.0
    else:
        recall = len(gt.intersection(pd)) / len(gt)


    if (4 * precision + recall) == 0.0:
        f2 = 0.0
    else:
        f2 = (5 * precision * recall) / (4 * precision + recall)
        
    return f2, precision, recall  
 


def evaluate_val(config,
                 model,
                 reference_dataloader,
                 query_dataloader,
                 gt_dict,
                 cleanup=True):
        
    print("\nExtract Features:")
    query_preds = predict(config, model, query_dataloader)
    reference_preds = predict(config, model, reference_dataloader) 
    
    f2, precision, recall = score_predictions(query_preds, reference_preds, gt_dict, config.margin, device=config.device, cleanup=cleanup)
        
    return f2, precision, recall


def score_predictions(query_preds, reference_preds, gt_dict, margin=0.16, device="cpu", cleanup=True, use_cap=True):
    
    topic_features, topic_ids, topic_language = query_preds
    content_features, content_ids, content_language = reference_preds
    
    language_count = Counter(topic_language).most_common()
    language_list = [l for l, _ in language_count]
    
    print("\n{}[Margin: {:.2f}]{}".format(30*"-", margin, 30*"-"))
        
    scores = []
    precision_list = []
    recall_list = []
    
    print("\nCalculate Scores")
    for language in language_list:
        
        scores_language = []
        precision_list_language = []
        recall_list_language = []
        selection_length = []
        
        language_reference_index = content_language==language
        language_query_index = topic_language==language
        
        language_reference_features = content_features[language_reference_index]
        language_query_features = topic_features[language_query_index]
        
        language_reference_labels = content_ids[language_reference_index]
        language_query_labels = topic_ids[language_query_index]
        
        if len(language_reference_features) > 1 and len(language_query_features) > 1:

            if language_reference_features.dim == 1:
                language_reference_features = language_reference_features.unsqueeze(0)
            
            if language_query_features.dim == 1:
                language_query_features = language_query_features.unsqueeze(0)
                

            sim_l = language_query_features @ language_reference_features.T
                            
            for i in range(len(language_query_labels)):
                
                topic = language_query_labels[i]
                
                gt = gt_dict[topic]
                
                sim = sim_l[i].clone()
                
                th_tmp = sim.max() - margin * sim.max()
                
                p_select = (sim >= th_tmp).squeeze()
                
                if use_cap and p_select.sum() > 128:
                    _, indices = torch.sort(sim, descending=True)
                    p_select = indices[:128]
                
                c_choice = language_reference_labels[p_select.cpu().numpy()]
                 
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
            
            print(f"{language.ljust(3)} Score: {f2:.5f} - Precision: {precision:.5f} - Recall: {recall:.3f} - Selected: {str(selection_length).rjust(3)} - ({len(language_query_features)}x{len(language_reference_features)})")   

    f2 = np.array(scores).mean() 
    precision = np.array(precision_list).mean()
    recall = np.array(recall_list).mean()
    
    print("-"*80)
    print("Eval Score: {:.5f} - Precision: {:.5f} - Recall: {:.3f}".format(f2, precision, recall))
    print("-"*80)
    
    if cleanup:
        del content_features, topic_features, sim_l
        torch.cuda.empty_cache()
        gc.collect()
    
        
    return f2, precision, recall


def evaluate_train(config,
                   model,
                   reference_dataloader,
                   query_dataloader,
                   gt_dict,
                   content2topic,
                   cleanup=True):
        
    print("\nExtract Features:")
    
    query_features, query_labels, query_language = predict(config, model, query_dataloader)
    reference_features, reference_labels, reference_language = predict(config, model, reference_dataloader) 
    
    scores = []
    precision_list = []
    recall_list = []
    
    missing_dict = defaultdict(set)
    wrong_dict = defaultdict(set)
    
    reference_labels = np.array(reference_labels)
    query_labels = np.array(query_labels)
    
    
    language_count = Counter(query_language).most_common()
    language_list = [l for l, _ in language_count]
    
  
    query_language = np.array(query_language)
    reference_language = np.array(reference_language)
     
    print("\nCalculate Scores")
    for language in language_list:
        
        scores_language = []
        precision_list_language = []
        recall_list_language = []
        selection_length = []
                
        language_reference_index = reference_language==language
        language_query_index = query_language==language
        
        language_reference_features = reference_features[language_reference_index]
        language_query_features = query_features[language_query_index]
        
        language_reference_labels = reference_labels[language_reference_index]
        language_query_labels = query_labels[language_query_index]
        

        if len(language_reference_features) > 1 and len(language_query_features) > 1:

            if language_reference_features.dim == 1:
                language_reference_features = language_reference_features.unsqueeze(0)
            
            if language_query_features.dim == 1:
                language_query_features = language_query_features.unsqueeze(0)
       
            sim_l = language_query_features @ language_reference_features.T
                
            for i in range(len(language_query_labels)):
                
                topic = language_query_labels[i]
                
                gt = gt_dict[topic]
                
                sim = sim_l[i].clone()
                
                th_tmp = sim.max() - config.margin * sim.max()
                
                p_select = (sim >= th_tmp).squeeze()
                
                if p_select.sum() > 128:
                    _, indices = torch.sort(sim, descending=True)
                    p_select = indices[:128]
                
                c_choice = language_reference_labels[p_select.cpu().numpy()]
                
                gt = set(gt)
                c_choice = set(c_choice)
                
                missing = gt - c_choice
                
                if len(missing) > 0:
                    missing_dict[topic] = missing
                    
                
                wrong = c_choice - gt
                
                if len(wrong) > 0:
                    
                    wrong_dict[topic] = set(" ".join(list(wrong)).split(" "))
                
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
         
            print(f"{language.ljust(3)} Score: {f2:.5f} - Precision: {precision:.5f} - Recall: {recall:.3f} - Selected: {str(selection_length).rjust(3)} - ({len(language_query_features)}x{len(language_reference_features)})")  

    f2 = np.array(scores).mean() 
    precision = np.array(precision_list).mean()
    recall = np.array(recall_list).mean()
    
    print("-"*80)
    print("Train Score: {:.5f} - Precision: {:.5f} - Recall: {:.3f}".format(f2, precision, recall))
    print("-"*80)
    
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, query_features, sim_l
        torch.cuda.empty_cache()
        gc.collect()
        
        
    missing_pairs = []
     
    for key, value in missing_dict.items():
        for v in value:
            missing_pairs.append((key, v))
    
    topic2wrong = dict()
    for topic_wrong, content_wrong in wrong_dict.items():
        
        candidates = []
        
        for cw in content_wrong:
            
            other_topics = content2topic[cw]
            
            for ot in other_topics:
            
                candidates.append((ot, cw))
    
        topic2wrong[topic_wrong] = candidates
        
    return missing_pairs, topic2wrong
