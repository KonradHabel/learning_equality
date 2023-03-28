import os
os.environ['TOKENIZERS_PARALLELISM']='false'

import sys
import torch
import time
import math
import shutil
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from transformers import AutoTokenizer

from retrieval.loss import InfoNCE
from retrieval.model import Net
from retrieval.trainer import train
from retrieval.utils import setup_system, Logger
from retrieval.evaluate import evaluate_val, evaluate_train
from retrieval.dataset import EqualDatasetTrain, EqualDatasetEval

@dataclass
class Configuration:
    
    #--------------------------------------------------------------------------
    # Models:
    #--------------------------------------------------------------------------    
    # 'sentence-transformers/LaBSE'
    # 'microsoft/mdeberta-v3-base' 
    # 'sentence-transformers/stsb-xlm-r-multilingual'
    # 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    # 'sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens'
    #--------------------------------------------------------------------------
    
    # Transformer
    transformer: str = 'sentence-transformers/LaBSE'
    pooling: str = 'cls'                   # 'mean' | 'cls' | 'pooler'
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    proj = None                            # None | int for lower dimension 
        
    margin: float = 0.16

    # Reduction of model size
    layers_to_keep = None                 # None -> org. model | (1,2,...,11) layers to keep
    
    # Model Destillation 
    transformer_teacher: str ='sentence-transformers/LaBSE'
    use_teacher: bool = False             # use destillation
    pooling_teacher: str = 'cls'          # 'mean' | 'cls' | 'pooler'
    proj_teacher = None                   # None | int for lower dimension 
    
    # Language Sampling
    init_pool = 0
    pool = (0,1,2,3)                      # (0,) for only train on original data without translation
    epoch_stop_switching: int = 36        # epochs no language switching more used (near end of training)
    
    # Debugging
    debug = None                          # False | 10000 for fast test
        
    # Training 
    seed: int = 42
    epochs: int = 40
    batch_size: int = 512
    mixed_precision: bool = True          # using fp16
    gradient_accumulation: int = 1
    gradient_checkpointing: bool = False  # use gradient checkpointing
    verbose: bool = True                  # show progressbar
    gpu_ids: tuple = (0,1,2,3)            # GPU ids for training
    
    # Eval
    eval_every_n_epoch: int = 1
    normalize_features: bool = True
    zero_shot: bool = False               # eval before first epoch

    # Optimizer 
    clip_grad = 100.                      # None | float
    decay_exclue_bias: bool = False
    
    # Loss
    label_smoothing: float = 0.1
    
    # Learning Rate
    lr: float = 0.0002                   
    scheduler: str = 'polynomial'        # 'polynomial' | 'constant' | None
    warmup_epochs: int = 2
    lr_end: float = 0.00005              #  only for 'polynomial'
    
    # Data
    language: str = 'all'                # 'all' | 'en', es', 'pt', 'fr', ....
    fold: int = 0                        # eval on fold x
    train_on_all: bool = False           # train on all data incl. data of fold x
    max_len: int = 96                    # max token lenght for topic and content
     
    # Sampling
    max_wrong: int = 128                 # limit for sampling of wrong content for specific topic
    custom_sampling: bool = True         # do custom shuffle to prevent having related content in batch
    sim_sample: bool = True              # upsample missing and combine hard negatives in batch
    sim_sample_start: int = 1            # if > 1 skip firt n epochs for sim_sampling
    
    # Save folder for model checkpoints
    model_path: str = './checkpoints'
    
    # Checkpoint to start from  
    checkpoint_start = None              # pre-trained checkpoint for model we want to train
    checkpoint_teacher = None            # pre-trained checkpoint for teacher

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    # for better performance
    cudnn_benchmark: bool = True
    
    # make cudnn deterministic
    cudnn_deterministic: bool = False


#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 


if __name__ == '__main__':

    model_path = '{}/{}/{}'.format(config.model_path,
                                   config.transformer,
                                   time.strftime('%H%M%S'))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), '{}/train.py'.format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    
    print('\n{}[Model: {}]{}'.format(20*'-', config.transformer, 20*'-'))

    model = Net(transformer_name=config.transformer,
                gradient_checkpointing=config.gradient_checkpointing,
                hidden_dropout_prob=config.hidden_dropout_prob,
                attention_dropout_prob=config.attention_dropout_prob,
                pooling=config.pooling,
                projection=config.proj)
    
                
    print(model.transformer.config) 


    # load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print('Start from:', config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=True)  
        
        
    #-----------------------------------------------------------------------------#
    # Drop Transformer Layers                                                     #
    #-----------------------------------------------------------------------------#

    if config.layers_to_keep is not None:
        print('Remove layers from model. Only keep these layers: {}'.format(config.layers_to_keep))
        new_layers = torch.nn.ModuleList([layer_module for i, layer_module in enumerate(model.transformer.encoder.layer) if i in config.layers_to_keep])
        model.transformer.encoder.layer = new_layers
        model.transformer.config.num_hidden_layers = len(config.layers_to_keep)
        
        print('\n{}[Reduced Model: {}]{}'.format(17*'-', config.transformer, 17*'-'))
        print(model.transformer.config)  
        
    #-----------------------------------------------------------------------------#
    # DP and model to device                                                      #
    #-----------------------------------------------------------------------------#
   
    # Data parallel
    print('GPUs available:', torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
               
    # Model to device   
    model = model.to(config.device)
    
    #-----------------------------------------------------------------------------#
    # Model destillation                                                          #
    #-----------------------------------------------------------------------------#
    
    # Teacher for destillation
    if config.use_teacher:
        teacher = Net(transformer_name=config.transformer_teacher,
                      gradient_checkpointing=False,
                      hidden_dropout_prob=0.0,
                      attention_dropout_prob=0.0,
                      pooling=config.pooling_teacher,
                      projection=config.proj_teacher)
        
        print('\n{}[Teacher: {}]{}'.format(23*'-', config.transformer , 23*'-'))
        print(teacher.transformer.config)
        
        if config.checkpoint_teacher is not None:
            print('Load Teacher-Checkpoint:', config.checkpoint_teacher)
            model_state_dict = torch.load(config.checkpoint_teacher)  
            teacher.load_state_dict(model_state_dict, strict=True) 
        else:
            print('You are using a checkpoint for the Teacher-Model that was not trained on that task!!!')
            
        
        for name, p in teacher.named_parameters():
            p.requires_grad = False
        
        if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
            teacher = torch.nn.DataParallel(teacher, device_ids=config.gpu_ids)
                              
        teacher = teacher.to(config.device)        
                           
    else:
        teacher = None
    
    #-----------------------------------------------------------------------------#
    # Tokenizer                                                                   #
    #-----------------------------------------------------------------------------#   
    
    tokenizer = AutoTokenizer.from_pretrained(config.transformer)

    #-----------------------------------------------------------------------------#
    # Data                                                                        #
    #-----------------------------------------------------------------------------#
    
    df_correlations = pd.read_csv('./data/correlations.csv')
    topics = df_correlations['topic_id'].values
    content = df_correlations['content_ids'].values

    # GT dict for eval
    gt_dict = dict()

    for i in range(len(topics)):
        content_tmp = content[i].split(' ')
        topic_tmp = topics[i]
        gt_dict[topic_tmp] = content_tmp
    
    # split if not train on all data
    if config.train_on_all:
        df_correlations_train = df_correlations
    else:
        df_correlations_train = df_correlations[df_correlations['fold'] != config.fold]
        
        
    if config.debug:
        print(f'DEBUG MODE: use only {config.debug} topics for training')
        #df_correlations = df_correlations.sample(n=config.debug)

    topics = df_correlations_train['topic_id'].values
    content = df_correlations_train['content_ids'].values

    content2topic = defaultdict(set)

    for i in range(len(topics)):
        
        content_tmp = content[i].split(' ')
        topic_tmp = topics[i]
        
        for c in content_tmp:
            content2topic[c].add(topic_tmp)

    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#
   
    # Train
    train_dataset = EqualDatasetTrain(df_correlations=df_correlations_train, 
                                      fold=config.fold,
                                      tokenizer=tokenizer,
                                      max_len=config.max_len,
                                      shuffle_batch_size=config.batch_size,
                                      pool=config.pool,
                                      init_pool=config.init_pool,
                                      train_on_all=config.train_on_all,
                                      language=config.language,
                                      debug=config.debug)
    
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=config.batch_size, 
                              shuffle=not config.custom_sampling,
                              num_workers=config.num_workers,
                              pin_memory=True,
                              collate_fn=train_dataset.smart_batching_collate
                              )
    
    print('\nTrain Pairs:', len(train_dataset ))
    
    
    # Eval
    val_dataset_topic = EqualDatasetEval(mode='topic',
                                         typ='val',
                                         fold=config.fold,
                                         tokenizer=tokenizer,
                                         max_len=config.max_len,
                                         pool=config.pool,
                                         init_pool=config.init_pool,
                                         train_on_all=config.train_on_all,
                                         language=config.language,
                                         debug=config.debug)
    
    val_dataset_content = EqualDatasetEval(mode='content',
                                           typ='val',
                                           fold=config.fold,
                                           tokenizer=tokenizer,
                                           max_len=config.max_len,
                                           pool=config.pool,
                                           init_pool=config.init_pool,
                                           train_on_all=config.train_on_all,
                                           language=config.language,
                                           debug=config.debug)
    
    
    val_loader_topic = DataLoader(dataset=val_dataset_topic, 
                                  batch_size=config.batch_size, 
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  collate_fn=val_dataset_topic.smart_batching_collate
                                  )
    
    val_loader_content = DataLoader(dataset=val_dataset_content, 
                                    batch_size=config.batch_size, 
                                    shuffle=False,
                                    num_workers=config.num_workers,
                                    pin_memory=True,
                                    collate_fn=val_dataset_content.smart_batching_collate
                                    )
    
    print('\nTopics Val:', len(val_dataset_topic))
    print('Content Val:', len(val_dataset_content))
    

    #-----------------------------------------------------------------------------#
    # Sim Sample                                                                  #
    #-----------------------------------------------------------------------------#
    
    train_dataset_topic = EqualDatasetEval(mode='topic',
                                           typ='train',
                                           fold=config.fold,
                                           tokenizer=tokenizer,
                                           max_len=config.max_len,
                                           pool=config.pool,
                                           init_pool=config.init_pool,
                                           train_on_all=config.train_on_all,
                                           language=config.language,
                                           debug=config.debug)
    
    train_dataset_content = EqualDatasetEval(mode='content',
                                             typ='train',
                                             fold=config.fold,
                                             tokenizer=tokenizer,
                                             max_len=config.max_len,
                                             pool=config.pool,
                                             init_pool=config.init_pool,
                                             train_on_all=config.train_on_all,
                                             language=config.language,
                                             debug=config.debug)
    
    
    train_loader_topic = DataLoader(dataset=train_dataset_topic, 
                                  batch_size=config.batch_size, 
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  collate_fn=train_dataset_topic.smart_batching_collate
                                  )
    
    train_loader_content = DataLoader(dataset=train_dataset_content, 
                                    batch_size=config.batch_size, 
                                    shuffle=False,
                                    num_workers=config.num_workers,
                                    pin_memory=True,
                                    collate_fn=train_dataset_content.smart_batching_collate
                                    )

    print('\nTopics Train:', len(train_dataset_topic))
    print('Content Train:', len(train_dataset_content))   
    
    


    
    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn,
                             device=config.device,
                             )

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None
        
    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias']
        optimizer_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)


    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps = math.floor((len(train_loader) * config.epochs) / config.gradient_accumulation)
    warmup_steps = len(train_loader) * config.warmup_epochs
       
    if config.scheduler == 'polynomial':
        print('\nScheduler: polynomial - max LR: {} - end LR: {}'.format(config.lr, config.lr_end))  
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)
        
    elif config.scheduler == 'constant':
        print('\nScheduler: constant - max LR: {}'.format(config.lr))   
        scheduler =  get_constant_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps)
           
    else:
        scheduler = None
        
    print('Warmup Epochs: {} - Warmup Steps: {}'.format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print('Train Epochs:  {} - Train Steps:  {}'.format(config.epochs, train_steps))
        
     
    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    
    if config.zero_shot:
        print('\n{}[{}]{}'.format(30*'-', 'Zero Shot', 30*'-'))  

        f2, precision, recall = evaluate_val(config,
                                            model,
                                            reference_dataloader=val_loader_content,
                                            query_dataloader=val_loader_topic,
                                            gt_dict=gt_dict,
                                            cleanup=True)
       
    #-----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    #-----------------------------------------------------------------------------#            
    # Initial values no sim_sampling for first or first n epochs
    missing_pairs, topic2wrong = None, None
    
    if config.custom_sampling:
        train_loader.dataset.shuffle(missing_list=missing_pairs,
                                     wrong_dict=topic2wrong,
                                     max_wrong=config.max_wrong)
            
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    t_train_start = time.time()
    
    start_epoch = 0   
    best_score = 0
    
    # language switch pool without original position 0
    pools = config.pool[1:] 
    current_pool_pointer = 0
    
    for epoch in range(1, config.epochs+1):
        
        print('\n{}[Epoch: {}]{}'.format(30*'-', epoch, 30*'-'))
        
        train_loss = train(config,
                           model,
                           dataloader=train_loader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler,
                           teacher=teacher)
    
        print('Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}'.format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))
        

        
        print('\n{}[{}]{}'.format(30*'-', 'Evaluate (Val)', 30*'-'))
        

        f2, precision, recall = evaluate_val(config,
                                            model,
                                            reference_dataloader=val_loader_content,
                                            query_dataloader=val_loader_topic,
                                            gt_dict=gt_dict,
                                            cleanup=True)
                                        
                                        
        if f2 > best_score:
            best_score = f2
            
            best_checkpoint = '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, f2)
            
            if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                torch.save(model.module.state_dict(), best_checkpoint)
            else:
                torch.save(model.state_dict(), best_checkpoint)
            
        elif f2 < (0.8 * best_score):
            print('Something went wrong:')
            print(f'Resett to: {best_checkpoint} -> and continue training'  )
            model_state_dict = torch.load(best_checkpoint)
            if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                model.module.load_state_dict(model_state_dict, strict=True) 
            else:
                model.load_state_dict(model_state_dict, strict=True)                                    
                                            
                                                                     
        if config.sim_sample:
            print('\n{}[{}]{}'.format(30*'-', 'Evaluate (Train)', 30*'-'))
            # Set pool for next epoch -> sim sample for that pool
            if len(config.pool) > 1:
                if epoch < config.epoch_stop_switching:
                
                    # come back to original pool 0 every uneven epoch
                    if epoch % 2 == 0:
                        next_pool = 0
                    else:
                        next_pool = pools[current_pool_pointer % len(pools)]
                        current_pool_pointer += 1
                    
                    # set train data for next epoch
                    train_loader_content.dataset.set_pool(next_pool)
                    train_loader_topic.dataset.set_pool(next_pool)
                    train_loader.dataset.set_pool(next_pool)
                else:
                    train_loader_content.dataset.set_pool(0)
                    train_loader_topic.dataset.set_pool(0)  
                    train_loader.dataset.set_pool(0)
            
            if epoch >= config.sim_sample_start:
                missing_pairs, topic2wrong = evaluate_train(config=config,
                                                            model=model,
                                                            reference_dataloader=train_loader_content,
                                                            query_dataloader=train_loader_topic,
                                                            gt_dict=gt_dict,
                                                            content2topic=train_loader.dataset.content2topic,
                                                            cleanup=True)
                                                        
        if config.custom_sampling:    
            train_loader.dataset.shuffle(missing_list=missing_pairs,
                                         wrong_dict=topic2wrong,
                                         max_wrong=config.max_wrong)
                                         
                                         
                
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))            
