import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast

def train(config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None, teacher=None):
    
    
    if teacher:
        teacher.eval()
        teacher_loss = torch.nn.MSELoss()

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if config.verbose:
        time.sleep(0.1)
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
        
    # for loop over one epoch
    for ids1, mask1, ids2, mask2 in bar:
        
        if scaler:
            
            #-----------------------------------------------------------------------------#
            # using fp16                                                                  #
            #-----------------------------------------------------------------------------#
            
            with autocast():
                
                ids1 = ids1.to(config.device)
                mask1 =  mask1.to(config.device)
                ids2 = ids2.to(config.device)
                mask2 =  mask2.to(config.device)
                 
                # Forward pass
                features1, features2, sequence_output = model(ids1, mask1, ids2, mask2)
                
            
                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1: 
                    logit_scale = model.module.logit_scale.exp()
                else:
                    logit_scale = model.logit_scale.exp()
                

                loss = loss_function(features1, features2, logit_scale) 
                
                
                if teacher:
                
                    with torch.no_grad():
                        teach1, teach2, teach_sequence_output = teacher(ids1, mask1, ids2, mask2) 
                    
                    loss_teach = 0.25 * teacher_loss(features1, teach1) + 0.25 * teacher_loss(features2, teach2) + 0.5 * teacher_loss(sequence_output, teach_sequence_output) 
                    
                    loss = loss_teach
                    
                    
                losses.update(loss.item())
                
            # Calculate gradients using backward pass      
            scaler.scale(loss).backward()
            
            if config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), config.clip_grad) 
            
            if step % config.gradient_accumulation == 0:
                # Update model parameters (weights)
                scaler.step(optimizer)
                scaler.update()

                # Zero gradients for next step
                optimizer.zero_grad()
                
                # Scheduler
                if config.scheduler == "polynomial" or config.scheduler == "cosine" or config.scheduler ==  "constant":
                    scheduler.step()
   
        else:
            
            #-----------------------------------------------------------------------------#
            # using fp32                                                                  #
            #-----------------------------------------------------------------------------#
            
            ids1 = ids1.to(config.device)
            mask1 =  mask1.to(config.device)
            ids2 = ids2.to(config.device)
            mask2 =  mask2.to(config.device)

            # Forward pass
            features1, features2 = model(ids1, mask1, ids2, mask2)
            
            if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1: 
                logit_scale = model.module.logit_scale.exp()
            else:
                logit_scale = model.logit_scale.exp()
            

            loss = loss_function(features1, features2, logit_scale) 
                
            if teacher:
                with torch.no_grad():
                    teach1, teach2 = teacher(ids1, mask1, ids2, mask2) 
                
                loss_teach = 0.25 * teacher_loss(features1, teach1) + 0.25 * teacher_loss(features2, teach2) + 0.5 * teacher_loss(sequence_output, teach_sequence_output) 
                
                loss = loss_teach
                
            losses.update(loss.item())

            # Calculate gradients using backward pass
            loss.backward()
            
                       
            if config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), config.clip_grad)        
                        
            if step % config.gradient_accumulation == 0:
                # Update model parameters (weights)
                optimizer.step()
                # Zero gradients for next step
                optimizer.zero_grad()
                
                # Scheduler
                if config.scheduler == "polynomial" or config.scheduler == "cosine" or config.scheduler ==  "constant":
                    scheduler.step()
        
        
        
        if config.verbose:
        
            if teacher:
                monitor = {
                           "loss": "{:.4f}".format(loss.item()),
                           "loss_teach": "{:.4f}".format(loss_teach.item()),
                           "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])
                           }
            else:
                monitor = {
                           "loss": "{:.4f}".format(loss.item()),
                           "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])
                           }            
             
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1
        
        if config.debug:
            if step > config.debug:
                break

    if config.verbose:
        bar.close()

    return losses.avg


