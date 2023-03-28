import torch
import numpy as np
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        attention_mask = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(last_hidden_state * attention_mask, 1)
        sum_mask = attention_mask.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings   
    
class AttentionPooling(nn.Module):
    def __init__(self):
        super(AttentionPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        return last_hidden_state[:, 0, :]   
    
class Net(nn.Module):
    def __init__(self,
                 transformer_name,
                 pretrained=True,
                 gradient_checkpointing=False,
                 hidden_dropout_prob=0.1,
                 attention_dropout_prob=0.1,
                 num_hidden_layers=None,
                 pooling="mean",
                 projection=None):
        
        super().__init__()

        self.config = AutoConfig.from_pretrained(transformer_name)
     
        self.config.update({"hidden_dropout_prob": hidden_dropout_prob,
                            "attention_probs_dropout_prob": attention_dropout_prob})
        
        if num_hidden_layers is not None:
            self.config.num_hidden_layers = num_hidden_layers
            
        if pretrained:
            self.transformer = AutoModel.from_pretrained(transformer_name, config=self.config)
        else:
            self.transformer = AutoModel.from_config(config=self.config)
        
        if pooling == "cls":
            self.pooler = AttentionPooling()
        elif pooling == "pooler":
            self.pooler = "pooler"
        else:
            self.pooler = MeanPooling()
    
        if gradient_checkpointing:
            self.transformer.gradient_checkpointing_enable()
         
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if projection is not None:
            self.proj = torch.nn.Linear(projection[0], projection[1], bias=False)
            self._init_weights(self.proj)
        else:
            self.proj = None
            
            
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    
    def forward(self, ids1, mask1, ids2=None, mask2=None):
        
        if ids2 is not None and mask2 is not None:
            
            ids = torch.cat([ids1, ids2], dim=0)
            mask = torch.cat([mask1, mask2], dim=0)
            
            transformer_out = self.transformer(ids, mask)
            
            if self.pooler == "pooler":
                pooled_output = transformer_out.pooler_output
            else:
                sequence_output = transformer_out.last_hidden_state
                pooled_output = self.pooler(sequence_output, mask)
        
            
            if self.proj is not None:
                pooled_output = self.proj(pooled_output)
  
                
            pooled_output1 = pooled_output[:len(ids1), :]
            pooled_output2 = pooled_output[len(ids1):, :]
            
            sequence_output = sequence_output * mask.unsqueeze(-1)
            
            return pooled_output1, pooled_output2, sequence_output
        
        else:
            transformer_out = self.transformer(ids1, mask1)

            if self.pooler == "pooler":
                pooled_output = transformer_out.pooler_output
            else:
                sequence_output = transformer_out.last_hidden_state
                pooled_output = self.pooler(sequence_output, mask1)

            if self.proj is not None:
                pooled_output = self.proj(pooled_output)
                
            return pooled_output