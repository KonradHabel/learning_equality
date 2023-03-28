from torch.utils.data import Dataset
import torch
import time
      
class EqualDatasetEval(Dataset):
    
    def __init__(self,
                 sorted_input,
                 pad_token_id,
                 max_len):
        
        super().__init__()
        
        self.input = sorted_input
        self.pad_token_id = pad_token_id
        self.max_len = max_len
        
        
    def __getitem__(self, index):
        return self.input[index]

    def __len__(self):
        return len(self.input)
    
    
    def smart_batching_collate(self, batch):
        
        sequences, targets, language, length = list(zip(*batch))
        
        b_max_len = min(max(length), self.max_len)
        bs = len(sequences)
        
        mask = torch.zeros((bs, self.max_len), dtype=torch.float)
        input_ids = torch.full((bs, self.max_len), self.pad_token_id, dtype=torch.long)
        
        for i in range(bs):
            mask[i, :length[i]] = 1
            input_ids[i, :length[i]] = sequences[i]
         
        # cut to actually longest in batch    
        return input_ids[:, :b_max_len], mask[:, :b_max_len], targets, language




def sort_input(text, names, language, tokenizers, max_len):
    
    t0 = time.time()
   
    print('Tokenizing {:,} training samples...'.format(len(text)))

    update_interval = len(text) // 10

    input_ids = []
    length = []
    for t in text:
        if ((len(input_ids) % update_interval) == 0):
            print('  Tokenized {:,} samples.'.format(len(input_ids)))

        input_id = tokenizers.encode(
            text=t,           
            add_special_tokens=True, 
            max_length=max_len,  
            truncation=True,     
            padding=False,
            return_tensors='pt'
        )   
        
        input_id = input_id.squeeze()                               
        input_ids.append(input_id)
        length.append(len(input_id))
        
    print('DONE.')
    print('{:>10,} samples'.format(len(input_ids)))
    

    sorted_input = sorted(zip(input_ids, names, language, length), key=lambda x: x[-1], reverse=True)
    print('Shortest sample:', len(sorted_input[0][0]))
    print('Longest sample:', len(sorted_input[-1][0]))
        

    t1 = time.time()
    print(f"Time: {t1-t0:.3f} sec")
    
    return sorted_input