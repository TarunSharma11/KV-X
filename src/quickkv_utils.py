
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math

# perform qk calculation and get indices
# this version will not update in inference mode

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class QuickKVCluster():
    def __init__(self, num_hidden_layers = 32, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', beta = 20, num_layers = 80, layer_idx=None):
        
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        
        self.steps = -1
        self.beta = beta
        
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.previous_attention_weights = None

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.previous_attention_weights = None

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        # reduce cache line size gradually for this layer
        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num
        
        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt - self.window_size) * 2 - max_num
    
        steps = (max_num - min_num) // (self.num_hidden_layers - 1)
        max_capacity_prompt = max_num - self.layer_idx * steps
       
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            # hyperparameter for controlling mask size is hardcoded here
            mask_size= math.floor((1.0*max_capacity_prompt)/4)
            middle_compressed_size= mask_size
            k_cur = key_states[:, :, -self.window_size:, :].to("cpu")
            v_cur = value_states[:, :, -self.window_size:, :].to("cpu")

            if (mask_size and max_capacity_prompt >= (4.0*mask_size)) :
                # apply mask on left and right KVs. then compress middle KVs
                left_k = key_states[:, :, :mask_size, :].to("cpu")
                left_v = value_states[:, :, :mask_size, :].to("cpu")
                right_k =  key_states[:, :, -mask_size:, :].to("cpu")
                right_v =  value_states[:, :, -mask_size:, :].to("cpu")

                middle_k = key_states[:, :, mask_size : -mask_size, :].to("cpu")
                middle_v =  value_states[:, :, mask_size : -mask_size, :].to("cpu")

                middle_weights = torch.matmul(query_states[..., -self.window_size:, :].to("cpu"), key_states[:, :, mask_size : -mask_size, :].to("cpu").transpose(2, 3)) / math.sqrt(head_dim)
                mask = torch.full((self.window_size, self.window_size), torch.finfo(middle_weights.dtype).min, device=middle_weights.device)
                mask_cond = torch.arange(mask.size(-1), device=middle_weights.device)
                mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
                mask = mask.to(middle_weights.device)
                attention_mask = mask[None, None, :, :].to("cpu")

                middle_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask
                middle_weights = nn.functional.softmax(middle_weights, dim=-1, dtype=torch.float32).to("cpu")
                middle_weights_sum = middle_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2).to("cpu")

                if self.pooling == 'avgpool':
                    attn_cache = F.avg_pool1d(middle_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1).to("cpu")
                elif self.pooling == 'maxpool':
                    attn_cache = F.max_pool1d(middle_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1).to("cpu")
                else:
                    raise ValueError('Pooling method not supported')
            
                indices = attn_cache.topk(middle_compressed_size, dim=-1).indices.to("cpu")
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                
                middle_kcompressed = middle_k[:, :, :, :].gather(dim = 2, index = indices).to("cpu")
                middle_vcompressed = middle_v[:, :, :, :].gather(dim = 2, index = indices).to("cpu")

                k_past_compress = torch.cat([left_k, middle_kcompressed], dim = 2)
                v_past_compress =  torch.cat([left_v, middle_vcompressed], dim = 2)

                k_past_compress = torch.cat([k_past_compress, right_k], dim = 2)
                v_past_compress =  torch.cat([v_past_compress, right_v], dim = 2)

                key_states = torch.cat([k_past_compress, k_cur], dim = 2).to("cuda:0")
                value_states = torch.cat([v_past_compress, v_cur], dim = 2).to("cuda:0")
            else:
                # do not apply mask, line is too small - just compress
                middle_weights = torch.matmul(query_states[..., -self.window_size:, :].to("cpu"), key_states.to("cpu").transpose(2, 3)) / math.sqrt(head_dim)
                mask = torch.full((self.window_size, self.window_size), torch.finfo(middle_weights.dtype).min, device=middle_weights.device)
                mask_cond = torch.arange(mask.size(-1), device=middle_weights.device)
                mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
                mask = mask.to(middle_weights.device)
                attention_mask = mask[None, None, :, :].to("cpu")

                middle_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask
                middle_weights = nn.functional.softmax(middle_weights, dim=-1, dtype=torch.float32).to(query_states.dtype).to("cpu")
                middle_weights_sum = middle_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)

                if self.pooling == 'avgpool':
                    attn_cache = F.avg_pool1d(middle_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                elif self.pooling == 'maxpool':
                    attn_cache = F.max_pool1d(middle_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                else:
                    raise ValueError('Pooling method not supported')
                
                indices = attn_cache.topk(max_capacity_prompt, dim=-1).indices.to("cpu")
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

                kcompressed = key_states.to("cpu").gather(dim = 2, index = indices)
                vcompressed = value_states.to("cpu").gather(dim = 2, index = indices)

                key_states = torch.cat([kcompressed, k_cur], dim = 2).to("cuda:0")
                value_states = torch.cat([vcompressed, v_cur], dim = 2).to("cuda:0")

            return key_states, value_states


def init_quickkv(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 1024
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
    
    
    self.kv_cluster = QuickKVCluster( 
        num_hidden_layers = num_hidden_layers,
        layer_idx = self.layer_idx,
        window_size = self.config.window_size, 
        max_capacity_prompt = 512, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling
        )
