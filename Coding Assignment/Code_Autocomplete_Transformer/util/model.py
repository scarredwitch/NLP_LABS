import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

import torchtext, datasets, math
from tqdm import tqdm

from queue import PriorityQueue
import operator


tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        #x = [batch size, query len, hid dim]
        
        return x, attention
    

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        #x = [batch size, seq len, hid dim]
        
        return x

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, 
                 pf_dim, dropout, device, pad_idx, max_length = 100):
                
        super().__init__()
        
        self.device = device
        self.output_dim = output_dim
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(100, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.pad_idx = pad_idx
    
    def make_mask(self, x):
        
        #x = [batch size, len]
        
        pad_mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)
        #pad_mask = [batch size, 1, 1, len]
        
        x_len = x.shape[1]
        
        sub_mask = torch.tril(torch.ones((x_len, x_len), device = self.device)).bool()
        #sub_mask = [len, len]
            
        mask = pad_mask & sub_mask
        #mask = [batch size, 1, len, len]
        
        return mask 
    
    def forward(self, x):
        
        #x = [batch size, len]
                
        batch_size = x.shape[0]
        x_len      = x.shape[1]
        
        #get mask here since we remove seq2seq class
        mask   = self.make_mask(x)
        #mask = [batch size, 1, len, len]

        pos = torch.arange(0, x_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)          
            
        x = self.dropout((self.tok_embedding(x) * self.scale) + self.pos_embedding(pos))
        #x = [batch size, len, hid dim]
        
        for layer in self.layers:
            x, attention = layer(x, mask)
        
        #x = [batch size, len, hid dim]
        #attention = [batch size, n heads, len, len]
        
        output = self.fc_out(x)
        #output = [batch size, len, output dim]
            
        return output, attention

    def beam_decode(self, penalty_alpha = 0.9, max_length = 5, beam_size = 5):
        
        # Start with SOS Harry Potter is
        prompt = 'Harry Potter is '
        
        tokens = tokenizer(prompt)
        indices = [SOS_IDX] + [vocab[t] for t in tokens]

        decoder_input = torch.Tensor([indices]).long().to(device)
        #decoder_input: [batch size, len] = [1, 1]
        scores = torch.Tensor([0.]).to(device)
        #scores: [1]
        
        for i in range(max_length):
            
            # print(f"========Length: {i}")
            
            # Decoder prediction
            logits, _ = self.forward(decoder_input)
            #[beam_size, current dec len=i, vocab_size]
                        
            logits = logits[:, -1] 
            # Last sequence step: [beam_size, current dec len=i, vocab_size] => [beam_size, vocab_size]
            
            # print(f"{logits.shape=}")

            # Softmax
            # Log softmax is better, since beam search accumulates probability
            # if simply softmax, the probability can get too small and then become unstable
            log_probs = torch.log_softmax(logits, dim=1)
    
            # Add length penalty, otherwise, always very short sentence will win...
            penalty   = ((5 + (i+1)) / (5 + 1)) ** penalty_alpha #see https://arxiv.org/abs/1609.08144
            log_probs = log_probs / penalty
            
            # print(f"{decoder_input[:, -1]=}")
            
            # Update score where EOS has not been reached
            log_probs[decoder_input[:, -1]==EOS_IDX, :] = -2 #discouraged it to end
            log_probs[decoder_input[:, -1]==UNK_IDX, :] = -10 #very discouraged to spit out unk
            scores = scores.unsqueeze(1) + log_probs 
            # scores: [beam_size, vocab_size]
            # log_probs: [beam_size, vocab_size]

            # print(f"{log_probs.shape=}")
            # print(f"{scores.shape=}")
            #log_probs: torch.Size([1, 29475])
            #scores.shape=torch.Size([1, 29475])
            
            # Flatten scores from [beams, vocab_size] to [beams * vocab_size] to get top k, and reconstruct beam indices and token indices
            # Since we flatten it, we have to retrieve the actual beam indices and token_indices using floor division and remainder
            # You can try on paper; it will make sense
            scores, indices = torch.topk(scores.reshape(-1), beam_size) #scores: [beam_size]; #indices: [beam_size]
            beam_indices  = torch.divide   (indices, self.output_dim, rounding_mode='floor') # indices // vocab_size
            token_indices = torch.remainder(indices, self.output_dim)                        # indices %  vocab_size
            
            # print(f"{scores=}")
            # print(f"{indices.shape=}")
            
            # print(f"{indices=}")
            # print(f"{beam_indices=}")
            # print(f"{token_indices=}")
            
            # Build the next decoder input
            # For efficiency, the trick is to concatenate all hypotheses into one string and sent to decoder at once
            # We can later chop it ...
            next_decoder_input = []
            for beam_index, token_index in zip(beam_indices, token_indices):
                # print(f"{beam_index=}")
                prev_decoder_input = decoder_input[beam_index]
                # print(f"{prev_decoder_input=}")
                if prev_decoder_input[-1]==EOS_IDX:
                    token_index = EOS_IDX # once EOS, always EOS
                token_index = torch.LongTensor([token_index]).long().to(device)
                next_decoder_input.append(torch.cat([prev_decoder_input, token_index]))
                # print("here: " + " ".join([vocab.lookup_token(i) for i in next_decoder_input[-1]]) + "; score: " + str(scores[beam_index].item()))
            decoder_input = torch.vstack(next_decoder_input)
            
            # print(f"{decoder_input=}")
            
             # If all beams are finished, and the length is at least 5, exit
            if i > 5:
                if (decoder_input[:, -1]==EOS_IDX).sum() == beam_size:
                    break
                
        # convert the top scored sequence to a list of text tokens
        decoder_output, _ = max(zip(decoder_input, scores), key=lambda x: x[1])
        decoder_output = decoder_output[1:].cpu().numpy() # remove SOS
        
        return [vocab.lookup_token(i) for i in decoder_output if i != EOS_IDX] # remove EOS if exists
    
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)        
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        
        #x = [batch size, len, hid dim]
        #mask = [batch size, 1, len, len]
        
        #multi attention, skip and then norm
        _x, attention = self.self_attention(x, x, x, mask)
        x = self.self_attn_layer_norm(x + self.dropout(_x))
        #x = [batch size, len, hid dim]
        #attention = [batch size, n heads, len, len]
    
        #positionwise feedforward
        _x = self.positionwise_feedforward(x)
        x = self.ff_layer_norm(x + self.dropout(_x))
        #x = [batch size, len, hid dim]
        
        return x, attention