import torch
import pickle
from utils.model import LSTMLanguageModel
from torchtext.data.utils import get_tokenizer
device = torch.device('cpu')

tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            #prediction: [batch size, seq len, vocab size]
            #prediction[:, -1]: [batch size, vocab size] #probability of last vocab
            
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']: #if it is unk, we sample again
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:    #if it is eos, we stop
                break

            indices.append(prediction) #autoregressive, thus output becomes input

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens

def predict(prompt,model,vocab,temperature=1):
    max_seq_len = 30
    seed = 0
    generation = generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed)
    return ' '.join(generation)

def load_lstm():
    #utils\vocab.pickle
    with open('./vocab.pickle', 'rb') as handle:
        vocab = pickle.load(handle)


    vocab_size = len(vocab)
    emb_dim = 1024                # 400 in the paper
    hid_dim = 1024                # 1150 in the paper
    num_layers = 2                # 3 in the paper
    dropout_rate = 0.65              
    lr = 1e-3                     
    model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate)
    save_path = f'./model/best-val-auto.pt'
    model.load_state_dict(torch.load(save_path,map_location=torch.device('cpu')))
    return model, vocab
    

if __name__ == "__main__":


    model,vocab_dict = load_lstm()
    prompt = 'import numpy'

#sample from this distribution higher probability will get more change
    temperatures = [0.4, 0.7, 1.0]
    for temp in temperatures:
        generation = generate(prompt, 30, temp, model, tokenizer, 
                            vocab_dict, device, 0)
        print(generation)
    # print(str(temperature)+'\n'+' '.join(generation)+'\n')
