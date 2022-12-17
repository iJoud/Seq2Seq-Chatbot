import random 
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)

    def forward(self, x, hidden, cell_state):
        x = self.embedding(x)
        x = x.view(1, 1, -1)
        x, (hidden, cell_state) = self.lstm(x, (hidden, cell_state))
        return x, hidden, cell_state
        

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.embedding = nn.Embedding(output_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim= 1)

    def forward(self, x, hidden, cell_state):
        x = self.embedding(x)
        x = x.view(1, 1, -1)
        x, (hidden, cell_state) = self.lstm(x, (hidden, cell_state))
        x = self.softmax(self.fc(x[0]))
        return x, hidden, cell_state
    
     
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.encoder = Encoder(self.input_size, self.hidden_size)
        self.decoder = Decoder(self.hidden_size, self.output_size)
        
    def forward(self, src, trg, src_len, trg_len, teacher_force=1):
        
        output = {
            'decoder_output':[]
        }
        
        encoder_hidden = torch.zeros([1, 1, self.hidden_size]).to(device) # 1 = number of LSTM layers
        cell_state = torch.zeros([1, 1, self.hidden_size]).to(device)  
        
        for i in range(src_len):
            encoder_output, encoder_hidden, cell_state = self.encoder(src[i], encoder_hidden, cell_state)

        decoder_input = torch.Tensor([[0]]).long().to(device) # 0 = SOS_token
        decoder_hidden = encoder_hidden
        
        for i in range(trg_len):
            decoder_output, decoder_hidden, cell_state = self.decoder(decoder_input, decoder_hidden, cell_state)
            output['decoder_output'].append(decoder_output)
            
            if self.training: # Model not in eval mode
                decoder_input = target_tensor[i] if random.random() > teacher_force else decoder_output.argmax(1) # teacher forcing
            else:
                _, top_index = decoder_output.data.topk(1)
                decoder_input = top_index.squeeze().detach()
                
        return output
