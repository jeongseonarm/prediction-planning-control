import math
import torch
import torch.nn as nn

class FC_projection(nn.Module):
    # [Latent vector로의 projection을 위한 class - from Transformers for Image Recognition at Scale(VIT)]
    def __init__(self, Config_dict):
        super().__init__()
        self.Linear = nn.Linear(Config_dict['Feature_len'], Config_dict['TF_dim'])

    def forward(self, Input_tensor):
        # [Input  - Tensor(Batch_size, Seq_len, Feature_len)]
        # [Output - Tensor(Batch_size, Seq_len, TF_dim)]
        return self.Linear(Input_tensor)

class FC_projection_encoder(nn.Module):
    # [Latent vector로의 projection을 위한 class - from Transformers for Image Recognition at Scale(VIT)]
    def __init__(self, Config_dict):
        super().__init__()
        self.Linear = nn.Linear(Config_dict['Feature_len']+Config_dict['Maneuver_num'], Config_dict['TF_dim'])

    def forward(self, Input_tensor):
        # [Input  - Tensor(Batch_size, Seq_len, Feature_len)]
        # [Output - Tensor(Batch_size, Seq_len, TF_dim)]
        return self.Linear(Input_tensor)
    
class FC_projection_encoder_max(nn.Module):
    # [Latent vector로의 projection을 위한 class - from Transformers for Image Recognition at Scale(VIT)]
    def __init__(self, Config_dict):
        super().__init__()
        self.Linear = nn.Linear(Config_dict['Feature_len']+1, Config_dict['TF_dim'])

    def forward(self, Input_tensor):
        # [Input  - Tensor(Batch_size, Seq_len, Feature_len)]
        # [Output - Tensor(Batch_size, Seq_len, TF_dim)]
        return self.Linear(Input_tensor)
    
class Encoder_projection(nn.Module):
    # [Latent vector로의 projection을 위한 class - from Transformers for Image Recognition at Scale(VIT)]
    def __init__(self, Config_dict):
        super().__init__()
        self.Linear = nn.Linear(Config_dict['Feature_len'], Config_dict['TF_dim'])

    def forward(self, Input_tensor):
        # [Input  - Tensor(Batch_size, Seq_len, Feature_len)]
        # [Output - Tensor(Batch_size, Seq_len, TF_dim)]
        return self.Linear(Input_tensor)
    
class Decoder_projection(nn.Module):
    # [Latent vector로의 projection을 위한 class - from Transformers for Image Recognition at Scale(VIT)]
    def __init__(self, Config_dict):
        super().__init__()
        self.Linear = nn.Linear(2, Config_dict['TF_dim'])

    def forward(self, Input_tensor):
        # [Input  - Tensor(Batch_size, Seq_len, Feature_len)]
        # [Output - Tensor(Batch_size, Seq_len, TF_dim)]
        return self.Linear(Input_tensor)


class LSTM_projection(nn.Module):
    # [Latent vector로의 projection을 위한 class - from Transformers for Image Recognition at Scale(VIT)]
    def __init__(self, Config_dict):
        super().__init__()
        self.LSTM = nn.LSTM(input_size = Config_dict['Feature_len'], hidden_size = Config_dict['TF_dim'], num_layers = 1, batch_first = True)

    def forward(self, Input_tensor):
        # [Input  - Tensor(Batch_size, Seq_len, Feature_len)]
        # [Output - Tensor(Batch_size, Seq_len, TF_dim)]
        O, _ = self.LSTM(Input_tensor)
        return O
    
class Positional_encoding(nn.Module):
    # [Projection된 tensor에 위치정보를 부여해주기 위한 class]
    def __init__(self, Config_dict):
        super().__init__()
        self.max_len = Config_dict['Max_len']
        self.TF_dim  = Config_dict['TF_dim']

        pe = torch.zeros(self.max_len, self.TF_dim)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.TF_dim, 2).float() * (-math.log(10000.0) / self.TF_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, Input_tensor):
        # [Input  - Tensor(Batch_size, Seq_len, TF_dim)]
        # [Output - Tensor(Batch_size, Seq_len, TF_dim)]
        Batch_size = Input_tensor.size(0)
        return Input_tensor + self.pe[:Batch_size, :]
    
class Generate_QKV(nn.Module):
    # [Projection된 tensor를 사용하여 Q, K, V를 생성해주기 위한 class]
    def __init__(self, Config_dict):
        super().__init__()
        # [Set params]
        self.TF_dim = Config_dict['TF_dim']
        
        # [Query, Key, Value]
        self.Q = nn.Linear(self.TF_dim, self.TF_dim)
        self.K = nn.Linear(self.TF_dim, self.TF_dim)
        self.V = nn.Linear(self.TF_dim, self.TF_dim)
        
    def forward(self, Input_tensor):
        # [Input  - Tensor(Batch_size, Seq_len, TF_dim)]
        # [Output - Tensor(Batch_size, Seq_len, TF_dim)]
        Q = self.Q(Input_tensor)
        K = self.K(Input_tensor)
        V = self.V(Input_tensor)
        return Q, K, V

class Multi_head_attention(nn.Module):
    # [Encoder, Decoder 내부의 MHA class]
    def __init__(self, Config_dict, Mask):
        super().__init__()
        # [Set params]
        self.Num_heads      = Config_dict['Num_heads']
        self.TF_dim         = Config_dict['TF_dim']
        self.Head_dim       = self.TF_dim // self.Num_heads
        self.Device         = Config_dict['Device']
        self.Mask           = Mask
        
        # [Set dimension, factor]
        self.QKV_dimension  = torch.tensor(self.Head_dim, device=self.Device)
        self.Scale_factor   = torch.sqrt(self.QKV_dimension)
        
        # [Softmax]
        self.Softmax = nn.Softmax(dim=-1)
        
        # [Linear]
        self.Linear = nn.Linear(self.TF_dim, self.TF_dim)
        
    def Look_ahead_mask(self, Seq_len, Device):
        Device      = Device
        Mask_matrix = torch.tril(torch.ones((Seq_len, Seq_len), device = Device)).bool()
        return Mask_matrix

    def forward(self, Q, K, V):     
        # [Input  - Tensor(Batch_size, Q_len, TF_dim), Tensor(Batch_size, K_len, TF_dim), Tensor(Batch_size, V_len, TF_dim)]
        # [Output - Tensor(Batch_size, Q_len, TF_dim), Attention_Score(Batch_size, Num_heads, Q_len, K_len)]
   
        # [Store batch size]
        Batch_size = Q.size(0)
        
        # [Scaled Dot-product Attention (TF_dim = Num_head*QKV_dimension)]
        # [Batch_size, Num_heads, Q/K/V_len, QKV_dimension]
        Q  = Q.view(Batch_size, self.Num_heads, -1, self.QKV_dimension)
        KT = K.view(Batch_size, self.Num_heads, self.QKV_dimension, -1)
        V  = V.view(Batch_size, self.Num_heads, -1, self.QKV_dimension)

        # [Attention score]
        # [Batch_size, Num_heads, Q_len, K_len]
        Attention_score = torch.matmul(Q, KT) / self.Scale_factor

        # [Look ahead mask in decoder]
        if self.Mask == 'Look_ahead_mask':
            Mask_matrix     = self.Look_ahead_mask(Attention_score.size(-1), self.Device)
            Attention_score = Attention_score.masked_fill(Mask_matrix==False, float("-inf"))
            
        # [Attention - Concat]
        # [(Batch_size, Num_heads, Q_len, K_len) X (Batch_size, Num_heads, V_len, QKV_dimension)]
        ## -> [Batch_size, Num_heads, Q_len, QKV_dimension]
        ## -> [Batch_size, Q_len, TF_dim]
        Attention = torch.matmul(self.Softmax(Attention_score), V).view(Batch_size, -1, self.TF_dim)
        
        # [Attention - Concat - Linear]
        Attention = self.Linear(Attention)
        
        return Attention, self.Softmax(Attention_score)

class FC_feed_forward_networks(nn.Module):
    # [Encoder, Decoder 내부의 FFN class (using FC)]
    def __init__(self, Config_dict):
        super().__init__()
        self.TF_dim  = Config_dict['TF_dim']
        self.FFN_dim = Config_dict['FFN_dim']
        
        self.Linear_1 = nn.Linear(self.TF_dim, self.FFN_dim)
        self.Linear_2 = nn.Linear(self.FFN_dim, self.TF_dim)
        self.Relu     = nn.ReLU()
        
    def forward(self, Input_tensor):
        # [Input  - Tensor(Batch_size, Q_len, TF_dim)]
        # [Output - Tensor(Batch_size, Q_len, TF_dim)]
        return self.Linear_2(self.Relu(self.Linear_1(Input_tensor)))
    
class LSTM_feed_forward_networks(nn.Module):
    # [Encoder, Decoder 내부의 FFN class (using LSTM)]
    def __init__(self, Config_dict):
        super().__init__()
        self.TF_dim  = Config_dict['TF_dim']
        self.FFN_dim = Config_dict['FFN_dim']

        self.LSTM_1 = nn.LSTM(input_size = self.TF_dim, hidden_size = self.FFN_dim, num_layers = 1, batch_first = True)
        self.LSTM_2 = nn.LSTM(input_size = self.FFN_dim, hidden_size = self.TF_dim, num_layers = 1, batch_first = True)
        self.Relu   = nn.ReLU()
        
    def forward(self, Input_tensor):
        # [Input  - Tensor(Batch_size, Q_len, TF_dim)]
        # [Output - Tensor(Batch_size, Q_len, TF_dim)]
        O, _ = self.LSTM_1(Input_tensor)
        O = self.Relu(O)
        O, _ = self.LSTM_2(O)
        
        return O

class Transformer_encoder_layer(nn.Module):
    # [Transformer encoder layer class]
    def __init__(self, Config_dict):
        super().__init__()
        self.Encoder_QKV = Generate_QKV(Config_dict)
        self.Encoder_MHA = Multi_head_attention(Config_dict, Mask=None)
        self.Encoder_FFN = FC_feed_forward_networks(Config_dict)
        self.Encoder_LN1 = nn.LayerNorm(Config_dict['TF_dim'])
        self.Encoder_LN2 = nn.LayerNorm(Config_dict['TF_dim'])
        
    def forward(self, Encoder_input_tensor):
        # [Input  - Tensor(Batch_size, Q_len, TF_dim)]
        # [Output - Tensor(Batch_size, Q_len, TF_dim)]
        Q, K, V = self.Encoder_QKV(Encoder_input_tensor)
        _Encoder_Output, _ = self.Encoder_MHA(Q, K, V)
        _Encoder_Output = self.Encoder_LN1(_Encoder_Output + Encoder_input_tensor)
        Encoder_Output = self.Encoder_FFN(_Encoder_Output)
        Encoder_Output = self.Encoder_LN2(Encoder_Output + _Encoder_Output)
        
        return Encoder_Output

class Transformer_encoder(nn.Module):
    def __init__(self, Config_dict):
        super().__init__()
        # [Set Encoder list]
        self.Encoder_list = nn.ModuleList([Transformer_encoder_layer(Config_dict) for _ in range(Config_dict['Num_enc'])])
        self.apply_xavier()
        
    def apply_xavier(self):
        print('- Apply xavier in encoder -')
        for layer in self.Encoder_list.named_parameters():
            if layer[1].dim() > 1:
                nn.init.xavier_uniform_(layer[1])
                
    def forward(self, Encoder_tensor):
        for Encoder in self.Encoder_list:
            Encoder_tensor = Encoder(Encoder_tensor)
            
        return Encoder_tensor

class Transformer_encoder_only(nn.Module):
    def __init__(self, Config_dict):
        super().__init__()
        # [Set Encoder list]
        self.Encoder_list = nn.ModuleList([Transformer_encoder_layer(Config_dict) for _ in range(Config_dict['Num_enc_only'])])
        self.apply_xavier()
        
    def apply_xavier(self):
        print('- Apply xavier in encoder -')
        for layer in self.Encoder_list.named_parameters():
            if layer[1].dim() > 1:
                nn.init.xavier_uniform_(layer[1])
                
    def forward(self, Encoder_tensor):
        for Encoder in self.Encoder_list:
            Encoder_tensor = Encoder(Encoder_tensor)
            
        return Encoder_tensor
    
class Transformer_decoder_layer(nn.Module):
    def __init__(self, Config_dict):
        super().__init__()
        self.Decoder_QKV  = Generate_QKV(Config_dict)
        # [Masked-Multi head attention]
        self.Decoder_MMHA = Multi_head_attention(Config_dict, Mask='Look_ahead_mask')
        # [Multi-head attention]
        self.Decoder_MHA  = Multi_head_attention(Config_dict, Mask=None)
        # [Feed forward]
        self.Decoder_FFN  = FC_feed_forward_networks(Config_dict)
        # [Layer Norm]
        self.Decoder_LN1  = nn.LayerNorm(Config_dict['TF_dim'])
        self.Decoder_LN2  = nn.LayerNorm(Config_dict['TF_dim'])
        self.Decoder_LN3  = nn.LayerNorm(Config_dict['TF_dim'])
        
    def forward(self, Decoder_input_tensor, Encoder_KV):
        # [Generate Q, K, V]
        Q, K, V = self.Decoder_QKV(Decoder_input_tensor)
        
        # [First layer]
        __Decoder_Output, Masekd_Decoder_Attention_score = self.Decoder_MMHA(Q, K, V)
        __Decoder_Output = self.Decoder_LN1(__Decoder_Output + Decoder_input_tensor)

        # [Second layer]
        _Decoder_Output, Decoder_Attention_score = self.Decoder_MHA(__Decoder_Output, Encoder_KV, Encoder_KV)
        _Decoder_Output = self.Decoder_LN2(_Decoder_Output + __Decoder_Output)
        
        # # [Third layer] 기존 인철형이 짜논 코드
        # Decoder_Output = self.Decoder_FFN(_Decoder_Output) + _Decoder_Output
        # Decoder_Output = self.Decoder_LN3(Decoder_Output)
        
        # [Third layer] FFN에 더해줬던 것을 LN3내로 이동
        Decoder_Output = self.Decoder_FFN(_Decoder_Output) 
        Decoder_Output = self.Decoder_LN3(Decoder_Output + _Decoder_Output)
        
        return Decoder_Output, Masekd_Decoder_Attention_score, Decoder_Attention_score
    
class Transformer_decoder_only_layer(nn.Module):
    def __init__(self, Config_dict):
        super().__init__()
        self.Decoder_QKV  = Generate_QKV(Config_dict)
        # [Multi-head attention]
        self.Decoder_MHA  = Multi_head_attention(Config_dict, Mask=None)
        # [Feed forward]
        self.Decoder_FFN  = FC_feed_forward_networks(Config_dict)
        # [Layer Norm]
        self.Decoder_LN1  = nn.LayerNorm(Config_dict['TF_dim'])
        self.Decoder_LN2  = nn.LayerNorm(Config_dict['TF_dim'])
        self.Decoder_LN3  = nn.LayerNorm(Config_dict['TF_dim'])
        
    def forward(self, Decoder_input_tensor):
        # [Generate Q, K, V]
        Q, K, V = self.Decoder_QKV(Decoder_input_tensor)
        
        # [First layer]
        __Decoder_Output, Masekd_Decoder_Attention_score = self.Decoder_MHA(Q, K, V)
        __Decoder_Output = self.Decoder_LN1(__Decoder_Output + Decoder_input_tensor)

        # [Second layer] (self attention으로 수정함)
        _Decoder_Output, Decoder_Attention_score = self.Decoder_MHA(__Decoder_Output, __Decoder_Output, __Decoder_Output)
        _Decoder_Output = self.Decoder_LN2(_Decoder_Output + __Decoder_Output)
        
        # [Third layer]
        Decoder_Output = self.Decoder_FFN(_Decoder_Output) 
        Decoder_Output = self.Decoder_LN3(Decoder_Output + _Decoder_Output)
        
        return Decoder_Output, Masekd_Decoder_Attention_score, Decoder_Attention_score
    
class Transformer_decoder_layer_no_mask(nn.Module):
    def __init__(self, Config_dict):
        super().__init__()
        self.Decoder_QKV  = Generate_QKV(Config_dict)
        # [Multi-head attention]
        self.Decoder_MHA  = Multi_head_attention(Config_dict, Mask=None)
        # [Feed forward]
        self.Decoder_FFN  = FC_feed_forward_networks(Config_dict)
        # [Layer Norm]
        self.Decoder_LN1  = nn.LayerNorm(Config_dict['TF_dim'])
        self.Decoder_LN2  = nn.LayerNorm(Config_dict['TF_dim'])
        self.Decoder_LN3  = nn.LayerNorm(Config_dict['TF_dim'])
        
    def forward(self, Decoder_input_tensor, Encoder_KV):
        # [Generate Q, K, V]
        Q, K, V = self.Decoder_QKV(Decoder_input_tensor)
        
        # [First layer]
        __Decoder_Output, Masekd_Decoder_Attention_score = self.Decoder_MHA(Q, K, V)
        __Decoder_Output = self.Decoder_LN1(__Decoder_Output + Decoder_input_tensor)

        # [Second layer]
        _Decoder_Output, Decoder_Attention_score = self.Decoder_MHA(__Decoder_Output, Encoder_KV, Encoder_KV)
        _Decoder_Output = self.Decoder_LN2(_Decoder_Output + __Decoder_Output)
        
        # [Third layer] 
        Decoder_Output = self.Decoder_FFN(_Decoder_Output) + _Decoder_Output
        Decoder_Output = self.Decoder_LN3(Decoder_Output)

        return Decoder_Output, Masekd_Decoder_Attention_score, Decoder_Attention_score
    
class Transformer_enc_dec_layer(nn.Module):
    def __init__(self, Config_dict):
        super().__init__()
        self.QKV  = Generate_QKV(Config_dict)
        # [Multi-head attention]
        self.MHA  = Multi_head_attention(Config_dict, Mask=None)
        # [Feed forward]
        self.FFN  = FC_feed_forward_networks(Config_dict)
        # [Layer Norm]
        self.LN  = nn.LayerNorm(Config_dict['TF_dim'])
        
    def forward(self, Encoder_input_tensor):
        # [Generate Q, K, V]
        Q, K, V = self.QKV(Encoder_input_tensor)
        
        # [First layer]
        Output_1, Attention_score = self.MHA(Q, K, V)
        Output_2 = self.LN(Output_1 + Encoder_input_tensor)

        # [Second layer]
        Output_3 = self.FFN(Output_2)
        Output_4 = self.LN(Output_3 + Output_2)
        
        Output_5, Attention_score = self.MHA(Output_2, Output_4, Output_4)
        Output_6 = self.LN(Output_5 + Output_4)
        
        # [Third layer] 
        Output_7 = self.FFN(Output_6)
        Decoder_Output = self.LN(Output_7 + Output_6)

        return Decoder_Output
    
class Transformer_enc_dec(nn.Module):
    def __init__(self, Config_dict):
        super().__init__()
        # [Set Encoder list]
        self.Enc_Dec_list = nn.ModuleList([Transformer_enc_dec_layer(Config_dict) for _ in range(Config_dict['Num_enc_dec'])])
        self.apply_xavier()
        
    def apply_xavier(self):
        print('- Apply xavier in encoder -')
        for layer in self.Enc_Dec_list.named_parameters():
            if layer[1].dim() > 1:
                nn.init.xavier_uniform_(layer[1])
                
    def forward(self, Encoder_tensor):
        for Enc_Dec in self.Enc_Dec_list:
            Encoder_tensor = Enc_Dec(Encoder_tensor)
            
        return Encoder_tensor
    
class Transformer_decoder(nn.Module):
    def __init__(self, Config_dict):
        super().__init__()
        # [Set Decoder list]
        self.Decoder_list = nn.ModuleList([Transformer_decoder_layer(Config_dict) for _ in range(Config_dict['Num_dec'])])
        self.apply_xavier()
        
    def apply_xavier(self):
        print('- Apply xavier in decoder -')
        for layer in self.Decoder_list.named_parameters():
            if layer[1].dim() > 1:
                nn.init.xavier_uniform_(layer[1])
                
    def forward(self, Decoder_tensor, Encoder_KV):
        for Decoder in self.Decoder_list:
            Decoder_tensor, Masekd_Decoder_Attention_score, Decoder_Attention_score = Decoder(Decoder_tensor, Encoder_KV)
            
        return Decoder_tensor, Masekd_Decoder_Attention_score, Decoder_Attention_score
    
class Transformer_decoder_only(nn.Module):
    def __init__(self, Config_dict):
        super().__init__()
        # [Set Decoder list]
        self.Decoder_list = nn.ModuleList([Transformer_decoder_only_layer(Config_dict) for _ in range(Config_dict['Num_dec'])])
        self.apply_xavier()
        
    def apply_xavier(self):
        print('- Apply xavier in decoder -')
        for layer in self.Decoder_list.named_parameters():
            if layer[1].dim() > 1:
                nn.init.xavier_uniform_(layer[1])
                
    def forward(self, Decoder_tensor):
        for Decoder in self.Decoder_list:
            Decoder_tensor, Masekd_Decoder_Attention_score, Decoder_Attention_score = Decoder(Decoder_tensor)
            
        return Decoder_tensor, Masekd_Decoder_Attention_score, Decoder_Attention_score
    

class Transformer_decoder_no_mask(nn.Module):
    def __init__(self, Config_dict):
        super().__init__()
        # [Set Decoder list]
        self.Decoder_list = nn.ModuleList([Transformer_decoder_layer_no_mask(Config_dict) for _ in range(Config_dict['Num_dec'])])
        self.apply_xavier()
        
    def apply_xavier(self):
        print('- Apply xavier in decoder -')
        for layer in self.Decoder_list.named_parameters():
            if layer[1].dim() > 1:
                nn.init.xavier_uniform_(layer[1])
                
    def forward(self, Decoder_tensor, Encoder_KV):
        for Decoder in self.Decoder_list:
            Decoder_tensor, Masekd_Decoder_Attention_score, Decoder_Attention_score = Decoder(Decoder_tensor, Encoder_KV)
            
        return Decoder_tensor, Masekd_Decoder_Attention_score, Decoder_Attention_score
    
