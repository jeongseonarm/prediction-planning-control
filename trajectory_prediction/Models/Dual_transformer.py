from Models.Base_network.Custom_transformer_xavier_v4 import *
import torch

class Dual_transformer(nn.Module):
    def __init__(self, Config):
        super().__init__()
        # [Params]
        self.Input_horizon      = Config['Input_horizon']
        self.Prediction_horizon = Config['Prediction_horizon']
        self.Maneuver_num = Config['Maneuver_num']
        
        self.TF_dim = Config['TF_dim']
        
        # [Projection]
        self.Encoder_only_projection = FC_projection(Config)
        self.Encoder_projection = FC_projection_encoder(Config)
        self.Decoder_projection = FC_projection(Config)
        
        # [Positional encoding]
        self.Positional_encoding = Positional_encoding(Config)
        
        # [TF encoder]
        self.Transformer_encoder = Transformer_encoder(Config)
        self.Transformer_encoder_only = Transformer_encoder_only(Config)

        # [TF decoder]
        self.Transformer_decoder = Transformer_decoder(Config)

        # [Pred maneuver]
        self.Prediction_maneuver = nn.Linear(self.TF_dim, self.Maneuver_num)
        
        # [Task head]
        self.Prediction_trajectory = nn.Linear(self.TF_dim, int(self.Prediction_horizon / self.Input_horizon) * 2)
        
    def forward(self, Encoder_only_input_tensor, Encoder_input_tensor, Decoder_input_tensor):
        Batch_size = Encoder_input_tensor.size(0)
        
        # [Projection]
        ## [Batch_size, Seq_len, TF_dim]
        Encoder_only_tensor = self.Encoder_only_projection(Encoder_only_input_tensor)

        # [Positional encoding]
        Encoder_only_tensor = self.Positional_encoding(Encoder_only_tensor)

        # [Encoder_only]
        Encoder_only_tensor = self.Transformer_encoder_only(Encoder_only_tensor)

        # [Predict intention]
        Encoder_only_tensor = self.Prediction_maneuver(Encoder_only_tensor)
        maneuver_tensor = torch.softmax(Encoder_only_tensor, dim=2)

        # [concat]
        Encoder_tensor = torch.cat((Encoder_input_tensor, maneuver_tensor), dim=2)

        # [Projection]
        ## [Batch_size, Seq_len, TF_dim]
        Encoder_tensor = self.Encoder_projection(Encoder_tensor)
        Decoder_tensor = self.Decoder_projection(Decoder_input_tensor)
        
        # [Positional encoding]
        Encoder_tensor = self.Positional_encoding(Encoder_tensor)
        Decoder_tensor = self.Positional_encoding(Decoder_tensor)
        
        # [Encoder - Decoder]
        Encoder_tensor = self.Transformer_encoder(Encoder_tensor)
        Decoder_tensor, Masekd_Decoder_Attention_score, Decoder_attention_score = self.Transformer_decoder(Decoder_tensor, Encoder_tensor)
        
        # [Prediction]
        Prediction = self.Prediction_trajectory(Decoder_tensor)
        Prediction = Prediction.view(Batch_size, -1, 2)
        
        return Prediction, Encoder_only_tensor
