import torch
import numpy as np
from Utils.Base import *


def Make_tensor(track_data, Config_dict):
    
    Track_data = track_data.loc[:, ['xCenter', 'yCenter', 'heading', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration']]
    Track_data['xCenter'] = standard_scale_for_vali(Track_data['xCenter'], Config_dict['mean'][0], Config_dict['std'][0])
    Track_data['yCenter'] = standard_scale_for_vali(Track_data['yCenter'], Config_dict['mean'][1], Config_dict['std'][1])
    Track_data['heading'] = standard_scale_for_vali(Track_data['heading'], Config_dict['mean'][2], Config_dict['std'][2])
    Track_data['xVelocity'] = standard_scale_for_vali(Track_data['xVelocity'], Config_dict['mean'][3], Config_dict['std'][3]) 
    Track_data['yVelocity'] = standard_scale_for_vali(Track_data['yVelocity'], Config_dict['mean'][4], Config_dict['std'][4]) 
    Track_data['xAcceleration'] = standard_scale_for_vali(Track_data['xAcceleration'], Config_dict['mean'][5], Config_dict['std'][5]) 
    Track_data['yAcceleration'] = standard_scale_for_vali(Track_data['yAcceleration'], Config_dict['mean'][6], Config_dict['std'][6]) 
        
    return torch.tensor(Track_data.to_numpy(), device= Config_dict['Device'], dtype=torch.float32)