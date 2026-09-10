import json
import torch
import numpy as np
import math

def Read_config(Config_dir):    
    with open(Config_dir) as f:
        Config = json.load(f)
    return Config

def standard_scale(Data):
    mean = np.mean(Data, axis=0)
    std = np.std(Data, axis=0)
    scaled = (Data - mean) / std
    
    data= {"mean": list(mean), "std": list(std)}
    save_path= f'./Config/Meanstd_config.json'

    with open(save_path, 'w') as json_file:
        json.dump(data, json_file)

    print('Train mean, std:', mean, ' ', std)
    return scaled, mean, std

def standard_scale_inference(Data):
    mean = np.mean(Data, axis=0)
    std = np.std(Data, axis=0)
    scaled = (Data - mean) / std
    # print('Train mean, std:', mean, ' ', std)
    return scaled

def standard_scale_for_vali(Data, mean, std):
    scaled = (Data - mean) / std
    # print('Vali mean, std:', mean, ' ', std)
    return scaled

def reverse_standard_scale(scaled, mean, std):
    Data = (scaled * std) + mean
    return Data

def Tensor_standard_scale_for_train(Tensor):
    Tensor_mean = torch.mean(Tensor, axis=0)
    Tensor_std  = torch.std(Tensor, axis=0)
    Scaled_tensor = (Tensor - Tensor_mean) / Tensor_std
    print('Train mean, std:', Tensor_mean, ' ', Tensor_std)
    return Scaled_tensor, Tensor_mean, Tensor_std

def Tensor_standard_scale_for_vali(Tensor, Tensor_mean, Tensor_std):
    Scaled_tensor = (Tensor - Tensor_mean) / Tensor_std
    print('Vali mean, std:', Tensor_mean, ' ', Tensor_std)
    return Scaled_tensor

def Tensor_reverse_standard_scale(Scaled_tensor, Tensor_mean, Tensor_std):
    return (Scaled_tensor * Tensor_std) + Tensor_mean

def softmax(x):
    e_x = [math.exp(i) for i in x]
    sum_e_x = sum(e_x)
    softmax_values = [i / sum_e_x for i in e_x]
    return softmax_values
