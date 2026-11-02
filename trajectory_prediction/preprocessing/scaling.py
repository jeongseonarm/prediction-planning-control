"""Inference-time configuration loading and feature scaling helpers."""

import json

import numpy as np


def Read_config(config_path):
    """Load one JSON configuration file."""
    with open(config_path, encoding="utf-8") as config_file:
        return json.load(config_file)


def standard_scale_for_vali(data, mean, std):
    """Apply the training-set normalization parameters to inference inputs."""
    return (data - mean) / std


def reverse_standard_scale(scaled, mean, std):
    """Convert normalized model outputs back to their original units."""
    return scaled * std + mean
