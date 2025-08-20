import os
import torch
import numpy as np
from datetime import datetime
from pytz import timezone

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# ex. target_word: .csv / in target_path find 123.csv file
def find_filepath(target_path, target_word):
    file_paths = []
    for file in os.listdir(target_path):
        if os.path.isfile(os.path.join(target_path, file)):
            if target_word in file:
                file_paths.append(target_path + file)
            
    return file_paths


def generate_padding_mask(seq, padding_value=0):
    """
    生成padding mask
    Args:
        seq: 序列张量 [batch_size, seq_len]
        padding_value: padding值
    Returns:
        mask张量，True表示padding位置
    """
    return (seq == padding_value)


def sample_counterfactual_sequence(seq, mask_ratio=0.2, padding_value=0):
    """
    为序列生成反事实变体（随机mask部分物品）
    Args:
        seq: 原始序列 [batch_size, seq_len]
        mask_ratio: mask比例
        padding_value: padding值
    Returns:
        反事实序列
    """
    batch_size, seq_len = seq.shape
    cf_seq = seq.clone()
    
    for i in range(batch_size):
        # 找到非padding位置
        non_padding_positions = (seq[i] != padding_value).nonzero(as_tuple=True)[0]
        if len(non_padding_positions) > 0:
            # 随机选择要mask的位置
            num_to_mask = max(1, int(len(non_padding_positions) * mask_ratio))
            mask_positions = torch.randperm(len(non_padding_positions))[:num_to_mask]
            actual_positions = non_padding_positions[mask_positions]
            
            # 将选中位置设为padding
            cf_seq[i, actual_positions] = padding_value
    
    return cf_seq


def calculate_sequence_importance(cf_weights, threshold=0.5):
    """
    计算序列中每个位置的重要性
    Args:
        cf_weights: 反事实权重 [seq_len]
        threshold: 重要性阈值
    Returns:
        重要位置的索引和权重
    """
    important_mask = cf_weights < threshold
    important_positions = important_mask.nonzero(as_tuple=True)[0]
    important_weights = cf_weights[important_positions]
    
    return important_positions, important_weights


def print_counterfactual_info(epoch, cf_loss, cf_weights_sample):
    """
    打印反事实训练信息
    """
    important_positions, important_weights = calculate_sequence_importance(cf_weights_sample)
    print(f"Epoch {epoch} - CF Loss: {cf_loss:.4f}")
    print(f"Important positions: {important_positions.tolist()}")
    print(f"Weights range: [{cf_weights_sample.min():.3f}, {cf_weights_sample.max():.3f}]")
    
    
    