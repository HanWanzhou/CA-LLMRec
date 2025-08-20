import contextlib
import logging
import os
import glob
import pickle

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from utils import *
from pre_train.sasrec.model import SASRec

# 全局Args类定义，用于pickle兼容性
class Args:
    def __init__(self):
        # SASRec的默认参数 - 自动检测GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hidden_units = 50
        self.maxlen = 50
        self.num_blocks = 2
        self.num_heads = 1
        self.dropout_rate = 0.5
        self.l2_emb = 0.0


def load_checkpoint(recsys, pre_trained):
    path = f'pre_train/{recsys}/{pre_trained}/'
    
    pth_file_path = find_filepath(path, '.pth')
    assert len(pth_file_path) == 1, 'There are more than two models in this dir. You need to remove other model files.\n'
    
    # 处理SASRec模型加载时的Args类问题
    try:
        kwargs, checkpoint = torch.load(pth_file_path[0], map_location="cpu")
    except (AttributeError, pickle.UnpicklingError) as e:
        logging.info(f"处理SASRec模型加载问题: {str(e)}")
        logging.info("使用自定义unpickler重新加载...")
        
        # 创建自定义unpickler来处理Args类
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == 'Args':
                    return Args  # 返回我们的全局Args类
                return super().find_class(module, name)
        
        try:
            # 使用自定义unpickler加载
            with open(pth_file_path[0], 'rb') as f:
                kwargs, checkpoint = CustomUnpickler(f).load()
        except Exception as e2:
            logging.info("自定义unpickler失败，尝试强制加载state_dict...")
            try:
                # 最后的备选方案：尝试直接加载为state_dict
                checkpoint = torch.load(pth_file_path[0], map_location="cpu")
                if isinstance(checkpoint, tuple) and len(checkpoint) == 2:
                    kwargs, checkpoint = checkpoint
                else:
                    # 如果不是预期格式，创建默认参数
                    kwargs = {'args': Args()}
                    
            except Exception as e3:
                logging.error(f"所有加载方法都失败: {str(e3)}")
                raise e3
    
    logging.info("load checkpoint from %s" % pth_file_path[0])
    return kwargs, checkpoint

class RecSys(nn.Module):
    def __init__(self, recsys_model, pre_trained_data, device):
        super().__init__()
        kwargs, checkpoint = load_checkpoint(recsys_model, pre_trained_data)
        kwargs['args'].device = device
        model = SASRec(**kwargs)
        model.load_state_dict(checkpoint)
            
        for p in model.parameters():
            p.requires_grad = False
            
        self.item_num = model.item_num
        self.user_num = model.user_num
        self.model = model.to(device)
        self.hidden_units = kwargs['args'].hidden_units
        
    def forward(self):
        print('forward')
    
    def predict_counter(self, user, seq, pool, cf_weights=None):
        """
        反事实预测方法 - 参考UCR实现
        Args:
            user: 用户ID
            seq: 用户序列
            pool: 候选物品池
            cf_weights: 反事实权重 [batch_size, seq_len]
        Returns:
            预测分数
        """
        if cf_weights is not None:
            # 应用反事实权重到序列嵌入
            seq_emb = self.model.item_emb(seq)  # [batch_size, seq_len, emb_dim]
            cf_weights_expanded = cf_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
            weighted_seq_emb = seq_emb * cf_weights_expanded
            
            # 使用加权后的序列嵌入进行预测
            # 这里需要适配SASRec的predict方法
            with torch.no_grad():
                scores = self.model.predict_with_embeddings(user, weighted_seq_emb, pool)
        else:
            # 正常预测
            with torch.no_grad():
                scores = self.model.predict(user, seq, pool)
                
        return scores
    
    def get_item_embeddings(self, item_ids):
        """
        获取物品嵌入
        Args:
            item_ids: 物品ID张量
        Returns:
            物品嵌入张量
        """
        with torch.no_grad():
            return self.model.item_emb(item_ids)