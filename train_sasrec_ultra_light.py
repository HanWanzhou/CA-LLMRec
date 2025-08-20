#!/usr/bin/env python3
"""
快速训练SASRec模型 - 专为超轻量化数据集设计
"""

import os
import sys
import time
import torch
import argparse
from tqdm import tqdm

# 添加路径
sys.path.append('./pre_train/sasrec')
from pre_train.sasrec.model import SASRec
from pre_train.sasrec.data_preprocess import *
from pre_train.sasrec.utils import *

# 将Args类移到函数外部以支持pickle序列化
class Args:
    def __init__(self, dataset_name, epochs=20, device='cuda'):
        self.dataset = dataset_name
        self.batch_size = 128  # 与A-LLMRec保持一致，GPU可以处理更大batch
        self.lr = 0.001
        self.maxlen = 50
        self.hidden_units = 50
        self.num_blocks = 2
        self.num_epochs = epochs
        self.num_heads = 1
        self.dropout_rate = 0.5
        self.l2_emb = 0.0
        self.device = device
        self.inference_only = False
        self.state_dict_path = None

def train_sasrec_ultra_light(dataset_name, device='cuda', epochs=20):
    """为超轻量化数据集快速训练SASRec模型"""
    
    print(f"🚀 开始为数据集 {dataset_name} 训练SASRec模型...")
    
    args = Args(dataset_name, epochs, device)
    
    # 检查数据是否存在
    data_path = f'./data/amazon/{dataset_name}.txt'
    
    # 如果直接路径不存在，尝试查找带有后缀的文件
    if not os.path.exists(data_path):
        import glob
        pattern = f'./data/amazon/{dataset_name}*.txt'
        matching_files = glob.glob(pattern)
        
        if matching_files:
            data_path = matching_files[0]
            # 更新dataset名称为实际文件名（去掉.txt后缀）
            dataset_name = os.path.basename(data_path).replace('.txt', '')
            print(f"📁 找到数据文件: {data_path}")
            print(f"🔄 更新数据集名称为: {dataset_name}")
        else:
            print(f"❌ 数据文件不存在: {data_path}")
            print(f"❌ 也未找到匹配的文件: {pattern}")
            return False
    
    try:
        # 数据预处理 - 直接从txt文件加载数据
        print("📊 从txt文件加载数据...")
        dataset = data_partition(args.dataset, path=data_path)
        
        [user_train, user_valid, user_test, usernum, itemnum] = dataset
        print(f'用户数: {usernum}, 物品数: {itemnum}')
        
        # 计算平均序列长度
        cc = 0.0
        for u in user_train:
            cc += len(user_train[u])
        print(f'平均序列长度: {cc / len(user_train):.2f}')
        
        # 数据加载器
        print("🔄 创建数据加载器...")
        sampler = WarpSampler(user_train, usernum, itemnum, 
                             batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
        
        # 模型初始化
        print("🧠 初始化SASRec模型...")
        model = SASRec(usernum, itemnum, args).to(args.device)
        
        # 参数初始化
        for name, param in model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass  # 跳过一维参数
        
        model.train()
        
        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
        
        # 训练循环
        print("🎯 开始训练...")
        num_batch = len(user_train) // args.batch_size
        
        for epoch in range(1, args.num_epochs + 1):
            t0 = time.time()
            running_loss = 0.0
            
            progress_bar = tqdm(range(num_batch), desc=f'Epoch {epoch}/{args.num_epochs}')
            
            for step in progress_bar:
                u, seq, pos, neg = sampler.next_batch()
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                
                pos_logits, neg_logits = model(u, seq, pos, neg)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
                
                optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = torch.nn.BCEWithLogitsLoss()(pos_logits[indices], pos_labels[indices])
                loss += torch.nn.BCEWithLogitsLoss()(neg_logits[indices], neg_labels[indices])
                
                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{running_loss/(step+1):.4f}'})
            
            # 每10个epoch评估一次
            if epoch % 10 == 0 or epoch == args.num_epochs:
                model.eval()
                print(f'\n📊 Epoch {epoch} 评估...')
                t_test = evaluate(model, dataset, args)
                t_valid = evaluate_valid(model, dataset, args)
                
                print(f'Epoch: {epoch}, Valid (NDCG@10: {t_valid[0]:.4f}, HR@10: {t_valid[1]:.4f}), '
                      f'Test (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})')
                model.train()
        
        # 保存模型 - 兼容A-LLMRec格式
        print("💾 保存模型...")
        
        # 同时保存到两个位置以确保兼容性
        # 1. 保存到pre_train目录 (A-LLMRec格式)
        output_dir_a = f'./pre_train/sasrec/{args.dataset}'
        os.makedirs(output_dir_a, exist_ok=True)
        
        # 2. 保存到output目录 (我们的格式)
        output_dir_b = f'./pre_train/sasrec/output/{args.dataset}'
        os.makedirs(output_dir_b, exist_ok=True)
        
        fname = f'SASRec.epoch={args.num_epochs}.lr={args.lr}.layer={args.num_blocks}.head={args.num_heads}.hidden={args.hidden_units}.maxlen={args.maxlen}.pth'
        
        # 保存到两个位置
        model_path_a = os.path.join(output_dir_a, fname)
        model_path_b = os.path.join(output_dir_b, fname)
        
        torch.save([model.kwargs, model.state_dict()], model_path_a)
        torch.save([model.kwargs, model.state_dict()], model_path_b)
        
        print(f"✅ 模型已保存到:")
        print(f"   📁 A-LLMRec兼容位置: {model_path_a}")
        print(f"   📁 备份位置: {model_path_b}")
        print(f"🔗 此模型可直接用于A-LLMRec训练")
        
        sampler.close()
        return True
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='快速训练SASRec模型')
    parser.add_argument('--dataset', required=True, help='数据集名称')
    parser.add_argument('--device', default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--epochs', default=20, type=int, help='训练轮数')
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA不可用，切换到CPU")
        args.device = 'cpu'
    
    success = train_sasrec_ultra_light(args.dataset, args.device, args.epochs)
    
    if success:
        print("🎉 SASRec模型训练完成!")
    else:
        print("💥 SASRec模型训练失败!")
        sys.exit(1)

if __name__ == '__main__':
    main() 