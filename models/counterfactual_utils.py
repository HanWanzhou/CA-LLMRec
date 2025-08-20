"""
CA-LLMRec 反事实工具模块 - Week 2
基于UCR项目的EXPGenerator设计，针对PyTorch 2.1.2 + CUDA 11.8 + RTX 4090优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import random


class AdvancedCounterfactualGenerator(nn.Module):
    """
    高级反事实生成器 - Week 2增强版
    基于UCR的EXPGenerator，增加了动态权重调整和稳定性控制
    """
    def __init__(self, seq_len: int, device: str, args=None):
        super(AdvancedCounterfactualGenerator, self).__init__()
        self.seq_len = seq_len
        self.device = device
        self.args = args
        
        # 可学习的权重向量 - 与UCR EXPGenerator完全一致
        self.delta = nn.Parameter(torch.FloatTensor(seq_len).uniform_(0, 1))
        
        # Week 2新增：动态权重调整参数
        self.temperature = nn.Parameter(torch.tensor(1.0))  # 温度参数，控制权重分布的尖锐程度
        self.sparsity_weight = getattr(args, 'cf_sparsity_weight', 1.0) if args else 1.0
        
        # 梯度裁剪参数 - 针对RTX 4090优化
        self.max_grad_norm = getattr(args, 'cf_max_grad_norm', 1.0) if args else 1.0
        
        # 稳定性控制
        self.eps = 1e-8
        
    def clamp_delta(self, padding_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        限制delta值并处理padding - 与UCR EXPGenerator.clamp_delta()完全一致
        Args:
            padding_ids: padding位置的索引张量 [num_padding_positions]
        Returns:
            clamped_delta: [seq_len] 限制后的权重
        """
        clamped_delta = torch.clamp(self.delta, 0, 1)
        if padding_ids is not None and len(padding_ids) > 0:
            # padding位置权重设为1（不修改），与UCR完全一致
            clamped_delta[padding_ids] = 1.0
        return clamped_delta
    
    def generate_counterfactual_weights(self, batch_size: int, 
                                      padding_ids: Optional[torch.Tensor] = None,
                                      use_temperature: bool = False,
                                      target_seq_len: Optional[int] = None) -> torch.Tensor:
        """
        生成反事实权重 - Week 2增强版
        Args:
            batch_size: 批次大小
            padding_ids: padding位置索引
            use_temperature: 是否使用温度调节
            target_seq_len: 目标序列长度，如果与self.seq_len不同则调整
        Returns:
            cf_weights: [batch_size, target_seq_len] 的权重矩阵
        """
        clamped_delta = self.clamp_delta(padding_ids)
        
        # 如果目标序列长度与初始化时不同，需要调整
        if target_seq_len is not None and target_seq_len != self.seq_len:
            if target_seq_len > self.seq_len:
                # 扩展：用1填充（不影响原有权重）
                padding_size = target_seq_len - self.seq_len
                padding = torch.ones(padding_size, device=self.device)
                clamped_delta = torch.cat([clamped_delta, padding], dim=0)
            else:
                # 截断：保留前target_seq_len个权重
                clamped_delta = clamped_delta[:target_seq_len]
        
        if use_temperature:
            # 使用温度参数调节权重分布
            clamped_delta = torch.sigmoid(clamped_delta / torch.clamp(self.temperature, min=self.eps))
        
        # 扩展到batch维度
        cf_weights = clamped_delta.unsqueeze(0).expand(batch_size, -1)
        return cf_weights
    
    def compute_advanced_loss(self, target_scores: torch.Tensor, 
                            rest_scores: torch.Tensor, 
                            args) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算高级反事实损失 - 基于UCR但增加稳定性控制
        Args:
            target_scores: 目标物品分数
            rest_scores: 其他物品分数
            args: 超参数配置
        Returns:
            l1_loss, pairwise_target_loss, pairwise_rest_loss, total_loss
        """
        # L1正则化项 - 鼓励稀疏解释
        l1_loss = torch.linalg.norm((1 - self.delta), ord=1) * self.sparsity_weight
        
        # 目标物品排名损失 - 增加稳定性控制
        pairwise_target_loss = torch.tensor(0.0, device=self.device)
        if hasattr(args, 'lam') and hasattr(args, 'alp_1') and hasattr(args, 'K'):
            if len(rest_scores) > 0:
                sorted_rest, _ = torch.sort(rest_scores, descending=True)
                k_th_score = sorted_rest[min(args.K-1, len(sorted_rest)-1)]
                margin = args.alp_1 + target_scores - k_th_score
                pairwise_target_loss = args.lam * F.relu(margin)
        
        # 其他物品损失 - 增加数值稳定性
        pairwise_rest_loss = torch.tensor(0.0, device=self.device)
        if hasattr(args, 'gam') and hasattr(args, 'lam') and len(rest_scores) > 0:
            pairwise_rest_loss = -args.gam * args.lam * torch.mean(rest_scores)
        
        # 总损失
        total_loss = l1_loss + pairwise_target_loss + pairwise_rest_loss
        
        return l1_loss, pairwise_target_loss, pairwise_rest_loss, total_loss
    
    def apply_gradient_clipping(self):
        """应用梯度裁剪 - 针对RTX 4090优化"""
        if self.delta.grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
    
    def get_explanation_weights(self, threshold: float = 0.5, 
                              top_k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取解释权重 - Week 2增强版
        Args:
            threshold: 重要性阈值
            top_k: 返回前k个重要位置
        Returns:
            important_positions: 重要位置索引
            important_weights: 对应权重值
        """
        with torch.no_grad():
            weights = self.clamp_delta()
            
            if top_k is not None:
                # 返回权重最小的top_k个位置（最重要的位置）
                _, indices = torch.topk(weights, k=min(top_k, len(weights)), largest=False)
                important_positions = indices
                important_weights = weights[indices]
            else:
                # 使用阈值过滤
                important_mask = weights < threshold
                important_positions = torch.nonzero(important_mask, as_tuple=False).squeeze(-1)
                if len(important_positions.shape) == 0:
                    important_positions = important_positions.unsqueeze(0)
                important_weights = weights[important_positions]
            
            return important_positions, important_weights


class CounterfactualDataSampler:
    """
    反事实数据采样器 - Week 2新增
    智能采样策略，提高反事实数据质量
    """
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.mask_ratio = getattr(args, 'cf_mask_ratio', 0.2)
        self.min_mask_items = getattr(args, 'cf_min_mask_items', 1)
        self.max_mask_items = getattr(args, 'cf_max_mask_items', 10)
        
    def generate_padding_ids(self, sequences: torch.Tensor, padding_value: int = 0) -> List[torch.Tensor]:
        """
        生成每个序列的padding位置 - 基于UCR设计
        Args:
            sequences: [batch_size, seq_len] 序列张量
            padding_value: padding值
        Returns:
            padding_ids_list: 每个序列的padding位置列表
        """
        padding_ids_list = []
        for seq in sequences:
            padding_positions = (seq == padding_value).nonzero(as_tuple=False).squeeze(-1)
            padding_ids_list.append(padding_positions)
        return padding_ids_list
    
    def smart_mask_sampling(self, user_seq: torch.Tensor, 
                           importance_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        智能mask采样 - 基于重要性分数
        Args:
            user_seq: 用户序列 [batch_size, seq_len]
            importance_scores: 重要性分数 [batch_size, seq_len]
        Returns:
            masked_seq: mask后的序列
        """
        batch_size, seq_len = user_seq.shape
        masked_seq = user_seq.clone()
        
        for i in range(batch_size):
            seq = user_seq[i]
            non_padding_mask = (seq != 0)
            non_padding_indices = non_padding_mask.nonzero(as_tuple=False).squeeze(-1)
            
            if len(non_padding_indices) == 0:
                continue
                
            # 计算要mask的物品数量
            num_items = len(non_padding_indices)
            num_mask = max(self.min_mask_items, 
                          min(self.max_mask_items, int(num_items * self.mask_ratio)))
            
            # 基于重要性采样或随机采样
            if importance_scores is not None:
                # 重要性采样：优先mask重要性低的物品
                item_importance = importance_scores[i][non_padding_indices]
                _, mask_indices = torch.topk(item_importance, k=num_mask, largest=False)
                mask_positions = non_padding_indices[mask_indices]
            else:
                # 随机采样
                mask_indices = torch.randperm(num_items)[:num_mask]
                mask_positions = non_padding_indices[mask_indices]
            
            # 应用mask
            masked_seq[i, mask_positions] = 0
            
        return masked_seq
    
    def generate_diverse_counterfactuals(self, user_seq: torch.Tensor, 
                                       num_variants: int = 3) -> List[torch.Tensor]:
        """
        生成多样化的反事实序列
        Args:
            user_seq: 原始用户序列 [batch_size, seq_len]
            num_variants: 生成变体数量
        Returns:
            cf_sequences: 反事实序列列表
        """
        cf_sequences = []
        
        for _ in range(num_variants):
            # 使用不同的mask比例
            mask_ratio = self.mask_ratio * (0.5 + torch.rand(1).item())
            original_ratio = self.mask_ratio
            self.mask_ratio = mask_ratio
            
            cf_seq = self.smart_mask_sampling(user_seq)
            cf_sequences.append(cf_seq)
            
            # 恢复原始比例
            self.mask_ratio = original_ratio
            
        return cf_sequences


class CounterfactualTrainingOptimizer:
    """
    反事实训练优化器 - Week 2核心组件
    针对RTX 4090和PyTorch 2.1.2优化的训练策略
    """
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = args.device
        
        # 优化器配置 - 针对RTX 4090优化
        self.cf_lr = getattr(args, 'cf_learning_rate', 0.001)
        self.weight_decay = getattr(args, 'cf_weight_decay', 1e-5)
        self.warmup_steps = getattr(args, 'cf_warmup_steps', 100)
        
        # 损失权重动态调整
        self.alignment_weight_scheduler = self._create_weight_scheduler()
        self.current_step = 0
        
        # 性能监控
        self.loss_history = []
        self.convergence_threshold = getattr(args, 'cf_convergence_threshold', 1e-6)
        
    def _create_weight_scheduler(self):
        """创建权重调度器"""
        def scheduler(step):
            # 线性增长的权重调度
            if step < self.warmup_steps:
                return self.model.cf_alignment_weight * (step / self.warmup_steps)
            else:
                return self.model.cf_alignment_weight
        return scheduler
    
    def optimize_counterfactual_step(self, cf_generator, target_scores, rest_scores):
        """
        单步反事实优化 - 基于UCR的优化流程
        Args:
            cf_generator: 反事实生成器
            target_scores: 目标分数
            rest_scores: 其他分数
        Returns:
            loss_dict: 损失字典
        """
        # 计算损失
        l1_loss, pairwise_target, pairwise_rest, total_loss = cf_generator.compute_advanced_loss(
            target_scores, rest_scores, self.args
        )
        
        # 反向传播
        total_loss.backward()
        
        # 梯度裁剪
        cf_generator.apply_gradient_clipping()
        
        # 更新当前步骤和权重
        self.current_step += 1
        current_weight = self.alignment_weight_scheduler(self.current_step)
        
        # 记录损失历史
        loss_dict = {
            'l1_loss': l1_loss.item(),
            'pairwise_target': pairwise_target.item(),
            'pairwise_rest': pairwise_rest.item(),
            'total_loss': total_loss.item(),
            'current_weight': current_weight
        }
        self.loss_history.append(loss_dict)
        
        return loss_dict
    
    def check_convergence(self, window_size: int = 10) -> bool:
        """检查训练收敛性"""
        if len(self.loss_history) < window_size:
            return False
            
        recent_losses = [loss['total_loss'] for loss in self.loss_history[-window_size:]]
        loss_std = np.std(recent_losses)
        return loss_std < self.convergence_threshold
    
    def get_training_stats(self) -> dict:
        """获取训练统计信息"""
        if not self.loss_history:
            return {}
            
        recent_losses = self.loss_history[-10:] if len(self.loss_history) >= 10 else self.loss_history
        
        stats = {
            'total_steps': self.current_step,
            'avg_loss': np.mean([loss['total_loss'] for loss in recent_losses]),
            'avg_l1': np.mean([loss['l1_loss'] for loss in recent_losses]),
            'avg_target': np.mean([loss['pairwise_target'] for loss in recent_losses]),
            'avg_rest': np.mean([loss['pairwise_rest'] for loss in recent_losses]),
            'converged': self.check_convergence()
        }
        
        return stats


def print_counterfactual_training_info(epoch: int, step: int, loss_dict: dict, stats: dict):
    """
    打印反事实训练信息 - Week 2增强版
    """
    print(f"CF Training - Epoch {epoch}, Step {step}")
    print(f"  Total Loss: {loss_dict['total_loss']:.6f}")
    print(f"  L1 Loss: {loss_dict['l1_loss']:.6f}")
    print(f"  Target Loss: {loss_dict['pairwise_target']:.6f}")
    print(f"  Rest Loss: {loss_dict['pairwise_rest']:.6f}")
    print(f"  Current Weight: {loss_dict['current_weight']:.4f}")
    
    if stats:
        print(f"  Avg Loss (10 steps): {stats['avg_loss']:.6f}")
        print(f"  Converged: {stats['converged']}")
        print(f"  Total Steps: {stats['total_steps']}")


def setup_rtx4090_optimization():
    """
    RTX 4090优化设置 - 针对PyTorch 2.1.2
    """
    # 启用Tensor Core优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # 内存优化
    torch.cuda.empty_cache()
    
    # 设置CUDA设备属性
    if torch.cuda.is_available():
        device_properties = torch.cuda.get_device_properties(0)
        print(f"GPU优化设置完成:")
        print(f"  设备: {device_properties.name}")
        print(f"  显存: {device_properties.total_memory / 1e9:.1f} GB")
        print(f"  计算能力: {device_properties.major}.{device_properties.minor}")
        print(f"  TensorCore: 已启用")
        print(f"  TF32: 已启用")
    
    return True 