"""
CA-LLMRec 反事实评估指标模块 - Week 2
基于UCR项目的评估方法，针对反事实推荐系统扩展
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
import json


class CounterfactualMetrics:
    """
    反事实评估指标计算器 - Week 2
    基于UCR的评估框架，扩展支持反事实推荐评估
    """
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.reset()
        
    def reset(self):
        """重置所有指标"""
        # 基础推荐指标
        self.ndcg_scores = []
        self.hit_scores = []
        self.precision_scores = []
        self.recall_scores = []
        
        # 反事实特有指标
        self.cf_ndcg_scores = []
        self.cf_hit_scores = []
        self.alignment_losses = []
        self.sparsity_scores = []
        
        # 对比指标
        self.improvement_scores = []
        self.fidelity_scores = []
        self.diversity_scores = []
        
        # 解释质量指标
        self.explanation_lengths = []
        self.explanation_coverage = []
        
    def compute_ndcg_at_k(self, predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
        """
        计算NDCG@K - 基于UCR的评估方法
        Args:
            predictions: 预测分数 [batch_size, num_items]
            targets: 真实标签 [batch_size, num_items]
            k: top-k
        Returns:
            NDCG@K分数
        """
        batch_size = predictions.shape[0]
        ndcg_sum = 0.0
        valid_users = 0
        
        for i in range(batch_size):
            pred = predictions[i]
            target = targets[i]
            
            # 找到正样本位置
            positive_items = (target > 0).nonzero(as_tuple=False).squeeze(-1)
            if len(positive_items) == 0:
                continue
                
            # 获取top-k预测
            _, top_k_indices = torch.topk(pred, k=min(k, len(pred)))
            
            # 计算DCG
            dcg = 0.0
            for j, item_idx in enumerate(top_k_indices):
                if item_idx in positive_items:
                    dcg += 1.0 / np.log2(j + 2)
            
            # 计算IDCG
            num_relevant = min(len(positive_items), k)
            idcg = sum(1.0 / np.log2(j + 2) for j in range(num_relevant))
            
            # 计算NDCG
            if idcg > 0:
                ndcg_sum += dcg / idcg
                valid_users += 1
        
        return ndcg_sum / valid_users if valid_users > 0 else 0.0
    
    def compute_hit_rate_at_k(self, predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
        """
        计算Hit Rate@K - 基于UCR的评估方法
        Args:
            predictions: 预测分数 [batch_size, num_items]
            targets: 真实标签 [batch_size, num_items]
            k: top-k
        Returns:
            Hit Rate@K分数
        """
        batch_size = predictions.shape[0]
        hit_sum = 0.0
        valid_users = 0
        
        for i in range(batch_size):
            pred = predictions[i]
            target = targets[i]
            
            # 找到正样本位置
            positive_items = (target > 0).nonzero(as_tuple=False).squeeze(-1)
            if len(positive_items) == 0:
                continue
                
            # 获取top-k预测
            _, top_k_indices = torch.topk(pred, k=min(k, len(pred)))
            
            # 检查是否命中
            hit = any(item_idx in positive_items for item_idx in top_k_indices)
            hit_sum += 1.0 if hit else 0.0
            valid_users += 1
        
        return hit_sum / valid_users if valid_users > 0 else 0.0
    
    def compute_counterfactual_fidelity(self, original_preds: torch.Tensor, 
                                      cf_preds: torch.Tensor, 
                                      explanations: List[List[int]], 
                                      k: int = 10) -> float:
        """
        计算反事实忠实度 - 基于UCR的忠实度评估
        Args:
            original_preds: 原始预测 [batch_size, num_items]
            cf_preds: 反事实预测 [batch_size, num_items]
            explanations: 解释列表，每个用户的重要物品位置
            k: top-k
        Returns:
            忠实度分数
        """
        batch_size = original_preds.shape[0]
        fidelity_sum = 0.0
        valid_cases = 0
        
        for i in range(batch_size):
            if i >= len(explanations):
                continue
                
            orig_pred = original_preds[i]
            cf_pred = cf_preds[i]
            explanation = explanations[i]
            
            if len(explanation) == 0:
                continue
            
            # 获取原始top-k
            _, orig_top_k = torch.topk(orig_pred, k=min(k, len(orig_pred)))
            # 获取反事实top-k
            _, cf_top_k = torch.topk(cf_pred, k=min(k, len(cf_pred)))
            
            # 计算IoU（交并比）
            orig_set = set(orig_top_k.cpu().numpy())
            cf_set = set(cf_top_k.cpu().numpy())
            
            intersection = len(orig_set.intersection(cf_set))
            union = len(orig_set.union(cf_set))
            
            if union > 0:
                iou = intersection / union
                # 忠实度 = 1 - IoU（变化越大，忠实度越高）
                fidelity = 1.0 - iou
                fidelity_sum += fidelity
                valid_cases += 1
        
        return fidelity_sum / valid_cases if valid_cases > 0 else 0.0
    
    def compute_explanation_sparsity(self, explanations: List[List[int]], 
                                   sequence_lengths: List[int]) -> float:
        """
        计算解释稀疏性
        Args:
            explanations: 解释列表
            sequence_lengths: 序列长度列表
        Returns:
            稀疏性分数（越小越稀疏）
        """
        if not explanations or not sequence_lengths:
            return 0.0
            
        sparsity_scores = []
        for explanation, seq_len in zip(explanations, sequence_lengths):
            if seq_len > 0:
                sparsity = len(explanation) / seq_len
                sparsity_scores.append(sparsity)
        
        return np.mean(sparsity_scores) if sparsity_scores else 0.0
    
    def compute_explanation_coverage(self, explanations: List[List[int]], 
                                   total_items: int) -> float:
        """
        计算解释覆盖率
        Args:
            explanations: 解释列表
            total_items: 总物品数
        Returns:
            覆盖率分数
        """
        if not explanations:
            return 0.0
            
        all_explained_items = set()
        for explanation in explanations:
            all_explained_items.update(explanation)
        
        coverage = len(all_explained_items) / total_items
        return coverage
    
    def compute_diversity_score(self, predictions: torch.Tensor, k: int = 10) -> float:
        """
        计算推荐多样性分数
        Args:
            predictions: 预测分数 [batch_size, num_items]
            k: top-k
        Returns:
            多样性分数
        """
        batch_size = predictions.shape[0]
        diversity_sum = 0.0
        
        for i in range(batch_size):
            pred = predictions[i]
            _, top_k_indices = torch.topk(pred, k=min(k, len(pred)))
            
            # 计算top-k物品的方差作为多样性指标
            top_k_scores = pred[top_k_indices]
            diversity = torch.std(top_k_scores).item()
            diversity_sum += diversity
        
        return diversity_sum / batch_size
    
    def update_metrics(self, original_preds: torch.Tensor, cf_preds: torch.Tensor,
                      targets: torch.Tensor, explanations: List[List[int]],
                      alignment_loss: float = 0.0, k: int = 10):
        """
        更新所有指标
        Args:
            original_preds: 原始预测
            cf_preds: 反事实预测
            targets: 真实标签
            explanations: 解释列表
            alignment_loss: 对齐损失
            k: top-k
        """
        # 基础推荐指标
        ndcg = self.compute_ndcg_at_k(original_preds, targets, k)
        hit_rate = self.compute_hit_rate_at_k(original_preds, targets, k)
        
        self.ndcg_scores.append(ndcg)
        self.hit_scores.append(hit_rate)
        
        # 反事实推荐指标
        cf_ndcg = self.compute_ndcg_at_k(cf_preds, targets, k)
        cf_hit_rate = self.compute_hit_rate_at_k(cf_preds, targets, k)
        
        self.cf_ndcg_scores.append(cf_ndcg)
        self.cf_hit_scores.append(cf_hit_rate)
        
        # 改进指标
        improvement = cf_ndcg - ndcg
        self.improvement_scores.append(improvement)
        
        # 忠实度指标
        fidelity = self.compute_counterfactual_fidelity(original_preds, cf_preds, explanations, k)
        self.fidelity_scores.append(fidelity)
        
        # 多样性指标
        diversity = self.compute_diversity_score(cf_preds, k)
        self.diversity_scores.append(diversity)
        
        # 对齐损失
        self.alignment_losses.append(alignment_loss)
        
        # 解释质量指标
        if explanations:
            avg_length = np.mean([len(exp) for exp in explanations])
            self.explanation_lengths.append(avg_length)
    
    def get_summary_metrics(self) -> Dict[str, float]:
        """
        获取汇总指标
        Returns:
            指标字典
        """
        summary = {}
        
        # 基础指标
        if self.ndcg_scores:
            summary['NDCG@10'] = np.mean(self.ndcg_scores)
            summary['Hit@10'] = np.mean(self.hit_scores)
        
        # 反事实指标
        if self.cf_ndcg_scores:
            summary['CF_NDCG@10'] = np.mean(self.cf_ndcg_scores)
            summary['CF_Hit@10'] = np.mean(self.cf_hit_scores)
        
        # 改进指标
        if self.improvement_scores:
            summary['NDCG_Improvement'] = np.mean(self.improvement_scores)
            summary['Improvement_Std'] = np.std(self.improvement_scores)
        
        # 忠实度和多样性
        if self.fidelity_scores:
            summary['Fidelity'] = np.mean(self.fidelity_scores)
        if self.diversity_scores:
            summary['Diversity'] = np.mean(self.diversity_scores)
        
        # 对齐损失
        if self.alignment_losses:
            summary['Alignment_Loss'] = np.mean(self.alignment_losses)
        
        # 解释质量
        if self.explanation_lengths:
            summary['Avg_Explanation_Length'] = np.mean(self.explanation_lengths)
        
        return summary
    
    def print_metrics_report(self):
        """打印详细的指标报告"""
        summary = self.get_summary_metrics()
        
        print("\n" + "="*60)
        print("CA-LLMRec 反事实评估报告 - Week 2")
        print("="*60)
        
        # 基础推荐性能
        print("\n📊 基础推荐性能:")
        if 'NDCG@10' in summary:
            print(f"  NDCG@10: {summary['NDCG@10']:.4f}")
        if 'Hit@10' in summary:
            print(f"  Hit@10: {summary['Hit@10']:.4f}")
        
        # 反事实推荐性能
        print("\n🔄 反事实推荐性能:")
        if 'CF_NDCG@10' in summary:
            print(f"  CF_NDCG@10: {summary['CF_NDCG@10']:.4f}")
        if 'CF_Hit@10' in summary:
            print(f"  CF_Hit@10: {summary['CF_Hit@10']:.4f}")
        
        # 改进效果
        print("\n📈 改进效果:")
        if 'NDCG_Improvement' in summary:
            print(f"  NDCG改进: {summary['NDCG_Improvement']:.4f}")
            print(f"  改进标准差: {summary.get('Improvement_Std', 0):.4f}")
        
        # 解释质量
        print("\n🔍 解释质量:")
        if 'Fidelity' in summary:
            print(f"  忠实度: {summary['Fidelity']:.4f}")
        if 'Diversity' in summary:
            print(f"  多样性: {summary['Diversity']:.4f}")
        if 'Avg_Explanation_Length' in summary:
            print(f"  平均解释长度: {summary['Avg_Explanation_Length']:.2f}")
        
        # 训练质量
        print("\n🎯 训练质量:")
        if 'Alignment_Loss' in summary:
            print(f"  对齐损失: {summary['Alignment_Loss']:.6f}")
        
        print("="*60)
    
    def save_metrics_to_file(self, filepath: str):
        """保存指标到文件"""
        summary = self.get_summary_metrics()
        
        # 添加详细数据
        detailed_metrics = {
            'summary': summary,
            'detailed_scores': {
                'ndcg_scores': self.ndcg_scores,
                'hit_scores': self.hit_scores,
                'cf_ndcg_scores': self.cf_ndcg_scores,
                'cf_hit_scores': self.cf_hit_scores,
                'improvement_scores': self.improvement_scores,
                'fidelity_scores': self.fidelity_scores,
                'diversity_scores': self.diversity_scores,
                'alignment_losses': self.alignment_losses,
                'explanation_lengths': self.explanation_lengths
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(detailed_metrics, f, indent=2, ensure_ascii=False)
        
        print(f"指标已保存到: {filepath}")
    
    def plot_metrics_comparison(self, save_path: str = None):
        """绘制指标对比图"""
        if not self.ndcg_scores or not self.cf_ndcg_scores:
            print("没有足够的数据用于绘图")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # NDCG对比
        axes[0, 0].plot(self.ndcg_scores, label='Original NDCG', alpha=0.7)
        axes[0, 0].plot(self.cf_ndcg_scores, label='Counterfactual NDCG', alpha=0.7)
        axes[0, 0].set_title('NDCG@10 Comparison')
        axes[0, 0].set_xlabel('Batch')
        axes[0, 0].set_ylabel('NDCG@10')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 改进分数分布
        if self.improvement_scores:
            axes[0, 1].hist(self.improvement_scores, bins=20, alpha=0.7, color='green')
            axes[0, 1].set_title('NDCG Improvement Distribution')
            axes[0, 1].set_xlabel('NDCG Improvement')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 忠实度趋势
        if self.fidelity_scores:
            axes[1, 0].plot(self.fidelity_scores, color='orange', alpha=0.7)
            axes[1, 0].set_title('Counterfactual Fidelity Trend')
            axes[1, 0].set_xlabel('Batch')
            axes[1, 0].set_ylabel('Fidelity Score')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 对齐损失趋势
        if self.alignment_losses:
            axes[1, 1].plot(self.alignment_losses, color='red', alpha=0.7)
            axes[1, 1].set_title('Alignment Loss Trend')
            axes[1, 1].set_xlabel('Batch')
            axes[1, 1].set_ylabel('Alignment Loss')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()


class CounterfactualEvaluator:
    """
    反事实评估器 - Week 2主要评估接口
    """
    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.metrics = CounterfactualMetrics(device)
        
    def evaluate_model(self, dataloader, k: int = 10, 
                      max_batches: Optional[int] = None) -> Dict[str, float]:
        """
        评估模型性能
        Args:
            dataloader: 数据加载器
            k: top-k
            max_batches: 最大评估批次数
        Returns:
            评估结果字典
        """
        self.model.eval()
        self.metrics.reset()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                # 获取批次数据
                user_ids = batch['user_ids']
                seqs = batch['seqs']
                pos_seqs = batch['pos_seqs']
                cf_seqs = batch['cf_seqs']
                
                # 原始预测
                original_preds = self._get_predictions(user_ids, seqs, pos_seqs)
                
                # 反事实预测
                cf_preds = self._get_predictions(user_ids, cf_seqs, pos_seqs)
                
                # 生成目标标签
                targets = self._generate_targets(pos_seqs)
                
                # 获取解释
                explanations = self._get_explanations(seqs, batch.get('padding_ids', None))
                
                # 计算对齐损失（如果可用）
                alignment_loss = 0.0
                if hasattr(self.model, 'counterfactual_alignment_loss'):
                    # 这里需要根据具体模型实现来调整
                    pass
                
                # 更新指标
                self.metrics.update_metrics(
                    original_preds, cf_preds, targets, 
                    explanations, alignment_loss, k
                )
        
        return self.metrics.get_summary_metrics()
    
    def _get_predictions(self, user_ids: torch.Tensor, seqs: torch.Tensor, 
                        pos_seqs: torch.Tensor) -> torch.Tensor:
        """获取模型预测"""
        # 这里需要根据具体模型实现来调整
        # 暂时返回随机预测作为示例
        batch_size = seqs.shape[0]
        num_items = self.model.item_num if hasattr(self.model, 'item_num') else 1000
        return torch.randn(batch_size, num_items).to(self.device)
    
    def _generate_targets(self, pos_seqs: torch.Tensor) -> torch.Tensor:
        """生成目标标签"""
        batch_size = pos_seqs.shape[0]
        num_items = self.model.item_num if hasattr(self.model, 'item_num') else 1000
        targets = torch.zeros(batch_size, num_items).to(self.device)
        
        # 将正样本位置标记为1
        for i in range(batch_size):
            pos_items = pos_seqs[i][pos_seqs[i] > 0]
            if len(pos_items) > 0:
                targets[i, pos_items] = 1.0
        
        return targets
    
    def _get_explanations(self, seqs: torch.Tensor, 
                         padding_ids: Optional[torch.Tensor] = None) -> List[List[int]]:
        """获取解释"""
        explanations = []
        batch_size = seqs.shape[0]
        
        for i in range(batch_size):
            # 这里需要根据反事实生成器来获取解释
            if hasattr(self.model, 'cf_generator'):
                important_positions, _ = self.model.cf_generator.get_explanation_weights(threshold=0.5)
                explanations.append(important_positions.cpu().numpy().tolist())
            else:
                explanations.append([])
        
        return explanations


def run_comprehensive_evaluation(model, dataloader, save_dir: str = './eval_results/'):
    """
    运行全面的反事实评估
    Args:
        model: 待评估模型
        dataloader: 数据加载器
        save_dir: 结果保存目录
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("🚀 开始CA-LLMRec反事实评估...")
    
    # 创建评估器
    evaluator = CounterfactualEvaluator(model)
    
    # 运行评估
    results = evaluator.evaluate_model(dataloader, k=10, max_batches=100)
    
    # 打印报告
    evaluator.metrics.print_metrics_report()
    
    # 保存结果
    metrics_file = os.path.join(save_dir, 'counterfactual_metrics.json')
    evaluator.metrics.save_metrics_to_file(metrics_file)
    
    # 绘制图表
    plot_file = os.path.join(save_dir, 'metrics_comparison.png')
    evaluator.metrics.plot_metrics_comparison(plot_file)
    
    print(f"✅ 评估完成！结果已保存到: {save_dir}")
    
    return results 