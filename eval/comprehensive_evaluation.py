"""
CA-LLMRec Week 4 - 大规模实验和评估模块
支持多数据集验证、基线对比、统计显著性测试
提供全面的实验分析和可视化
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import pickle
import os
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import ndcg_score, precision_score, recall_score, f1_score
import time
import warnings
warnings.filterwarnings('ignore')


class ComprehensiveEvaluator:
    """
    全面评估器
    支持多种推荐指标、统计测试、可视化分析
    """
    
    def __init__(self, metrics: List[str] = None, k_values: List[int] = None):
        """
        Args:
            metrics: 要计算的指标列表
            k_values: top-k值列表
        """
        self.metrics = metrics or ['hit_rate', 'ndcg', 'precision', 'recall', 'f1', 'mrr', 'map']
        self.k_values = k_values or [1, 3, 5, 10, 20]
        
        self.results_history = defaultdict(list)
        self.baseline_results = {}
        
        print(f"全面评估器初始化完成")
        print(f"  支持指标: {self.metrics}")
        print(f"  Top-K值: {self.k_values}")
    
    def compute_hit_rate(self, predictions: np.ndarray, targets: np.ndarray, k: int) -> float:
        """计算命中率@K"""
        hits = 0
        total = 0
        
        for pred, target in zip(predictions, targets):
            if len(target) > 0:
                top_k_pred = pred[:k]
                if any(item in target for item in top_k_pred):
                    hits += 1
                total += 1
        
        return hits / total if total > 0 else 0.0
    
    def compute_ndcg(self, predictions: np.ndarray, targets: np.ndarray, k: int) -> float:
        """计算NDCG@K"""
        ndcg_scores = []
        
        for pred, target in zip(predictions, targets):
            if len(target) == 0:
                continue
            
            # 创建相关性分数
            relevance_scores = []
            for item in pred[:k]:
                relevance_scores.append(1 if item in target else 0)
            
            if sum(relevance_scores) == 0:
                ndcg_scores.append(0.0)
            else:
                # 计算NDCG
                true_relevances = np.array([relevance_scores])
                ndcg = ndcg_score(true_relevances, true_relevances, k=k)
                ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def compute_precision(self, predictions: np.ndarray, targets: np.ndarray, k: int) -> float:
        """计算精确率@K"""
        precisions = []
        
        for pred, target in zip(predictions, targets):
            if len(target) == 0:
                continue
            
            top_k_pred = pred[:k]
            hits = sum(1 for item in top_k_pred if item in target)
            precision = hits / k
            precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def compute_recall(self, predictions: np.ndarray, targets: np.ndarray, k: int) -> float:
        """计算召回率@K"""
        recalls = []
        
        for pred, target in zip(predictions, targets):
            if len(target) == 0:
                continue
            
            top_k_pred = pred[:k]
            hits = sum(1 for item in top_k_pred if item in target)
            recall = hits / len(target)
            recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def compute_f1(self, predictions: np.ndarray, targets: np.ndarray, k: int) -> float:
        """计算F1分数@K"""
        precision = self.compute_precision(predictions, targets, k)
        recall = self.compute_recall(predictions, targets, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def compute_mrr(self, predictions: np.ndarray, targets: np.ndarray, k: int) -> float:
        """计算平均倒数排名@K"""
        mrr_scores = []
        
        for pred, target in zip(predictions, targets):
            if len(target) == 0:
                continue
            
            reciprocal_rank = 0
            for i, item in enumerate(pred[:k]):
                if item in target:
                    reciprocal_rank = 1 / (i + 1)
                    break
            
            mrr_scores.append(reciprocal_rank)
        
        return np.mean(mrr_scores) if mrr_scores else 0.0
    
    def compute_map(self, predictions: np.ndarray, targets: np.ndarray, k: int) -> float:
        """计算平均精确率@K"""
        ap_scores = []
        
        for pred, target in zip(predictions, targets):
            if len(target) == 0:
                continue
            
            hits = 0
            precision_sum = 0
            
            for i, item in enumerate(pred[:k]):
                if item in target:
                    hits += 1
                    precision_sum += hits / (i + 1)
            
            if hits > 0:
                ap_scores.append(precision_sum / min(len(target), k))
            else:
                ap_scores.append(0.0)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    def evaluate_model(self, predictions: np.ndarray, targets: np.ndarray, 
                      model_name: str = "model") -> Dict[str, Dict[str, float]]:
        """
        评估单个模型
        Args:
            predictions: 预测结果 [num_users, num_candidates]
            targets: 真实标签 [num_users, num_relevant_items]
            model_name: 模型名称
        Returns:
            评估结果字典
        """
        results = {}
        
        for k in self.k_values:
            results[f"@{k}"] = {}
            
            for metric in self.metrics:
                if metric == 'hit_rate':
                    score = self.compute_hit_rate(predictions, targets, k)
                elif metric == 'ndcg':
                    score = self.compute_ndcg(predictions, targets, k)
                elif metric == 'precision':
                    score = self.compute_precision(predictions, targets, k)
                elif metric == 'recall':
                    score = self.compute_recall(predictions, targets, k)
                elif metric == 'f1':
                    score = self.compute_f1(predictions, targets, k)
                elif metric == 'mrr':
                    score = self.compute_mrr(predictions, targets, k)
                elif metric == 'map':
                    score = self.compute_map(predictions, targets, k)
                else:
                    score = 0.0
                
                results[f"@{k}"][metric] = score
        
        # 存储结果历史
        self.results_history[model_name].append(results)
        
        return results
    
    def compare_models(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        比较多个模型的性能
        Args:
            model_results: 模型结果字典
        Returns:
            比较结果
        """
        comparison = {
            'ranking': {},
            'improvements': {},
            'statistical_tests': {}
        }
        
        # 对每个指标进行排名
        for k in self.k_values:
            comparison['ranking'][f"@{k}"] = {}
            
            for metric in self.metrics:
                scores = {model: results[f"@{k}"][metric] 
                         for model, results in model_results.items()}
                
                # 按分数排序
                ranked_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                comparison['ranking'][f"@{k}"][metric] = ranked_models
        
        # 计算相对改进
        if len(model_results) >= 2:
            model_names = list(model_results.keys())
            baseline_model = model_names[0]  # 假设第一个是基线
            
            for model_name in model_names[1:]:
                comparison['improvements'][model_name] = {}
                
                for k in self.k_values:
                    comparison['improvements'][model_name][f"@{k}"][metric] = {}
                    
                    for metric in self.metrics:
                        baseline_score = model_results[baseline_model][f"@{k}"][metric]
                        current_score = model_results[model_name][f"@{k}"][metric]
                        
                        if baseline_score > 0:
                            improvement = (current_score - baseline_score) / baseline_score * 100
                        else:
                            improvement = 0.0
                        
                        comparison['improvements'][model_name][f"@{k}"][metric] = improvement
        
        return comparison
    
    def statistical_significance_test(self, results1: List[float], results2: List[float],
                                    test_type: str = 'ttest') -> Dict[str, float]:
        """
        统计显著性测试
        Args:
            results1: 模型1的结果列表
            results2: 模型2的结果列表
            test_type: 测试类型 ('ttest', 'wilcoxon')
        Returns:
            测试结果
        """
        if test_type == 'ttest':
            statistic, p_value = stats.ttest_rel(results1, results2)
        elif test_type == 'wilcoxon':
            statistic, p_value = stats.wilcoxon(results1, results2)
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
    
    def generate_report(self, model_results: Dict[str, Dict], 
                       save_path: str = None) -> str:
        """
        生成评估报告
        Args:
            model_results: 模型结果字典
            save_path: 保存路径
        Returns:
            报告内容
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CA-LLMRec 模型评估报告")
        report_lines.append("=" * 80)
        
        # 模型性能表格
        report_lines.append("\n📊 模型性能对比")
        report_lines.append("-" * 50)
        
        for k in self.k_values:
            report_lines.append(f"\n🎯 Top-{k} 指标:")
            
            # 创建表格头
            header = f"{'模型':<15}"
            for metric in self.metrics:
                header += f"{metric.upper():<10}"
            report_lines.append(header)
            report_lines.append("-" * len(header))
            
            # 填充表格内容
            for model_name, results in model_results.items():
                row = f"{model_name:<15}"
                for metric in self.metrics:
                    score = results[f"@{k}"][metric]
                    row += f"{score:<10.4f}"
                report_lines.append(row)
        
        # 模型比较
        if len(model_results) >= 2:
            comparison = self.compare_models(model_results)
            
            report_lines.append(f"\n🏆 模型排名 (基于NDCG@10):")
            if '@10' in comparison['ranking'] and 'ndcg' in comparison['ranking']['@10']:
                rankings = comparison['ranking']['@10']['ndcg']
                for i, (model, score) in enumerate(rankings, 1):
                    report_lines.append(f"  {i}. {model}: {score:.4f}")
            
            # 改进分析
            report_lines.append(f"\n📈 相对改进分析:")
            baseline_model = list(model_results.keys())[0]
            report_lines.append(f"  基线模型: {baseline_model}")
            
            for model_name, improvements in comparison['improvements'].items():
                report_lines.append(f"\n  {model_name} 相对于 {baseline_model}:")
                for k in [5, 10]:  # 重点关注@5和@10
                    if f"@{k}" in improvements:
                        for metric in ['hit_rate', 'ndcg']:
                            if metric in improvements[f"@{k}"]:
                                imp = improvements[f"@{k}"][metric]
                                report_lines.append(f"    {metric.upper()}@{k}: {imp:+.2f}%")
        
        # 生成建议
        report_lines.append(f"\n💡 优化建议:")
        best_model = max(model_results.keys(), 
                        key=lambda x: model_results[x]['@10']['ndcg'])
        report_lines.append(f"  • 最佳模型: {best_model}")
        report_lines.append(f"  • 建议重点优化召回率和精确率的平衡")
        report_lines.append(f"  • 考虑集成多个模型的预测结果")
        
        report_lines.append("\n" + "=" * 80)
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"✅ 评估报告已保存到: {save_path}")
        
        return report_content


class MultiDatasetEvaluator:
    """
    多数据集评估器
    支持在多个数据集上进行交叉验证
    """
    
    def __init__(self, datasets: Dict[str, Any]):
        """
        Args:
            datasets: 数据集字典 {dataset_name: dataset_data}
        """
        self.datasets = datasets
        self.evaluator = ComprehensiveEvaluator()
        self.cross_dataset_results = defaultdict(dict)
        
        print(f"多数据集评估器初始化完成")
        print(f"  数据集: {list(datasets.keys())}")
    
    def evaluate_across_datasets(self, model, model_name: str = "model") -> Dict[str, Dict]:
        """
        在所有数据集上评估模型
        Args:
            model: 要评估的模型
            model_name: 模型名称
        Returns:
            跨数据集评估结果
        """
        results = {}
        
        for dataset_name, dataset_data in self.datasets.items():
            print(f"\n🔄 在数据集 {dataset_name} 上评估 {model_name}...")
            
            # 获取测试数据
            test_data = dataset_data.get('test', dataset_data)
            
            # 模型预测
            predictions, targets = self._get_model_predictions(model, test_data)
            
            # 评估
            dataset_results = self.evaluator.evaluate_model(
                predictions, targets, f"{model_name}_{dataset_name}"
            )
            
            results[dataset_name] = dataset_results
            
            # 打印关键指标
            ndcg_10 = dataset_results['@10']['ndcg']
            hit_rate_10 = dataset_results['@10']['hit_rate']
            print(f"  NDCG@10: {ndcg_10:.4f}, Hit Rate@10: {hit_rate_10:.4f}")
        
        self.cross_dataset_results[model_name] = results
        return results
    
    def _get_model_predictions(self, model, test_data) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取模型预测结果
        Args:
            model: 模型
            test_data: 测试数据
        Returns:
            (predictions, targets)
        """
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            # 这里需要根据具体的数据格式进行调整
            for batch in test_data:
                if isinstance(batch, dict):
                    user_seq = batch.get('user_seq', batch.get('seqs'))
                    target_items = batch.get('targets', batch.get('pos_seqs'))
                else:
                    # 简化处理
                    user_seq, target_items = batch
                
                # 生成推荐
                top_k_items, _ = model.generate_recommendations(user_seq, top_k=20)
                
                predictions.extend(top_k_items.cpu().numpy())
                targets.extend(target_items.cpu().numpy())
        
        return np.array(predictions), np.array(targets)
    
    def compute_average_performance(self, model_name: str) -> Dict[str, Dict]:
        """
        计算模型在所有数据集上的平均性能
        Args:
            model_name: 模型名称
        Returns:
            平均性能结果
        """
        if model_name not in self.cross_dataset_results:
            raise ValueError(f"Model {model_name} not found in results")
        
        model_results = self.cross_dataset_results[model_name]
        average_results = {}
        
        # 计算每个k值的平均指标
        for k in self.evaluator.k_values:
            average_results[f"@{k}"] = {}
            
            for metric in self.evaluator.metrics:
                scores = [results[f"@{k}"][metric] for results in model_results.values()]
                average_results[f"@{k}"][metric] = np.mean(scores)
                average_results[f"@{k}"][f"{metric}_std"] = np.std(scores)
        
        return average_results
    
    def generate_cross_dataset_report(self, save_path: str = None) -> str:
        """
        生成跨数据集评估报告
        Args:
            save_path: 保存路径
        Returns:
            报告内容
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CA-LLMRec 跨数据集评估报告")
        report_lines.append("=" * 80)
        
        # 数据集概览
        report_lines.append(f"\n📊 数据集概览:")
        for dataset_name in self.datasets.keys():
            report_lines.append(f"  • {dataset_name}")
        
        # 各模型在各数据集上的表现
        report_lines.append(f"\n🎯 各数据集详细结果:")
        
        for model_name, dataset_results in self.cross_dataset_results.items():
            report_lines.append(f"\n  模型: {model_name}")
            report_lines.append(f"  {'-' * 40}")
            
            for dataset_name, results in dataset_results.items():
                report_lines.append(f"\n    数据集: {dataset_name}")
                for k in [5, 10]:
                    if f"@{k}" in results:
                        ndcg = results[f"@{k}"]["ndcg"]
                        hit_rate = results[f"@{k}"]["hit_rate"]
                        report_lines.append(f"      NDCG@{k}: {ndcg:.4f}, Hit Rate@{k}: {hit_rate:.4f}")
        
        # 平均性能对比
        report_lines.append(f"\n🏆 平均性能对比:")
        report_lines.append(f"  {'模型':<20} {'NDCG@10':<12} {'Hit Rate@10':<15} {'稳定性':<10}")
        report_lines.append(f"  {'-' * 60}")
        
        for model_name in self.cross_dataset_results.keys():
            avg_results = self.compute_average_performance(model_name)
            ndcg_10 = avg_results["@10"]["ndcg"]
            ndcg_10_std = avg_results["@10"]["ndcg_std"]
            hit_rate_10 = avg_results["@10"]["hit_rate"]
            
            stability = "高" if ndcg_10_std < 0.02 else "中" if ndcg_10_std < 0.05 else "低"
            
            report_lines.append(f"  {model_name:<20} {ndcg_10:<12.4f} {hit_rate_10:<15.4f} {stability:<10}")
        
        report_lines.append("\n" + "=" * 80)
        
        report_content = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"✅ 跨数据集评估报告已保存到: {save_path}")
        
        return report_content


class BaselineComparison:
    """
    基线对比模块
    与经典推荐算法进行对比
    """
    
    def __init__(self):
        self.baseline_models = {}
        self.comparison_results = {}
        
        print("基线对比模块初始化完成")
    
    def add_baseline_model(self, name: str, model: Any):
        """
        添加基线模型
        Args:
            name: 模型名称
            model: 模型实例
        """
        self.baseline_models[name] = model
        print(f"✅ 添加基线模型: {name}")
    
    def compare_with_baselines(self, ca_llmrec_model, test_data, 
                              evaluator: ComprehensiveEvaluator) -> Dict[str, Dict]:
        """
        与基线模型进行对比
        Args:
            ca_llmrec_model: CA-LLMRec模型
            test_data: 测试数据
            evaluator: 评估器
        Returns:
            对比结果
        """
        all_results = {}
        
        # 评估CA-LLMRec
        print("🔄 评估CA-LLMRec模型...")
        ca_predictions, targets = self._get_predictions(ca_llmrec_model, test_data)
        all_results['CA-LLMRec'] = evaluator.evaluate_model(ca_predictions, targets, 'CA-LLMRec')
        
        # 评估基线模型
        for baseline_name, baseline_model in self.baseline_models.items():
            print(f"🔄 评估基线模型: {baseline_name}...")
            try:
                baseline_predictions, _ = self._get_predictions(baseline_model, test_data)
                all_results[baseline_name] = evaluator.evaluate_model(
                    baseline_predictions, targets, baseline_name
                )
            except Exception as e:
                print(f"⚠️ 基线模型 {baseline_name} 评估失败: {e}")
                continue
        
        self.comparison_results = all_results
        return all_results
    
    def _get_predictions(self, model, test_data) -> Tuple[np.ndarray, np.ndarray]:
        """获取模型预测结果"""
        # 这里需要根据具体模型接口进行调整
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in test_data:
                # 根据模型类型调用不同的预测方法
                if hasattr(model, 'generate_recommendations'):
                    top_k_items, _ = model.generate_recommendations(batch['user_seq'], top_k=20)
                elif hasattr(model, 'predict'):
                    top_k_items = model.predict(batch['user_seq'])
                else:
                    # 简化处理
                    top_k_items = torch.randint(1, 1000, (len(batch['user_seq']), 20))
                
                predictions.extend(top_k_items.cpu().numpy())
                targets.extend(batch['targets'].cpu().numpy())
        
        return np.array(predictions), np.array(targets)
    
    def generate_comparison_report(self, save_path: str = None) -> str:
        """
        生成基线对比报告
        Args:
            save_path: 保存路径
        Returns:
            报告内容
        """
        if not self.comparison_results:
            return "❌ 尚未进行基线对比，请先调用compare_with_baselines方法"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CA-LLMRec vs 基线模型对比报告")
        report_lines.append("=" * 80)
        
        # 性能对比表格
        report_lines.append(f"\n📊 性能对比 (Top-10):")
        report_lines.append(f"{'模型':<20} {'NDCG@10':<12} {'Hit Rate@10':<15} {'Precision@10':<15}")
        report_lines.append("-" * 65)
        
        # 按NDCG@10排序
        sorted_results = sorted(
            self.comparison_results.items(),
            key=lambda x: x[1]['@10']['ndcg'],
            reverse=True
        )
        
        for model_name, results in sorted_results:
            ndcg_10 = results['@10']['ndcg']
            hit_rate_10 = results['@10']['hit_rate']
            precision_10 = results['@10']['precision']
            
            report_lines.append(f"{model_name:<20} {ndcg_10:<12.4f} {hit_rate_10:<15.4f} {precision_10:<15.4f}")
        
        # 相对改进分析
        if 'CA-LLMRec' in self.comparison_results:
            ca_results = self.comparison_results['CA-LLMRec']
            report_lines.append(f"\n📈 CA-LLMRec相对改进:")
            
            for model_name, results in self.comparison_results.items():
                if model_name == 'CA-LLMRec':
                    continue
                
                ndcg_improvement = (ca_results['@10']['ndcg'] - results['@10']['ndcg']) / results['@10']['ndcg'] * 100
                hit_improvement = (ca_results['@10']['hit_rate'] - results['@10']['hit_rate']) / results['@10']['hit_rate'] * 100
                
                report_lines.append(f"  vs {model_name}:")
                report_lines.append(f"    NDCG@10改进: {ndcg_improvement:+.2f}%")
                report_lines.append(f"    Hit Rate@10改进: {hit_improvement:+.2f}%")
        
        report_lines.append("\n" + "=" * 80)
        
        report_content = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"✅ 基线对比报告已保存到: {save_path}")
        
        return report_content


def create_comprehensive_evaluation_suite(datasets: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    创建全面评估套件
    Args:
        datasets: 数据集字典
    Returns:
        评估套件
    """
    evaluation_suite = {
        'evaluator': ComprehensiveEvaluator(),
        'baseline_comparison': BaselineComparison(),
    }
    
    if datasets:
        evaluation_suite['multi_dataset_evaluator'] = MultiDatasetEvaluator(datasets)
    
    print("✅ 全面评估套件创建完成")
    print("  包含组件: 综合评估器、基线对比、多数据集评估")
    
    return evaluation_suite


if __name__ == "__main__":
    # 测试评估模块
    print("🧪 测试大规模实验和评估模块...")
    
    # 创建模拟数据
    np.random.seed(42)
    num_users = 100
    num_items = 1000
    
    # 模拟预测结果
    predictions = np.random.randint(1, num_items, (num_users, 20))
    targets = [np.random.choice(range(1, num_items), size=np.random.randint(1, 5), replace=False) 
               for _ in range(num_users)]
    targets = np.array(targets, dtype=object)
    
    # 测试综合评估器
    print("\n1. 测试综合评估器...")
    evaluator = ComprehensiveEvaluator()
    
    results = evaluator.evaluate_model(predictions, targets, "TestModel")
    print(f"✅ 评估完成，NDCG@10: {results['@10']['ndcg']:.4f}")
    
    # 测试模型对比
    print("\n2. 测试模型对比...")
    predictions2 = np.random.randint(1, num_items, (num_users, 20))
    results2 = evaluator.evaluate_model(predictions2, targets, "TestModel2")
    
    comparison = evaluator.compare_models({
        'TestModel': results,
        'TestModel2': results2
    })
    
    print(f"✅ 模型对比完成，排名: {comparison['ranking']['@10']['ndcg']}")
    
    # 测试报告生成
    print("\n3. 测试报告生成...")
    report = evaluator.generate_report({
        'TestModel': results,
        'TestModel2': results2
    })
    
    print("✅ 报告生成完成")
    print("报告摘要:")
    print(report.split('\n')[0])
    print(report.split('\n')[1])
    print("...")
    
    print("\n🎉 大规模实验和评估模块测试完成！") 