#!/usr/bin/env python3
"""
CA-LLMRec 超轻量化训练脚本
使用5%数据集进行快速训练，目标6小时内完成全部训练和评估
"""

import os
import sys
import torch
import argparse
import time
import json
from tqdm import tqdm

# 添加项目路径
sys.path.append('.')

from utils import *
from train_model import *
from models.a_llmrec_model import A_llmrec_model


def create_ultra_light_args(max_users=1000, max_items=3000):
    """创建超轻量化训练参数"""
    parser = argparse.ArgumentParser()
    
    # 基础设置
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--multi_gpu", action='store_false')  # 单GPU训练
    parser.add_argument('--gpu_num', type=int, default=0)
    
    # 模型设置
    parser.add_argument("--llm", type=str, default='opt', help='flan_t5, opt, vicuna')
    parser.add_argument("--recsys", type=str, default='sasrec')
    
    # 超轻量化数据集设置
    dataset_suffix = f"_ultra_light_{max_users}u_{max_items}i"
    parser.add_argument("--rec_pre_trained_data", type=str, 
                       default=f'Movies_and_TV{dataset_suffix}')
    
    # 训练阶段设置
    parser.add_argument("--pretrain_stage1", action='store_true')
    parser.add_argument("--pretrain_stage2", action='store_true')
    parser.add_argument("--inference", action='store_true')
    
    # 超轻量化超参数 - 针对快速训练优化
    parser.add_argument('--batch_size1', default=128, type=int)  # Stage 1大批次
    parser.add_argument('--batch_size2', default=8, type=int)    # Stage 2适中批次
    parser.add_argument('--batch_size_infer', default=16, type=int)
    parser.add_argument('--maxlen', default=50, type=int)        # 与预训练模型保持一致
    parser.add_argument('--num_epochs', default=20, type=int)    # 减少训练轮数
    parser.add_argument("--stage1_lr", type=float, default=0.001)  # 提高学习率
    parser.add_argument("--stage2_lr", type=float, default=0.0005) # 提高学习率
    
    # CA-LLMRec反事实参数 - 优化版
    parser.add_argument("--enable_counterfactual", action='store_true',
                       help='Enable counterfactual augmented training')
    parser.add_argument("--cf_alignment_weight", type=float, default=0.05,
                       help='Weight for counterfactual alignment loss (降低以加速收敛)')
    parser.add_argument("--cf_mask_ratio", type=float, default=0.15,
                       help='Mask ratio for counterfactual sequence generation (降低复杂度)')
    
    # 性能优化参数
    parser.add_argument("--num_workers", type=int, default=4, help='DataLoader workers')
    parser.add_argument("--pin_memory", action='store_true', help='Pin memory for faster GPU transfer')
    parser.add_argument("--compile_model", action='store_true', help='Use torch.compile for acceleration')
    
    # 早停和调度器
    parser.add_argument("--early_stopping_patience", type=int, default=5, 
                       help='Early stopping patience')
    parser.add_argument("--use_scheduler", action='store_true', 
                       help='Use learning rate scheduler')
    
    return parser.parse_args(args=[])


def check_ultra_light_dataset(dataset_name):
    """检查超轻量化数据集是否存在"""
    data_path = f'./data/amazon/{dataset_name}.txt'
    mapping_path = f'./data/amazon/{dataset_name}_mappings.json'
    
    if not os.path.exists(data_path):
        print(f"❌ 数据集不存在: {data_path}")
        print("请先运行: python create_ultra_light_dataset.py")
        return False
    
    if not os.path.exists(mapping_path):
        print(f"❌ 映射文件不存在: {mapping_path}")
        return False
    
    # 加载统计信息
    try:
        with open(mapping_path, 'r') as f:
            mappings = json.load(f)
        
        stats = mappings['stats']
        print(f"✅ 数据集检查通过:")
        print(f"  用户数: {mappings['user_count']}")
        print(f"  物品数: {mappings['item_count']}")
        print(f"  交互数: {mappings['interaction_count']}")
        print(f"  用户压缩率: {stats['compression_ratio_users']:.1%}")
        print(f"  物品压缩率: {stats['compression_ratio_items']:.1%}")
        
        return True
    except Exception as e:
        print(f"❌ 读取映射文件失败: {e}")
        return False


def prepare_sasrec_for_ultra_light(args):
    """为超轻量化数据集准备SASRec模型"""
    
    # A-LLMRec期望的目录结构：pre_train/sasrec/{dataset_name}/
    target_dir = f'./pre_train/sasrec/{args.rec_pre_trained_data}'
    
    # 检查A-LLMRec期望的目录是否存在并且有模型文件
    if os.path.exists(target_dir):
        import glob
        pth_files = glob.glob(os.path.join(target_dir, '*.pth'))
        if pth_files:
            if len(pth_files) == 1:
                print(f"✅ 找到A-LLMRec兼容的SASRec模型: {pth_files[0]}")
                return True
            else:
                print(f"⚠️ 发现多个模型文件，A-LLMRec需要唯一模型文件:")
                for pth in pth_files:
                    print(f"   📄 {pth}")
                print("请保留最好的模型，删除其他模型文件")
                return False
    
    # 检查我们的output目录是否有训练好的模型
    output_dir = f'./pre_train/sasrec/output/{args.rec_pre_trained_data}'
    if os.path.exists(output_dir):
        import glob
        pth_files = glob.glob(os.path.join(output_dir, '*.pth'))
        if pth_files:
            print(f"📋 在output目录发现训练好的SASRec模型:")
            for pth in pth_files:
                print(f"   📄 {pth}")
            
            # 选择epoch数最高的模型
            best_model = None
            best_epoch = 0
            for pth in pth_files:
                filename = os.path.basename(pth)
                if 'epoch=' in filename:
                    try:
                        epoch_str = filename.split('epoch=')[1].split('.')[0]
                        epoch = int(epoch_str)
                        if epoch > best_epoch:
                            best_epoch = epoch
                            best_model = pth
                    except:
                        pass
            
            if best_model:
                print(f"🏆 选择最佳模型 (epoch={best_epoch}): {best_model}")
                
                # 创建A-LLMRec期望的目录
                os.makedirs(target_dir, exist_ok=True)
                
                # 复制最佳模型到A-LLMRec期望的位置
                import shutil
                target_path = os.path.join(target_dir, os.path.basename(best_model))
                shutil.copy(best_model, target_path)
                print(f"✅ 模型已复制到A-LLMRec期望位置: {target_path}")
                return True
    
    print("🔧 未找到现有SASRec模型，需要训练...")
    
    # 检查原始Movies_and_TV模型是否存在（作为fallback）
    fallback_dirs = [
        './pre_train/sasrec/output/Movies_and_TV',
        './pre_train/sasrec/Movies_and_TV'
    ]
    
    for fallback_dir in fallback_dirs:
        if os.path.exists(fallback_dir):
            import glob
            pth_files = glob.glob(os.path.join(fallback_dir, '*.pth'))
            if pth_files:
                print(f"📋 发现原始SASRec模型，将复制并调整...")
                
                # 选择最好的模型
                best_model = pth_files[0]  # 简单选择第一个
                
                # 创建目录
                os.makedirs(target_dir, exist_ok=True)
                
                # 复制模型
                import shutil
                target_path = os.path.join(target_dir, os.path.basename(best_model))
                shutil.copy(best_model, target_path)
                print(f"✅ SASRec模型准备完成: {target_path}")
                return True
    
    print("⚠️ 未找到任何预训练SASRec模型，开始自动训练...")
    
    # 自动训练SASRec模型
    try:
        from train_sasrec_ultra_light import train_sasrec_ultra_light
        
        # 使用较少的epochs进行快速训练
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        success = train_sasrec_ultra_light(args.rec_pre_trained_data, device=device, epochs=20)
        
        if success:
            print(f"✅ SASRec模型自动训练完成")
            return True
        else:
            print("❌ SASRec模型自动训练失败")
            return False
            
    except Exception as e:
        print(f"❌ 自动训练SASRec时出错: {e}")
        print("建议手动运行: python train_sasrec_ultra_light.py --dataset", args.rec_pre_trained_data)
        return False


def train_stage1_ultra_fast(args):
    """超快速Stage 1训练"""
    print("\n🚀 开始Stage 1训练 (超轻量化)")
    print("=" * 50)
    
    start_time = time.time()
    
    # 设置训练参数
    args.pretrain_stage1 = True
    args.pretrain_stage2 = False
    args.inference = False
    
    try:
        # 调用原始训练函数
        if args.multi_gpu:
            train_model_phase1(args)
        else:
            train_model_phase1_(0, 1, args)
        
        stage1_time = time.time() - start_time
        print(f"\n✅ Stage 1训练完成，耗时: {stage1_time/60:.1f}分钟")
        return True
        
    except Exception as e:
        print(f"❌ Stage 1训练失败: {e}")
        return False


def train_stage2_ultra_fast(args):
    """超快速Stage 2训练"""
    print("\n🚀 开始Stage 2训练 (超轻量化)")
    print("=" * 50)
    
    start_time = time.time()
    
    # 设置训练参数
    args.pretrain_stage1 = False
    args.pretrain_stage2 = True
    args.inference = False
    
    try:
        # 调用原始训练函数
        if args.multi_gpu:
            train_model_phase2(args)
        else:
            train_model_phase2_(0, 1, args)
        
        stage2_time = time.time() - start_time
        print(f"\n✅ Stage 2训练完成，耗时: {stage2_time/60:.1f}分钟")
        return True
        
    except Exception as e:
        print(f"❌ Stage 2训练失败: {e}")
        return False


def evaluate_ultra_fast(args):
    """超快速评估"""
    print("\n🔍 开始模型评估")
    print("=" * 50)
    
    start_time = time.time()
    
    # 设置评估参数
    args.pretrain_stage1 = False
    args.pretrain_stage2 = False
    args.inference = True
    
    try:
        # 调用原始评估函数
        inference(args)
        
        eval_time = time.time() - start_time
        print(f"\n✅ 评估完成，耗时: {eval_time/60:.1f}分钟")
        
        # 运行评估脚本
        print("📊 计算评估指标...")
        os.system("python eval.py")
        
        return True
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        return False


def main():
    """超轻量化训练主函数"""
    parser = argparse.ArgumentParser(description='CA-LLMRec超轻量化训练')
    parser.add_argument('--max_users', type=int, default=1000, 
                       help='最大用户数量 (默认: 1000)')
    parser.add_argument('--max_items', type=int, default=3000, 
                       help='最大物品数量 (默认: 3000)')
    parser.add_argument('--skip_stage1', action='store_true', 
                       help='跳过Stage 1训练')
    parser.add_argument('--skip_stage2', action='store_true', 
                       help='跳过Stage 2训练')
    parser.add_argument('--skip_eval', action='store_true', 
                       help='跳过评估')
    
    cmd_args = parser.parse_args()
    
    print("🚀 CA-LLMRec 超轻量化训练")
    print("=" * 60)
    print(f"目标：使用{cmd_args.max_users}用户，{cmd_args.max_items}物品的数据集")
    print(f"预计总训练时间：4-6小时")
    
    # 创建训练参数
    args = create_ultra_light_args(cmd_args.max_users, cmd_args.max_items)
    
    # 启用CA-LLMRec功能
    args.enable_counterfactual = True
    
    print(f"\n📋 训练配置:")
    print(f"  数据集: {args.rec_pre_trained_data}")
    print(f"  设备: {args.device}")
    print(f"  反事实功能: {'✅' if args.enable_counterfactual else '❌'}")
    print(f"  批次大小: Stage1={args.batch_size1}, Stage2={args.batch_size2}")
    print(f"  训练轮数: {args.num_epochs}")
    print(f"  序列长度: {args.maxlen}")
    
    # 检查数据集
    if not check_ultra_light_dataset(args.rec_pre_trained_data):
        print("\n请先运行数据集生成脚本:")
        print(f"python create_ultra_light_dataset.py --max_users {cmd_args.max_users} --max_items {cmd_args.max_items}")
        return
    
    # 准备SASRec模型
    if not prepare_sasrec_for_ultra_light(args):
        print("⚠️ SASRec模型准备失败，但继续训练...")
    
    # 开始训练
    total_start_time = time.time()
    success_stages = []
    
    try:
        # Stage 1: 协同过滤嵌入对齐
        if not cmd_args.skip_stage1:
            print(f"\n{'='*60}")
            print("🎯 Stage 1: 协同过滤嵌入对齐 + 反事实增强")
            print(f"预计时间: 1-2小时")
            
            if train_stage1_ultra_fast(args):
                success_stages.append("Stage 1")
            else:
                print("❌ Stage 1训练失败，停止训练")
                return
        
        # Stage 2: LLM推荐训练
        if not cmd_args.skip_stage2:
            print(f"\n{'='*60}")
            print("🎯 Stage 2: LLM推荐训练")
            print(f"预计时间: 2-3小时")
            
            if train_stage2_ultra_fast(args):
                success_stages.append("Stage 2")
            else:
                print("❌ Stage 2训练失败，但可以继续评估Stage 1结果")
        
        # 评估
        if not cmd_args.skip_eval:
            print(f"\n{'='*60}")
            print("🎯 模型评估")
            print(f"预计时间: 30分钟")
            
            if evaluate_ultra_fast(args):
                success_stages.append("Evaluation")
            else:
                print("❌ 评估失败")
    
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程出现错误: {e}")
    
    # 总结
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print("🏁 训练完成总结")
    print(f"总耗时: {total_time/3600:.1f}小时 ({total_time/60:.1f}分钟)")
    print(f"成功阶段: {', '.join(success_stages) if success_stages else '无'}")
    
    if "Stage 1" in success_stages:
        print("✅ 反事实增强的协同过滤嵌入训练成功")
    if "Stage 2" in success_stages:
        print("✅ LLM推荐训练成功")
    if "Evaluation" in success_stages:
        print("✅ 模型评估完成")
        print("📊 请查看evaluation结果了解模型性能")
    
    if total_time < 6 * 3600:  # 6小时
        print(f"🎉 训练在目标时间内完成！({total_time/3600:.1f}小时 < 6小时)")
    else:
        print(f"⚠️ 训练超出目标时间 ({total_time/3600:.1f}小时 > 6小时)")
    
    print("\n下一步:")
    print("1. 查看训练日志了解详细信息")
    print("2. 分析evaluation结果")
    print("3. 如需要可以调整超参数重新训练")


if __name__ == "__main__":
    main() 