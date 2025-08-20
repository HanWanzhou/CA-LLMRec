#!/usr/bin/env python3
"""
运行CA-LLMRec推理并生成反事实解释
这个脚本展示如何启用反事实解释功能
"""

import subprocess
import sys
import os

def run_inference_with_explanations():
    """运行带反事实解释的推理"""
    
    print("🎯 CA-LLMRec 反事实解释推理")
    print("=" * 50)
    
    # 检查必要文件
    print("📋 检查必要文件...")
    
    required_files = [
        './models_saved/A_llmrec_model.pth',
        './main.py',
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 缺少必要文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n请先完成模型训练：")
        print("   python main.py --pretrain_stage1 --enable_counterfactual")
        print("   python main.py --pretrain_stage2")
        return False
    
    print("✅ 必要文件检查完成")
    
    # 清空之前的输出
    if os.path.exists('./recommendation_output.txt'):
        print("🗑️  清空之前的推荐输出...")
        os.remove('./recommendation_output.txt')
    
    # 构建推理命令
    cmd = [
        sys.executable, 'main.py',
        '--inference',
        '--rec_pre_trained_data', 'Movies_and_TV_ultra_light_1000u_3000i',
        '--enable_counterfactual',  # 关键：启用反事实功能
        '--batch_size_infer', '2'
    ]
    
    print("🚀 运行反事实解释推理...")
    print(f"命令: {' '.join(cmd)}")
    print()
    
    try:
        # 运行推理
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ 推理完成成功")
            
            # 检查输出文件
            if os.path.exists('./recommendation_output.txt'):
                print("📄 分析推荐输出...")
                
                with open('./recommendation_output.txt', 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 统计
                total_recommendations = content.count('Answer: ')
                cf_explanations = content.count('=== 反事实解释 ===')
                
                print(f"   总推荐数: {total_recommendations}")
                print(f"   反事实解释数: {cf_explanations}")
                
                if cf_explanations > 0:
                    print(f"🎉 成功生成 {cf_explanations} 条反事实解释！")
                    
                    # 显示示例
                    show_explanation_example(content)
                    
                else:
                    print("⚠️  未发现反事实解释，可能是功能未正确启用")
                    
            else:
                print("❌ 未生成推荐输出文件")
                
        else:
            print("❌ 推理执行失败")
            print("错误输出:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ 推理超时（5分钟）")
        return False
    except Exception as e:
        print(f"❌ 推理执行异常: {e}")
        return False
    
    return True

def show_explanation_example(content):
    """显示反事实解释示例"""
    
    print("\n📋 反事实解释示例:")
    print("-" * 40)
    
    lines = content.split('\n')
    
    # 找到第一个反事实解释
    cf_start = -1
    for i, line in enumerate(lines):
        if '=== 反事实解释 ===' in line:
            cf_start = i
            break
    
    if cf_start >= 0:
        # 显示推荐上下文
        context_start = max(0, cf_start - 5)
        for i in range(context_start, cf_start):
            if lines[i].strip() and ('Answer:' in lines[i] or 'LLM:' in lines[i]):
                print(lines[i])
        
        # 显示反事实解释
        print()
        for i in range(cf_start, min(cf_start + 8, len(lines))):
            if lines[i].strip():
                print(lines[i])
        
        print("-" * 40)

def show_usage():
    """显示使用说明"""
    
    print("\n💡 反事实解释功能使用说明:")
    print("=" * 50)
    
    print("1️⃣  训练阶段（启用反事实）:")
    print("   python main.py --pretrain_stage1 --enable_counterfactual")
    print("   python main.py --pretrain_stage2")
    print()
    
    print("2️⃣  推理阶段（启用反事实解释）:")
    print("   python main.py --inference --enable_counterfactual")
    print("   或运行此脚本: python run_inference_with_explanations.py")
    print()
    
    print("3️⃣  查看结果:")
    print("   查看 recommendation_output.txt 中的 '=== 反事实解释 ===' 部分")
    print()
    
    print("📊 反事实解释包含:")
    print("   - 权重分布: 每个历史物品的重要性权重")
    print("   - 关键物品: 对推荐最重要的历史物品")
    print("   - 可移除物品: 可以移除而不影响推荐的物品")
    print("   - 反事实推荐: 移除可移除物品后的推荐结果")
    print("   - 自然语言解释: 易于理解的推荐解释")
    print("   - 控制保真度: 反事实操作的有效性度量")

def main():
    """主函数"""
    
    print("🔬 CA-LLMRec 反事实解释演示")
    print("展示如何生成和显示反事实解释功能")
    print()
    
    # 显示使用说明
    show_usage()
    
    # 询问是否运行
    user_input = input("\n是否现在运行反事实解释推理？(y/n): ").strip().lower()
    
    if user_input in ['y', 'yes', '是', '1']:
        success = run_inference_with_explanations()
        
        if success:
            print("\n🎉 反事实解释功能演示完成！")
            print("请查看 recommendation_output.txt 文件中的解释部分")
        else:
            print("\n❌ 演示过程中遇到问题，请检查训练是否完成")
    else:
        print("\n📝 手动运行步骤:")
        print("1. 确保模型已训练完成")
        print("2. 运行: python main.py --inference --enable_counterfactual")
        print("3. 查看: recommendation_output.txt")

if __name__ == "__main__":
    main() 