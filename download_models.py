#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载A-LLMRec所需的预训练模型
"""

import os
import sys
from sentence_transformers import SentenceTransformer

def download_sentence_transformer():
    """下载SentenceTransformer模型"""
    print("🔄 开始下载SentenceTransformer模型...")
    
    try:
        # 下载nq-distilbert-base-v1模型 (A-LLMRec中使用的文本编码器)
        # 设置较长的超时时间
        import socket
        socket.setdefaulttimeout(300)  # 5分钟超时
        
        model = SentenceTransformer('nq-distilbert-base-v1')
        
        # 测试模型是否可用
        test_embedding = model.encode("test sentence")
        if test_embedding is not None and len(test_embedding) > 0:
            print("✅ SentenceTransformer模型下载完成!")
            
            # 尝试获取缓存位置（不同版本API可能不同）
            try:
                if hasattr(model, 'cache_folder'):
                    print(f"📁 模型缓存位置: {model.cache_folder}")
                elif hasattr(model, '_cache_folder'):
                    print(f"📁 模型缓存位置: {model._cache_folder}")
                else:
                    # 使用默认的transformers缓存位置
                    import os
                    from transformers import TRANSFORMERS_CACHE
                    print(f"📁 模型缓存位置: {TRANSFORMERS_CACHE}")
            except:
                print("📁 模型缓存位置: 使用系统默认位置")
                
            print(f"🧪 模型测试成功 (embedding维度: {len(test_embedding)})")
            return True
        else:
            print("❌ 模型下载完成但测试失败")
            return False
            
    except Exception as e:
        print(f"⚠️ SentenceTransformer下载遇到问题: {e}")
        
        # 尝试检查是否已经部分下载成功
        try:
            print("🔍 检查是否已有可用的模型文件...")
            model = SentenceTransformer('nq-distilbert-base-v1', device='cpu')
            test_embedding = model.encode("test sentence")
            
            if test_embedding is not None and len(test_embedding) > 0:
                print("✅ 发现已下载的模型可以正常使用!")
                
                # 尝试获取缓存位置
                try:
                    if hasattr(model, 'cache_folder'):
                        print(f"📁 模型位置: {model.cache_folder}")
                    elif hasattr(model, '_cache_folder'):
                        print(f"📁 模型位置: {model._cache_folder}")
                    else:
                        print("📁 模型位置: 使用系统默认位置")
                except:
                    print("📁 模型位置: 使用系统默认位置")
                    
                print(f"🧪 模型测试成功 (embedding维度: {len(test_embedding)})")
                return True
            else:
                print("❌ 已下载的模型无法正常使用")
                return False
                
        except Exception as e2:
            print(f"❌ 模型检查也失败: {e2}")
            print("💡 建议:")
            print("   1. 检查网络连接")
            print("   2. 稍后重试")
            print("   3. 或者手动下载模型")
            return False

def check_torch_cuda():
    """检查PyTorch CUDA支持"""
    import torch
    
    print("🔍 检查PyTorch和CUDA环境...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️ CUDA不可用，将使用CPU训练")

def main():
    print("🚀 开始下载A-LLMRec所需的预训练模型...")
    
    # 检查环境
    check_torch_cuda()
    
    # 下载SentenceTransformer模型
    success = download_sentence_transformer()
    
    if success:
        print("\n🎉 所有模型下载完成!")
        print("\n📋 接下来的步骤:")
        print("1. 训练SASRec基础模型:")
        print("   python train_sasrec_ultra_light.py --dataset Movies_and_TV_ultra_light --epochs 50")
        print("2. 训练CA-LLMRec模型:")
        print("   python train_ca_llmrec_ultra_light.py --dataset Movies_and_TV_ultra_light")
    else:
        print("\n💥 模型下载失败!")
        sys.exit(1)

if __name__ == '__main__':
    main() 