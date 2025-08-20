#!/usr/bin/env python3
"""
CA-LLMRec 超轻量化数据集生成脚本
将Amazon Movies and TV数据集缩减到原来的5%（约1000用户，3000物品）
目标：6小时内完成全部训练
"""

import os
import sys
import json
import gzip
import random
from collections import defaultdict
from tqdm import tqdm
import argparse

# 添加项目路径
sys.path.append('.')
sys.path.append('./pre_train/sasrec')

def parse(path):
    """解析gzip压缩的JSON文件"""
    g = gzip.open(path, 'rb')
    for l in tqdm(g, desc="解析数据"):
        yield json.loads(l)

def create_ultra_light_dataset(max_users=1000, max_items=3000, min_interactions=5):
    """
    创建超轻量化数据集 - 5%数据量
    Args:
        max_users: 最大用户数量（默认1000，原始约12万）
        max_items: 最大物品数量（默认3000，原始约5万）
        min_interactions: 最小交互次数（保证数据质量）
    """
    fname = 'Movies_and_TV'
    print(f"🔧 创建超轻量化数据集: {fname}")
    print(f"  目标用户数量: {max_users}")
    print(f"  目标物品数量: {max_items}")
    print(f"  最小交互次数: {min_interactions}")
    print(f"  预计数据量: 原始的5%")
    
    # 数据文件路径
    data_dir = './data/amazon'
    reviews_file = f'{data_dir}/{fname}.json.gz'
    reviews_file_5core = f'{data_dir}/{fname}_5.json.gz'
    meta_file = f'{data_dir}/meta_{fname}.json'
    meta_file_gz = f'{data_dir}/meta_{fname}.json.gz'
    
    # 检查文件是否存在 - 优先使用5-core版本
    if os.path.exists(reviews_file_5core):
        reviews_file = reviews_file_5core
        print(f"✅ 使用5-core数据集: {reviews_file}")
    elif os.path.exists(reviews_file):
        print(f"✅ 使用完整数据集: {reviews_file}")
    else:
        print(f"❌ 数据文件不存在: {reviews_file} 或 {reviews_file_5core}")
        print("请先下载Amazon Movies and TV数据集")
        return False
    
    # 检查元数据文件是否存在 - 支持.json.gz格式
    if os.path.exists(meta_file_gz):
        meta_file = meta_file_gz
        print(f"✅ 使用压缩元数据: {meta_file}")
    elif os.path.exists(meta_file):
        print(f"✅ 使用元数据: {meta_file}")
    else:
        print(f"❌ 元数据文件不存在: {meta_file} 或 {meta_file_gz}")
        print("请先下载Amazon Movies and TV元数据")
        return False
    
    # Step 1: 统计所有用户和物品的交互次数
    print("📊 第一遍扫描：统计交互次数...")
    countU = defaultdict(int)
    countP = defaultdict(int)
    
    for l in parse(reviews_file):
        # 过滤低评分（可选）
        if l['overall'] < 3:
            continue
        asin = l['asin']
        rev = l['reviewerID']
        countU[rev] += 1
        countP[asin] += 1
    
    print(f"  原始用户数: {len(countU)}")
    print(f"  原始物品数: {len(countP)}")
    
    # Step 2: 选择最活跃的用户和最热门的物品
    print("🎯 选择核心用户和物品...")
    
    # 按交互次数排序，选择最活跃的用户
    sorted_users = sorted(countU.items(), key=lambda x: x[1], reverse=True)
    selected_users = set()
    for user, count in sorted_users:
        if len(selected_users) >= max_users:
            break
        if count >= min_interactions:
            selected_users.add(user)
    
    # 按交互次数排序，选择最热门的物品
    sorted_items = sorted(countP.items(), key=lambda x: x[1], reverse=True)
    selected_items = set()
    for item, count in sorted_items:
        if len(selected_items) >= max_items:
            break
        if count >= min_interactions:
            selected_items.add(item)
    
    print(f"  选中用户: {len(selected_users)}")
    print(f"  选中物品: {len(selected_items)}")
    
    # Step 3: 重新扫描，只保留选中的用户-物品交互
    print("🔄 第二遍扫描：构建轻量化数据集...")
    
    usermap = {}
    usernum = 0
    itemmap = {}
    itemnum = 0
    User = {}
    review_dict = {}
    name_dict = {'title': {}, 'description': {}}
    
    # 加载元数据
    print("📚 加载物品元数据...")
    meta_dict = {}
    try:
        if meta_file.endswith('.gz'):
            # 处理gzip压缩文件
            with gzip.open(meta_file, 'rt', encoding='utf-8') as f:
                for line in tqdm(f, desc="加载元数据"):
                    data = json.loads(line.strip())
                    if data['asin'] in selected_items:
                        meta_dict[data['asin']] = data
        else:
            # 处理普通JSON文件
            with open(meta_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="加载元数据"):
                    data = json.loads(line.strip())
                    if data['asin'] in selected_items:
                        meta_dict[data['asin']] = data
    except Exception as e:
        print(f"⚠️ 元数据加载失败: {e}")
        meta_dict = {}
    
    # 处理评论数据
    processed_interactions = 0
    for l in parse(reviews_file):
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']
        
        # 只处理选中的用户和物品
        if rev not in selected_users or asin not in selected_items:
            continue
        
        # 过滤低评分
        if l['overall'] < 3:
            continue
        
        # 映射用户ID
        if rev not in usermap:
            usernum += 1
            usermap[rev] = usernum
            User[usernum] = []
        userid = usermap[rev]
        
        # 映射物品ID
        if asin not in itemmap:
            itemnum += 1
            itemmap[asin] = itemnum
        itemid = itemmap[asin]
        
        # 添加交互记录
        User[userid].append([itemid, time])
        processed_interactions += 1
        
        # 保存评论信息
        review_dict[(userid, itemid)] = {
            'rating': l['overall'],
            'review': l.get('reviewText', ''),
            'summary': l.get('summary', '')
        }
        
        # 保存物品信息
        if asin in meta_dict:
            meta_data = meta_dict[asin]
            name_dict['title'][itemid] = meta_data.get('title', 'No Title')
            # 处理description字段（可能是列表格式）
            description = meta_data.get('description', [])
            if isinstance(description, list):
                if len(description) == 0:
                    name_dict['description'][itemid] = 'Empty description'
                else:
                    name_dict['description'][itemid] = description[0]
            else:
                name_dict['description'][itemid] = description if description else 'Empty description'
        else:
            name_dict['title'][itemid] = 'No Title'
            name_dict['description'][itemid] = 'Empty description'
    
    print(f"  处理交互数: {processed_interactions}")
    print(f"  最终用户数: {usernum}")
    print(f"  最终物品数: {itemnum}")
    
    # Step 4: 按时间排序用户序列
    print("⏰ 按时间排序用户序列...")
    for userid in User:
        User[userid].sort(key=lambda x: x[1])  # 按时间排序
        User[userid] = [x[0] for x in User[userid]]  # 只保留物品ID
    
    # Step 5: 保存轻量化数据集
    output_suffix = f"_ultra_light_{max_users}u_{max_items}i"
    output_file = f"{data_dir}/{fname}{output_suffix}.txt"
    
    print(f"💾 保存轻量化数据集: {output_file}")
    
    # 保存用户-物品交互对数据（符合A-LLMRec标准格式）
    with open(output_file, 'w') as f:
        for user in range(1, usernum + 1):
            if user in User and len(User[user]) > 0:
                for item in User[user]:
                    f.write(f'{user} {item}\n')  # 每行一个用户-物品对
    
    # 保存映射信息
    mapping_file = f"{data_dir}/{fname}{output_suffix}_mappings.json"
    mappings = {
        'usermap': usermap,
        'itemmap': itemmap,
        'user_count': usernum,
        'item_count': itemnum,
        'interaction_count': processed_interactions,
        'name_dict': name_dict,
        'stats': {
            'original_users': len(countU),
            'original_items': len(countP),
            'selected_users': len(selected_users),
            'selected_items': len(selected_items),
            'compression_ratio_users': len(selected_users) / len(countU),
            'compression_ratio_items': len(selected_items) / len(countP)
        }
    }
    
    with open(mapping_file, 'w') as f:
        json.dump(mappings, f, indent=2)
    
    # 保存评论信息（用于LLM训练）
    review_file = f"{data_dir}/{fname}{output_suffix}_reviews.json"
    # 转换元组键为字符串键以支持JSON序列化
    review_dict_serializable = {f"{k[0]}_{k[1]}": v for k, v in review_dict.items()}
    with open(review_file, 'w') as f:
        json.dump(review_dict_serializable, f, indent=2)
    
    # 保存物品文本信息（用于A-LLMRec训练）
    text_name_file = f"{data_dir}/{fname}{output_suffix}_text_name_dict.json.gz"
    import pickle
    with gzip.open(text_name_file, 'wb') as f:
        pickle.dump(name_dict, f)
    
    print("✅ 超轻量化数据集创建完成！")
    print(f"📊 数据集统计:")
    print(f"  用户数: {usernum} (压缩率: {len(selected_users)/len(countU):.1%})")
    print(f"  物品数: {itemnum} (压缩率: {len(selected_items)/len(countP):.1%})")
    print(f"  交互数: {processed_interactions}")
    print(f"  平均每用户交互数: {processed_interactions/usernum:.1f}")
    print(f"  数据文件: {output_file}")
    print(f"  映射文件: {mapping_file}")
    print(f"  评论文件: {review_file}")
    print(f"  文本文件: {text_name_file}")
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='创建超轻量化Amazon Movies and TV数据集')
    parser.add_argument('--max_users', type=int, default=1000, 
                       help='最大用户数量 (默认: 1000)')
    parser.add_argument('--max_items', type=int, default=3000, 
                       help='最大物品数量 (默认: 3000)')
    parser.add_argument('--min_interactions', type=int, default=5, 
                       help='最小交互次数 (默认: 5)')
    
    args = parser.parse_args()
    
    print("🚀 CA-LLMRec 超轻量化数据集生成")
    print("=" * 60)
    
    # 创建数据目录
    os.makedirs('./data/amazon', exist_ok=True)
    
    # 生成数据集
    success = create_ultra_light_dataset(
        max_users=args.max_users,
        max_items=args.max_items,
        min_interactions=args.min_interactions
    )
    
    if success:
        print("\n🎉 数据集创建成功！")
        print("现在可以使用以下命令开始训练:")
        print(f"python train_ca_llmrec_ultra_light.py --max_users {args.max_users}")
    else:
        print("\n❌ 数据集创建失败！")
        print("请检查数据文件是否存在")

if __name__ == "__main__":
    main() 