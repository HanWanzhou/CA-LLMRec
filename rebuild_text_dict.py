#!/usr/bin/env python3
"""
重建A-LLMRec文本映射字典，使用真实的电影/电视剧标题
基于已有的ultra_light数据集和映射文件
"""

import json
import gzip
import pickle
from collections import defaultdict
import pandas as pd

def load_existing_mappings():
    """加载现有的映射文件"""
    print("Loading existing mappings...")
    
    # 加载映射文件
    with open('./data/amazon/Movies_and_TV_ultra_light_1000u_3000i_mappings.json', 'r') as f:
        mappings = json.load(f)
    
    print(f"Loaded mappings for {mappings['user_count']} users and {mappings['item_count']} items")
    
    # 获取物品映射 (ASIN -> item_id)
    itemmap = mappings['itemmap']  # {'asin': item_id, ...}
    
    # 检查现有的name_dict
    name_dict = mappings.get('name_dict', {'title': {}, 'description': {}})
    
    return itemmap, name_dict, mappings

def load_item_ids_from_txt():
    """从.txt文件加载所有出现的物品ID"""
    print("Loading item IDs from training data...")
    
    item_ids = set()
    with open('./data/amazon/Movies_and_TV_ultra_light_1000u_3000i.txt', 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                user_id, item_id = parts
                item_ids.add(int(item_id))
    
    print(f"Found {len(item_ids)} unique items in training data")
    return item_ids

def enhance_text_dict_with_metadata():
    """使用元数据增强文本字典"""
    print("Loading existing mappings and data...")
    itemmap, existing_name_dict, mappings = load_existing_mappings()
    valid_item_ids = load_item_ids_from_txt()
    
    print("Loading original metadata to enhance titles...")
    asin_to_info = {}
    
    # 读取原始元数据
    with gzip.open('./data/amazon/meta_Movies_and_TV.json.gz', 'rt') as f:
        for line_num, line in enumerate(f):
            if line_num % 10000 == 0:
                print(f"Processed {line_num} metadata entries...")
            
            try:
                item = json.loads(line)
                asin = item.get('asin')
                title = item.get('title', 'No Title')
                description = item.get('description', [])
                
                # 处理description字段
                if isinstance(description, list):
                    description = description[0] if len(description) > 0 else 'No Description'
                elif not description:
                    description = 'No Description'
                
                if asin:
                    asin_to_info[asin] = {
                        'title': title,
                        'description': description
                    }
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(asin_to_info)} metadata entries")
    
    # 构建增强的文本字典
    enhanced_text_dict = {
        'title': {},
        'description': {}
    }
    
    # 反向映射：item_id -> asin
    id2asin = {v: k for k, v in itemmap.items()}
    
    enhanced_count = 0
    for item_id in valid_item_ids:
        item_id_str = str(item_id)
        
        # 首先尝试从现有数据获取
        if item_id in existing_name_dict['title']:
            current_title = existing_name_dict['title'][item_id]
            current_desc = existing_name_dict['description'].get(item_id, 'No Description')
        else:
            current_title = 'No Title'
            current_desc = 'No Description'
        
        # 如果现有标题是默认值，尝试从元数据增强
        if current_title in ['No Title', 'Empty title'] and item_id in id2asin:
            asin = id2asin[item_id]
            if asin in asin_to_info:
                new_title = asin_to_info[asin]['title']
                new_desc = asin_to_info[asin]['description']
                
                if new_title and new_title != 'No Title':
                    current_title = new_title
                    enhanced_count += 1
                
                if new_desc and new_desc != 'No Description':
                    current_desc = new_desc
        
        enhanced_text_dict['title'][item_id] = current_title
        enhanced_text_dict['description'][item_id] = current_desc
    
    print(f"Enhanced {enhanced_count} items with better metadata")
    print(f"Total items in enhanced dictionary: {len(enhanced_text_dict['title'])}")
    
    # 保存增强的文本字典
    output_file = './data/amazon/Movies_and_TV_ultra_light_1000u_3000i_text_name_dict_enhanced.json'
    with open(output_file, 'wb') as f:
        pickle.dump(enhanced_text_dict, f)
    
    print(f"Saved enhanced text dictionary to: {output_file}")
    
    # 显示一些示例
    print("\nSample enhanced titles:")
    count = 0
    for item_id, title in enhanced_text_dict['title'].items():
        if title not in ['No Title', 'Empty title'] and count < 10:
            print(f"  Item {item_id}: {title}")
            count += 1
        if count >= 10:
            break
    
    # 统计信息
    real_titles_count = sum(1 for title in enhanced_text_dict['title'].values() 
                           if title not in ['No Title', 'Empty title'])
    print(f"\nStatistics:")
    print(f"  Items with real titles: {real_titles_count}/{len(enhanced_text_dict['title'])}")
    print(f"  Real title ratio: {real_titles_count/len(enhanced_text_dict['title'])*100:.1f}%")
    
    return enhanced_text_dict

if __name__ == "__main__":
    enhance_text_dict_with_metadata() 