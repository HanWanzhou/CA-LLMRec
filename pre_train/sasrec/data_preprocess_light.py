import os
import os.path
import gzip
import json
import pickle
from tqdm import tqdm
from collections import defaultdict
import random

def parse(path):
    g = gzip.open(path, 'rb')
    for l in tqdm(g):
        yield json.loads(l)
        
def preprocess_light(fname, max_users=5000, max_items=10000):
    """
    轻量化数据预处理 - 限制用户和物品数量
    Args:
        fname: 数据集名称 (Movies_and_TV)
        max_users: 最大用户数量（默认5000，原始约12万）
        max_items: 最大物品数量（默认10000，原始约5万）
    """
    print(f"🔧 开始轻量化预处理: {fname}")
    print(f"  限制用户数量: {max_users}")
    print(f"  限制物品数量: {max_items}")
    
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    line = 0

    file_path = f'../../data/amazon/{fname}.json.gz'
    
    # Step 1: 统计交互次数
    print("📊 统计用户和物品交互次数...")
    for l in parse(file_path):
        line += 1
        if ('Beauty' in fname) or ('Toys' in fname):
            if l['overall'] < 3:
                continue
        asin = l['asin']
        rev = l['reviewerID']
        countU[rev] += 1
        countP[asin] += 1
    
    # Step 2: 选择活跃用户和热门物品
    print("🎯 选择活跃用户和热门物品...")
    
    # 按交互次数排序，选择最活跃的用户
    sorted_users = sorted(countU.items(), key=lambda x: x[1], reverse=True)
    selected_users = set([user for user, count in sorted_users[:max_users] if count >= 5])
    
    # 按交互次数排序，选择最热门的物品
    sorted_items = sorted(countP.items(), key=lambda x: x[1], reverse=True)
    selected_items = set([item for item, count in sorted_items[:max_items] if count >= 5])
    
    print(f"  选中用户: {len(selected_users)}")
    print(f"  选中物品: {len(selected_items)}")
    
    # Step 3: 重新处理数据，只保留选中的用户和物品
    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = dict()
    review_dict = {}
    name_dict = {'title':{}, 'description':{}}
    
    # 加载元数据
    f = open(f'../../data/amazon/meta_{fname}.json', 'r')
    json_data = f.readlines()
    f.close()
    data_list = [json.loads(line[:-1]) for line in json_data]
    meta_dict = {}
    for l in data_list:
        meta_dict[l['asin']] = l
    
    print("🔄 重新处理选中的数据...")
    processed_interactions = 0
    for l in parse(file_path):
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']
        
        # 只处理选中的用户和物品
        if rev not in selected_users or asin not in selected_items:
            continue
            
        if ('Beauty' in fname) or ('Toys' in fname):
            if l['overall'] < 3:
                continue
        
        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = []
        
        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[asin] = itemid
        User[userid].append([time, itemid])
        
        processed_interactions += 1
        
        # 处理评论数据
        if itemmap[asin] in review_dict:
            try:
                review_dict[itemmap[asin]]['review'][usermap[rev]] = l['reviewText']
            except:
                pass
            try:
                review_dict[itemmap[asin]]['summary'][usermap[rev]] = l['summary']
            except:
                pass
        else:
            review_dict[itemmap[asin]] = {'review': {}, 'summary':{}}
            try:
                review_dict[itemmap[asin]]['review'][usermap[rev]] = l['reviewText']
            except:
                pass
            try:
                review_dict[itemmap[asin]]['summary'][usermap[rev]] = l['summary']
            except:
                pass
        
        # 处理元数据
        try:
            if len(meta_dict[asin]['description']) == 0:
                name_dict['description'][itemmap[asin]] = 'Empty description'
            else:
                name_dict['description'][itemmap[asin]] = meta_dict[asin]['description'][0]
            name_dict['title'][itemmap[asin]] = meta_dict[asin]['title']
        except:
            pass
    
    # 保存轻量化的文本字典
    output_suffix = f"_light_{max_users}u_{max_items}i"
    with open(f'../../data/amazon/{fname}{output_suffix}_text_name_dict.json.gz', 'wb') as tf:
        pickle.dump(name_dict, tf)
    
    # 按时间排序用户行为
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])
        
    print(f"✅ 轻量化预处理完成:")
    print(f"  最终用户数: {usernum}")
    print(f"  最终物品数: {itemnum}")
    print(f"  总交互数: {processed_interactions}")
    
    # 保存轻量化的行为数据
    f = open(f'../../data/amazon/{fname}{output_suffix}.txt', 'w')
    for user in User.keys():
        for i in User[user]:
            f.write('%d %d\n' % (user, i[1]))
    f.close()
    
    return {
        'usernum': usernum,
        'itemnum': itemnum,
        'interactions': processed_interactions,
        'output_suffix': output_suffix
    }

if __name__ == "__main__":
    # 轻量化预处理 Movies_and_TV
    result = preprocess_light('Movies_and_TV', max_users=5000, max_items=10000)
    print(f"🎉 轻量化数据集创建完成: Movies_and_TV{result['output_suffix']}") 