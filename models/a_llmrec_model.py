import random
import pickle
import os

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import numpy as np

from models.recsys_model import *
from models.llm4rec import *
from sentence_transformers import SentenceTransformer


class two_layer_mlp(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.fc1 = nn.Linear(dims, 128)
        self.fc2 = nn.Linear(128, dims)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x1 = self.fc2(x)
        return x, x1


class CounterfactualGenerator(nn.Module):
    """
    反事实生成器 - 基于UCR的EXPGenerator设计
    用于生成反事实序列权重，实现可解释推荐
    """
    def __init__(self, seq_len, device):
        super(CounterfactualGenerator, self).__init__()
        self.seq_len = seq_len
        self.device = device
        
        # 可学习的权重向量，用于生成反事实序列
        # 初始化为均匀分布[0,1]，表示每个历史物品的重要性
        self.delta = nn.Parameter(torch.FloatTensor(seq_len).uniform_(0, 1))
        
    def clamp_delta(self, padding_mask=None):
        """
        限制delta值在[0,1]范围内，并处理padding位置
        Args:
            padding_mask: 填充位置的mask，填充位置权重设为1（不影响）
        """
        clamped_delta = torch.clamp(self.delta, 0, 1)
        if padding_mask is not None:
            clamped_delta = clamped_delta.masked_fill(padding_mask, 1.0)
        return clamped_delta
    
    def generate_counterfactual_weights(self, batch_size=1, padding_mask=None, target_seq_len=None):
        """
        生成反事实权重
        Args:
            batch_size: 批次大小
            padding_mask: 填充mask
            target_seq_len: 目标序列长度，如果与self.seq_len不同则调整
        Returns:
            反事实权重张量 [batch_size, target_seq_len]
        """
        clamped_delta = self.clamp_delta(padding_mask)
        
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
        
        # 扩展到batch维度
        cf_weights = clamped_delta.unsqueeze(0).expand(batch_size, -1)
        return cf_weights
    
    def get_explanation_weights(self, threshold=0.5):
        """
        获取解释权重，用于生成文本解释
        Args:
            threshold: 重要性阈值
        Returns:
            重要物品的权重和索引
        """
        clamped_delta = self.clamp_delta()
        important_items = (clamped_delta < threshold).nonzero(as_tuple=True)[0]
        return important_items, clamped_delta[important_items]


class A_llmrec_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        rec_pre_trained_data = args.rec_pre_trained_data
        self.args = args
        self.device = args.device
        
        # 使用增强版的文本字典文件
        text_dict_path = f'./data/amazon/{args.rec_pre_trained_data}_text_name_dict_enhanced.json'
        # 如果增强版不存在，回退到原版
        if not os.path.exists(text_dict_path):
            text_dict_path = f'./data/amazon/{args.rec_pre_trained_data}_text_name_dict.json'
        
        with open(text_dict_path,'rb') as ft:
            self.text_name_dict = pickle.load(ft)
        
        self.recsys = RecSys(args.recsys, rec_pre_trained_data, self.device)
        self.item_num = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units
        self.sbert_dim = 768
        
        self.mlp = two_layer_mlp(self.rec_sys_dim)
        if args.pretrain_stage1:
            self.sbert = SentenceTransformer('nq-distilbert-base-v1')
            self.mlp2 = two_layer_mlp(self.sbert_dim)
        
        self.mse = nn.MSELoss()
        
        # 反事实相关组件 - CA-LLMRec扩展 (Week 2增强)
        self.maxlen = args.maxlen
        self.enable_counterfactual = getattr(args, 'enable_counterfactual', False)
        
        if self.enable_counterfactual:
            # Week 1: 基础反事实生成器
            self.cf_generator = CounterfactualGenerator(self.maxlen, self.device)
            
            # Week 2: 高级反事实生成器（可选）
            self.use_advanced_cf = getattr(args, 'use_advanced_cf', False)
            if self.use_advanced_cf:
                from models.counterfactual_utils import AdvancedCounterfactualGenerator, CounterfactualDataSampler
                self.advanced_cf_generator = AdvancedCounterfactualGenerator(self.maxlen, self.device, args)
                self.cf_data_sampler = CounterfactualDataSampler(args)
            
            # 反事实对齐损失权重
            self.cf_alignment_weight = getattr(args, 'cf_alignment_weight', 0.1)
            
            # 反事实mask比例
            self.cf_mask_ratio = getattr(args, 'cf_mask_ratio', 0.2)
            
            # Week 2新增：高级训练参数
            self.cf_learning_rate = getattr(args, 'cf_learning_rate', 0.001)
            self.cf_sparsity_weight = getattr(args, 'cf_sparsity_weight', 1.0)
            self.cf_max_grad_norm = getattr(args, 'cf_max_grad_norm', 1.0)
        self.NDCG = 0
        self.HIT = 0
        self.rec_NDCG = 0
        self.rec_HIT = 0
        self.lan_NDCG=0
        self.lan_HIT=0
        self.num_user = 0
        self.yes = 0
        
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        
        if args.pretrain_stage2 or args.inference:
            self.llm = llm4rec(device=self.device, llm_model=args.llm)
            
            self.log_emb_proj = nn.Sequential(
                nn.Linear(self.rec_sys_dim, self.llm.llm_model.config.hidden_size),
                nn.LayerNorm(self.llm.llm_model.config.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.llm.llm_model.config.hidden_size, self.llm.llm_model.config.hidden_size)
            )
            nn.init.xavier_normal_(self.log_emb_proj[0].weight)
            nn.init.xavier_normal_(self.log_emb_proj[3].weight)

            self.item_emb_proj = nn.Sequential(
                nn.Linear(128, self.llm.llm_model.config.hidden_size),
                nn.LayerNorm(self.llm.llm_model.config.hidden_size),
                nn.GELU(),
                nn.Linear(self.llm.llm_model.config.hidden_size, self.llm.llm_model.config.hidden_size)
            )
            nn.init.xavier_normal_(self.item_emb_proj[0].weight)
            nn.init.xavier_normal_(self.item_emb_proj[3].weight)
            
    def save_model(self, args, epoch1=None, epoch2=None):
        out_dir = f'./models/saved_models/'
        create_dir(out_dir)
        out_dir += f'{args.rec_pre_trained_data}_{args.recsys}_{epoch1}_'
        if args.pretrain_stage1:
            torch.save(self.sbert.state_dict(), out_dir + 'sbert.pt')
            torch.save(self.mlp.state_dict(), out_dir + 'mlp.pt')
            torch.save(self.mlp2.state_dict(), out_dir + 'mlp2.pt') 
        
        out_dir += f'{args.llm}_{epoch2}_'
        if args.pretrain_stage2:
            torch.save(self.log_emb_proj.state_dict(), out_dir + 'log_proj.pt')
            torch.save(self.item_emb_proj.state_dict(), out_dir + 'item_proj.pt')
            
    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        out_dir = f'./models/saved_models/{args.rec_pre_trained_data}_{args.recsys}_{phase1_epoch}_'
        
        mlp = torch.load(out_dir + 'mlp.pt', map_location = args.device)
        self.mlp.load_state_dict(mlp)
        del mlp
        for name, param in self.mlp.named_parameters():
            param.requires_grad = False

        if args.inference:
            out_dir += f'{args.llm}_{phase2_epoch}_'
            
            log_emb_proj_dict = torch.load(out_dir + 'log_proj.pt', map_location = args.device)
            self.log_emb_proj.load_state_dict(log_emb_proj_dict)
            del log_emb_proj_dict
            
            item_emb_proj_dict = torch.load(out_dir + 'item_proj.pt', map_location = args.device)
            self.item_emb_proj.load_state_dict(item_emb_proj_dict)
            del item_emb_proj_dict

    def find_item_text(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'
        if title_flag and description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}, {self.text_name_dict[d].get(i,d_)}"' for i in item]
        elif title_flag and not description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}"' for i in item]
        elif not title_flag and description_flag:
            return [f'"{self.text_name_dict[d].get(i,d_)}"' for i in item]
    
    def find_item_text_single(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'
        if title_flag and description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}, {self.text_name_dict[d].get(item,d_)}"'
        elif title_flag and not description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}"'
        elif not title_flag and description_flag:
            return f'"{self.text_name_dict[d].get(item,d_)}"'
        
    def get_item_emb(self, item_ids):
        with torch.no_grad():
            # 检查item_ids是否已经是张量
            if isinstance(item_ids, torch.Tensor):
                if item_ids.device != self.device:
                    item_ids = item_ids.to(self.device)
                if item_ids.dtype != torch.long:
                    item_ids = item_ids.long()
                item_embs = self.recsys.model.item_emb(item_ids)
            else:
            item_embs = self.recsys.model.item_emb(torch.LongTensor(item_ids).to(self.device))
            item_embs, _ = self.mlp(item_embs)
        
        return item_embs
    
    def generate_counterfactual_embedding(self, user_seq, item_emb, padding_mask=None, use_advanced=False):
        """
        生成反事实嵌入 - Week 2增强版
        Args:
            user_seq: 用户序列 [batch_size, seq_len]
            item_emb: 物品嵌入 [batch_size, seq_len, emb_dim]
            padding_mask: 填充mask
            use_advanced: 是否使用高级反事实生成器
        Returns:
            反事实嵌入
        """
        if not self.enable_counterfactual:
            return item_emb
            
        batch_size = user_seq.shape[0]
        
        # Week 2: 选择使用基础或高级生成器
        if use_advanced and self.use_advanced_cf:
            # 使用高级反事实生成器
            # 生成padding_ids（基于UCR格式）
            padding_ids_list = []
            for i in range(batch_size):
                seq = user_seq[i]
                padding_positions = (seq == 0).nonzero(as_tuple=False).squeeze(-1)
                padding_ids_list.append(padding_positions)
            
            # 为每个样本生成反事实权重
            cf_emb_list = []
            actual_seq_len = item_emb.size(1)
            for i in range(batch_size):
                padding_ids = padding_ids_list[i] if len(padding_ids_list[i]) > 0 else None
                cf_weights = self.advanced_cf_generator.generate_counterfactual_weights(
                    1, padding_ids, use_temperature=True, target_seq_len=actual_seq_len
                )
                # 应用权重
                cf_weights_expanded = cf_weights.unsqueeze(-1)  # [1, actual_seq_len, 1]
                cf_emb_single = item_emb[i:i+1] * cf_weights_expanded
                cf_emb_list.append(cf_emb_single)
            
            cf_emb = torch.cat(cf_emb_list, dim=0)
        else:
            # 使用基础反事实生成器 (Week 1)
            actual_seq_len = item_emb.size(1)
            cf_weights = self.cf_generator.generate_counterfactual_weights(
                batch_size, padding_mask, target_seq_len=actual_seq_len
            )
            
            cf_weights_expanded = cf_weights.unsqueeze(-1)  # [batch_size, actual_seq_len, 1]
            cf_emb = item_emb * cf_weights_expanded
        
        return cf_emb
    
    def counterfactual_alignment_loss(self, pos_emb, neg_emb, pos_text_emb, neg_text_emb, user_seq, padding_mask=None):
        """
        计算反事实对齐损失
        实现三元对齐：CF嵌入 ↔ 文本嵌入 ↔ 反事实CF嵌入
        """
        if not self.enable_counterfactual:
            return torch.tensor(0.0, device=self.device)
        
        # 确保user_seq是torch张量
        if isinstance(user_seq, np.ndarray):
            user_seq = torch.from_numpy(user_seq).to(self.device)
        elif not isinstance(user_seq, torch.Tensor):
            user_seq = torch.tensor(user_seq, device=self.device)
        
        # SASRec在mode='item'时返回reshape后的嵌入 [batch_size * seq_len, emb_dim]
        # 需要重新reshape为 [batch_size, seq_len, emb_dim]
        batch_size = user_seq.size(0)
        seq_len = user_seq.size(1)
        emb_dim = pos_emb.size(-1)
        
        # 重新reshape嵌入
        pos_emb_reshaped = pos_emb.view(batch_size, seq_len, emb_dim)
        neg_emb_reshaped = neg_emb.view(batch_size, seq_len, emb_dim)
        
        # 生成反事实嵌入
        cf_pos_emb = self.generate_counterfactual_embedding(user_seq, pos_emb_reshaped, padding_mask)
        cf_neg_emb = self.generate_counterfactual_embedding(user_seq, neg_emb_reshaped, padding_mask)
        
        # 对序列嵌入进行聚合（取最后一个非padding位置的嵌入）
        # 获取序列长度
        seq_len = cf_pos_emb.size(1)
        indices = [seq_len-1 for _ in range(cf_pos_emb.size(0))]  # 使用最后一个位置
        
        # 聚合嵌入
        cf_pos_agg = cf_pos_emb[range(cf_pos_emb.size(0)), indices]  # [batch_size, emb_dim]
        cf_neg_agg = cf_neg_emb[range(cf_neg_emb.size(0)), indices]  # [batch_size, emb_dim]
        
        # 投影到文本空间
        cf_pos_proj, _ = self.mlp(cf_pos_agg)
        cf_neg_proj, _ = self.mlp(cf_neg_agg)
        
        # 计算反事实对齐损失
        cf_alignment_loss = self.mse(cf_pos_proj, pos_text_emb.detach()) + \
                           self.mse(cf_neg_proj, neg_text_emb.detach())
        
        return cf_alignment_loss
    
    def forward(self, data, optimizer=None, batch_iter=None, mode='phase1'):
        if mode == 'phase1':
            self.pre_train_phase1(data, optimizer, batch_iter)
        if mode == 'phase2':
            self.pre_train_phase2(data, optimizer, batch_iter)
        if mode =='generate':
            self.generate(data)

    def pre_train_phase1(self,data,optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
        
        self.sbert.train()
        optimizer.zero_grad()

        u, seq, pos, neg = data
        indices = [self.maxlen*(i+1)-1 for i in range(u.shape[0])]
        
        with torch.no_grad():
            log_emb, pos_emb, neg_emb = self.recsys.model(u, seq, pos, neg, mode='item')
            
        log_emb_ = log_emb[indices]
        pos_emb_ = pos_emb[indices]
        neg_emb_ = neg_emb[indices]
        pos_ = pos.reshape(pos.size)[indices]
        neg_ = neg.reshape(neg.size)[indices]
        
        start_inx = 0
        end_inx = 60
        iterss = 0
        mean_loss = 0
        bpr_loss = 0
        gt_loss = 0
        rc_loss = 0
        text_rc_loss = 0
        cf_loss_total = 0  # 反事实损失累积
        original_loss = 0
        while start_inx < len(log_emb_):
            log_emb = log_emb_[start_inx:end_inx]
            pos_emb = pos_emb_[start_inx:end_inx]
            neg_emb = neg_emb_[start_inx:end_inx]
            
            pos__ = pos_[start_inx:end_inx]
            neg__ = neg_[start_inx:end_inx]
            
            start_inx = end_inx
            end_inx += 60
            iterss +=1
            
            pos_text = self.find_item_text(pos__)
            neg_text = self.find_item_text(neg__)
            
            pos_token = self.sbert.tokenize(pos_text)
            pos_text_embedding= self.sbert({'input_ids':pos_token['input_ids'].to(self.device),'attention_mask':pos_token['attention_mask'].to(self.device)})['sentence_embedding']
            neg_token = self.sbert.tokenize(neg_text)
            neg_text_embedding= self.sbert({'input_ids':neg_token['input_ids'].to(self.device),'attention_mask':neg_token['attention_mask'].to(self.device)})['sentence_embedding']
            
            pos_text_matching, pos_proj = self.mlp(pos_emb)
            neg_text_matching, neg_proj = self.mlp(neg_emb)
            
            pos_text_matching_text, pos_text_proj = self.mlp2(pos_text_embedding)
            neg_text_matching_text, neg_text_proj = self.mlp2(neg_text_embedding)
            
            pos_logits, neg_logits = (log_emb*pos_proj).mean(axis=1), (log_emb*neg_proj).mean(axis=1)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=pos_logits.device), torch.zeros(neg_logits.shape, device=pos_logits.device)

            loss = self.bce_criterion(pos_logits, pos_labels)
            loss += self.bce_criterion(neg_logits, neg_labels)
            
            matching_loss = self.mse(pos_text_matching,pos_text_matching_text) + self.mse(neg_text_matching,neg_text_matching_text)
            reconstruction_loss = self.mse(pos_proj,pos_emb) + self.mse(neg_proj,neg_emb)
            text_reconstruction_loss = self.mse(pos_text_proj,pos_text_embedding.data) + self.mse(neg_text_proj,neg_text_embedding.data)
            
            # 添加反事实对齐损失 - CA-LLMRec扩展
            if self.enable_counterfactual:
                # 获取完整序列嵌入用于反事实计算
                with torch.no_grad():
                    seq_log_emb, seq_pos_emb, seq_neg_emb = self.recsys.model(u, seq, pos, neg, mode='item')
                
                # 构造用户序列和padding mask
                batch_user_seq = seq
                
                # 使用序列嵌入计算反事实损失
                cf_loss = self.counterfactual_alignment_loss(
                    seq_pos_emb, seq_neg_emb,  # 使用完整序列嵌入
                    pos_text_matching_text, neg_text_matching_text,
                    batch_user_seq
                )
            else:
                cf_loss = torch.tensor(0.0, device=self.device)
            
            # 统一计算总损失
            total_loss = loss + matching_loss + 0.5*reconstruction_loss + 0.2*text_reconstruction_loss + self.cf_alignment_weight * cf_loss
            total_loss.backward()
            optimizer.step()
            
            mean_loss += total_loss.item()
            bpr_loss += loss.item()
            gt_loss += matching_loss.item()
            rc_loss += reconstruction_loss.item()
            text_rc_loss += text_reconstruction_loss.item()
            if self.enable_counterfactual:
                cf_loss_total += cf_loss.item()
            
        if self.enable_counterfactual:
            cf_loss_avg = cf_loss_total / iterss if iterss > 0 else 0
            print("CA-LLMRec loss in epoch {}/{} iteration {}/{}: {} / BPR loss: {} / Matching loss: {} / Item reconstruction: {} / Text reconstruction: {} / Counterfactual alignment: {}".format(epoch, total_epoch, step, total_step, mean_loss/iterss, bpr_loss/iterss, gt_loss/iterss, rc_loss/iterss, text_rc_loss/iterss, cf_loss_avg))
        else:
        print("loss in epoch {}/{} iteration {}/{}: {} / BPR loss: {} / Matching loss: {} / Item reconstruction: {} / Text reconstruction: {}".format(epoch, total_epoch, step, total_step, mean_loss/iterss, bpr_loss/iterss, gt_loss/iterss, rc_loss/iterss, text_rc_loss/iterss))
    
    def make_interact_text(self, interact_ids, interact_max_num):
        interact_item_titles_ = self.find_item_text(interact_ids, title_flag=True, description_flag=False)
        interact_text = []
        if interact_max_num == 'all':
            for title in interact_item_titles_:
                interact_text.append(title + '[HistoryEmb]')
        else:
            for title in interact_item_titles_[-interact_max_num:]:
                interact_text.append(title + '[HistoryEmb]')
            interact_ids = interact_ids[-interact_max_num:]
            
        interact_text = ','.join(interact_text)
        return interact_text, interact_ids
    
    def make_candidate_text(self, interact_ids, candidate_num, target_item_id, target_item_title):
        neg_item_id = []
        while len(neg_item_id)<50:
            t = np.random.randint(1, self.item_num+1)
            if not (t in interact_ids or t in neg_item_id):
                neg_item_id.append(t)
        random.shuffle(neg_item_id)
        
        candidate_ids = [target_item_id]
        candidate_text = [target_item_title + '[CandidateEmb]']

        for neg_candidate in neg_item_id[:candidate_num - 1]:
            candidate_text.append(self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + '[CandidateEmb]')
            candidate_ids.append(neg_candidate)
                
        random_ = np.random.permutation(len(candidate_text))
        candidate_text = np.array(candidate_text)[random_]
        # 保持candidate_ids为张量格式以兼容反事实解释生成
        candidate_ids_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=self.device)
        candidate_ids = candidate_ids_tensor[torch.from_numpy(random_).to(self.device)]
            
        return ','.join(candidate_text), candidate_ids
    
    def pre_train_phase2(self, data, optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
        
        optimizer.zero_grad()
        u, seq, pos, neg = data
        mean_loss = 0
        
        text_input = []
        text_output = []
        interact_embs = []
        candidate_embs = []
        self.llm.eval()
        
        with torch.no_grad():
            log_emb = self.recsys.model(u,seq,pos,neg, mode = 'log_only')
            
        for i in range(len(u)):
            target_item_id = pos[i][-1]
            target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
            
            interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10)
            candidate_num = 20
            candidate_text, candidate_ids = self.make_candidate_text(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title)
            
            input_text = ''
            input_text += ' is a user representation.'
                
            if self.args.rec_pre_trained_data == 'Movies_and_TV':
                input_text += 'This user has watched '
            elif self.args.rec_pre_trained_data == 'Video_Games':
                input_text += 'This user has played '
            elif self.args.rec_pre_trained_data == 'Luxury_Beauty' or self.args.rec_pre_trained_data == 'Toys_and_Games':
                input_text += 'This user has bought '
                
            input_text += interact_text
            
            if self.args.rec_pre_trained_data == 'Movies_and_TV':
                input_text +=' in the previous. Recommend one next movie for this user to watch next from the following movie title set, '
            elif self.args.rec_pre_trained_data == 'Video_Games':
                input_text +=' in the previous. Recommend one next game for this user to play next from the following game title set, '            
            elif self.args.rec_pre_trained_data == 'Luxury_Beauty' or self.args.rec_pre_trained_data == 'Toys_and_Games':
                input_text +=' in the previous. Recommend one next item for this user to buy next from the following item title set, '
                    
            input_text += candidate_text
            input_text += '. The recommendation is '

            text_input.append(input_text)
            text_output.append(target_item_title)

            interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)))
            candidate_embs.append(self.item_emb_proj(self.get_item_emb(candidate_ids)))
            
        samples = {'text_input': text_input, 'text_output': text_output, 'interact': interact_embs, 'candidate':candidate_embs}
        log_emb = self.log_emb_proj(log_emb)
        loss_rm = self.llm(log_emb, samples)
        loss_rm.backward()
        optimizer.step()
        mean_loss += loss_rm.item()
        print("A-LLMRec model loss in epoch {}/{} iteration {}/{}: {}".format(epoch, total_epoch, step, total_step, mean_loss))
        
    def generate(self, data):
        u, seq, pos, neg, rank = data
        
        answer = []
        text_input = []
        interact_embs = []
        candidate_embs = []
        
        # 反事实解释相关变量 - 先存储基础数据，等LLM生成后再计算
        cf_data = []
        
        with torch.no_grad():
            log_emb = self.recsys.model(u,seq,pos,neg, mode = 'log_only')
            for i in range(len(u)):
                target_item_id = pos[i]
                target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
                
                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10)
                candidate_num = 20
                candidate_text, candidate_ids = self.make_candidate_text(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title)
                
                # ========== 存储反事实解释所需的数据 ==========
                if self.enable_counterfactual:
                    cf_data.append({
                        'user_id': u[i].item(),
                        'user_sequence': seq[i][seq[i]>0],
                        'interact_ids': interact_ids,
                        'ground_truth_id': target_item_id,
                        'ground_truth_title': target_item_title,
                        'candidate_ids': candidate_ids
                    })
                else:
                    cf_data.append(None)
                
                input_text = ''
                input_text += ' is a user representation.'
                if self.args.rec_pre_trained_data == 'Movies_and_TV':
                    input_text += 'This user has watched '
                elif self.args.rec_pre_trained_data == 'Video_Games':
                    input_text += 'This user has played '
                elif self.args.rec_pre_trained_data == 'Luxury_Beauty' or self.args.rec_pre_trained_data == 'Toys_and_Games':
                    input_text += 'This user has bought '
                    
                input_text += interact_text
                
                if self.args.rec_pre_trained_data == 'Movies_and_TV':
                    input_text +=' in the previous. Recommend one next movie for this user to watch next from the following movie title set, '
                elif self.args.rec_pre_trained_data == 'Video_Games':
                    input_text +=' in the previous. Recommend one next game for this user to play next from the following game title set, '            
                elif self.args.rec_pre_trained_data == 'Luxury_Beauty' or self.args.rec_pre_trained_data == 'Toys_and_Games':
                    input_text +=' in the previous. Recommend one next item for this user to buy next from the following item title set, '
                
                input_text += candidate_text
                input_text += '. The recommendation is '
                
                answer.append(target_item_title)
                text_input.append(input_text)
                
                interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)))
                candidate_embs.append(self.item_emb_proj(self.get_item_emb(candidate_ids)))
        
        log_emb = self.log_emb_proj(log_emb)
        atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device)
        atts_llm = atts_llm.unsqueeze(1)
        log_emb = log_emb.unsqueeze(1)
        
        with torch.no_grad():
            self.llm.llm_tokenizer.padding_side = "left"
            llm_tokens = self.llm.llm_tokenizer(
                text_input,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)
            
            with torch.cuda.amp.autocast():
                inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens.input_ids)
                
                llm_tokens, inputs_embeds = self.llm.replace_hist_candi_token(llm_tokens, inputs_embeds, interact_embs, candidate_embs)
                    
                attention_mask = llm_tokens.attention_mask
                inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)
                    
                outputs = self.llm.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=False,
                    top_p=0.9,
                    temperature=1,
                    num_beams=1,
                    max_length=512,
                    min_length=1,
                    pad_token_id=self.llm.llm_tokenizer.eos_token_id,
                    repetition_penalty=1.5,
                    length_penalty=1,
                    num_return_sequences=1,
                )

            outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
            output_text = self.llm.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]

        # ========== 在LLM生成结果后计算反事实解释 ==========
        cf_explanations = []
        for i in range(len(text_input)):
            if self.enable_counterfactual and cf_data[i] is not None:
                # 使用LLM的实际预测结果进行反事实解释
                llm_prediction = output_text[i].strip('"').strip("'")  # 去除可能的引号
                cf_explanation = self.generate_counterfactual_explanation(
                    user_id=cf_data[i]['user_id'],
                    user_sequence=cf_data[i]['user_sequence'],
                    interact_ids=cf_data[i]['interact_ids'],
                    llm_prediction=llm_prediction,
                    ground_truth_title=cf_data[i]['ground_truth_title'],
                    candidate_ids=cf_data[i]['candidate_ids']
                )
                cf_explanations.append(cf_explanation)
            else:
                cf_explanations.append(None)

        for i in range(len(text_input)):
            f = open(f'./recommendation_output_cf.txt','a', encoding='utf-8')
            f.write(text_input[i])
            f.write('\n\n')
            
            f.write('Answer: '+ answer[i])
            f.write('\n\n')
            
            f.write('LLM: '+str(output_text[i]))
            f.write('\n\n')
            
            # ========== 新增：反事实解释输出 ==========
            if self.enable_counterfactual and cf_explanations[i] is not None:
                f.write('=== 反事实解释 ===\n')
                cf_exp = cf_explanations[i]
                f.write(f'权重分布: {[round(w, 3) for w in cf_exp["counterfactual_weights"]]}\n')
                f.write(f'关键物品: {cf_exp["important_items"]}\n')
                f.write(f'可移除物品: {cf_exp["removable_items"]}\n')
                f.write(f'反事实推荐: {cf_exp["counterfactual_recommendation"]}\n')
                f.write(f'解释: {cf_exp["explanation_text"]}\n')
                f.write(f'控制保真度: {cf_exp["control_fidelity"]:.3f}\n')
                f.write('\n')
            
            f.close()

        return output_text
    
    def generate_counterfactual_explanation(self, user_id, user_sequence, interact_ids, llm_prediction, ground_truth_title, candidate_ids):
        """
        为单个用户生成反事实解释
        
        Args:
            user_id: 用户ID
            user_sequence: 用户序列（非零部分）
            interact_ids: 交互物品ID列表
            llm_prediction: LLM的实际预测结果
            ground_truth_title: 真实答案标题（用于对比）
            candidate_ids: 候选物品ID列表
            
        Returns:
            反事实解释字典
        """
        try:
            # 1. 使用LLM的实际预测结果作为正常推荐
            normal_rec_title = llm_prediction
            
            # 尝试从候选物品中找到对应的ID
            normal_rec_id = None
            for cand_id in candidate_ids:
                cand_title = self.find_item_text_single(cand_id, title_flag=True, description_flag=False)
                if cand_title.strip('"').strip("'") == normal_rec_title:
                    normal_rec_id = cand_id
                    break
            
            # 如果找不到对应ID，使用第一个候选物品ID作为默认值
            if normal_rec_id is None and len(candidate_ids) > 0:
                normal_rec_id = candidate_ids[0]
            
            # 仍然需要构造序列用于反事实计算
            user_tensor = torch.tensor([user_id], device=self.device)
            seq_len = len(user_sequence)
            
            # 构造完整序列（padding到maxlen）
            full_seq = torch.zeros(self.maxlen, dtype=torch.long, device=self.device)
            if seq_len > 0:
                start_idx = max(0, self.maxlen - seq_len)
                # 确保user_sequence是torch张量
                if isinstance(user_sequence, torch.Tensor):
                    user_seq_tensor = user_sequence.to(self.device)
                else:
                    user_seq_tensor = torch.tensor(user_sequence, dtype=torch.long, device=self.device)
                full_seq[start_idx:] = user_seq_tensor[:min(seq_len, self.maxlen)]
            full_seq = full_seq.unsqueeze(0)  # [1, maxlen]
            
            # 准备候选物品张量用于反事实计算
            if isinstance(candidate_ids, torch.Tensor):
                candidate_tensor_cpu = candidate_ids.cpu()
            else:
                candidate_tensor_cpu = torch.tensor(candidate_ids, dtype=torch.long)
            
            # 2. 生成反事实权重
            if hasattr(self, 'cf_generator'):
                cf_weights = self.cf_generator.generate_counterfactual_weights(
                    batch_size=1, 
                    target_seq_len=self.maxlen
                ).squeeze(0)  # [maxlen]
                
                # 只保留有效序列位置的权重
                effective_weights = cf_weights[start_idx:] if seq_len > 0 else cf_weights[:0]
            else:
                # 如果没有cf_generator，使用简单的权重模拟
                effective_weights = torch.rand(seq_len, device=self.device)
            
            # 3. 识别关键物品和可移除物品
            threshold = 0.5
            if len(effective_weights) > 0:
                important_mask = effective_weights >= threshold
                removable_mask = effective_weights < threshold
                
                important_indices = torch.where(important_mask)[0]
                removable_indices = torch.where(removable_mask)[0]
                
                # 安全地获取物品ID
                important_items = []
                for idx in important_indices:
                    if idx.item() < len(user_sequence):
                        item_id = user_sequence[idx.item()]
                        # 处理张量和numpy数组
                        if isinstance(item_id, torch.Tensor):
                            item_id = item_id.item()
                        elif hasattr(item_id, 'item'):
                            item_id = item_id.item()
                        else:
                            item_id = int(item_id)
                        important_items.append(self.find_item_text_single(item_id, title_flag=True, description_flag=False))
                
                removable_items = []
                for idx in removable_indices:
                    if idx.item() < len(user_sequence):
                        item_id = user_sequence[idx.item()]
                        # 处理张量和numpy数组
                        if isinstance(item_id, torch.Tensor):
                            item_id = item_id.item()
                        elif hasattr(item_id, 'item'):
                            item_id = item_id.item()
                        else:
                            item_id = int(item_id)
                        removable_items.append(self.find_item_text_single(item_id, title_flag=True, description_flag=False))
            else:
                important_items = []
                removable_items = []
                important_indices = torch.tensor([], device=self.device)
                removable_indices = torch.tensor([], device=self.device)
            
            # 4. 生成反事实推荐
            if len(removable_indices) > 0:
                # 创建反事实序列（移除可移除物品）
                cf_seq = full_seq.clone()
                for idx in removable_indices:
                    if start_idx + idx.item() < self.maxlen:
                        cf_seq[0, start_idx + idx.item()] = 0
                
                # 计算反事实推荐
                with torch.no_grad():
                    # SASRec的predict方法期望CPU张量
                    user_tensor_cpu = user_tensor.cpu()
                    cf_seq_cpu = cf_seq.cpu()
                    cf_scores = self.recsys.model.predict(user_tensor_cpu, cf_seq_cpu, candidate_tensor_cpu)
                    cf_rec_idx = torch.argmax(cf_scores)
                    if isinstance(candidate_ids, torch.Tensor):
                        cf_rec_id = candidate_ids[cf_rec_idx.cpu().item()].item()
                    else:
                        cf_rec_id = candidate_ids[cf_rec_idx.cpu().item()]
                    cf_rec_title = self.find_item_text_single(cf_rec_id, title_flag=True, description_flag=False)
            else:
                # 如果没有可移除物品，反事实推荐等于正常推荐
                cf_rec_title = normal_rec_title
                cf_rec_id = normal_rec_id
            
            # 5. 计算控制保真度
            control_fidelity = 1.0 if cf_rec_id != normal_rec_id else 0.0
            
            # 6. 生成自然语言解释
            explanation_text = self.generate_natural_language_explanation(
                normal_rec_title, cf_rec_title, important_items, removable_items, control_fidelity
            )
            
            # 7. 构建解释字典
            explanation = {
                "user_id": user_id,
                "normal_recommendation": normal_rec_title,
                "counterfactual_recommendation": cf_rec_title,
                "counterfactual_weights": effective_weights.detach().cpu().numpy().tolist() if len(effective_weights) > 0 else [],
                "important_items": important_items,
                "removable_items": removable_items,
                "explanation_text": explanation_text,
                "control_fidelity": control_fidelity
            }
            
            return explanation
            
        except Exception as e:
            print(f"Warning: Failed to generate counterfactual explanation for user {user_id}: {e}")
            # 返回默认解释
            return {
                "user_id": user_id,
                "normal_recommendation": ground_truth_title,
                "counterfactual_recommendation": ground_truth_title,
                "counterfactual_weights": [],
                "important_items": [],
                "removable_items": [],
                "explanation_text": "无法生成反事实解释",
                "control_fidelity": 0.0
            }
    
    def generate_natural_language_explanation(self, normal_rec, cf_rec, important_items, removable_items, control_fidelity):
        """生成自然语言解释"""
        try:
            if control_fidelity > 0:
                # 推荐发生了变化
                important_str = "、".join(important_items[:3]) if important_items else "关键历史记录"
                removable_str = "、".join(removable_items[:3]) if removable_items else "部分历史记录"
                
                explanation = f"推荐'{normal_rec}'主要基于您对{important_str}的喜好。如果没有{removable_str}的观看历史，系统会推荐'{cf_rec}'。这说明{removable_str}对当前推荐有重要影响。"
            else:
                # 推荐没有变化
                important_str = "、".join(important_items[:3]) if important_items else "您的观看历史"
                removable_str = "、".join(removable_items[:3]) if removable_items else "部分历史记录"
                
                explanation = f"推荐'{normal_rec}'主要基于您对{important_str}的喜好。即使没有{removable_str}的观看历史，推荐结果也不会改变，说明当前推荐非常稳定。"
            
            return explanation
            
        except Exception as e:
            return f"推荐'{normal_rec}'基于您的观看偏好生成。"