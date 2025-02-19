"""
目标
1. 添加动量模型,和用来跟新参数的方法
2. 构建动态列队使用hard negative 用来训练
3. 写动量模型对应的ITC loss和MLM loss
"""

from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F



class IRRA_albef_backbone(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()
        self.model_pairs=[]
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        # 1.为clip动量模型初始化一份
        self.base_model_m, _ = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.model_pairs.append([self.base_model, self.base_model_m])

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        # 动量参数
        self.momentum = args.momentum
        self.queue_size = args.queue_size
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # 创建动态队列
        self.register_buffer("image_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(self.embed_dim, self.queue_size))
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        # 温度系数
        self.temp = nn.Parameter(torch.ones([]) * args.temp)

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            #2.为id head再初始化一份动量模型
            self.classifier_m = nn.Linear(self.embed_dim, self.num_classes)
            self.model_pairs.append([self.classifier, self.classifier_m])

            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            #irr的第一层
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            #3.为cross attn初始化一份动量模型
            self.cross_attn_m = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.model_pairs.append([self.cross_attn, self.cross_attn_m])

            #跨模态transformer,默认是四层
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            #4.为跨模态transformer初始化一份动量模型
            self.cross_modal_transformer_m = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            self.model_pairs.append([self.cross_modal_transformer, self.cross_modal_transformer_m])

            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            #5.为mlm head再初始化一份动量模型
            self.mlm_head_m = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),  # 与主模型共享参数
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))   
            self.model_pairs.append([self.mlm_head, self.mlm_head_m])

            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

            # 同步参数
            self._init_momentum_models()


    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    
    def cross_former(self, q, k, v):
        #模态融合的函数
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        return x
    
    def cross_former_m(self, q, k, v):
        #动量模型的模态融合的函数
        x = self.cross_attn_m(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer_m(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        return x
    
    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def _init_momentum_models(self):
        """初始化所有动量模型参数"""
        # 设置所有动量模型参数不更新梯度
        for model_pair in self.model_pairs:
            for param_m in model_pair[1].parameters():
                param_m.requires_grad = False
        self._copy_params_to_momentum_models()

    def _copy_params_to_momentum_models(self):
        """复制所有主模型参数到对应的动量模型"""
        for model_pair in self.model_pairs:
            # model_pair[0] 是主模型，model_pair[1] 是动量模型
            for param, param_m in zip(model_pair[0].parameters(), 
                                    model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # 完全复制初始参数

    @torch.no_grad()
    def _momentum_update(self):
        """执行动量更新所有模型对"""
        for model_pair in self.model_pairs:
            # 对每个模型对中的参数进行动量更新
            for param, param_m in zip(model_pair[0].parameters(),
                                    model_pair[1].parameters()):
                # 动量更新公式
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        """更新动态队列"""
        # 收集多GPU特征
        if self.args.distributed:
            image_feats = concat_all_gather(image_feat)
            text_feats = concat_all_gather(text_feat)
        else:
            image_feats = image_feat
            text_feats = text_feat
        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)
        
        # 替换队列内容
        self.image_queue[:, ptr:ptr+batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr+batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
        
    def compute_itc_loss(self, image_feat, text_feat, image_feat_m, text_feat_m):
        """计算改进的ITC损失"""
        # 动量特征投影
        # image_feat_m = F.normalize(self.vision_proj_m(image_feat_m), dim=-1)
        # text_feat_m = F.normalize(self.text_proj_m(text_feat_m), dim=-1)
        
        # 拼接队列特征
        image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
        text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)
        
        # 计算相似度
        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp
        
        # 创建目标分布
        sim_targets = torch.zeros(sim_i2t.size()).to(image_feat.device)
        sim_targets.fill_diagonal_(1)
        
        # 计算交叉熵损失
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets, dim=1).mean()
        return (loss_i2t + loss_t2i)/2
    
    def compute_mlm_loss(self, mlm_output, mlm_labels, mlm_output_m):
        """改进的MLM损失（动量蒸馏）"""
        # 动量模型预测
        with torch.no_grad():
            soft_labels = F.softmax(mlm_output_m, dim=-1)
            
        # 软交叉熵损失
        loss = -torch.sum(soft_labels * F.log_softmax(mlm_output, dim=-1), dim=-1)
        mask = (mlm_labels != -100).float()
        return (loss * mask).mean()

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        # 动量模型前向
        with torch.no_grad():
            self._momentum_update()
            image_feats_m, text_feats_m = self.base_model_m(images, caption_ids)
        
        # 更新队列
        self._dequeue_and_enqueue(image_feats_m[:,0,:], text_feats_m[:,0,:])

        # 更新队列
        self._dequeue_and_enqueue(image_feats_m[:,0,:], text_feats_m[:,0,:])
        
        # # 投影主模型特征
        # i_feats = F.normalize(self.vision_proj(image_feats[:,0,:]), dim=-1)
        # t_feats = F.normalize(self.text_proj(text_feats[:,0,:]), dim=-1)

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            itc_loss_m = self.compute_itc_loss(i_feats, t_feats, 
                                                  image_feats_m[:,0,:], 
                                                  text_feats_m[:,0,:])
            itc_loss = objectives.compute_itc(i_feats, t_feats, logit_scale)
            ret.update({'itc_loss': (1-self.args.m_alpha)*itc_loss + self.args.m_alpha*itc_loss_m})
        
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        
        if 'mlm' in self.current_task:
             # MLM前向
            mlm_ids = batch['mlm_ids']
            mlm_feats = self.base_model.encode_text(mlm_ids)
            
            # 主模型MLM
            x_forwoard = self.cross_former(mlm_feats, image_feats, image_feats)
            mlm_output = self.mlm_head(x_forwoard)
            
            # 动量模型MLM 
            with torch.no_grad():
                mlm_feats_m = self.base_model_m.encode_text(mlm_ids)
                x_m = self.cross_former_m(mlm_feats_m, image_feats_m, image_feats_m)
                mlm_output_m = self.mlm_head_m(x_m)
            
            # 计算改进的MLM损失
            scores = mlm_output.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            mlm_loss_m = self.compute_mlm_loss(scores, mlm_labels, mlm_output_m)


            x = self.cross_former(mlm_feats, image_feats, image_feats)
            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]
            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            mlm_loss=objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight
            ret.update({'mlm_loss': (1-self.args.m_alpha)*mlm_loss + self.args.m_alpha*mlm_loss_m})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})


        return ret


def build_albef_model(args, num_classes=11003):
    model = IRRA_albef_backbone(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model

@torch.no_grad()
def concat_all_gather(tensor):
    """分布式特征收集"""
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)