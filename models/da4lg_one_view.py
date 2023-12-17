import numpy as np
import json
import os
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup

import wandb
from torchvision.models import resnet50
import clip
import random
from models.lora import LoRA_ViT_timm
from models.lora_clip import LoRA_ViT_timm as LoRA_ViT_timm_CLIP
import models.aggregator as agg
from models.clip_caption import ClipCaptionModel, ClipCaptionPrefix
import timm
from timm.models.vision_transformer import _cfg

"""
使用eccv方案的adapter
"""
K = 1
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, image_source1, image_source2, text_embedding):

        B, N, C = image_source1.shape
        q = self.wq(text_embedding).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(image_source1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(image_source2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def update_parameter(param, step_size, opt=None, reserve=False):
    flag_update = False
    if step_size is not None:
        if param is not None:
            if opt['grad_params'][0] == None:
                if not reserve:
                    del opt['grad_params'][0]
                updated_param = param
            else:
                updated_param = param - step_size * opt['grad_params'][0]
                if not reserve:
                    del opt['grad_params'][0]
            flag_update = True
    if not flag_update:
        return param

    return updated_param


class MetaLinear(nn.Linear):
    def __init__(self, in_feat, reduction_dim, bias=False):
        super().__init__(in_feat, reduction_dim, bias=bias)

    def forward(self, inputs, opt = None, reserve = False):
        if opt != None and opt['meta']:
            updated_weight = update_parameter(self.weight, self.w_step_size, opt, reserve)
            updated_bias = update_parameter(self.bias, self.b_step_size, opt, reserve)

            return F.linear(inputs, updated_weight, updated_bias)
        else:
            return F.linear(inputs, self.weight, self.bias)

class HyperRouter(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.planes = planes
        self.fc1 = MetaLinear(planes, planes//16)
        self.fc2 = MetaLinear(planes//16, planes*K)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(-1)
    
    def forward(self, x, opt=None):

        x = x.squeeze(1)
        # print("------------------------------------------xxxxxxxxxxxxxxxxx", x.shape)
        weight = self.relu(F.normalize(self.fc1(x, opt), 2, -1))
        # print("------------------------------------------weight      1", weight.shape)
        weight = self.fc2(weight, opt).reshape(-1, self.planes, K)
        # print("------------------------------------------weight      2", weight.shape)
        x = self.softmax(torch.einsum('bi,bil->bl', x, weight))
        # print("------------------------------------------x", x.shape)

        return x



class BatchNormPoint(nn.Module):
    def __init__(self, feat_size, sync_bn=False):
        super().__init__()
        self.feat_size = feat_size
        self.bn = nn.BatchNorm1d(feat_size)

    def forward(self, x):
        assert len(x.shape) == 3
        s1, s2, s3 = x.shape[0], x.shape[1], x.shape[2]
        assert s3 == self.feat_size
        x = x.view(s1 * s2, self.feat_size)
        x = self.bn(x)
        return x.view(s1, s2, s3)


class Adapter(nn.Module):
    """
    Inter-view Adapter
    """

    def __init__(self):
        super().__init__()

        self.num_views = K
        self.in_features = 512
        self.adapter_ratio = 0.6
        self.fusion_init = 0.5
        self.dropout = 0.075

        
        self.fusion_ratio = nn.Parameter(torch.tensor([self.fusion_init] * self.num_views), requires_grad=True)
        
        self.global_f = nn.Sequential(
                BatchNormPoint(self.in_features),
                nn.Dropout(self.dropout),
                nn.Flatten(),
                nn.Linear(in_features=self.in_features * self.num_views,
                          out_features=self.in_features),
                nn.BatchNorm1d(self.in_features),
                nn.ReLU(),
                nn.Dropout(self.dropout))

        self.view_f = nn.Sequential(
                nn.Linear(in_features=self.in_features,
                          out_features=self.in_features),
                nn.ReLU(),
                nn.Linear(in_features=self.in_features,
                          out_features=self.in_features * self.num_views),
                nn.ReLU())


    def forward(self, feat2, feat1):
        """
        feat1: pretrain clip domain, feat2: finetune resnet domain
        """
        # print("================self.feat1", feat1.shape)
        img_feat = feat2.reshape(-1, self.num_views, self.in_features)
        # res_feat = feat1.reshape(-1, self.num_views * self.in_features)
        # print("================self.img_feat", img_feat.shape) torch.Size([64, 8, 512])
        # print("================self.fusion_ratio", self.fusion_ratio.shape) torch.Size([8])
        
        # Global feature
        global_feat = self.global_f(img_feat * self.fusion_ratio.reshape(1, -1, 1))
        # View-wise adapted features
        view_feat = self.view_f(global_feat)
        view_feat = view_feat.reshape(-1, self.num_views, self.in_features)
        
        img_feat = view_feat * self.adapter_ratio + feat1 * (1 - self.adapter_ratio)

        return img_feat



class DA4LGOneview(LightningModule):

    def __init__(self, cfg, train_ds, val_ds):
        print(f"++++++++++++++++++++++++++++++++++++++++++ {self.__class__.__name__} ++++++++++++++++++++++++++++++++++++++++++")
        self.optimizer = None
        super().__init__()

        self.cfg = cfg
        self.train_ds = train_ds
        self.val_ds = val_ds

        self.dropout = self.cfg['train']['dropout']

        # input dimensions
        self.feats_backbone = self.cfg['train']['feats_backbone']
        self.img_feat_dim = 512
        self.lang_feat_dim = 512
        self.num_views = 1

        self.max_seq_len = 23
        self.prefix_length = 40
        # choose aggregation method
        agg_cfg = dict(self.cfg['train']['aggregator'])
        agg_cfg['input_dim'] = self.img_feat_dim
        self.aggregator_type = self.cfg['train']['aggregator']['type']
        self.aggregator = agg.names[self.aggregator_type](agg_cfg)

        # build network
        self.build_model()

        # val progress
        self.best_val_acc = -1.0
        self.best_val_res = None

        # test progress
        self.best_test_acc = -1.0
        self.best_test_res = None

        # results save path
        self.save_path = Path(os.getcwd())

        # log with wandb
        self.log_data = self.cfg['train']['log']
        if self.log_data:
            self.run = wandb.init(
                project=self.cfg['wandb']['logger']['project'],
                config=self.cfg['train'],
                settings=wandb.Settings(show_emoji=False),
                reinit=True
            )
            wandb.run.name = self.cfg['wandb']['logger']['run_name']

    def build_model(self):
        self.caption_model = ClipCaptionModel(prefix_length=self.prefix_length, clip_length=40, prefix_size=512 * 2,
                                  num_layers=8)

        # image encoder
        self.cross_attn = CrossAttention(dim=512)
        self.router = HyperRouter(planes=512)

        self.img_fc_vit_imagenet = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 512),
        )

        self.img_fc_vit = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 512),
        )

        self.domain_fc = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 512),
        )

        self.img_fc = nn.Sequential(
            nn.Identity()
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2/', 
                                                       config = 'gpt2/config.json')
        
        clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self._vit = clip_model.visual
        num_params = sum(p.numel() for p in self._vit.parameters() if p.requires_grad)
        print(f"trainable parameters: {num_params}") #trainable parameters: 86859496

        self._vit = LoRA_ViT_timm_CLIP(self._vit, r=4)
        num_params = sum(p.numel() for p in self._vit.parameters() if p.requires_grad)
        print(f"trainable parameters: {num_params}") # trainable parameters: 147456

        # language encoder
        self.lang_fc = nn.Sequential(
            nn.Identity()
        )

        self.adapter = Adapter()

        # finetuning layers for classification
        self.cls_fc = nn.Sequential(
            nn.Linear(self.img_feat_dim + self.lang_feat_dim + self.lang_feat_dim, 512),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
        )

        self.aux_fc = nn.Sequential(
            nn.Linear(self.img_feat_dim + self.lang_feat_dim, 512),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
        )
        self.bceloss = nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        params_to_optimize = [p for p in self.parameters() if p.requires_grad]
        # TODO wd = 1e-3, 1e-4, 0.01, 0.05
        # TODO 1e-3
        # TODO Big model that is regularized.

        if self.cfg['train']['optim'] == 'adam':
            self.optimizer = torch.optim.Adam(params_to_optimize, lr=self.cfg['train']['lr'],
                                              weight_decay=self.cfg['train']['weight_decay'])

        elif self.cfg['train']['optim'] == 'adamW':
            self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, params_to_optimize), lr=self.cfg['train']['lr'],
                                               weight_decay=self.cfg['train']['weight_decay'])

        # Linear scheduler.
        def linear_warmup(step):
            return min(step / self.cfg['train']['warmup_steps'], 1.0)

        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, linear_warmup)
        scheduler_cfg = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return ([self.optimizer], [scheduler_cfg])

    def smoothed_cross_entropy(self, pred, target, alpha=0.1):
        # From ShapeGlot (Achlioptas et. al)
        # https://github.com/optas/shapeglot/blob/master/shapeglot/models/neural_utils.py
        n_class = pred.size(1) + 2 # spl 将类别+1，看看能不能train的更好
        one_hot = target
        one_hot = one_hot * ((1.0 - alpha) + alpha / n_class) + (1.0 - one_hot) * alpha / n_class  # smoothed
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1)
        return torch.mean(loss)

    def _criterion(self, out):
        try:
            probs = out['probs']
            labels = out['labels']
            aux_prob = out["aux_prob"]
            ans = out["ans"]
            logits = out["logits"]
            tokens = out["tokens"]
            caption_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            
            loss = self.smoothed_cross_entropy(probs, labels)
            # print("aux_prob", aux_prob)
            # print("ans", ans)
            aux_loss = self.bceloss(aux_prob, ans)
            return {
                'loss': loss + 0.01 * aux_loss + 0.01 * caption_loss
            }
        except:
            print("probs", probs)
            print("labels", labels)
            print("aux_prob", aux_prob)
            print("ans", ans)
            
            exit()

    def forward(self, batch):
        (img1_n_raw_feats, img2_n_raw_feats), (img1_n_feats, img2_n_feats), lang_feats, ans, (key1, key2), annotation, is_visual = batch
        tokens = []
        tokens = []
        mask = []
        for _a in annotation:
            _t = torch.tensor(self.tokenizer.encode(_a), dtype=torch.int64)
            padding = self.max_seq_len - _t.shape[0]
            if padding > 0:
                _t = torch.cat((_t, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                _t = _t[:self.max_seq_len]
            _m = _t.ge(0)  # mask is zero where we out of sequence
            _t[~_m] = 0
            _m = _m.float()
            _m = torch.cat((torch.ones(self.prefix_length), _m), dim=0)  # adding prefix mask
            tokens.append(_t)
            mask.append(_m)

        tokens = torch.stack(tokens, 0).to(device=self.device)
        mask = torch.stack(mask, 0).to(device=self.device)

        
        img1_n_raw_feats = img1_n_raw_feats.to(device=self.device)
        B, NI, C, H, W = img1_n_raw_feats.size()
        img2_n_raw_feats = img2_n_raw_feats.to(device=self.device)
        B, NI, C, H, W = img2_n_raw_feats.size()

        random_indices = torch.randint(0, NI, (B,)).to(device=self.device)
        img1_n_raw_feats = img1_n_raw_feats[torch.arange(B), random_indices].unsqueeze(1)
        img2_n_raw_feats = img2_n_raw_feats[torch.arange(B), random_indices].unsqueeze(1)
        NI = 1
        # clip域
        img1_n_raw_feats_clip = img1_n_raw_feats.reshape(B * NI, C, H, W)
        img1_n_raw_feats_clip = self._vit(img1_n_raw_feats_clip)
        img1_n_raw_feats_clip = img1_n_raw_feats_clip.reshape(B, NI, -1)
        img1_n_raw_feats_clip = self.img_fc_vit(img1_n_raw_feats_clip)

        img2_n_raw_feats_clip = img2_n_raw_feats.reshape(B * NI, C, H, W)
        img2_n_raw_feats_clip = self._vit(img2_n_raw_feats_clip)
        img2_n_raw_feats_clip = img2_n_raw_feats_clip.reshape(B, NI, -1)
        img2_n_raw_feats_clip = self.img_fc_vit(img2_n_raw_feats_clip)


        # to device
        img1_n_feats = img1_n_feats.to(device=self.device).float()
        img2_n_feats = img2_n_feats.to(device=self.device).float()
        lang_feats = lang_feats.to(device=self.device).float()

        img2_n_feats = img2_n_feats[torch.arange(B), random_indices].unsqueeze(1)
        img1_n_feats = img1_n_feats[torch.arange(B), random_indices].unsqueeze(1)

        # cross_attention
        output = self.cross_attn(torch.cat([img1_n_raw_feats_clip, img2_n_raw_feats_clip], dim=1), 
                                 torch.cat([img1_n_feats, img2_n_feats], dim=1), 
                                 lang_feats
                                 )
        output = self.router(output)
        # print("---------------------------", output.shape) torch.Size([128, 16])

        # joint visual features
        joint_feats_1 = self.adapter(img1_n_feats, img1_n_raw_feats_clip)
        joint_feats_2 = self.adapter(img2_n_feats, img2_n_raw_feats_clip)
        # print("joint_feats_1, joint_feats_1", joint_feats_1.shape, joint_feats_1.shape)

        # joint_feats = torch.cat([joint_feats_1, joint_feats_2], dim=-1)
        # joint_feats = joint_feats * output
        # print("-------------------joint_feats_1 * output", joint_feats_1.shape, output.shape)
        joint_feats_1_router = joint_feats_1 * output.unsqueeze(-1)
        joint_feats_2_router = joint_feats_2 * output.unsqueeze(-1)
        # joint visual features
        joint_feats_1_adapter = self.adapter(img1_n_feats, img1_n_raw_feats_clip)
        joint_feats_2_adapter = self.adapter(img2_n_feats, img2_n_raw_feats_clip)
        # print("joint_feats_1, joint_feats_1", joint_feats_1.shape, joint_feats_1.shape)

        
        # aggregate
        img1_feats = self.aggregator(img1_n_feats)
        img2_feats = self.aggregator(img2_n_feats)

        img1_n_raw_feats = self.aggregator(img1_n_raw_feats)
        img2_n_raw_feats = self.aggregator(img2_n_raw_feats)

        joint_feats_1_router = self.aggregator(joint_feats_1_router)
        joint_feats_2_router = self.aggregator(joint_feats_2_router)

        joint_feats_1_adapter = self.aggregator(joint_feats_1_adapter)
        joint_feats_2_adapter = self.aggregator(joint_feats_2_adapter)
        
        # lang encoding
        lang_enc = self.lang_fc(lang_feats)

        # normalize
        if self.cfg['train']['normalize_feats']:
            img1_feats = img1_feats / img1_feats.norm(dim=-1, keepdim=True)
            img2_feats = img2_feats / img2_feats.norm(dim=-1, keepdim=True)
            lang_enc = lang_enc / lang_enc.norm(dim=-1, keepdim=True)

            img1_n_raw_feats = img1_n_raw_feats / img1_n_raw_feats.norm(dim=-1, keepdim=True)
            img2_n_raw_feats = img2_n_raw_feats / img2_n_raw_feats.norm(dim=-1, keepdim=True)

            joint_feats_1_router = joint_feats_1_router / joint_feats_1_router.norm(dim=-1, keepdim=True)
            joint_feats_2_router = joint_feats_2_router / joint_feats_2_router.norm(dim=-1, keepdim=True)

            joint_feats_1_adapter = joint_feats_1_adapter / joint_feats_1_adapter.norm(dim=-1, keepdim=True)
            joint_feats_2_adapter = joint_feats_2_adapter / joint_feats_2_adapter.norm(dim=-1, keepdim=True)

        # caption model
        domain_feats_1 = torch.cat([joint_feats_1_router, joint_feats_1_adapter], dim=1)
        domain_feats_2 = torch.cat([joint_feats_2_router, joint_feats_2_adapter], dim=1)

        domain_feats_1 = self.domain_fc(domain_feats_1)
        domain_feats_2 = self.domain_fc(domain_feats_2)
        outputs = self.caption_model(tokens=tokens,prefix=torch.cat([domain_feats_1, domain_feats_2], dim=1), mask=mask)
        logits = outputs.logits[:, self.prefix_length - 1: -1]


        # img1 prob
        # print(f"{img1_feats.size(), img1_n_raw_feats.size()}")
        img1_enc = self.img_fc(img1_feats)
        img1_prob = self.cls_fc(torch.cat([img1_enc, domain_feats_1, lang_enc ], dim=-1))

        # img2 prob
        img2_enc = self.img_fc(img2_feats)
        img2_prob = self.cls_fc(torch.cat([img2_enc, domain_feats_2, lang_enc], dim=-1))

        # aux prob
        if random.randint(0,1) == 0:
            aux_prob = self.aux_fc(torch.cat([domain_feats_1, lang_enc], dim=-1)).squeeze(1)
        else:
            aux_prob = self.aux_fc(torch.cat([domain_feats_2, lang_enc], dim=-1)).squeeze(1)

        # cat probs
        probs = torch.cat([img1_prob, img2_prob], dim=-1)

        # num steps taken (8 for all views)
        bs = lang_enc.shape[0]
        num_steps = torch.ones((bs)).to(dtype=torch.long, device=lang_enc.device)
        if self.aggregator_type in ['maxpool', 'mean', 'gru']:
            num_steps = num_steps * 8
        elif self.aggregator_type in ['two_random_index']:
            num_steps = num_steps * 2

        test_mode = (ans[0] == -1)
        if not test_mode:
            # one-hot labels of answers
            labels = F.one_hot(ans)

            return {
                'probs': probs,
                'labels': labels,
                'is_visual': is_visual,
                'num_steps': num_steps,
                "aux_prob": aux_prob,
                "ans": ans.float(),
                "logits": logits,
                "tokens": tokens
            }
        else:
            return {
                'probs': probs,
                'num_steps': num_steps,
                "aux_prob": aux_prob,
                "ans": ans.float(),
                "logits": logits,
                "tokens": tokens
            }

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)

        # classifier loss
        losses = self._criterion(out)

        if self.log_data:
            wandb.log({
                'tr/loss': losses['loss'],
            })
        self.log("train_loss", losses['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return dict(
            loss=losses['loss']
        )

    def check_correct(self, b, labels, probs):
        right_prob = probs[b][labels[b].argmax()]
        wrong_prob = probs[b][labels[b].argmin()]
        correct = right_prob > wrong_prob
        return correct

    def validation_step(self, batch, batch_idx):
        all_view_results = {}
        for view in range(self.num_views):
            out = self.forward(batch)
            losses = self._criterion(out)

            loss = losses['loss']
            probs = out['probs']
            labels = out['labels']
            visual = out['is_visual']
            num_steps = out['num_steps']

            probs = F.softmax(probs, dim=-1)
            metrics = self.compute_metrics(labels, loss, probs, visual, num_steps)
            all_view_results[view] = metrics

        mean_val_loss = np.mean([m['val_loss'].detach().cpu().float() for m in all_view_results.values()])
        mean_val_acc = np.mean([m['val_acc'] for m in all_view_results.values()])

        return dict(
            val_loss=mean_val_loss,
            val_acc=mean_val_acc,
            all_view_results=all_view_results,
        )

    def compute_metrics(self, labels, loss, probs, visual, num_steps):
        batch_size = probs.shape[0]
        val_total, val_correct, val_pl_correct = 0, 0, 0.
        visual_total, visual_correct, pl_visual_correct = 0, 0, 0.
        nonvis_total, nonvis_correct, pl_nonvis_correct = 0, 0, 0.
        for b in range(batch_size):
            correct = self.check_correct(b, labels, probs)

            if correct:
                val_correct += 1
                val_pl_correct += 1. / num_steps[b]
            val_total += 1

            if bool(visual[b]):
                if correct:
                    visual_correct += 1
                    pl_visual_correct += 1. / num_steps[b]
                visual_total += 1
            else:
                if correct:
                    nonvis_correct += 1
                    pl_nonvis_correct += 1. / num_steps[b]
                nonvis_total += 1

        val_acc = float(val_correct) / val_total
        val_pl_acc = float(val_pl_correct) / val_total
        val_visual_acc = float(visual_correct) / visual_total
        val_pl_visual_acc = float(pl_visual_correct) / visual_total
        val_nonvis_acc = float(nonvis_correct) / nonvis_total
        val_pl_nonvis_acc = float(pl_nonvis_correct) / nonvis_total

        return dict(
            val_loss=loss,
            val_acc=val_acc,
            val_pl_acc=val_pl_acc,
            val_correct=val_correct,
            val_pl_correct=val_pl_correct,
            val_total=val_total,
            val_visual_acc=val_visual_acc,
            val_pl_visual_acc=val_pl_visual_acc,
            val_visual_correct=visual_correct,
            val_pl_visual_correct=pl_visual_correct,
            val_visual_total=visual_total,
            val_nonvis_acc=val_nonvis_acc,
            val_pl_nonvis_acc=val_pl_nonvis_acc,
            val_nonvis_correct=nonvis_correct,
            val_pl_nonvis_correct=pl_nonvis_correct,
            val_nonvis_total=nonvis_total,
        )

    def validation_epoch_end(self, all_outputs, mode='vl'):
        n_view_res = {}
        sanity_check = True
        for view in range(self.num_views):

            view_res = {
                'val_loss': 0.0,

                'val_correct': 0,
                'val_pl_correct': 0,
                'val_total': 0,

                'val_visual_correct': 0,
                'val_pl_visual_correct': 0,
                'val_visual_total': 0,

                'val_nonvis_correct': 0,
                'val_pl_nonvis_correct': 0,
                'val_nonvis_total': 0,
            }

            for output in all_outputs:
                metrics = output['all_view_results'][view]

                view_res['val_loss'] += metrics['val_loss'].item()

                view_res['val_correct'] += metrics['val_correct']
                view_res['val_pl_correct'] += int(metrics['val_pl_correct'])
                view_res['val_total'] += metrics['val_total']
                if view_res['val_total'] > 128:
                    sanity_check = False

                view_res['val_visual_correct'] += metrics['val_visual_correct']
                view_res['val_pl_visual_correct'] += int(metrics['val_pl_visual_correct'])
                view_res['val_visual_total'] += metrics['val_visual_total']

                view_res['val_nonvis_correct'] += metrics['val_nonvis_correct']
                view_res['val_pl_nonvis_correct'] += int(metrics['val_pl_nonvis_correct'])
                view_res['val_nonvis_total'] += metrics['val_nonvis_total']

            view_res['val_loss'] = float(view_res['val_loss']) / len(all_outputs)

            view_res['val_acc'] = float(view_res['val_correct']) / view_res['val_total']
            view_res['val_pl_acc'] = float(view_res['val_pl_correct']) / view_res['val_total']

            view_res['val_visual_acc'] = float(view_res['val_visual_correct']) / view_res['val_visual_total']
            view_res['val_pl_visual_acc'] = float(view_res['val_pl_visual_correct']) / view_res['val_visual_total']

            view_res['val_nonvis_acc'] = float(view_res['val_nonvis_correct']) / view_res['val_nonvis_total']
            view_res['val_pl_nonvis_acc'] = float(view_res['val_pl_nonvis_correct']) / view_res['val_nonvis_total']

            n_view_res[view] = view_res

        mean_val_loss = np.mean([r['val_loss'] for r in n_view_res.values()])

        val_acc = sum([r['val_correct'] for r in n_view_res.values()]) / float(sum([r['val_total'] for r in n_view_res.values()]))
        val_visual_acc = sum([r['val_visual_correct'] for r in n_view_res.values()]) / float(sum([r['val_visual_total'] for r in n_view_res.values()]))
        val_nonvis_acc = sum([r['val_nonvis_correct'] for r in n_view_res.values()]) / float(sum([r['val_nonvis_total'] for r in n_view_res.values()]))

        val_pl_acc = sum([r['val_pl_correct'] for r in n_view_res.values()]) / float(sum([r['val_total'] for r in n_view_res.values()]))
        val_pl_visual_acc = sum([r['val_pl_visual_correct'] for r in n_view_res.values()]) / float(sum([r['val_visual_total'] for r in n_view_res.values()]))
        val_pl_nonvis_acc = sum([r['val_pl_nonvis_correct'] for r in n_view_res.values()]) / float(sum([r['val_nonvis_total'] for r in n_view_res.values()]))

        self.log("val_acc", val_acc, on_step=False, on_epoch=True, prog_bar=False, logger=False)

        res = {
            f'{mode}/loss': mean_val_loss,
            f'{mode}/acc': val_acc,
            f'{mode}/acc_visual': val_visual_acc,
            f'{mode}/acc_nonvis': val_nonvis_acc,
            f'{mode}/pl_acc': val_pl_acc,
            f'{mode}/pl_acc_visual': val_pl_visual_acc,
            f'{mode}/pl_acc_nonvis': val_pl_nonvis_acc,
            f'{mode}/all_view_res': n_view_res,
        }

        if not sanity_check:  # only check best conditions and dump data if this isn't a sanity check

            # test (ran once at the end of training)
            if mode == 'test':
                self.best_test_res = dict(res)

            # val (keep track of best results)
            else:
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_val_res = dict(res)

            # results to save
            results_dict = self.best_test_res if mode == 'test' else self.best_val_res

            best_loss = results_dict[f'{mode}/loss']
            best_acc = results_dict[f'{mode}/acc']
            best_acc_visual = results_dict[f'{mode}/acc_visual']
            best_acc_nonvis = results_dict[f'{mode}/acc_nonvis']
            best_pl_acc = results_dict[f'{mode}/pl_acc']
            best_pl_acc_visual = results_dict[f'{mode}/pl_acc_visual']
            best_pl_acc_nonvis = results_dict[f'{mode}/pl_acc_nonvis']

            seed = self.cfg['train']['random_seed']
            json_file = os.path.join(self.save_path, f'{mode}-results-{seed}.json')

            # save results
            with open(json_file, 'w') as f:
                json.dump(results_dict, f, sort_keys=True, indent=4)

            # print best result
            print("\nBest-----:")
            print(f'Best {mode} Acc: {best_acc:0.5f} ({best_pl_acc:0.5f}) | Visual {best_acc_visual:0.5f} ({best_pl_acc_visual:0.5f}) | Nonvis: {best_acc_nonvis:0.5f} ({best_pl_acc_nonvis:0.5f}) | Val Loss: {best_loss:0.8f} ')
            print("------------")

        if self.log_data:
            wandb.log(res)
        return dict(
            val_loss=mean_val_loss,
            val_acc=val_acc,
            val_visual_acc=val_visual_acc,
            val_nonvis_acc=val_nonvis_acc,
            val_pl_acc=val_pl_acc,
            val_pl_visual_acc=val_pl_visual_acc,
            val_pl_nonvis_acc=val_pl_nonvis_acc,
        )


    def test_step(self, batch, batch_idx):
        all_view_results = {}
        for view in range(self.num_views):
            out = self.forward(batch)
            losses = self._criterion(out)

            loss = losses['loss']
            probs = out['probs']
            labels = out['labels']
            visual = out['is_visual']
            num_steps = out['num_steps']

            probs = F.softmax(probs, dim=-1)
            metrics = self.compute_metrics(labels, loss, probs, visual, num_steps)
            all_view_results[view] = metrics

        mean_val_loss = np.mean([m['val_loss'].detach().cpu().float() for m in all_view_results.values()])
        mean_val_acc = np.mean([m['val_acc'] for m in all_view_results.values()])

        return dict(
            val_loss=mean_val_loss,
            val_acc=mean_val_acc,
            all_view_results=all_view_results,
        )
    
    def test_epoch_end(self, all_outputs, mode='vl'):
        n_view_res = {}
        sanity_check = True
        for view in range(self.num_views):

            view_res = {
                'val_loss': 0.0,

                'val_correct': 0,
                'val_pl_correct': 0,
                'val_total': 0,

                'val_visual_correct': 0,
                'val_pl_visual_correct': 0,
                'val_visual_total': 0,

                'val_nonvis_correct': 0,
                'val_pl_nonvis_correct': 0,
                'val_nonvis_total': 0,
            }

            for output in all_outputs:
                metrics = output['all_view_results'][view]

                view_res['val_loss'] += metrics['val_loss'].item()

                view_res['val_correct'] += metrics['val_correct']
                view_res['val_pl_correct'] += int(metrics['val_pl_correct'])
                view_res['val_total'] += metrics['val_total']
                if view_res['val_total'] > 128:
                    sanity_check = False

                view_res['val_visual_correct'] += metrics['val_visual_correct']
                view_res['val_pl_visual_correct'] += int(metrics['val_pl_visual_correct'])
                view_res['val_visual_total'] += metrics['val_visual_total']

                view_res['val_nonvis_correct'] += metrics['val_nonvis_correct']
                view_res['val_pl_nonvis_correct'] += int(metrics['val_pl_nonvis_correct'])
                view_res['val_nonvis_total'] += metrics['val_nonvis_total']

            view_res['val_loss'] = float(view_res['val_loss']) / len(all_outputs)

            view_res['val_acc'] = float(view_res['val_correct']) / view_res['val_total']
            view_res['val_pl_acc'] = float(view_res['val_pl_correct']) / view_res['val_total']

            view_res['val_visual_acc'] = float(view_res['val_visual_correct']) / view_res['val_visual_total']
            view_res['val_pl_visual_acc'] = float(view_res['val_pl_visual_correct']) / view_res['val_visual_total']

            view_res['val_nonvis_acc'] = float(view_res['val_nonvis_correct']) / view_res['val_nonvis_total']
            view_res['val_pl_nonvis_acc'] = float(view_res['val_pl_nonvis_correct']) / view_res['val_nonvis_total']

            n_view_res[view] = view_res

        mean_val_loss = np.mean([r['val_loss'] for r in n_view_res.values()])

        val_acc = sum([r['val_correct'] for r in n_view_res.values()]) / float(sum([r['val_total'] for r in n_view_res.values()]))
        val_visual_acc = sum([r['val_visual_correct'] for r in n_view_res.values()]) / float(sum([r['val_visual_total'] for r in n_view_res.values()]))
        val_nonvis_acc = sum([r['val_nonvis_correct'] for r in n_view_res.values()]) / float(sum([r['val_nonvis_total'] for r in n_view_res.values()]))

        val_pl_acc = sum([r['val_pl_correct'] for r in n_view_res.values()]) / float(sum([r['val_total'] for r in n_view_res.values()]))
        val_pl_visual_acc = sum([r['val_pl_visual_correct'] for r in n_view_res.values()]) / float(sum([r['val_visual_total'] for r in n_view_res.values()]))
        val_pl_nonvis_acc = sum([r['val_pl_nonvis_correct'] for r in n_view_res.values()]) / float(sum([r['val_nonvis_total'] for r in n_view_res.values()]))

        self.log("val_acc", val_acc, on_step=False, on_epoch=True, prog_bar=False, logger=False)

        res = {
            f'{mode}/loss': mean_val_loss,
            f'{mode}/acc': val_acc,
            f'{mode}/acc_visual': val_visual_acc,
            f'{mode}/acc_nonvis': val_nonvis_acc,
            f'{mode}/pl_acc': val_pl_acc,
            f'{mode}/pl_acc_visual': val_pl_visual_acc,
            f'{mode}/pl_acc_nonvis': val_pl_nonvis_acc,
            f'{mode}/all_view_res': n_view_res,
        }

        if not sanity_check:  # only check best conditions and dump data if this isn't a sanity check

            # test (ran once at the end of training)
            if mode == 'test':
                self.best_test_res = dict(res)

            # val (keep track of best results)
            else:
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_val_res = dict(res)

            # results to save
            results_dict = self.best_test_res if mode == 'test' else self.best_val_res

            best_loss = results_dict[f'{mode}/loss']
            best_acc = results_dict[f'{mode}/acc']
            best_acc_visual = results_dict[f'{mode}/acc_visual']
            best_acc_nonvis = results_dict[f'{mode}/acc_nonvis']
            best_pl_acc = results_dict[f'{mode}/pl_acc']
            best_pl_acc_visual = results_dict[f'{mode}/pl_acc_visual']
            best_pl_acc_nonvis = results_dict[f'{mode}/pl_acc_nonvis']

            seed = self.cfg['train']['random_seed']
            json_file = os.path.join(self.save_path, f'{mode}-results-{seed}.json')

            # save results
            with open(json_file, 'w') as f:
                json.dump(results_dict, f, sort_keys=True, indent=4)

            # print best result
            print("\nBest-----:")
            print(f'Best {mode} Acc: {best_acc:0.5f} ({best_pl_acc:0.5f}) | Visual {best_acc_visual:0.5f} ({best_pl_acc_visual:0.5f}) | Nonvis: {best_acc_nonvis:0.5f} ({best_pl_acc_nonvis:0.5f}) | Val Loss: {best_loss:0.8f} ')
            print("------------")

        if self.log_data:
            wandb.log(res)
        return dict(
            val_loss=mean_val_loss,
            val_acc=val_acc,
            val_visual_acc=val_visual_acc,
            val_nonvis_acc=val_nonvis_acc,
            val_pl_acc=val_pl_acc,
            val_pl_visual_acc=val_pl_visual_acc,
            val_pl_nonvis_acc=val_pl_nonvis_acc,
        )




"""
    def test_step(self, batch, batch_idx):
        all_view_results = {}
        for view in range(self.num_views):
            out = self.forward(batch)
            probs = out['probs']
            num_steps = out['num_steps']
            objects = batch[3]
            annotation = batch[4]

            probs = F.softmax(probs, dim=-1)
            pred_ans = probs.argmax(-1)

            all_view_results[view] = dict(
                annotation=annotation,
                objects=objects,
                pred_ans=pred_ans,
                num_steps=num_steps,
            )

        return dict(
            all_view_results=all_view_results,
        )

    def test_epoch_end(self, all_outputs, mode='test'):
        test_results = {v: list() for v in range(self.num_views)}

        for out in all_outputs:
            for view in range(self.num_views):
                view_res = out['all_view_results']
                bs = view_res[view]['pred_ans'].shape[0]
                for b in range(bs):
                    test_results[view].append({
                        'annotation': view_res[view]['annotation'][b],
                        'objects': (
                            view_res[view]['objects'][0][b],
                            view_res[view]['objects'][1][b],
                        ),
                        'pred_ans': int(view_res[view]['pred_ans'][b]),
                        'num_steps': int(view_res[view]['num_steps'][b]),
                    })

        test_pred_save_path = self.save_path
        if not os.path.exists(test_pred_save_path):
            os.makedirs(test_pred_save_path)

        model_type = self.__class__.__name__.lower()
        json_file = os.path.join(test_pred_save_path, f'{model_type}_test_results.json')
        with open(json_file, 'w') as f:
            json.dump(test_results, f, sort_keys=True, indent=4)
"""

    