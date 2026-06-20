import torch
import torch.nn as nn
import torch.nn.functional as F
# import pdb;pdb.set_trace()
import math
import numpy as np
from maskrcnn_benchmark.modeling.utils import cat, concat_box_prediction_layers, permute_and_flatten
from timm.models.layers import DropPath
import umap
import matplotlib.pyplot as plt
from transformers.activations import ACT2FN
from datetime import datetime
import os
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def _make_conv(input_dim, output_dim, k, stride=1):
    pad = (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim, (k, k), padding=(pad, pad), stride=(stride, stride)),
        nn.BatchNorm2d(output_dim),
        nn.ReLU(inplace=True)
    )


def _make_mlp(input_dim, output_dim, drop):
    return nn.Sequential(nn.Linear(input_dim, output_dim),
                         nn.BatchNorm1d(output_dim),
                         nn.ReLU(inplace=True),
                         nn.Dropout(drop),
                         nn.Linear(output_dim, output_dim),
                         nn.BatchNorm1d(output_dim),
                         nn.ReLU(inplace=True))


def _make_coord(batch, height, width):
    # relative position encoding
    xv, yv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
    xv_min = (xv.float() * 2 - width) / width
    yv_min = (yv.float() * 2 - height) / height
    xv_max = ((xv + 1).float() * 2 - width) / width
    yv_max = ((yv + 1).float() * 2 - height) / height
    xv_ctr = (xv_min + xv_max) / 2
    yv_ctr = (yv_min + yv_max) / 2
    hmap = torch.ones(height, width) * (1. / height)
    wmap = torch.ones(height, width) * (1. / width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0), \
                                               xv_max.unsqueeze(0), yv_max.unsqueeze(0), \
                                               xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0), \
                                               hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0))
    coord = coord.unsqueeze(0).repeat(batch, 1, 1, 1)
    return coord


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def func_attention(query, context, smooth=1, raw_feature_norm="softmax", eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax()(attn * smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT

def umap_two_matrices_and_save(query_states, key_states, output_dir='umap_results'):
    """
    将两个矩阵的特征降维并绘制在一张图上
    
    参数:
    query_states: 查询矩阵 (形状为 [batch_size, num_features_query, feature_dim])
    key_states: 关键矩阵 (形状为 [batch_size, num_features_key, feature_dim])
    output_dir: 保存图像的目录
    """
    # 创建保存图像的目录
    os.makedirs(output_dir, exist_ok=True)

    # 将矩阵转换为 numpy 数组
    query_states_np = query_states.cpu().numpy()
    key_states_np = key_states.cpu().numpy()

    # 获取当前时间
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 为每个 head 分别应用 UMAP
    for head_idx in range(query_states_np.shape[0]):
        if head_idx == 13:
            # 取出每个 head 的特征
            query_state_head = query_states_np[head_idx].reshape(-1, query_states_np.shape[-1])
            key_state_head = key_states_np[head_idx].reshape(-1, key_states_np.shape[-1])

            # 结合两个特征矩阵
            combined_data = np.vstack((query_state_head, key_state_head))

            # 应用 UMAP
            reducer = umap.UMAP()
            embedding = reducer.fit_transform(combined_data)

            # 分别绘制并保存图像
            plt.figure()
            plt.scatter(embedding[:query_state_head.shape[0], 0], embedding[:query_state_head.shape[0], 1], s=5, label='Query States')
            plt.scatter(embedding[query_state_head.shape[0]:, 0], embedding[query_state_head.shape[0]:, 1], s=5, label='Key States')
            plt.legend()
            plt.title(f'UMAP projection for head {head_idx}')
            save_path = os.path.join(output_dir, f'umap_projection_head{head_idx}_{current_time}.png')
            plt.savefig(save_path)
            plt.close()   

def umap_three_matrices_and_save(matrix1, matrix2, matrix3, output_dir='umap_results'):
    """
    将三个矩阵的特征降维并绘制在一张图上
    
    参数:
    matrix1, matrix2, matrix3: 待降维的三个矩阵 (形状为 [batch_size, num_features, feature_dim])
    output_dir: 保存图像的目录
    """
    # 创建保存图像的目录
    os.makedirs(output_dir, exist_ok=True)

    # 将矩阵转换为 numpy 数组
    matrix1_np = matrix1.cpu().numpy()
    matrix2_np = matrix2.cpu().numpy()
    matrix3_np = matrix3.cpu().numpy()

    # 获取当前时间
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 为每个 head 分别应用 UMAP
    for head_idx in range(matrix1_np.shape[0]):
        if head_idx == 13:
            # 取出每个 head 的特征
            matrix1_head = matrix1_np[head_idx].reshape(-1, matrix1_np.shape[-1])
            matrix2_head = matrix2_np[head_idx].reshape(-1, matrix2_np.shape[-1])
            matrix3_head = matrix3_np[head_idx].reshape(-1, matrix3_np.shape[-1])

            # 结合三个特征矩阵
            combined_data = np.vstack((matrix1_head, matrix2_head, matrix3_head))

            # 应用 UMAP
            reducer = umap.UMAP()
            embedding = reducer.fit_transform(combined_data)

            # 绘制并保存图像
            plt.figure()
            plt.scatter(embedding[:matrix1_head.shape[0], 0], embedding[:matrix1_head.shape[0], 1], s=5, label='Combined Pyramid V States')
            plt.scatter(embedding[matrix1_head.shape[0]:matrix1_head.shape[0] + matrix2_head.shape[0], 0], embedding[matrix1_head.shape[0]:matrix1_head.shape[0] + matrix2_head.shape[0], 1], s=5, label='V Selected States')
            plt.scatter(embedding[matrix1_head.shape[0] + matrix2_head.shape[0]:, 0], embedding[matrix1_head.shape[0] + matrix2_head.shape[0]:, 1], s=5, label='L States')
            plt.legend()
            plt.title(f'UMAP projection for head {head_idx}')
            save_path = os.path.join(output_dir, f'umap_projection_head{head_idx}_{current_time}.png')
            plt.savefig(save_path)
            plt.close()
            
def save_attn_weights_to_file(attn_weights, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'ab') as f:
        np.save(f, attn_weights.cpu().numpy())

def append_float_to_npy(float_value, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    float_value_array = np.atleast_1d(float_value.cpu().numpy())  
    if os.path.exists(filename):
        
        existing_data = np.load(filename)
        
        existing_data = np.atleast_1d(existing_data)  
        new_data = np.concatenate((existing_data, float_value_array), axis=0)
    else:
        
        new_data = float_value_array
    
    with open(filename, 'wb') as f:
        np.save(f, new_data)
                                    
class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = cfg.MODEL.DYHEAD.FUSE_CONFIG.STABLE_SOFTMAX_2D
        self.clamp_min_for_underflow = cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW
        self.clamp_max_for_overflow = cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, l, attention_mask_l=None):
        bsz, tgt_len, embed_dim = v.size()

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape) #[8*1, 18134, 256]
        value_l_states = value_l_states.view(*proj_shape) #[8*1, 256, 256]
        
        batch_size, num_features_key, feature_dim = key_states.shape
    
        # 计算 attention_mask_l 中值为 1 的数量
        num_selected_features = attention_mask_l.sum(dim=1).max().item()
        
        # 使用掩码选择特征
        selected_key_states = []
        for batch_idx in range(batch_size):
            selected_features = key_states[batch_idx, attention_mask_l[0].bool()]
            selected_key_states.append(selected_features[:num_selected_features])
        
        # 将列表转换为张量
        selected_key_states = torch.stack(selected_key_states)
        attn_l = torch.bmm(query_states, selected_key_states.transpose(1, 2))

        #query_states: [8*1, 18134, 256]
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        # attn_weights_l = nn.functional.softmax(attn_weights.transpose(1, 2), dim=-1)

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()
        
        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000) # Do not increase 50000, data type half has quite limited range

        #stable_softmax_2d
        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = (attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0])
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(attn_weights_l, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(attn_weights_l, max=50000) # Do not increase 50000, data type half has quite limited range

        attn_weights_l = attn_weights_l.softmax(dim=-1)

        # import pdb;pdb.set_trace()
        if attention_mask_l is not None:
            assert (attention_mask_l.dim() == 2)
            attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_v = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)


        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l


# Bi-Direction MHA (text->image, image->text)
class BiAttentionBlock(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, hidden_dim=None, dropout=0.1,
                 drop_path=.0, init_values=1e-4, cfg=None):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(v_dim=v_dim,
                                         l_dim=l_dim,
                                         embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         cfg=cfg)

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

    def forward(self, v, l, attention_mask_l=None, dummy_tensor=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l, attention_mask_l=attention_mask_l)
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return v, l

class BiAttentionBlockForCheckpoint(nn.Module):
    layer = 0
    cal_attn = 0
    
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, hidden_dim=None, dropout=0.1,
                 drop_path=.0, init_values=1e-4, cfg=None):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlockForCheckpoint, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(v_dim=v_dim,
                                         l_dim=l_dim,
                                         embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         cfg=cfg)

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

        self.cfg = cfg
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL:
            if not self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DO_LANG_PROJ_OUTSIDE_CHECKPOINT:
                self.shrink_lang = FeatureResizer(l_dim * 5, l_dim, 0.1)

    def forward(self, q0, q1, q2, q3, q4, l, attention_mask_l=None, dummy_tensor=None, attention_mask_voc=None, voc=None,):

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL:
            visu_feat = []
            lang_feat = []
            for ii, feat in enumerate([q0, q1, q2, q3, q4]):
                bs, _, h, w = feat.shape
                q = feat.flatten(2).transpose(1, 2)
                
                new_v, new_l = self.single_attention_call(q, l, attention_mask_l=attention_mask_l)
                new_v = new_v.transpose(1, 2).contiguous().view(bs, -1, h, w)
                lang_feat.append(new_l)
                visu_feat.append(new_v)
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DO_LANG_PROJ_OUTSIDE_CHECKPOINT:
                pass
            else:
                lang_feat = self.shrink_lang(torch.cat(lang_feat, dim = -1)) # From multiple dimensions
                lang_feat = [lang_feat, None, None, None, None]
        else:
            #TODO: 选词模块实现在这里面
            visu_feat = []
            size_per_level, visual_features_flatten = [], []
            for ii, feat_per_level in enumerate([q0, q1, q2, q3, q4]):
                bs, c, h, w = feat_per_level.shape
                size_per_level.append([h, w])
                feat = permute_and_flatten(feat_per_level, bs, 1, c, h, w)
                visual_features_flatten.append(feat)
            visual_features_flatten = cat(visual_features_flatten, dim=1)
            new_v, new_l, new_voc = self.single_attention_call(visual_features_flatten, l, attention_mask_l=attention_mask_l, attention_mask_voc= attention_mask_voc, voc=voc, size_per_level=size_per_level)
            # [bs, N, C] -> [bs, C, N]·
            new_v = new_v.transpose(1, 2).contiguous()

            start = 0
            for (h, w) in size_per_level:
                new_v_per_level = new_v[:, :, start:start + h * w].view(bs, -1, h, w).contiguous()
                visu_feat.append(new_v_per_level)
                start += h * w
            
            lang_feat = [new_l, new_voc, None, None, None]

        return visu_feat[0], visu_feat[1], visu_feat[2], visu_feat[3], visu_feat[4], lang_feat[0], lang_feat[1], lang_feat[2], lang_feat[3], lang_feat[4]
    
            
    def apply_umap_and_save(self, combined_pyramid_v_states, l_states, l_states_f_reshape, v_selected_state_reshape, output_dir):
        # 创建保存图像的目录
        os.makedirs(output_dir, exist_ok=True)

        # 将四个矩阵转换为 numpy 数组
        combined_pyramid_v_states_np = combined_pyramid_v_states.cpu().numpy()
        l_states_np = l_states.cpu().numpy()
        l_states_f_reshape_np = l_states_f_reshape.cpu().numpy()
        v_selected_state_reshape_np = v_selected_state_reshape.cpu().numpy()

        # 获取当前时间
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 为每个 head 分别应用 UMAP
        for head_idx in range(combined_pyramid_v_states_np.shape[0]):
            # 取出每个 head 的特征
            combined_pyramid_v_state_head = combined_pyramid_v_states_np[head_idx].reshape(-1, combined_pyramid_v_states_np.shape[-1])
            l_state_head = l_states_np[head_idx].reshape(-1, l_states_np.shape[-1])
            l_state_f_head = l_states_f_reshape_np[head_idx].reshape(-1, l_states_f_reshape_np.shape[-1])
            v_selected_state_head = v_selected_state_reshape_np[head_idx].reshape(-1, v_selected_state_reshape_np.shape[-1])

            # 结合四个特征矩阵
            combined_data = np.vstack((combined_pyramid_v_state_head, l_state_head, l_state_f_head, v_selected_state_head))

            # 应用 UMAP
            reducer = umap.UMAP()
            embedding = reducer.fit_transform(combined_data)

            # 保存每个特征矩阵的结果
            start_idx = 0

            plt.figure()
            plt.scatter(embedding[start_idx:start_idx + combined_pyramid_v_state_head.shape[0], 0], embedding[start_idx:start_idx + combined_pyramid_v_state_head.shape[0], 1], s=5)
            plt.title(f'UMAP projection for head {head_idx} - Combined Pyramid V States')
            save_path = os.path.join(output_dir, f'umap_projection_{current_time}_head{head_idx}_combined_pyramid_v_states.png')
            plt.savefig(save_path)
            plt.close()

            start_idx += combined_pyramid_v_state_head.shape[0]

            plt.figure()
            plt.scatter(embedding[start_idx:start_idx + l_state_head.shape[0], 0], embedding[start_idx:start_idx + l_state_head.shape[0], 1], s=5)
            plt.title(f'UMAP projection for head {head_idx} - L States')
            save_path = os.path.join(output_dir, f'umap_projection_{current_time}_head{head_idx}_l_states.png')
            plt.savefig(save_path)
            plt.close()

            start_idx += l_state_head.shape[0]

            plt.figure()
            plt.scatter(embedding[start_idx:start_idx + l_state_f_head.shape[0], 0], embedding[start_idx:start_idx + l_state_f_head.shape[0], 1], s=5)
            plt.title(f'UMAP projection for head {head_idx} - L States F Reshape')
            save_path = os.path.join(output_dir, f'umap_projection_{current_time}_head{head_idx}_l_states_f_reshape.png')
            plt.savefig(save_path)
            plt.close()

            start_idx += l_state_f_head.shape[0]

            plt.figure()
            plt.scatter(embedding[start_idx:start_idx + v_selected_state_head.shape[0], 0], embedding[start_idx:start_idx + v_selected_state_head.shape[0], 1], s=5)
            plt.title(f'UMAP projection for head {head_idx} - V Selected State Reshape')
            save_path = os.path.join(output_dir, f'umap_projection_{current_time}_head{head_idx}_v_selected_state_reshape.png')
            plt.savefig(save_path)
            plt.close()
    
    def single_attention_call(self, v, l, attention_mask_l=None, dummy_tensor=None, attention_mask_voc = None, voc = None, size_per_level=None):
        BiAttentionBlockForCheckpoint.layer += 1 
        if BiAttentionBlockForCheckpoint.layer > 6:
            BiAttentionBlockForCheckpoint.layer -= 6 
        print(BiAttentionBlockForCheckpoint.layer)
        # import pdb;pdb.set_trace()
        selayer = 6
        # if BiAttentionBlockForCheckpoint.layer == 4  or BiAttentionBlockForCheckpoint.layer == 5:# best setting
        if BiAttentionBlockForCheckpoint.layer < 7 or BiAttentionBlockForCheckpoint.layer > 0:# no fusion
        # if BiAttentionBlockForCheckpoint.layer < 0 or BiAttentionBlockForCheckpoint.layer > 7:# all fusion   
            BiAttentionBlockForCheckpoint.cal_attn = 0
            v = self.layer_norm_v(v)
            l = self.layer_norm_l(l)
            delta_v, delta_l = self.attn(v, l, attention_mask_l=attention_mask_l)
            # v, l = v + delta_v, l + delta_l
            v = v + self.drop_path(self.gamma_v * delta_v)
            l = l + self.drop_path(self.gamma_l * delta_l)
            # 计算融合后 attn
            BiAttentionBlockForCheckpoint.cal_attn = 1
            self.attn(v,l,attention_mask_l = attention_mask_l)
            
            voc = voc + self.drop_path(self.gamma_l * delta_l)

            # self.visualize_features(v, l, voc)
            return v, l, voc        
        else:
            v = self.layer_norm_v(v) 
            l = self.layer_norm_l(l) #1,256,768
            BiAttentionBlockForCheckpoint.cal_attn = 0
            if voc is None:
                voc = l
                voc = voc - voc
            else:
                voc = self.layer_norm_l(voc)
            
            ctx = 1
            if attention_mask_voc is None:
                attention_mask_voc = attention_mask_l.clone()
                zero_indices = (attention_mask_voc == 0).nonzero(as_tuple=True)[1][:ctx]
                attention_mask_voc[:, zero_indices] = 1
                # attention_mask_voc[:,:ctx] = 1 #1,256 
            # voc = l

            # New, AttriGLIP, voc 嵌入 attn_l 做attention
            attention_mask_pad = 1 - attention_mask_voc
            # attn_l = voc
            attn_l = attention_mask_l.unsqueeze(2) * l + (attention_mask_voc-attention_mask_l).unsqueeze(2) * voc + attention_mask_pad.unsqueeze(2) * l
            attention_mask_attn_l = attention_mask_voc
            
            # self.visualize_features(v, l, attn_l, self.layer)
            
            # annotation from here
            # New, AttriGLIP, pyramid visual-linguistc matching
            start = 0
            pyramid_v_t = []
            pyramid_v_states = []
            pyramid_vl_attns = [] #各个层次的视觉特征与词表的attention矩阵
            
            bsz, tgt_len, embed_dim = v.size()
            proj_shape = (bsz * self.attn.num_heads, -1, self.attn.head_dim)
            
            ## linguistic hidden, state
            l_states = self.attn._shape(self.attn.l_proj(attn_l), -1, bsz)
            # l_states = self.attn._shape(self.attn.l_proj(l), -1, bsz)
            l_states = l_states.view(*proj_shape)
            pyramid_v_state_all = []
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            ## pyramid visual hidden states
            for (h, w) in size_per_level:
                       
                pyramid_v_per_level = v[:, start:start + h * w, :]
                pyramid_v_state_per_level = self.attn.v_proj(pyramid_v_per_level) * self.attn.scale
                pyramid_v_state_per_level = self.attn._shape(pyramid_v_state_per_level, h * w, bsz).view(*proj_shape)
                # import pdb;pdb.set_trace()
                pyramid_v_state_all.append(pyramid_v_state_per_level)
                             
                # 计算 vl-attn-metrics
                pyramid_vl_attn_per_level = torch.bmm(pyramid_v_state_per_level, l_states.transpose(1,2)) #>> [bs*num_heads, h * w, seq_len]
                # print(f'Layer {BiAttentionBlockForCheckpoint.layer} 的 level {h} {w} 总和为: {torch.mean(pyramid_vl_attn_per_level).item()}')
                pyramid_vl_attn_per_level = pyramid_vl_attn_per_level.transpose(1, 2) #>> [bs*num_heads, seq_len, h * w]
                # import pdb;pdb.set_trace()
                pyramid_vl_attn_per_level = pyramid_vl_attn_per_level.contiguous().view(self.attn.num_heads, -1, h, w) #>> [bs*num_heads, seq_len, h, w]
                # pyramid_vl_attn_per_level = pyramid_vl_attn_per_level.view(self.attn.num_heads, -1, h, w).contiguous() #>> [bs*num_heads, seq_len, h, w]
                pyramid_vl_attn_per_level = pyramid_vl_attn_per_level.permute(0, 2, 3, 1) #>> [bs*num_heads, h, w, seq_len]
                pyramid_vl_attns.append(pyramid_vl_attn_per_level)
                
                # 计算 visual-hidden-state
                pyramid_v_state_per_level = pyramid_v_state_per_level.transpose(1, 2).contiguous().view(self.attn.num_heads, -1, h, w)
                pyramid_v_state_per_level = pyramid_v_state_per_level.permute(0, 2, 3, 1)
                pyramid_v_states.append(pyramid_v_state_per_level)
                

                start += h * w
            combined_pyramid_v_states = torch.cat(pyramid_v_state_all, dim=1)
            
            # import pdb;pdb.set_trace()
            topK = 5 #5
            level = 0
            prev_level_sign = None
            pyramid_v_signs = [] #>> 从金字塔顶层开始，延续到最底层；存每一层的重要性矩阵, [[bs, h1, w1], ..., [bs, h5, w5]]
            pyramid_v_selected_indices = [] #>> 从金字塔顶层开始，延续到最底层；存每一层选的词, [[bs, K1], ..., [bs, K5]]
            for pyramid_vl_attn_per_level, (h, w) in zip(pyramid_vl_attns[::-1], size_per_level[::-1]):
                
                pyramid_vl_attn_per_level = pyramid_vl_attn_per_level.contiguous().view(bsz, self.attn.num_heads, h, w, -1)
                
                ## 本来是pyramid_v_sign_per_level *= attention_mask_attn_k.unsqueeze(1).unsqueeze(2).unsqueeze(3), 但是可能sign值会小于0，不如直接在0位加一个很大的负进去
                # pyramid_v_sign_per_level = pyramid_vl_attn_per_level + ((-1e5) * (1-attention_mask_attn_l.unsqueeze(1).unsqueeze(2).unsqueeze(3))) #这句有什么意义 之前的
                pyramid_v_sign_per_level = pyramid_vl_attn_per_level * attention_mask_attn_l.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                
                pyramid_v_sign_per_level = torch.sum(pyramid_v_sign_per_level, dim=-1)
                # pyramid_v_sign_per_level = torch.sum(pyramid_vl_attn_per_level, dim=-1) #之前的
                pyramid_v_sign_per_level = torch.mean(pyramid_v_sign_per_level, dim=1)
                
                
                ## 融合先前层次的重要性
                if prev_level_sign is not None:
                    ## 对上一层次的重要性权重进行上采样
                    prev_level_sign_up = F.interpolate(prev_level_sign.unsqueeze(dim=1), 
                                                    size=(h, w), mode='nearest').squeeze(dim=1)
                    # import pdb;pdb.set_trace()
                    pyramid_v_sign_per_level = (pyramid_v_sign_per_level * 0.5 + prev_level_sign_up * 0.5)
                    # pyramid_v_sign_per_level = pyramid_v_sign_per_level * prev_level_sign_up #之前的
                    
                ## 选词
                pyramid_v_sign_flatten_per_level = pyramid_v_sign_per_level.view(bsz, -1)
                
                #阈值
                valid_counts = torch.sum(pyramid_v_sign_flatten_per_level > -500).item()
                k= topK*4**level
                if valid_counts < k:
                    k = valid_counts
                _, topk_indices_per_level, = torch.topk(pyramid_v_sign_flatten_per_level, 
                                                        k=k, 
                                                        dim=1, largest=True, sorted=True)
                
                prev_level_sign = pyramid_v_sign_per_level
                pyramid_v_signs.append(pyramid_v_sign_per_level)
                pyramid_v_selected_indices.append(topk_indices_per_level)
                level = level + 1
            
            start = 0
            v_selected = []
            for indices, (h,w) in zip(pyramid_v_selected_indices, size_per_level[::-1]):
                indices = indices.unsqueeze(-1).expand(-1, -1, embed_dim)
                topk_v_per_level = torch.gather(v[:, start:start + h * w, :], 1, indices)
                v_selected.append(topk_v_per_level)
                start += h * w
            v_selected = torch.cat(v_selected, dim=1)
            
            
            # 利用选出来的视觉词，选择topK文本词
            topK_l = 5 #5
            v_selected_state = (self.attn.v_proj(v_selected) * self.attn.scale
                                ).view(bsz, -1, self.attn.num_heads, self.attn.head_dim).transpose(1,2).contiguous()
            
            l_states_dim = l_states.size(-1)
            l_states_f = l_states.view(bsz, self.attn.num_heads, -1, l_states_dim).contiguous()
            l_states_f_reshape = l_states_f.view(bsz * self.attn.num_heads, -1, self.attn.head_dim)
            v_selected_state_reshape = v_selected_state.view(bsz * self.attn.num_heads, -1, self.attn.head_dim)
            l_vs_attn = torch.bmm(l_states_f_reshape, v_selected_state_reshape.transpose(1,2))
            num_selected_features = attention_mask_attn_l.sum(dim=1).max().item()
            batch_size, num_features_key, feature_dim = l_states.shape
            # 使用掩码选择特征
            selected_key_states = []
            for batch_idx in range(batch_size):
                selected_features = l_states[batch_idx, attention_mask_attn_l[0].bool()]
                selected_key_states.append(selected_features[:num_selected_features])
            selected_key_states = torch.stack(selected_key_states)
            l_vs_attn = l_vs_attn.view(bsz, self.attn.num_heads, l_vs_attn.size(1), l_vs_attn.size(2))
            l_vs_attn = torch.mean(l_vs_attn, dim=-1) #用一个文本token与所有的视觉token相似度的均值来代表一个文本的重要性
            l_vs_attn = torch.mean(l_vs_attn, dim=1) #合并一个文本token的所有的head
            
            l_vs_attn_selected = l_vs_attn * attention_mask_attn_l
            l_vs_attn = l_vs_attn +((-1e5) * (1-attention_mask_attn_l)) #去掉使用[pad]填充的位置的文本重要性

            sk = torch.sum(attention_mask_attn_l, dim=-1, keepdim=True) # 1, 1
            sk = torch.clamp(sk, max=topK_l)
            
            last_index = torch.sum(attention_mask_attn_l == 1)
            s_attention_mask_attn_l = torch.zeros_like(attention_mask_attn_l, dtype=torch.int64)  # 设置dtype为torch.int64
            self.layer += 1 
            layer = self.layer

            # 循环处理每个批次, 选择词 #推理时bsz为2
            for i in range(bsz):
                k = sk[i].item()  # 获取当前批次的k值

                # 统计大于 -3 的元素数量
                thr_l_attn = -3
                valid_count = torch.sum(l_vs_attn[i] > thr_l_attn).item()

                # 如果大于 thr_l_attn 的元素数量小于 k，调整 k
                if valid_count < k:
                    k = valid_count

                # 应用 torch.topk
                _, top_indices = torch.topk(l_vs_attn[i], k=k, largest=True, sorted=True)

                # 排除掉最后一个索引和0索引
                last_index = l_vs_attn.size(1)
                top_indices = top_indices[top_indices != last_index - 1]
                top_indices = top_indices[top_indices != 0]

                # 更新掩码
                s_attention_mask_attn_l[i, top_indices] = 1
            
            # MHA module, 模态融合  
            
            _, delta_l = self.attn(v_selected, attn_l, attention_mask_l=attention_mask_attn_l)
            delta_v, _ = self.attn(v, attn_l, attention_mask_l=s_attention_mask_attn_l)
            
            v = v + self.drop_path(self.gamma_v * delta_v)
            l = l + self.drop_path(self.gamma_l * delta_l)
            
            BiAttentionBlockForCheckpoint.cal_attn = 1
            self.attn(v,l,attention_mask_l = attention_mask_l)

            voc = voc + self.drop_path(self.gamma_l * delta_l)
            # import pdb;pdb.set_trace()
            return v, l, voc


# Single Direction MHA
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module for both image and text
    """

    def __init__(self, q_dim, k_dim, embed_dim, num_heads, dropout=0.1, 
        clamp_min_for_underflow = False, clamp_max_for_overflow = False):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_dim = q_dim
        self.k_dim = k_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.q_proj = nn.Linear(self.q_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.q_dim)
        self.clamp_min_for_underflow = clamp_min_for_underflow
        self.clamp_max_for_overflow = clamp_max_for_overflow

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def forward(self, q, k, v, attention_mask=None, return_attention=False):
        bsz, tgt_len, embed_dim = q.size()

        query_states = self.q_proj(q) * self.scale
        key_states = self._shape(self.k_proj(k), -1, bsz)
        value_states = self._shape(self.v_proj(v), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000) # Do not increase 50000, data type half has quite limited range

        if attention_mask is not None:
            # [bsz, src_len]
            assert (attention_mask.dim() == 2)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if return_attention:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)


        return attn_output, attn_weights


class AttentionMLP(nn.Module):
    def __init__(self, q_dim, hidden_dim, dropout=0.1):
        super(AttentionMLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(q_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, q_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class AttentionT2I(nn.Module):
    def __init__(self, q_dim, k_dim, embed_dim, num_heads, hidden_dim=None, dropout=0.1,
                 drop_path=.0, init_values=1e-4, mode="i2t", use_layer_scale = False,
                 clamp_min_for_underflow = False, clamp_max_for_overflow = False):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(AttentionT2I, self).__init__()

        # pre_layer norm
        self.layer_norm_q_1 = nn.LayerNorm(q_dim)
        self.layer_norm_k_1 = nn.LayerNorm(k_dim)
        self.attn = MultiHeadAttention(q_dim=q_dim,
                                       k_dim=k_dim,
                                       embed_dim=embed_dim,
                                       num_heads=num_heads,
                                       clamp_min_for_underflow=clamp_min_for_underflow,
                                       clamp_max_for_overflow=clamp_max_for_overflow)
        self.mode = mode

        # add layer scale for training stability
        self.use_layer_scale = use_layer_scale
        if self.use_layer_scale:
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.gamma = nn.Parameter(init_values * torch.ones((q_dim)), requires_grad=True)


    def forward(self, q0, q1, q2, q3, q4, k, v, attention_mask, dummy_arg=None):
        qs = []
        for q_index, q in enumerate([q0, q1, q2, q3, q4]):
            bs, _, h, w = q.shape
            # (batch, seq_len, embed_size)
            q = q.flatten(2).transpose(1, 2)
            q = self.layer_norm_q_1(q)
            k, v = self.layer_norm_k_1(k), self.layer_norm_k_1(v)
            delta_q = self.attn(q, k, v, attention_mask=attention_mask)[0]
            if self.use_layer_scale:
                q = q + self.drop_path(self.gamma * delta_q)
            else:
                q = q + delta_q
            q = q.transpose(1, 2).contiguous().view(bs, -1, h, w)
            qs.append(q)


        return qs[0], qs[1], qs[2], qs[3], qs[4]
