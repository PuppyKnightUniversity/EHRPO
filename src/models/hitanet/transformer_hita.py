import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn.init as init
from models.hitanet.visit_level_transformer import VisitTransformer

class Embedding(torch.nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx,
                                        max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                                        sparse=sparse, _weight=_weight)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):

        super(PositionalEncoding, self).__init__()


        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.from_numpy(position_encoding.astype(np.float32))

        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len):

        '''
            TODO: max_len is not equal with real max_len
        '''
        max_len = torch.max(input_len)
        #print('input_len:', input_len)
        #print('max_len:', max_len)
        # tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor

        pos = np.zeros([len(input_len), max_len]) # (batch_size, max_len)
        for ind, length in enumerate(input_len):
            for pos_ind in range(1, length + 1):
                pos[ind, pos_ind - 1] = pos_ind # 位置从1开始，0表示填充

        input_pos = torch.tensor(pos, dtype=torch.long, device=input_len.device) # (batch_size, max_len) 如果元素为0，则代表是padding的
        return self.position_encoding(input_pos), input_pos


class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        # reshape attention, mean by multi head
        attention = attention.view(batch_size, num_heads, attention.size(1), attention.size(2))
        attention = attention.mean(dim=1)


        return output, attention


class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        '''
            args:
                inputs: (batchsize, visits, embedding)
            returns:
                output: (batchsize, visits, embedding)
                attention: (num_head * batchsize, visits, visits)
        '''

        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


def padding_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


class EncoderHitanet_visit_level_attention(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=1,
                 model_dim=128,
                 num_heads=4,
                 ffn_dim=1024,
                 dropout=0.0):
        super(EncoderHitanet_visit_level_attention, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             range(num_layers)])
        self.pre_embedding = Embedding(vocab_size, model_dim)
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(model_dim))
        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)

        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.time_layer = torch.nn.Linear(64, model_dim)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.max_seq_len = max_seq_len
        
        self.visit_level_transformer = VisitTransformer(model_dim=model_dim,
                                                        nhead = 4,
                                                        num_layers = 1,
                                                        dim_feedforward = 2 * model_dim,
                                                        dropout = 0.0,
                                                        activation="relu")
        
    
    def get_visit_embedding_by_transformer(self, diagnosis_codes):


        # print('diagnosis_codes shape:', diagnosis_codes.shape)
        
        '''
            Mask generation
        '''
        code_padding_mask = torch.all(diagnosis_codes == 0, dim=-1)  # (patient, visit, code) 
        patient_num, visit_num, code_num = code_padding_mask.shape

        '''
            Flatten `patient * visit`
        '''
        code_padding_mask = code_padding_mask.view(patient_num * visit_num, code_num)  # (batch_size, code)
        diagnosis_codes = diagnosis_codes.reshape(patient_num * visit_num, code_num, -1).contiguous()  # (batch_size, code, embedding)
        
        '''
            Skip mechanism: 仅处理包含有效 `code` 的 `visit`
        '''
        valid_samples = ~code_padding_mask.all(dim=1)  # (batch_size,)

        if valid_samples.any():  # 至少有一个有效 visit 才能送入 Transformer
            filtered_diagnosis_codes = diagnosis_codes[valid_samples]  # (valid_batch_size, code, embedding)
            filtered_code_padding_mask = code_padding_mask[valid_samples]  # (valid_batch_size, code)

            '''
                Apply Transformer
            '''
            filtered_diagnosis_codes = filtered_diagnosis_codes.permute(1, 0, 2)  # (code, valid_batch_size, embedding)
            visit_embedding, atten_wights_list = self.visit_level_transformer(src=filtered_diagnosis_codes, src_key_padding_mask=filtered_code_padding_mask)
            
            assert not torch.isnan(visit_embedding).any()

            '''
                Restore original shape (patient, visit, embedding)
            '''
            visit_embedding = visit_embedding.mean(dim=0)  # (valid_batch_size, embedding)

            # 创建 full_visit_embedding，确保所有样本都有对应 embedding
            full_visit_embedding = torch.zeros(patient_num * visit_num, visit_embedding.shape[-1], device=diagnosis_codes.device)
            full_visit_embedding[valid_samples] = visit_embedding  # 确保形状匹配

            '''
                calculate visit-level code importance score
            '''
            full_visit_code_importance_score = torch.zeros(patient_num * visit_num, code_num, device=diagnosis_codes.device)
            
            filtered_visit_code_importance_score = torch.zeros(visit_embedding.shape[0], code_num, device=diagnosis_codes.device)
            
            # use last layer attention weights
            last_layer_attn_weights = atten_wights_list[-1]
            layer_importance = last_layer_attn_weights.sum(dim=-2)
            seq_sum = layer_importance.sum(dim=-1, keepdim=True)
            layer_importance = layer_importance / seq_sum

            filtered_visit_code_importance_score += layer_importance
            
            '''
            for i, attn_weights in enumerate(atten_wights_list ):
                # browse each layer attention weights
                # each layer attention weights shape: (batch_size, seq_length, seq_length)
                layer_importance = attn_weights.sum(dim=-2)  # 求每个token对其他tokens的注意力和
                # normalization
                seq_sum = layer_importance.sum(dim=-1, keepdim=True)
                layer_importance = layer_importance / seq_sum
                filtered_visit_code_importance_score += layer_importance  # 累积加总
            filtered_visit_code_importance_score /= len(atten_wights_list)
            '''


            # 归一化分数（可选）
            full_visit_code_importance_score[valid_samples] = filtered_visit_code_importance_score

        else:
            '''
                If all visits in the batch are padding, return 0 vector
            '''
            full_visit_embedding = torch.zeros(patient_num * visit_num, diagnosis_codes.shape[-1], device=diagnosis_codes.device)
            full_visit_code_importance_score = torch.zeros(patient_num * visit_num, code_num, device=diagnosis_codes.device)

        '''
            Reshape to original (patient, visit, embedding)
        '''
        visit_embedding = full_visit_embedding.view(patient_num, visit_num, -1)
        code_importance_score = full_visit_code_importance_score.view(patient_num, visit_num, -1)

        return visit_embedding, code_importance_score



    def forward(self, 
                diagnosis_codes, 
                mask, 
                mask_code, 
                seq_time_step, 
                input_len):

        '''
            TODO: apply transformer to visit-level
        '''
        
        '''
            get embedding for each visit
        '''
        output, code_level_importance_score = self.get_visit_embedding_by_transformer(diagnosis_codes)
        output = output + self.bias_embedding

        '''
            time embedding
        '''
        seq_time_step = seq_time_step/180
        # f_t = 1 - tanh( ( W_f * theta_t / 180 + b_f))^2
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        # r_t = W_r * f_t + b_r
        time_feature = self.time_layer(time_feature)
        output += time_feature

        '''
            position embedding
        '''
        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        output += output_pos

        self_attention_mask = padding_mask(ind_pos, ind_pos)

        '''
            visit-level transformer
        '''
        atten_weights_list = []
        outputs = []

        for encoder in self.encoder_layers:
            # for each layer of transformer, cal embedding
            output, attention = encoder(output, self_attention_mask)
            atten_weights_list.append(attention)
            outputs.append(output)

        '''
            TODO: 
                1. normalization attention importance score
            get visit-level attention importance score
        '''

        '''
            calculate visit level importance score
        '''
        batch_size, visit_len = atten_weights_list[0].shape[0], atten_weights_list[0].shape[1]

        visit_level_importance_score = torch.zeros(batch_size, visit_len, device=output.device)

        # TODO:这里也使用最后一层是不是会好一些？
        # 还需要额外写一些assert, 确保每一个visit内所有code的权重之和是1

        last_layer_attn_weights = atten_weights_list[-1]
        layer_importance = last_layer_attn_weights.sum(dim=-2)
        seq_sum = layer_importance.sum(dim=-1, keepdim=True)
        layer_importance = layer_importance / seq_sum
        visit_level_importance_score += layer_importance

        for i, attn_weights in enumerate(atten_weights_list):
            # attn_weights shape (batch_size, visit_len, visit_len)
            layer_importance = attn_weights.sum(dim=-2)
            # layer_importance shape (batch_size, visit_len)
            seq_sum = layer_importance.sum(dim=-1, keepdim=True)
            layer_importance = layer_importance / seq_sum
            visit_level_importance_score += layer_importance

        visit_level_importance_score /= len(atten_weights_list)
        
        return output, code_level_importance_score, visit_level_importance_score


class TransformerHitaBlock(nn.Module):
    def __init__(self, options):
        super(TransformerHitaBlock, self).__init__()
        self.feature_encoder = EncoderHitanet_visit_level_attention(options['n_diagnosis_codes'] + 1, 51, num_layers=options['layer'])
        self.self_layer = torch.nn.Linear(128, 1)
        self.classify_layer = torch.nn.Linear(128, 2)
        self.quiry_layer = torch.nn.Linear(128, 64)
        self.quiry_weight_layer = torch.nn.Linear(128, 2)
        self.relu = nn.ReLU(inplace=True)
        # dropout layer
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, 
                seq_dignosis_codes, 
                seq_time_step, 
                batch_labels, 
                options, 
                mask,
                original_lengths):        
        

        diagnosis_codes = seq_dignosis_codes
        if options['use_gpu']:
            pass
        else:
            diagnosis_codes = torch.LongTensor(diagnosis_codes)
            pass


        features, code_level_importance_score, visit_level_importance_score = self.feature_encoder(diagnosis_codes, 
                                                                                                   None, 
                                                                                                   mask, 
                                                                                                   seq_time_step, 
                                                                                                   original_lengths)

        final_statues = features
        final_statues = final_statues.sum(1)

        return code_level_importance_score, visit_level_importance_score, final_statues















