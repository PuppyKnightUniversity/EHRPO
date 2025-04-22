import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionTransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, nhead, dim_feedforward, dropout, activation):
        super(AttentionTransformerEncoderLayer, self).__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        self.self_attn = self.transformer_encoder_layer.self_attn

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 获取注意力权重
        attn_output, attn_weights = self.self_attn(
            src, 
            src, 
            src, 
            attn_mask=src_mask, 
            key_padding_mask=src_key_padding_mask,
            average_attn_weights = True
        )
        
        # TransformerEncoderLayer的其他操作
        output = self.transformer_encoder_layer.linear2(self.transformer_encoder_layer.dropout(F.relu(self.transformer_encoder_layer.linear1(attn_output))))
        output = self.transformer_encoder_layer.norm2(output + self.transformer_encoder_layer.dropout(attn_output))
        
        return output, attn_weights


class VisitTransformer(nn.Module):
    def __init__(self, model_dim, nhead, num_layers, dim_feedforward, dropout, activation):
        super(VisitTransformer, self).__init__()
        # 使用自定义的 AttentionTransformerEncoderLayer
        self.visit_level_transformer_layers = nn.ModuleList([
            AttentionTransformerEncoderLayer(model_dim, nhead, dim_feedforward, dropout, activation) 
            for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None, src_key_padding_mask=None):        
        '''
            src (seq_length, batch_size, model_dim)
        '''
        attn_weights_list = []
        
        output = src

        for layer in self.visit_level_transformer_layers:
            output, attn_weights = layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            # print(attn_weights.shape) # (batch_size, seq_length, seq_length)
            attn_weights_list.append(attn_weights)  # 保存每一层的注意力权重
        
        return output, attn_weights_list

if __name__ == '__main__':
    # 示例用法
    model_dim = 512
    nhead = 4
    num_layers = 1
    dim_feedforward = 1024
    dropout = 0.0
    activation = 'relu'

    # 创建模型
    model = VisitTransformer(model_dim, nhead, num_layers, dim_feedforward, dropout, activation)

    # 输入数据
    batch_size = 300
    seq_length = 10
    src = torch.rand(seq_length, batch_size, model_dim)  

    # 进行前向传播
    output, attn_weights_list = model(src)

    # 输出结果
    # print("Transformer output shape:", output.shape)  # 应该是 (batch_size, seq_length, model_dim)
    
    # 初始化一个重要性分数数组
    importance_scores = torch.zeros(src.shape[1], src.shape[0])  # (batch_size, seq_length)

    # 对每一层的注意力权重求和，得到每个token的重要性分数
    for i, attn_weights in enumerate(attn_weights_list):
        # print(f"Attention weights shape for layer {i+1}: {attn_weights.shape}")
        
        # 对每层的注意力权重进行求和，计算每个token的重要性分数
        # 每层的权重 shape: (batch_size, seq_length, seq_length)
        layer_importance = attn_weights.sum(dim=-1)  # 求每个token对其他tokens的注意力和
        importance_scores += layer_importance  # 累积加总
    
    # 归一化分数（可选）
    importance_scores /= len(attn_weights_list)
    
    # 输出每个token的重要性
    print("Importance scores shape:", importance_scores.shape)  # 应该是 (batch_size, seq_length)
    print("Importance scores:", importance_scores)