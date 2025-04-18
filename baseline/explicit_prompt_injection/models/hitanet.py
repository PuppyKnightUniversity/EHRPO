import math
import torch
from torch import nn
from typing import Dict, List, Optional, Tuple
from pyhealth.models import BaseModel
from pyhealth.datasets import SampleEHRDataset
from models.transformer_hita import TransformerHitaBlock

class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        if mask is not None:
            p_attn = p_attn.masked_fill(mask == 0, 0)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(3)]
        )
        self.output_linear = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        if mask is not None:
            mask = mask.unsqueeze(1)
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, mask=None):
        x = self.w_2(self.dropout(self.activation(self.w_1(x))))
        if mask is not None:
            mask = mask.sum(dim=-1) > 0
            x[~mask] = 0
        return x


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    """Transformer block.

    MultiHeadedAttention + PositionwiseFeedForward + SublayerConnection

    Args:
        hidden: hidden size of transformer.
        attn_heads: head sizes of multi-head attention.
        dropout: dropout rate.
    """

    def __init__(self, hidden, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=4 * hidden, dropout=dropout
        )
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        """Forward propagation.

        Args:
            x: [batch_size, seq_len, hidden]
            mask: [batch_size, seq_len, seq_len]

        Returns:
            A tensor of shape [batch_size, seq_len, hidden]
        """
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, lambda _x: self.feed_forward(_x, mask=mask))
        return self.dropout(x)


class TransformerLayer(nn.Module):
    """Transformer layer.

    Paper: Ashish Vaswani et al. Attention is all you need. NIPS 2017.

    This layer is used in the Transformer model. But it can also be used
    as a standalone layer.

    Args:
        feature_size: the hidden feature size.
        heads: the number of attention heads. Default is 1.
        dropout: dropout rate. Default is 0.5.
        num_layers: number of transformer layers. Default is 1.

    Examples:
        >>> from pyhealth.models import TransformerLayer
        >>> input = torch.randn(3, 128, 64)  # [batch size, sequence len, feature_size]
        >>> layer = TransformerLayer(64)
        >>> emb, cls_emb = layer(input)
        >>> emb.shape
        torch.Size([3, 128, 64])
        >>> cls_emb.shape
        torch.Size([3, 64])
    """

    def __init__(self, feature_size, heads=1, dropout=0.5, num_layers=1):
        super(TransformerLayer, self).__init__()
        self.transformer = nn.ModuleList(
            [TransformerBlock(feature_size, heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self, x: torch.tensor, mask: Optional[torch.tensor] = None
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, feature_size].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            emb: a tensor of shape [batch size, sequence len, feature_size],
                containing the output features for each time step.
            cls_emb: a tensor of shape [batch size, feature_size], containing
                the output features for the first time step.
        """
        if mask is not None:
            mask = torch.einsum("ab,ac->abc", mask, mask)
        for transformer in self.transformer:
            x = transformer(x, mask)
        emb = x
        cls_emb = x[:, 0, :]
        return emb, cls_emb


class HitaTransformer(BaseModel):
    def __init__(
        self,
        dataset: SampleEHRDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        embedding_dim: int = 128,
        **kwargs
    ):
        super(HitaTransformer, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.embedding_dim = embedding_dim

        # validate kwargs for Transformer layer
        if "feature_size" in kwargs:
            raise ValueError("feature_size is determined by embedding_dim")

        # the key of self.feat_tokenizers only contains the code based inputs
        self.feat_tokenizers = {}
        self.label_tokenizer = self.get_label_tokenizer()
        # the key of self.embeddings only contains the code based inputs
        self.embeddings = nn.ModuleDict()
        # the key of self.linear_layers only contains the float/int based inputs
        self.linear_layers = nn.ModuleDict()

        # add feature transformation layers
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            # sanity check
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "Transformer only supports str code, float and int as input types"
                )
            elif (input_info["type"] == str) and (input_info["dim"] not in [2, 3]):
                raise ValueError(
                    "Transformer only supports 2-dim or 3-dim str code as input types"
                )
            elif (input_info["type"] in [float, int]) and (
                input_info["dim"] not in [2, 3]
            ):
                raise ValueError(
                    "Transformer only supports 2-dim or 3-dim float and int as input types"
                )
            # for code based input, we need Type
            # for float/int based input, we need Type, input_dim
            self.add_feature_transform_layer(feature_key, input_info)

        self.transformer = nn.ModuleDict()

        self.options = {'n_diagnosis_codes':embedding_dim, 'layer':2, 'dropout_rate':0.1, 'use_gpu':True}

        for feature_key in feature_keys:
            self.transformer[feature_key] = TransformerHitaBlock(self.options)

        output_size = self.get_output_size(self.label_tokenizer)

        # transformer's output feature size is still embedding_dim
        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)

    def truncate_padding_tokenize(self, 
                                  x, 
                                  feature_key,
                                  max_visit_length_for_truncation, 
                                  max_code_length_for_truncation):
        '''
            truncate & padding & tokenize
        '''
        max_length_for_truncation = [max_visit_length_for_truncation, max_code_length_for_truncation]
        # cal original lengths, if length > max_visit_length_for_truncation, then set to max_visit_length_for_truncation, if length <= max_visit_length_for_truncation, then set to length
        original_visit_lengths = torch.tensor([ len(token) if len(token) <= max_visit_length_for_truncation else max_visit_length_for_truncation for token in x],
                                         dtype=torch.long, 
                                         device=self.device)

        # truncate visit length
        x = [tokens[-max_visit_length_for_truncation :] for tokens in x]

        # truncate code length
        x = [[tokens[-max_code_length_for_truncation :] for tokens in visits] for visits in x]

        raw_data_after_truncation = x

        # padding & tokenize  
        x = self.feat_tokenizers[feature_key].batch_encode_3d(x, max_length = max_length_for_truncation)
        
        # convert to tensor
        x = torch.tensor(x, dtype=torch.long, device=self.device)

        return x, original_visit_lengths, raw_data_after_truncation

    def ehr_code_embedding(self, x, feature_key):
        '''
            embedding for each code
        '''
        x = self.embeddings[feature_key](x) # （patient, visit, code, embedding）
        return x
    
    
    def deal_with_delta_days(self, delta_days, max_visit_length_for_truncation):
            
        # truncation
        delta_days = [visits[-max_visit_length_for_truncation :] for visits in delta_days]
        # padding
        batch_max_visit_length = max([len(tokens) for tokens in delta_days])
        delta_days = [tokens + [[10000]] * (batch_max_visit_length - len(tokens))
                        for tokens in delta_days]
                
        delta_days = torch.tensor(delta_days, dtype=torch.long, device=self.device)
        padding_delta_days = delta_days

        return padding_delta_days
    
    
    def prepare_hitanet_input(self, x, feature_key, batch_time, max_visit_length_for_truncation, max_code_length_for_truncation):
        '''
            prepare hitanet input
        '''

        '''
            truncation & padding & tokening
        '''
        x, original_visit_lengths, raw_data_after_truncation = self.truncate_padding_tokenize(x,
                                                                                              feature_key,
                                                                                              max_visit_length_for_truncation, 
                                                                                              max_code_length_for_truncation)        
        
        '''
            embedding for each code
        '''
        x = self.ehr_code_embedding(x, feature_key) # （patient, visit, code, embedding）

        '''
            deal with delta days
        '''
        padding_delta_days = self.deal_with_delta_days(batch_time, max_visit_length_for_truncation)

        return x, padding_delta_days, original_visit_lengths, raw_data_after_truncation

    def map_patient_embedding_to_result(self, patient_emb, batch_label):
        '''
            map patient embedding to result
        '''
        logits = self.fc(patient_emb)
        y_true = self.prepare_labels(batch_label, self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        return results

    def process_attention(self, 
                          feature_key, 
                          code_level_importance_score, 
                          visit_level_importance_score, 
                          raw_data_after_truncation):
        
        # load tokenizer
        tokenizer = self.feat_tokenizers[feature_key]

        '''
            extract attention information
        '''

        # store all batch's attention information
        batch_patient_dict_list = []
        
        # browse each patient
        for pid, patient_data in enumerate(raw_data_after_truncation):
            # store each patient's attention information
            patient_dict = {}
            
            patient_atten_list = []
            
            # Dictionary to store total scores for each code
            code_total_scores = {}
            
            # Dictionary to store total scores for each visit
            visit_total_scores = {}
            
            # First pass: collect all visit scores and code scores for normalization
            visit_scores = []
            for vid, visit_data in enumerate(patient_data):
                visit_score = visit_level_importance_score[pid][vid].item()
                visit_scores.append(visit_score)
            
            # Normalize visit scores
            total_visit_score = sum(visit_scores)
            if total_visit_score > 0:
                normalized_visit_scores = [score / total_visit_score for score in visit_scores]
            else:
                normalized_visit_scores = visit_scores
            
            # browse each visit
            for vid, visit_data in enumerate(patient_data):
                # extract visit score (already normalized)
                visit_dict = {}
                visit_total_scores[vid] = visit_dict['score'] = normalized_visit_scores[vid]

                # browse each code
                visit_dict['code'] = []
                code_scores = []
                for cid, code in enumerate(visit_data):
                    # extract code score
                    code_score = code_level_importance_score[pid][vid][cid].item()
                    code_scores.append(code_score)
                    visit_dict['code'].append({code:code_score})
                
                # Normalize code scores within this visit
                total_code_score = sum(code_scores)
                if total_code_score > 0:  # Avoid division by zero
                    normalized_code_scores = [score / total_code_score for score in code_scores]
                    for i, code_dict in enumerate(visit_dict['code']):
                        code = list(code_dict.keys())[0]
                        visit_dict['code'][i] = {code: normalized_code_scores[i]}
                        
                        # Calculate and accumulate total score for this code using normalized scores
                        if code not in code_total_scores:
                            code_total_scores[code] = 0
                        code_total_scores[code] += normalized_visit_scores[vid] * normalized_code_scores[i]
                
                # append visit dict to all_atten_list
                patient_atten_list.append(visit_dict)
            
            patient_dict['detail_attention_list'] = patient_atten_list
            patient_dict['code_total_scores'] = code_total_scores
            patient_dict['visit_total_scores'] = visit_total_scores
            # append patient attention list to all_atten_list
            batch_patient_dict_list.append(patient_dict)
        
        return batch_patient_dict_list
    
    def extract_importance_visit_and_code(self, 
                                          batch_patient_dict_list, ):
        '''
            extract importance visit and code
        '''
        patient_descriptions = []
        for patient_idx, patient_dict in enumerate(batch_patient_dict_list):
            # build patient description
            description = f"According to Medical Expert, the patient's visit importance ranking is as follows:"
            print(f"\nPatient {patient_idx} visit importance ranking:")
            
            # Sort visits by score in descending order
            sorted_visits = sorted(patient_dict['visit_total_scores'].items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)
            
            # Print visit rankings
            for rank, (visit_idx, score) in enumerate(sorted_visits, 1):
                print(f"Rank {rank}: Visit {visit_idx} (Score: {score:.4f})")
            
            print(f"\nPatient {patient_idx} code importance ranking:")
            
            # Sort codes by total score in descending order
            sorted_codes = sorted(patient_dict['code_total_scores'].items(), 
                                key=lambda x: x[1], 
                                reverse=True)
            
            # Print code rankings
            for rank, (code, score) in enumerate(sorted_codes, 1):
                print(f"Rank {rank}: Code {code} (Total Score: {score:.4f})")
    
    def analyze_attention(self, 
                          feature_key, 
                          code_level_importance_score, 
                          visit_level_importance_score, 
                          raw_data_after_truncation):
        '''
            analyze attention
        '''
        batch_patient_dict_list = self.process_attention(feature_key, 
                                                         code_level_importance_score, 
                                                         visit_level_importance_score, 
                                                         raw_data_after_truncation)
        self.extract_importance_visit_and_code(batch_patient_dict_list)

    def wrap_patient_attention_prompt(self, patient_attention_dict):
        '''
            wrap patient attention prompt
        '''
        batch_patient_attention_prompt = []
        '''
            TODO: add patient attention information from EHR model
        '''
        for feature_key in patient_attention_dict.keys():
            code_level_importance_score, visit_level_importance_score, raw_data_after_truncation = patient_attention_dict[feature_key]

        return batch_patient_attention_prompt
    
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:

        patient_emb = []
        patient_attention_dict = {}
        batch_time = kwargs['delta_days']
        batch_label = kwargs[self.label_key]

        for feature_key in self.feature_keys:

            '''
                Notice: each time only handle one feature_key, e.g conditions, drugs, procedures
            '''
            x = kwargs[feature_key]

            # load tokenizer
            tokenizer = self.feat_tokenizers[feature_key]


            ''' 
                prepare hitanet input
            '''
            max_visit_length_for_truncation = 10
            max_code_length_for_truncation = 512

            x, padding_delta_days, original_visit_lengths, raw_data_after_truncation = self.prepare_hitanet_input(x, 
                                                                                                                  feature_key,
                                                                                                                  batch_time,
                                                                                                                  max_visit_length_for_truncation,
                                                                                                                  max_code_length_for_truncation)

            '''
                apply hitanet
            '''
            mask = None
            code_level_importance_score, visit_level_importance_score, x = self.transformer[feature_key](x, 
                                                                                                         padding_delta_days,
                                                                                                         self.prepare_labels(kwargs[self.label_key],self.label_tokenizer), 
                                                                                                         self.options, 
                                                                                                         mask,
                                                                                                         original_visit_lengths)
            patient_emb.append(x)

            '''
                analyze attention
            '''
            self.analyze_attention(feature_key, 
                                   code_level_importance_score, 
                                   visit_level_importance_score,
                                   raw_data_after_truncation)
            
            patient_attention_dict[feature_key] = [code_level_importance_score, visit_level_importance_score, raw_data_after_truncation]
        
        patient_emb = torch.cat(patient_emb, dim=1)
        results = self.map_patient_embedding_to_result(patient_emb, batch_label)

        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        
        '''
            TODO: add patient attention information from EHR model
        '''
        batch_patient_attention_prompt = self.wrap_patient_attention_prompt(patient_attention_dict)
        results["patient_attention_prompt"] = batch_patient_attention_prompt

        return results
            

                                        



