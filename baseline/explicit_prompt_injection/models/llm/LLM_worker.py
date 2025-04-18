import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import Dict, List
from openai import OpenAI
from pyhealth.medcode import InnerMap
from pyhealth.datasets import SampleEHRDataset
from pyhealth.trainer import get_metrics_fn
from pyhealth.metrics import (binary_metrics_fn, multiclass_metrics_fn,
                              multilabel_metrics_fn, regression_metrics_fn)
import torch.nn.functional as F
from pyhealth.models import BaseModel
import numpy as np
import re
import torch.nn as nn

class LLM_worker(BaseModel):
    def __init__(
        self,
        ehr_model:nn.Module,
        ehr_model_path:str,
        dataset: SampleEHRDataset,
        feature_keys: List[str],
        label_key: str,   
        mode = "binary",     
        llm_name = 'qwen2-5-7b-instruct',
        llm_local_path = '/data1/dhx/llmbase/hub/Qwen/Qwen2___5-7B-Instruct',
        is_api = True,
        api_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjE3YzQzODY2LTE2YTctNGI1Ni05ZjFjLTk1ODIzODY0OTJmZCJ9.OusI4q1_w_wzFk46cE4hfV8t1tRGkwBlZoJG9UxZI7o',
        metrics = None,
        inference_type = 'straight_forward',
        task_name = 'mortality_prediction',
        EHR_model_prompt_injection = False,
        **kwargs
    ):  
        super(LLM_worker, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        
        self.ehr_model = ehr_model
        self.llm_name = llm_name
        self.is_api = is_api
        self.label_tokenizer = self.get_label_tokenizer()
        self.metrics = metrics
        self.mode = mode
        self.inference_type = inference_type
        self.task_name = task_name
        self.feature_keys = feature_keys
        self.dataset = dataset
        self.EHR_model_prompt_injection = EHR_model_prompt_injection

        self.get_basic_info()
        self.initialize_llm(llm_name, llm_local_path, api_key, is_api)
        self.initialize_ehr_model(ehr_model, ehr_model_path)

    def initialize_llm(self, llm_name:str, llm_local_path:str, api_key:str, is_api:bool):
        '''
            initialize llm
        '''
        if llm_name not in ['deepseek-r1-local-preview', 
                            'qwen2-5-7b-instruct',
                            'qwen2-5-1.5b-instruct']:
            raise ValueError(
                    f"LLM type {llm_name} not implemented"
                )
        
        if llm_name == 'deepseek-r1-local-preview':
            if is_api:
                self.api_key = api_key
                self.client = OpenAI(api_key=self.api_key, base_url="http://162.105.88.35:3000/api")
        
        elif llm_name == 'qwen2-5-7b-instruct':   
            model_path = llm_local_path

            self.tokenizer = AutoTokenizer.from_pretrained( model_path,
                                                            use_fast=False,
                                                            trust_remote_code=True) 
            
            self.model = AutoModelForCausalLM.from_pretrained( model_path,
                                                               device_map="auto",
                                                               trust_remote_code=True)
        elif llm_name == 'qwen2-5-1.5b-instruct':
            model_path = llm_local_path

            self.tokenizer = AutoTokenizer.from_pretrained( model_path,
                                                            use_fast=False,
                                                            trust_remote_code=True) 
            
            self.model = AutoModelForCausalLM.from_pretrained( model_path,
                                                               device_map="auto",
                                                               trust_remote_code=True)
            
        print('llm model initialized successfully.')
          
            
    def initialize_ehr_model(self, ehr_model, ehr_model_path):
        '''
            initialize ehr model
        '''
        self.ehr_model = ehr_model
        state_dict = torch.load(ehr_model_path, map_location='cuda:2')
        self.ehr_model.load_state_dict(state_dict)
        print('EHR model initialized successfully.')
        

    def get_basic_info(self):
        print('------ basic llm information ------')
        print('llm name: ', self.llm_name)
        print('task name: ', self.task_name)
        print('inference type', self.inference_type)



    def get_visit_level_probing_prompt(self,
                                       patient_nl):
        
        # visit sequence like: <Visit 1> <Visit 2> ... <Visit n>
        visit_len = len(patient_nl)
        visit_placeholders = [f"<Visit {i+1}>" for i in range(visit_len)]
        visit_sequence = "\n".join(visit_placeholders)

        # wrap prompt
        visit_level_probing_prompt = (
        "Here is the current patient's <Visit Sequence>:\n"
        f"{visit_sequence}\n"
        "Now, based on prior reasoning process, "
        "the most critical visits are"
        )
        
        return visit_level_probing_prompt

    def get_llm_medical_attention_with_logits(self, 
                                              patient_nl,
                                              patient_prompt,
                                              inference_type = 'deep_seek_r1',
                                              candicate_ans_list = ['A', 'B']):
        '''
            TODO: calculate llm towards EHR visits attention 
        args:
            patient_nl: list[visit_nl 1, visit_nl 2, ..., visit_nl n]
                list of visit_nl
            patient_prompt: str
                prompt of each patient for llm
            inference_type: str
                how llm inference
            candicate_ans_list: list
                list of candicate answers

        returns:
            visit_level_attention:
                which visit is more important
            feature_level_attention
                which feature is more important

        notice:
            visit list of each patient : [visit1, visit2, ..., visit n]
            feature list of each patient: [codename1, codename2, codename3]

            forwardpass to cal attention
        '''
        

        '''
            visit-level attention probing

                if visit_len == 1, assert visit attention == 1
                else rank visits and caculate visit importance via rank

            feature-level attention probing 
        '''
        # get llm response
        logits, history_add_llm_response = self.get_answer_logits(prompt=patient_prompt, 
                                                                  inference_type = inference_type, 
                                                                  candicate_ans_list=candicate_ans_list,
                                                                  return_llm_answer = True)
        '''
            history_add_llm_response
                is a template format including question and answer by llm
        '''
        print('llm response: ', history_add_llm_response)

        # cal llm visit-level attention
        '''
            TODO:
                1.provide llm response concated with visit-level probing prompt
                2.probing llm attention
        '''

        # get visit level probing prompt
        visit_level_probing_prompt = self.get_visit_level_probing_prompt(patient_nl)
        print('visit-level probing prompt: ', visit_level_probing_prompt)

        # concact full probing prompt
        full_probing_prompt = history_add_llm_response + "\n\n" + visit_level_probing_prompt

        # calculate attention
        inputs = self.tokenizer(full_probing_prompt, return_tensors="pt")
        outputs = self.model(**inputs, output_attentions=True, return_dict=True)
        # shape: [num_layers, batch, num_heads, seq_len, seq_len]
        all_attentions = outputs.attentions  # decoder self-attention for decoder-only models like GPT

        # extract attention for each visit i
        visit_tokens = [f"<Visit {i+1}>" for i in range(len(patient_nl))]
        input_ids = inputs["input_ids"][0]  # shape: [seq_len]
        

        # Step 1: 找出每个 "<Visit i>" token 的位置

        
        decoded_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # 把每个 visit 对应的位置存下来
        visit_token_positions = []
        for visit_token in visit_tokens:
            # encode 单个 token 看它被 tokenized 为什么
            tok_ids = self.tokenizer.encode(visit_token, add_special_tokens=False)
            if len(tok_ids) == 1:
                visit_id = tok_ids[0]
                positions = (input_ids == visit_id).nonzero(as_tuple=True)[0].tolist()
                if positions:
                    visit_token_positions.append(positions[0])  # 只取第一个匹配位置
                else:
                    visit_token_positions.append(None)
            else:
                visit_token_positions.append(None)  # token被拆分无法单独定位

        print("Visit token positions:", visit_token_positions)

        # 选择最后一层 self-attention，shape: [batch=1, heads, seq_len, seq_len]
        last_layer_attention = all_attentions[-1][0]  # shape: [heads, seq_len, seq_len]

        # 对所有 head 取平均
        mean_attention = last_layer_attention.mean(dim=0)  # shape: [seq_len, seq_len]

        # 针对每个 visit token 的位置，计算它被其他 token注意的程度（或者它注意别人）
        visit_attentions = []
        for pos in visit_token_positions:
            if pos is not None:
                # 方法1：看这个 token 被其他 token 注意（列 attention）
                attn_score = mean_attention[:, pos].sum().item()
                visit_attentions.append(attn_score)
            else:
                visit_attentions.append(0.0)
        print("Visit attentions:", visit_attentions)
        return logits


    def get_answer_logits(self, 
                          prompt = None, 
                          inference_type = 'deep_seek_r1', 
                          candicate_ans_list = ['A', 'B'],
                          return_llm_answer = False):
        
        self.model.eval()
        # only consider first generated token
        if inference_type == 'straight_forward':
            # tokenize
            print(prompt)
            tokens = self.tokenizer(prompt, return_tensors="pt")
            input_ids = tokens['input_ids']
            
            # forward
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits

            # only use last token's logits
            next_token_logits = logits[:, -1, :]

        elif inference_type == 'deep_seek_r1':
            '''
                TODO: prompt should be concated with generated answer
            '''
            # generate answer
            prompt_in_chat_template, response_in_chat_template = self.get_response(prompt,
                                                                                   return_prompt_in_chat_template=True)
            # make sure response end with <answer>
            if "<answer>" not in response_in_chat_template:
                if response_in_chat_template.endswith("<|im_end|>"):
                    response_in_chat_template = response_in_chat_template.rstrip("<|im_end|>").strip()

                response = response_in_chat_template + "<answer> "
            else:
                response = re.split(r"(<answer>)", response_in_chat_template)[0] + "<answer> "
            
            assert response.endswith("<answer> "), f"Error: response does not end with '<answer> '. Actual response: {response}"
            
            
            full_text = prompt_in_chat_template + response
            print(full_text)

            # forward
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            next_token_logits = logits[:, -1, :]

        # get candidate answer token logits
        candidate_ans_logits_list = []
        for candidate_ans in candicate_ans_list:
            candidate_ans_ids = self.tokenizer.convert_tokens_to_ids(candidate_ans)
            candidate_ans_logits = next_token_logits[0, candidate_ans_ids]
            candidate_ans_logits_list.append(candidate_ans_logits)
            
        # normalization
        candidate_ans_probs_list = F.softmax(torch.tensor(candidate_ans_logits_list), dim=0).tolist()

        # only get label 1 (A) answer probility
        positive_probility = candidate_ans_probs_list[0]

        if return_llm_answer:
            #  get llm predict label based on positive_probility
            if positive_probility > 0.5:
                llm_predict_label = 'A'
            else:
                llm_predict_label = 'B'

            if inference_type == 'straight_forward':
                history_add_llm_response  =  llm_predict_label
            elif inference_type == 'deep_seek_r1':
                history_add_llm_response  = full_text + llm_predict_label +  ' </answer>'

            return positive_probility, history_add_llm_response

        return positive_probility 


    def batch_get_next_token_with_logits(self, 
                                         batch_prompt, 
                                         inference_type = 'deep_seek_r1', 
                                         candicate_ans_list = ['A', 'B']):
        batch_logits = []
        for prompt in batch_prompt:
            logits = self.get_answer_logits(prompt=prompt, 
                                            inference_type = inference_type, 
                                            candicate_ans_list=candicate_ans_list)
            print(logits)
            batch_logits.append(logits)
        batch_logits = torch.tensor(batch_logits).reshape(-1,1)
        return batch_logits

    def batch_get_next_token_with_logits_with_attention(self, 
                                                        batch_nl,
                                                        batch_prompt, 
                                                        inference_type = 'deep_seek_r1', 
                                                        candicate_ans_list = ['A', 'B']):
        batch_logits = []
        for patient_nl,patient_prompt in zip(batch_nl, batch_prompt):
            logits = self.get_llm_medical_attention_with_logits(          
                                                                patient_nl = patient_nl,
                                                                patient_prompt = patient_prompt,
                                                                inference_type = inference_type, 
                                                                candicate_ans_list=candicate_ans_list)
            print(logits)
            batch_logits.append(logits)
        batch_logits = torch.tensor(batch_logits).reshape(-1,1)
        return batch_logits


    def get_response(self, 
                     prompt = None,
                     return_prompt_in_chat_template = False):
        '''
            args:
                return_prompt_in_chat_template: bool
                    True, only return response
                    False, will return prompt and response
        '''
        '''
            for single prompt, return llm answer
        '''
        if self.is_api:
            # api request answer
            if self.llm_name == "deepseek-r1-local-preview":
                response = self.client.chat.completions.create(
                model="deepseek-r1-local-preview",
                messages=[{"role": "user", "content": prompt}],)
                response = response.choices[0].message.content
        else:
            # local generate answer
            if self.llm_name == "qwen2-5-7b-instruct":
                messages = [{"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}, ]
                text = self.tokenizer.apply_chat_template(messages,
                                                          tokenize=False,
                                                          add_generation_prompt=True,)
                inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens= 2048,
                        temperature = 0.7,   
                        top_p = 0.8, 
                        top_k = 20,              
                        do_sample=True, 
                        repetition_penalty = 1.05,           
                        eos_token_id=self.tokenizer.eos_token_id)
                
                
                # only return new generated token by llm
                input_ids = inputs.input_ids
                prompt_in_chat_template = response = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)[0]
                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
                response_in_chat_template = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
                
                if return_prompt_in_chat_template:
                    return prompt_in_chat_template, response_in_chat_template
        return response
    
    def batch_get_response(self, batch_prompt):
        '''
            for batch of prompts, return batch of llm answers
        '''
        batch_response = []
        for prompt in batch_prompt:
            try:
                response = self.get_response(prompt)
                print(response)
                batch_response.append(response)
            except Exception as e:
                print(f"Error: {str(e)}")
        return batch_response
    
    def transform_codes2nl(self, visit_cut_len = 10, code_cut_len =512, warning_msg=False, **kwargs):
        '''
            Transform medical code to English(natural language)
        '''
        # Initial mapping table
        self.code_vocs = self.dataset.code_vocs
        diagnosis_map_table = InnerMap.load(self.code_vocs['conditions'])
        procedures_map_table = InnerMap.load(self.code_vocs['procedures'])
        prescriptions_map_table = InnerMap.load(self.code_vocs['drugs'])

        
        batch_nl = []  
        patient_num = len(kwargs['visit_id'])

        for pid in range(patient_num):
            # for each patient
            patient_nl = []
            visit_id = kwargs['visit_id'][pid]
            patient_id = kwargs['patient_id'][pid]
            
            # each patient conditions, procedures, drugs cut off to max visit_cut_len
            conditions_list = kwargs['conditions'][pid][-visit_cut_len:]
            procedures_list = kwargs['procedures'][pid][-visit_cut_len:]
            drugs_list = kwargs['drugs'][pid][-visit_cut_len:]

            assert len(conditions_list) == len(procedures_list) == len(drugs_list)
            
            for conditions, procedures, drugs in zip(conditions_list, procedures_list, drugs_list):
                # for each visit
                visit_nl = {}

                visit_nl['conditions'] = []
                visit_nl['procedures'] = []
                visit_nl['drugs'] = []

                # cut off code to code_cut_len
                conditions = conditions[-code_cut_len:]
                procedures = procedures[-code_cut_len:]
                drugs = drugs[-code_cut_len:]

                # map code to EN
                for diagnosis_code in conditions:
                    try:
                        diagnosis_nl = diagnosis_map_table.lookup(diagnosis_code)
                        visit_nl['conditions'].append(diagnosis_nl)
                    except Exception as e:
                        if warning_msg:
                            print('Warning: ',diagnosis_code, 'could not be found')

                for procedures_code in procedures:
                    try:
                        procedures_nl = procedures_map_table.lookup(procedures_code)
                        visit_nl['procedures'].append(procedures_nl)
                    except Exception as e:
                        if warning_msg:
                            print('Warning: ',procedures_code, 'could not be found')

                for prescriptions_code in drugs:
                    try:
                        prescriptions_nl = prescriptions_map_table.lookup(prescriptions_code)
                        visit_nl['drugs'].append(prescriptions_nl)
                    except Exception as e:
                        if warning_msg:
                            print('Warning: ',prescriptions_code, 'could not be found')
                
                patient_nl.append(visit_nl)
            
            batch_nl.append(patient_nl)
        
        return batch_nl


    def task_head_generate(self, task_name='mortality_prediction'):
        '''
            get task specific prompt for llm
            telling llm what to do
        '''
        if task_name not in ['mortality_prediction', 'readmission_prediction']:
            raise ValueError(
                    f"task type {task_name} not implemented"
                )
        
        # for different task_name, get different task_head
        if task_name == 'mortality_prediction':
            # task_head = '<Question>: What is the likelihood that the patient will die within the next 14 days? Select one of the following options: A. Probability greater than 50%. B. Probability less than 50%.\n'
            task_head = '<Question>: will the patient die within the next 14 days? Select one of the following options: A. will die. B. will not die.\n'
        
        elif task_name == 'readmission_prediction':
            task_head = '<Question>: Will the patient be readmitted to the hospital within two weeks? Select one of the following options: A. Yes. B. No.\n'
        return task_head
    
    def inference_head_generate(self, inference_type = 'straight_forward'):
        '''
            tell llm how to inference, like 'think step by step' or 'give answer straight_forward'
        '''
        if inference_type not in ['straight_forward', 'deep_seek_r1']:
            raise ValueError(
                    f"task type {inference_type} not implemented"
                )
        
        if inference_type == 'straight_forward':
            # straight forward give answer without any explannation
            inference_head = 'Important: Provide only the letter corresponding to your chosen answer. Do not include any explanation or additional text. Your answer is:'
        elif inference_type == 'deep_seek_r1':
            # deep_seek_r1 style 'think and answer'
            inference_head = 'Important: First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. '
        return inference_head
    
    def prompt_generate(self, 
                        batch_nl, 
                        task_name='mortality_prediction', 
                        inference_type = 'straight_forward'):
        '''
            wrap all patient EHR data into prompt for llm 
        '''

        batch_prompt = []

        identity_head = 'You are a medical expert with extensive knowledge in analyzing electronic health records (EHR).\n'
        
        data_description_head =\
                'The structured temporal electronic health records (EHR) data is identified by <Visit Sequence>. It includes the patient’s multiple visits to medical facilities, capturing diagnosed diseases, laboratory test information, and medication details.\n'

        task_head = self.task_head_generate(task_name=task_name)

        inference_head = self.inference_head_generate(inference_type=inference_type)

        for patient_nl in tqdm(batch_nl, desc="Wrap LLM prompt", total=len(batch_nl)):
            
            current_patient_information = "Here is the current patient's <Visit Sequence>:\n"
            
            for i,visit_nl in enumerate(patient_nl):
                current_patient_information += '<Visit ' + str(i+1) + '>:\n'
                for (key, value) in visit_nl.items():
                    current_patient_information += '\t' +  '[' + str(key) + ']' + ':'+ '\n'
                    for code in value:
                        current_patient_information += '\t\t' + code + '\n'

            patient_prompt = identity_head + data_description_head + current_patient_information + task_head + inference_head
            batch_prompt.append(patient_prompt)
        
        return batch_prompt


    def batch_forward(self, 
                      task_name = 'mortality_prediction', 
                      EHR_model_prompt_injection = False,
                      inference_type = 'deep_seek_r1',  
                      **kwargs):
        # TODO: process EHR data as small EHR model does

        if EHR_model_prompt_injection:
            # forwardpass EHR model
            self.ehr_model.eval()
            with torch.no_grad():
                # TODO: put attention prompt into outputs
                outputs = self.ehr_model(**kwargs)
                loss = outputs['loss']
                y_true = outputs['y_true'].cpu().numpy()
                y_prob = outputs['y_prob'].cpu().numpy()


        # transform codes to natural language
        batch_nl = self.transform_codes2nl(**kwargs)

        # generate prompt
        batch_prompt = self.prompt_generate(batch_nl,
                                            task_name = task_name, 
                                            inference_type = inference_type)
        
        # TODO: after generating complete answer, how to get logits

        # batch_answer = self.batch_get_response(batch_prompt)


        batch_logits = self.batch_get_next_token_with_logits_with_attention(
                                                                            batch_nl = batch_nl,
                                                                            batch_prompt = batch_prompt,
                                                                            inference_type = inference_type)
        y_prob = batch_logits
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)

        return y_prob, y_true


    def inference(self, dataloader):

        y_true_all = []
        y_prob_all = []

        # inference
        for data in tqdm(dataloader, desc="Evaluation"):
            y_prob, y_true = self.batch_forward(task_name=self.task_name,
                                                inference_type=self.inference_type,
                                                EHR_model_prompt_injection=self.EHR_model_prompt_injection,
                                                **data)
            y_prob = y_prob.cpu().numpy()
            y_true = y_true.cpu().numpy()
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)

        y_true_all = np.concatenate(y_true_all, axis=0)
        y_prob_all = np.concatenate(y_prob_all, axis=0)

        outputs = [y_true_all, y_prob_all]


        print(y_true_all)
        print(y_prob_all)

        # evaluate
        metrics_fn = get_metrics_fn(self.mode)
        scores = metrics_fn(y_true_all, y_prob_all, metrics=self.metrics)
        print(scores)

        return outputs



if __name__ == "__main__":
    
    llm_worker = LLM_worker(llm_name='qwen2-5-7b-instruct', dataset=None, feature_keys=None, label_key=None)

    llm_worker.get_next_token_with_logits('The patient was diagnosed with')

    