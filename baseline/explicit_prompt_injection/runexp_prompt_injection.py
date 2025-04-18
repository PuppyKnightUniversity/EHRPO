import os
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader

from pyhealth.utils import load_pickle
from pyhealth.datasets import collate_fn_dict
from pyhealth.trainer import Trainer

from models.hitanet import HitaTransformer

from models.LLM_worker import LLM_worker

from utils.dataset_remake import MIMIC3Dataset
from utils.task_remake import (patient_train_val_test_split,
                               patient_level_mortality_prediction_mimic3,
                               patient_level_readmission_prediction_mimic3)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,1,0,4,5'

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_logger():
    logger = logging.getLogger("pyhealth")
    logger.setLevel(logging.DEBUG)
    print(os.getcwd())
    return logger

def set_dataset(dataset = 'mimic3', task_name = 'readmission_prediction'):
    if dataset == 'mimic3':
        mimic3base = MIMIC3Dataset(
            root="/data1/xiaobei/codebase/EHRPO/baseline/hitanet_modified/data/raw_data/MIMICIII_data",
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        )
        
        x_key = ["conditions", "procedures", "drugs"]

        if task_name == 'readmission_prediction':
            dataset_sample = mimic3base.set_task(task_fn=patient_level_readmission_prediction_mimic3)
        elif task_name == 'mortality_prediction':
            dataset_sample = mimic3base.set_task(task_fn=patient_level_mortality_prediction_mimic3)
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    
    return dataset_sample, x_key

def set_dataloader(dataset_sample, seed = 1128):
    train_ds, val_ds, test_ds = patient_train_val_test_split(dataset_sample, [0.8, 0.1, 0.1], seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=300,
        shuffle=False,
        collate_fn=collate_fn_dict,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=300,
        shuffle=False,
        collate_fn=collate_fn_dict,
        drop_last=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=200,
        shuffle=False,
        collate_fn=collate_fn_dict,
        drop_last=True
    )

    return train_loader, val_loader, test_loader

def set_ehr_model(dataset_sample, x_key):
    
    ehr_model = HitaTransformer(
        dataset=dataset_sample,
        feature_keys=x_key,
        label_key="label",
        mode="binary",
    )

    return ehr_model

def set_llm_worker(ehr_model, 
                   dataset_sample, 
                   x_key, 
                   task_name, 
                   inference_type = 'deep_seek_r1',
                   EHR_model_prompt_injection = False):
    llm_worker = LLM_worker(
        ehr_model = ehr_model,
        ehr_model_path = "/data1/xiaobei/codebase/EHRPO_v2/baseline/explicit_prompt_injection/output/20250416-123103/last.ckpt",
        dataset=dataset_sample,
        feature_keys=x_key,
        label_key="label",
        mode="binary",
        llm_name = 'qwen2-5-7b-instruct',
        is_api = False,
        metrics=[
            "pr_auc",
            "roc_auc",
            "accuracy",
            "balanced_accuracy",
            "f1",
            "precision",
            "recall",
            "cohen_kappa",
            "jaccard",
            "ECE"
        ],
        inference_type = inference_type,
        task_name = task_name,  
        EHR_model_prompt_injection = EHR_model_prompt_injection
    )

    return llm_worker



def runexp(dataset = 'mimic3', 
           seed = 1128, 
           task_name = 'readmission_prediction',
           inference_type = 'deep_seek_r1',
           EHR_model_prompt_injection = False):
    
    set_logger()
    set_random_seed(seed)

    dataset_sample, x_key = set_dataset(dataset, task_name)
    
    train_loader, val_loader, test_loader = set_dataloader(dataset_sample, seed)

    ehr_model = set_ehr_model(dataset_sample, x_key)

    llm_worker = set_llm_worker(ehr_model, 
                                dataset_sample, 
                                x_key, 
                                task_name, 
                                inference_type,
                                EHR_model_prompt_injection = EHR_model_prompt_injection)

    llm_worker.inference(test_loader)


if __name__=='__main__':
    
    runexp(dataset = 'mimic3', 
           seed = 1128, 
           task_name = 'mortality_prediction',
           inference_type = 'straight_forward',
           EHR_model_prompt_injection = True)

