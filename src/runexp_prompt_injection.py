import os
import torch
import logging
import numpy as np
import pickle
from torch.utils.data import DataLoader

from pyhealth.utils import load_pickle
from pyhealth.datasets import collate_fn_dict

from models.hitanet.hitanet import HitaTransformer
from models.llm.LLM_worker import LLM_worker

from utils.trainer import Trainer
from utils.dataset_remake import MIMIC3Dataset
from utils.task_remake import (patient_train_val_test_split,
                               patient_level_mortality_prediction_mimic3,
                               patient_level_readmission_prediction_mimic3)

from args.ehrpo_args import parse_args

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_logger():
    logger = logging.getLogger("pyhealth")
    logger.setLevel(logging.DEBUG)
    print(os.getcwd())
    return logger

def set_dataset(dataset = 'mimic3', dataset_path = None, task_name = 'readmission_prediction'):
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(os.getcwd(), 'cache/data')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache file name based on dataset and task
    cache_file = os.path.join(cache_dir, f'{dataset}_{task_name}.pkl')
    
    # Check if cache exists
    if os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}")
        with open(cache_file, 'rb') as f:
            dataset_sample, x_key = pickle.load(f)
        return dataset_sample, x_key
    
    if dataset == 'mimic3':
        mimic3base = MIMIC3Dataset(
            root= dataset_path,
            tables= ["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            code_mapping= {"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        )
        
        x_key = ["conditions", "procedures", "drugs"]

        if task_name == 'readmission_prediction':
            dataset_sample = mimic3base.set_task(task_fn=patient_level_readmission_prediction_mimic3)
        elif task_name == 'mortality_prediction':
            dataset_sample = mimic3base.set_task(task_fn=patient_level_mortality_prediction_mimic3)
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    
    # Save to cache
    print(f"Saving dataset to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump((dataset_sample, x_key), f)
    
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

def set_ehr_model(dataset_sample, 
                  x_key, 
                  dataloader, 
                  dataset = None,
                  task_name = None):
    
    ehr_model = HitaTransformer(
        dataset=dataset_sample,
        feature_keys=x_key,
        label_key="label",
        mode="binary",
    )

    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(os.getcwd(), 'cache/models')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache file name based on dataset and task
    dataset_name = dataset
    task_name = task_name
    cache_file = os.path.join(cache_dir, f'{dataset_name}_{task_name}.ckpt')
    
    # Check if cache exists
    if os.path.exists(cache_file):
        print(f"Loading cached model from {cache_file}")
        ehr_model.load_state_dict(torch.load(cache_file))
        return ehr_model

    else:
        print("Start Training EHR Model from Scratch...")

        train_dataloader, val_dataloader, test_dataloader = dataloader
        ehr_model.train_mode = 'FineTune'
        torch.backends.cudnn.benchmark = False
        ehr_model_trainer = Trainer(model=ehr_model, metrics=[
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
        )
        ehr_model_trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            epochs=30,
        )

        print("Evaluating EHR Model...")
        ehr_model_trainer.evaluate(test_dataloader)

        # Save model to cache
        print(f"Saving model to cache: {cache_file}")
        torch.save(ehr_model.state_dict(), cache_file)

        # Export EHR Model
        ehr_model = ehr_model_trainer.export_model()

    return ehr_model

def set_llm_worker(llm_name,
                   llm_local_path,
                   ehr_model, 
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
        llm_name = llm_name,
        llm_local_path = llm_local_path,
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



def runexp(llm_name = 'qwen2-5-7b-instruct',
           llm_local_path = None,
           dataset = 'mimic3', 
           dataset_path = None,
           seed = 1128, 
           task_name = 'readmission_prediction',
           inference_type = 'deep_seek_r1',
           EHR_model_prompt_injection = False,):
    
    set_logger()
    set_random_seed(seed)

    dataset_sample, x_key = set_dataset(dataset, dataset_path, task_name)
    
    train_loader, val_loader, test_loader = set_dataloader(dataset_sample, seed)

    ehr_model = set_ehr_model(dataset_sample=dataset_sample, 
                              x_key=x_key,
                              dataloader=[train_loader, val_loader, test_loader],
                              dataset = dataset,
                              task_name = task_name)

    llm_worker = set_llm_worker(llm_name = llm_name,
                                llm_local_path = llm_local_path,
                                ehr_model = ehr_model, 
                                dataset_sample = dataset_sample, 
                                x_key = x_key, 
                                task_name = task_name, 
                                inference_type = inference_type,
                                EHR_model_prompt_injection = EHR_model_prompt_injection)

    llm_worker.inference(test_loader)


if __name__=='__main__':

    args = parse_args()

    runexp(llm_name = args.llm_name,
           llm_local_path = args.llm_local_path,
           dataset = args.dataset, 
           dataset_path = args.dataset_path,
           seed = args.seed, 
           task_name = args.task_name,
           inference_type = args.inference_type,
           EHR_model_prompt_injection = args.EHR_model_prompt_injection)

