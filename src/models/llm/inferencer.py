import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader

from pyhealth.utils import load_pickle
from pyhealth.datasets import collate_fn_dict
from pyhealth.trainer import Trainer

from LLM_worker import LLM_worker
from star_trainer import STaRTrainer

import sys
sys.path.append('/data/fy/fy/codebase/EHRPO/src')

from utils.dataset_remake import MIMIC4Dataset, eICUDataset, MIMIC3Dataset
from utils.task_remake import patient_train_val_test_split
from utils.task_remake import patient_level_mortality_prediction_mimic3, patient_level_mortality_prediction_mimic4, patient_level_mortality_prediction_eicu
from utils.task_remake import patient_level_readmission_prediction_mimic3

import argparse


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference or STaR training on medical data")
    parser.add_argument("--mode", type=str, choices=["inference", "star"], default="star",
                        help="Run mode: inference or STaR training")
    parser.add_argument("--star_iterations", type=int, default=3,
                        help="Number of STaR iterations to run")
    parser.add_argument("--output_dir", type=str, default="./star_outputs",
                        help="Directory to save STaR outputs")
    return parser.parse_args()

def deepseek_inferencer():
    print("GPU count:", torch.cuda.device_count())
    print("Current GPU index:", torch.cuda.current_device())
    args = parse_args()
    
    logger = logging.getLogger("pyhealth")
    logger.setLevel(logging.DEBUG)
    print(os.getcwd())
    open("mcts_search_log.txt", 'w')

    dataset = 'mimic3'  

    seed = 1128
    set_random_seed(seed)

    if dataset == 'mimic4':
        mimic4_base = MIMIC4Dataset(
              root="/data1/jxk/MedicalData/physionet.org/files/mimiciv/2.2/hosp",
              tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"],
              code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
         )
        
        mimic4sample = mimic4_base.set_task(task_fn=patient_level_mortality_prediction_mimic4)
        x_key = ["conditions", "procedures", "drugs"]
        mimic4sample = load_pickle(r"sample_dataset_mimiciv_mortality_multifea.pkl")

    elif dataset == 'mimic3':
        mimic3base = MIMIC3Dataset(
            root="/data/fy/fy/codebase/Hitanet_rebuild/data/raw_data/MIMICIII_data",
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        )
        
        mimic4sample = mimic3base.set_task(task_fn=patient_level_readmission_prediction_mimic3)

        x_key = ["conditions", "procedures", "drugs"]

    elif dataset == 'eICU':
        mimic3base = eICUDataset(
            root="/home/jxk/physionet.org/files/eicu-crd/2.0",
            tables=["diagnosis", "medication", "lab", "treatment", "physicalExam", "admissionDx"],
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        )
        mimic4sample = mimic3base.set_task(task_fn=patient_level_mortality_prediction_eicu)
        x_key = ["conditions"]

    train_ds, val_ds, test_ds = patient_train_val_test_split(mimic4sample, [0.8, 0.1, 0.1], seed)

    from torch.utils.data import Subset
    partial_test_ds = Subset(test_ds, indices=range(8))
    partial_train_ds = Subset(train_ds, indices=range(128))

    partial_train_loader = DataLoader(
        partial_train_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn_dict,
        drop_last=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=8,
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
        batch_size=20,
        shuffle=False,
        collate_fn=collate_fn_dict,
        drop_last=True
    )
    partial_test_loader = DataLoader(
        partial_test_ds,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn_dict,
        drop_last=True
    )

    model = LLM_worker(
        ehr_model=None,
        dataset=mimic4sample,
        feature_keys=x_key,
        label_key="label",
        mode="binary",
        llm_name = 'qwen2-5-1.5b-instruct',
        is_api = False,
        inference_type = 'mcts',
        task_name = 'readmission_prediction',
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
        ]
    )

    if args.mode == "inference":
        print("Starting Inference")
        model.inference(test_loader)
    elif args.mode == "star":
        print("Starting STaR Training")
        star_trainer = STaRTrainer(
            llm_worker=model,
            train_loader=partial_train_loader,
            test_loader=test_loader,
            partial_test_loader=partial_test_loader,
            output_dir=args.output_dir,
            n_iterations=args.star_iterations,
            p_rationalization=1.0,
            mcts_rollouts=5,
            max_depth=4
        )
        star_trainer.run()


if __name__=='__main__':
    deepseek_inferencer()

