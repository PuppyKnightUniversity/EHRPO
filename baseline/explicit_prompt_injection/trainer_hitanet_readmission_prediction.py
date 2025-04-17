# Task1: Data Load
import os
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader

from pyhealth.utils import load_pickle
from pyhealth.datasets import collate_fn_dict


from models.hitanet import HitaTransformer

from utils.dataset_remake import MIMIC4Dataset, eICUDataset, MIMIC3Dataset
from utils.task_remake import patient_train_val_test_split
from utils.task_remake import patient_level_mortality_prediction_mimic3, patient_level_mortality_prediction_mimic4, patient_level_mortality_prediction_eicu
from utils.task_remake import patient_level_readmission_prediction_mimic3
from utils.trainer import Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def hitanet_trainer():
    
    logger = logging.getLogger("pyhealth")
    logger.setLevel(logging.DEBUG)
    print(os.getcwd())

    dataset = 'mimic3'  

    seed = 789
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
            root="/data1/xiaobei/codebase/EHRPO/baseline/hitanet_modified/data/raw_data/MIMICIII_data",
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

    model = HitaTransformer(
        dataset=mimic4sample,
        feature_keys=x_key,
        label_key="label",
        mode="binary",
    )

    print("Start FineTune")
    model.train_mode = 'FineTune'
    torch.backends.cudnn.benchmark = False
    ehr_model_trainer = Trainer(model=model, metrics=[
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
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        epochs=30,
        #monitor="pr_auc",
    )

    ehr_model_trainer.evaluate(test_loader)

if __name__=='__main__':
    hitanet_trainer()

