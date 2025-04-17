import numpy as np
from pyhealth.data import Event, Visit, Patient
from itertools import chain
from typing import Optional, Tuple, Union, List
from sklearn.model_selection import train_test_split
import numpy as np
import torch

from pyhealth.datasets import SampleBaseDataset

def patient_level_mortality_prediction_mimic4(patient, dataset='mimic4'):
    """
    patient is a <pyhealth.data.Patient> object
    """
    samples = []

    # if the patient only has one visit, we drop it     # fixme
    if len(patient) == 1:
        return []

    newpatient = Patient(
        patient_id=patient.patient_id,
    )
    encounter_resort = []
    for visit in patient:
        encounter_resort.append((visit.encounter_time - patient[0].encounter_time).days)
    reidx = sorted(range(len(encounter_resort)), key=lambda k: encounter_resort[k])
    for newid in reidx:
        newpatient.add_visit(patient[newid])
    patient = newpatient

    # step 1: define label
    idx_last_visit = len(patient) - 1
    if patient[idx_last_visit].discharge_status not in [0, 1]:
        mortality_label = 0
    else:
        mortality_label = int(patient[idx_last_visit].discharge_status)

    # step 2: obtain features
    conditions_merged = []
    procedures_merged = []
    drugs_merged = []
    delta_days = []
    for idx, visit in enumerate(patient):
        if idx == len(patient) - 1: break
        if dataset == 'mimic3':
            conditions = visit.get_code_list(table="DIAGNOSES_ICD")
            procedures = visit.get_code_list(table="PROCEDURES_ICD")
            drugs = visit.get_code_list(table="PRESCRIPTIONS")
        if dataset == 'mimic4':
            conditions = visit.get_code_list(table="diagnoses_icd")
            procedures = visit.get_code_list(table="procedures_icd")
            drugs = visit.get_code_list(table="prescriptions")

        conditions_merged += [conditions]
        procedures_merged += [procedures]
        drugs_merged += [drugs]

        if idx==0:
            delta_days += [[1]]
        if idx >= 1:
            delta_days += [[abs((visit.encounter_time - patient_tmp_out_time).days)]]

        patient_tmp_out_time = visit.encounter_time

    if drugs_merged == [] or procedures_merged == [] or conditions_merged == []:        # todo
        return []

    # uniq_conditions = list(set(conditions_merged))
    # uniq_procedures = list(set(procedures_merged))
    # uniq_drugs = list(set(drugs_merged))
    uniq_conditions = conditions_merged
    uniq_procedures = procedures_merged
    uniq_drugs = drugs_merged

    # step 3: exclusion criteria
    if len(uniq_conditions) * len(uniq_procedures) * len(uniq_drugs) == 0:
        return []

    # step 4: assemble the sample
    samples.append(
        {
            "visit_id": visit.visit_id,
            "patient_id": patient.patient_id,
            "conditions": uniq_conditions,
            "procedures": uniq_procedures,
            "delta_days": delta_days,
            "drugs": uniq_drugs,
            "label": mortality_label,
        }
    )
    return samples


def patient_level_mortality_prediction_mimic3(patient, dataset='mimic3'):
    """
    patient is a <pyhealth.data.Patient> object
    """
    samples = []

    # if the patient only has one visit, we drop it
    if len(patient) == 1:
        return []

    visit_len = len(patient)
    newpatient = Patient(
        patient_id=patient.patient_id,
    )
    encounter_resort = []

    for visit in patient:
        encounter_resort.append((visit.encounter_time - patient[0].encounter_time).days)

    reidx = sorted(range(len(encounter_resort)), key=lambda k: encounter_resort[k])
    for newid in reidx:
        newpatient.add_visit(patient[newid])
    patient = newpatient

    # step 1: define label
    idx_last_visit = len(patient) - 1
    if patient[idx_last_visit].discharge_status not in [0, 1]:
        mortality_label = 0
    else:
        mortality_label = int(patient[idx_last_visit].discharge_status)

    # step 2: obtain features
    conditions_merged = []
    procedures_merged = []
    drugs_merged = []
    delta_days = []
    # patient_tmp_out_time = 1
    patient_last_visit_encounter_time = patient[idx_last_visit].encounter_time

    for idx, visit in enumerate(patient):
        if idx == len(patient) - 1: break # 对最后一次visit，不做处理

        if dataset == 'mimic3':
            conditions = visit.get_code_list(table="DIAGNOSES_ICD")
            procedures = visit.get_code_list(table="PROCEDURES_ICD")
            drugs = visit.get_code_list(table="PRESCRIPTIONS")

        if dataset == 'mimic4':
            conditions = visit.get_code_list(table="diagnoses_icd")
            procedures = visit.get_code_list(table="procedures_icd")
            drugs = visit.get_code_list(table="prescriptions")

        conditions_merged += [conditions]
        procedures_merged += [procedures]
        drugs_merged += [drugs]

        delta_days += [ [ abs((visit.encounter_time - patient_last_visit_encounter_time).days) ] ]
        '''
        if idx==0:
            delta_days += [[1]]
        if idx >= 1:
            delta_days += [[abs((visit.encounter_time - patient_tmp_out_time).days)]] # 当前的visit距离上一次就诊的时间
        
        patient_tmp_out_time = visit.encounter_time # 更新patient_tmp_out_time为这一次visit time
        '''

        # 

    if drugs_merged == [] or procedures_merged == [] or conditions_merged == []:        # todo
        return []

    # uniq_conditions = list(set(conditions_merged))
    # uniq_procedures = list(set(procedures_merged))
    # uniq_drugs = list(set(drugs_merged))
    uniq_conditions = conditions_merged
    uniq_procedures = procedures_merged
    uniq_drugs = drugs_merged

    # step 3: exclusion criteria
    if len(uniq_conditions) * len(uniq_procedures) * len(uniq_drugs) == 0:
        return []

    # step 4: assemble the sample
    samples.append(
        {
            "visit_id": visit.visit_id,
            "patient_id": patient.patient_id,
            "conditions": uniq_conditions,
            "procedures": uniq_procedures,
            "delta_days": delta_days,
            "drugs": uniq_drugs,
            "label": mortality_label,
        }
    )
    return samples


def patient_level_mortality_prediction_eicu(patient, dataset='eicu'):
    """
    patient is a <pyhealth.data.Patient> object
    """
    samples = []

    # if the patient only has one visit, we drop it
    if len(patient) == 1:
        return []

    newpatient = Patient(
        patient_id=patient.patient_id,
    )
    encounter_resort = []
    for visit in patient:
        encounter_resort.append((visit.encounter_time - patient[0].encounter_time).days)
    reidx = sorted(range(len(encounter_resort)), key=lambda k: encounter_resort[k])
    for newid in reidx:
        newpatient.add_visit(patient[newid])
    patient = newpatient

    # step 1: define label
    idx_last_visit = len(patient) - 1
    if patient[idx_last_visit].discharge_status not in [0, 1]:
        mortality_label = 0
    else:
        mortality_label = int(patient[idx_last_visit].discharge_status)

    # step 2: obtain features
    conditions_merged = []
    procedures_merged = []
    drugs_merged = []
    delta_days = []
    for idx, visit in enumerate(patient):
        if idx == len(patient) - 1: break
        if dataset == 'mimic3':
            conditions = visit.get_code_list(table="DIAGNOSES_ICD")
            procedures = visit.get_code_list(table="PROCEDURES_ICD")
            drugs = visit.get_code_list(table="PRESCRIPTIONS")
        if dataset == 'mimic4':
            conditions = visit.get_code_list(table="diagnoses_icd")
            procedures = visit.get_code_list(table="procedures_icd")
            drugs = visit.get_code_list(table="prescriptions")
        if dataset == 'eicu':
            conditions = visit.get_code_list(table="diagnosis")
            procedures = visit.get_code_list(table="physicalExam")
            drugs = visit.get_code_list(table="medication")

        conditions_merged += [conditions]
        procedures_merged += [procedures]
        drugs_merged += [drugs]

        if idx==0:
            delta_days += [[1]]
        if idx >= 1:
            delta_days += [[abs((visit.encounter_time - patient_tmp_out_time).days)]]
        patient_tmp_out_time = visit.encounter_time

    if drugs_merged == [] or procedures_merged == [] or conditions_merged == []:        # todo
        return []

    # uniq_conditions = list(set(conditions_merged))
    # uniq_procedures = list(set(procedures_merged))
    # uniq_drugs = list(set(drugs_merged))
    uniq_conditions = conditions_merged
    uniq_procedures = procedures_merged
    uniq_drugs = drugs_merged

    # step 3: exclusion criteria
    if len(uniq_conditions) * len(uniq_procedures) * len(uniq_drugs) == 0:
        return []
    flat_uniq_conditions = list(np.array(uniq_conditions, dtype=object).flatten())
    flat_uniq_procedures = list(np.array(uniq_procedures, dtype=object).flatten())
    flat_uniq_drugs = list(np.array(uniq_drugs, dtype=object).flatten())

    if flat_uniq_conditions == [] and flat_uniq_procedures == [] and flat_uniq_drugs == []:
        return []

    # step 4: assemble the sample
    samples.append(
        {
            "visit_id": visit.visit_id,
            "patient_id": patient.patient_id,
            "conditions": uniq_conditions,
            "procedures": uniq_procedures,
            "delta_days": delta_days,
            "label": mortality_label,
        }
    )
    return samples


def single_visit_mortality_prediction_mimic3_fn(patient, dataset='eicu'):
    samples = []

    if len(patient) > 1:
        return []
    else:
        visit: Visit = patient[0]
        if visit.discharge_status not in [0, 1]:
            mortality_label = 0
            # fixme
            # return []
        else:
            mortality_label = int(visit.discharge_status)
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # exclude: visits without condition, procedure, and drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            return []
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": mortality_label,
            }
        )
    # no cohort selection
    return samples


def single_visit_mortality_prediction_mimic4_fn(patient, dataset='eicu'):
    samples = []

    if len(patient) > 1:
        return []
    else:
        visit: Visit = patient[0]
        if visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(visit.discharge_status)
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # exclude: visits without condition, procedure, and drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            return []
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": mortality_label,
            }
        )
    # no cohort selection
    return samples


def patient_train_val_test_split(
        dataset: SampleBaseDataset,
        ratios: Union[Tuple[float, float, float], List[float]],
        seed: Optional[int] = None,
):
    """Splits the dataset by patient.

    Args:
        dataset: a `SampleBaseDataset` object
        ratios: a list/tuple of ratios for train / val / test
        seed: random seed for shuffling the dataset

    Returns:
        train_dataset, val_dataset, test_dataset: three subsets of the dataset of
            type `torch.utils.data.Subset`.

    Note:
        The original dataset can be accessed by `train_dataset.dataset`,
            `val_dataset.dataset`, and `test_dataset.dataset`.
    """
    if seed is not None:
        np.random.seed(seed)
    # assert sum(ratios) == 1.0, "ratios must sum to 1.0"
    patient_indx = list(range(0, len(dataset), 1))
    label_list = [sample["label"] for sample in dataset]
    temp_index, test_index, y_temp, y_test = \
        train_test_split(patient_indx, label_list, test_size=ratios[2], stratify=label_list, random_state=seed)
    train_index, val_index, y_train, y_val = train_test_split(temp_index, y_temp,
                                                              test_size=ratios[1] / ratios[0] + ratios[1],
                                                              stratify=y_temp,
                                                              random_state=seed)

    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)
    return train_dataset, val_dataset, test_dataset

