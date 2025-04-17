# pyhealth task for timeseries
from pyhealth.data import Patient, Visit
from collections import OrderedDict
from operator import attrgetter

def sort_visits_by_encounter_time(patient: Patient):
    '''Sort patient's visit by encountered time
    '''
    raw_vist_oderedcist = patient.visits
    sorted_vist_oderedcist = OrderedDict(sorted(raw_vist_oderedcist.items(), key=lambda x: attrgetter('encounter_time')(x[1])))
    patient.visits = sorted_vist_oderedcist
    return patient

#TODO 合并服务器上已写的相对时间处理
'''
根据preprocess_new中时间预处理的步骤，我们预处理如下:
用encounter_time作为衡量的指标
1. 将绝对的日期改成相对于该用户第一次访问时间的相对时间days
2. 筛选history_visits时用>observation days(365)来筛选出合法的histpry_visits
3. 每个病人我们只考虑倒数50个visits我们认为它是值得考虑的
4. 最后再用合法的日期的final-之前的相对时间
if len(patient_time_list) > 0:
    final_time = patient_time_list[-1]
    patient_time_list = [final_time - visit_time for visit_time in patient_time_list]
pyhealth对数据的结构有严格的要求，我们这里后续处理需要在loader中完成
'''
def mortality_prediction_time_series_mimic3_fn(patient: Patient):
    ''' Custom version for time series mortality prediction

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id,
            visit_series, labels

    Notice visit_series should be a list of this patient's visits

    '''
    samples = []
    # time_steps = []
    # start_time = None
    observation_days = 365
    max_visit_num = 50
    patient.sort_visits()
    patient.check_visits()
    for i in range(len(patient)-1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i+1]
        history_visits = [patient[index] for index in range(i+1)]

        if next_visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(next_visit.discharge_status)
        if mortality_label == 1:
            final_label = [1]
        else:
            final_label = [0]
        #convert to relative times, step 1
        # if start_time == None:
        #     start_time = visit.encounter_time
        #     time_steps.append(0)
        # else:
        #     time_steps.append((visit.encounter_time - start_time).days)

        # make sure time series are meaningful
        valid = True
        for index in range(len(history_visits)-1):
            # this visit time must be earlier than later visit
            assert history_visits[index].encounter_time < history_visits[index+1].encounter_time
        for index in range(len(history_visits)):
            # patient can not be dead firstly then alive later
            if history_visits[index].discharge_status == 1:
                valid = False

        # For each visit in history_visits, we should extract all types of medical codes and concat them together
        history_visits_code_list = []
        history_visits = []
        # visit_sequence_len_dim2_array = []
        for pre_visit in range(i, -1, -1):
            previsit = patient[pre_visit]
            if (visit.encounter_time - previsit.encounter_time).days >= observation_days: #距离太远的记录没什么参考价值,step2
                break
            # For each visit, get medical code list
            conditions = previsit.get_code_list(table="DIAGNOSES_ICD")
            procedures = previsit.get_code_list(table="PROCEDURES_ICD")
            drugs = previsit.get_code_list(table="PRESCRIPTIONS")
            # Concat all medical codes
            all_codes_of_each_vist = conditions + procedures + drugs
            history_visits.insert(0, previsit)
            history_visits_code_list.insert(0, all_codes_of_each_vist)
            # visit_sequence_len_dim2_array.insert(0, len(all_codes_of_each_vist))

        #time handle step 4
        # final_time = time_steps[-1]
        # final_time_steps = [final_time - visit_time for visit_time in time_steps]

        # visit_sequence_len_dim2_array = visit_sequence_len_dim2_array #to avoid key has vectors of different lengths
        # final_time_steps = final_time_steps #to avoid key has vectors of different lengths

        if valid:
            samples.append(
                {
                    "visit_id": visit.visit_id,
                    "visit_sequence": history_visits, # list[visit object]
                    "visit_sequence_codes": history_visits_code_list, # list[list[code]]
                    "patient_id": patient.patient_id,
                    # "visit_sequence_len_dim2_array": visit_sequence_len_dim2_array, # list[int]
                    "label": final_label,
                    # "time": final_time_steps,
                }
            )
    return samples[-max_visit_num:] #limit the number of effective visits, step3

if __name__ == "__main__":
    from pyhealth.datasets import MIMIC3Dataset

    base_dataset = MIMIC3Dataset(
        root="/data/fy/fy/database/mimic/MIMICIII_data",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        dev=True,
        code_mapping={"ICD9CM": "CCSCM", "NDC": "ATC"},
        refresh_cache=False,
    )

    sample_dataset = base_dataset.set_task(mortality_prediction_time_series_mimic3_fn)