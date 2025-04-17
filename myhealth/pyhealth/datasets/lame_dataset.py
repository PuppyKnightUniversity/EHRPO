# This script is used for building lame dataset
# transform each sample's visit medical code sequence to NLP
from typing import Dict, List
from tqdm import tqdm
from pyhealth.medcode import InnerMap
from pyhealth.datasets.sample_dataset_ts import SampleEHRDataset_ts
import json
import pickle
from pyhealth.llm import XiaoBei
from pyhealth.lameutils import Vocab
import numpy as np

class LameDataset(SampleEHRDataset_ts):
    def __init__(self,
                 samples: List[Dict],
                 code_vocs=None,
                 dataset_name="",
                 task_name="",
                 load_medical_knowledge= False,
                 load_medical_knowledge_path = './xiaobei.pkl'):

        super().__init__(samples, code_vocs, dataset_name, task_name)
        self.llm_list = ['xiaobei']
        self.code_all_set = set()
        # load medical knowledge
        if load_medical_knowledge:
            with open(load_medical_knowledge_path, 'rb') as f:
                samples_with_medical_knowledge = pickle.load(f)
            print("self.samples length:\n", len(self.samples))
            print("samples_with_medical_knowledge length:\n", len(samples_with_medical_knowledge))
            for i in range(len(self.samples)):
                self.samples[i]['medical_knowledge_dict'] = samples_with_medical_knowledge[i]['medical_knowledge_dict']
                self.samples[i]['medical_knowledge_dict']['medical_knowledge'] = self.samples[i]['medical_knowledge_dict']['medical_knowledge'][100:]
            #ablation study
            for i in range(len(self.samples)):
                self.samples[i]['medical_knowledge_dict'] = samples_with_medical_knowledge[i]['medical_knowledge_dict']
                self.samples[i]['medical_knowledge_dict']['medical_knowledge'] = ""
        else:
            # if no knowledge exist, generate medical knowledge
            self.transform_codes2nl()
            self.transform_nl2prompt()
            for sample in self.samples:
                assert len(sample['visit_sequence']) == len(sample['visit_sequence_nl'])
            self.generate_medical_knowledge(dump_samples=True)

        self.tokenize_medical_code()
        self.tokenize_medical_knowledge()
        self.adapt()
        
    # 1.transform medical code to natural language
    def transform_codes2nl(self, warning_msg = False):
        '''Transform medical code to English(natural language)
        '''
        # Initial mapping table
        diagnosis_map_table = InnerMap.load(self.code_vocs['conditions'])
        procedures_map_table = InnerMap.load(self.code_vocs['procedures'])
        prescriptions_map_table = InnerMap.load(self.code_vocs['drugs'])

        for sample in tqdm(self.samples, desc="Transform medical codes to English", total=len(self.samples)):
            visit_sequence = sample['visit_sequence']
            visit_sequence_nl = []
            for visit in visit_sequence:
                # 3 types of medical codes
                diagnosis_code_list = visit.get_code_list('DIAGNOSES_ICD')
                procedures_code_list = visit.get_code_list('PROCEDURES_ICD')
                prescriptions_code_list = visit.get_code_list('PRESCRIPTIONS')

                visit_nl = {}
                visit_nl['diagnosis'] = []
                visit_nl['procedures'] = []
                visit_nl['prescriptions'] = []

                for diagnosis_code in diagnosis_code_list:
                    try:
                        diagnosis_nl = diagnosis_map_table.lookup(diagnosis_code)
                        visit_nl['diagnosis'].append(diagnosis_nl)
                    except Exception as e:
                        if warning_msg:
                            print('Warning: ',diagnosis_code, 'could not be found')

                for procedures_code in procedures_code_list:
                    try:
                        procedures_nl = procedures_map_table.lookup(procedures_code)
                        visit_nl['procedures'].append(procedures_nl)
                    except Exception as e:
                        if warning_msg:
                            print('Warning: ',procedures_code, 'could not be found')

                for prescriptions_code in prescriptions_code_list:
                    try:
                        prescriptions_nl = prescriptions_map_table.lookup(prescriptions_code)
                        visit_nl['prescriptions'].append(prescriptions_nl)
                    except Exception as e:
                        if warning_msg:
                            print('Warning: ',prescriptions_code, 'could not be found')

                visit_sequence_nl.append(visit_nl)

            sample['visit_sequence_nl'] = visit_sequence_nl

    # 2.generate prompt for llm base on natural language form of medical concept
    def transform_nl2prompt(self,
                            ICL = False,
                            dump_prompt = False,
                            dump_path = './prompt.json'):

        # ************* For Test *************
        dump_data = []

        # for sample in self.samples:
        for sample in tqdm(self.samples, desc="Transform English to LLM prompt", total=len(self.samples)):

            visit_sequence_nl = sample['visit_sequence_nl']

            identity_statement = 'You are an expert model in the field of medicine, endowed with extensive medical knowledge and excellent capabilities for reasoning and analyzing medical issues. Your focus is on providing users with detailed and useful medical knowledge. Now, you are required to analyze a patient’s structured temporal electronic health records (EHR) data , assess the patient’s health condition, and provide useful medical knowledge.\n'
            if ICL:
                # TODO: add similar patient data as in context learning example
                data_description = 'The structured temporal electronic health records (EHR) data is identified by <Visit Sequence>. It includes the patient’s multiple visits to medical facilities, capturing diagnosed diseases, laboratory test information, and medication details. The disease label data is indicated by <Ground Truth>. This data only appears in <Contextual Learning Examples>, representing the disease information diagnosed in the patient’s next visit.\n'

                output_requirement = '<Contextual Learning Examples> provide information about patients similar to the current one. From these examples,\
                     you are to summarize the progression patterns of similar diseases. Then, combining the current patient’s <Visit Sequence> and <Clinic Note>\
                      information, provide personalized medical knowledge. In this personalized medical knowledge, you need to: first, give an overall assessment\
                       of the current health condition of the patient; second, evaluate the risks of potential diseases the patient might face; and finally, \
                       provide the most relevant medical knowledge.\n'
            else:
                # TODO: only target patient's information
                data_description =\
                    'The structured temporal electronic health records (EHR) data is identified by <Visit Sequence>. It includes the patient’s multiple visits to medical facilities, capturing diagnosed diseases, laboratory test information, and medication details.\n'

                output_requirement = \
                    'Based on the current patient’s <Visit Sequence> information, you need to: First, give an overall assessment of the current health condition of the patient; Second, evaluate the risks of potential diseases the patient might face; Finally, provide the most relevant medical knowledge.\n'

            current_patient_information = "Here is the current patient's <Visit Sequence>:\n"

            for i,visit_nl in enumerate(visit_sequence_nl):
                current_patient_information += '<Visit' + str(i+1) + '>:\n'
                for (key, value) in visit_nl.items():
                    current_patient_information += '\t' +  '[' + str(key) + ']' + ':'+ '\n'
                    for code in value:
                        current_patient_information += '\t\t' + code + '\n'

            # final prompt
            final_prompt = identity_statement + data_description + output_requirement+ current_patient_information
            sample['prompt'] = final_prompt

            # ************* For Test *************
            sample_dump_dict = {}
            sample_dump_dict['prompt'] = final_prompt
            dump_data.append(sample_dump_dict)

        # if we need to dump prompt
        if dump_prompt:
            with open(dump_path, 'w') as f:
                json.dump(dump_data, f, indent=4)

    # 3.generate medical knowledge based on each patient's llm prompt

    def generate_medical_knowledge(self,
                                   LLM_name = 'xiaobei',
                                   dump_samples = False,
                                   dump_path = './xiaobei.pkl'):

        # Based on LLM name, initialize a LLM object
        LLM = None
        assert LLM_name in self.llm_list
        if LLM_name == 'xiaobei':
            LLM = XiaoBei()

        # For each patient visit sequence, generate medical knowledge
        medical_knowledge_json_list = []
        for sample in tqdm(self.samples, desc="Generating Medical Knowledge", total=len(self.samples)):
            # Medical knowledge generation
            prompt = sample['prompt']
            medical_knowledge = LLM.request(prompt)
            # Use a new dict to store medical knowledge
            medical_knowledge_dict = dict()
            medical_knowledge_dict['llm'] = LLM_name
            medical_knowledge_dict['medical_knowledge'] = medical_knowledge
            sample['medical_knowledge_dict'] = medical_knowledge_dict
            # ************* For Test *************
            medical_knowledge_json_list.append(medical_knowledge_dict)
            with open('med_knowledge.json', 'w') as f:
                json.dump(medical_knowledge_json_list, f, indent=4)

        # Dump samples
        if dump_samples:
            with open(dump_path, 'wb') as f:
                pickle.dump(self.samples, f)
        return None

    # TODO: 4.offline-load load knowledge from file
    def load_medical_knowledge(self, samples_with_medical_knowledge_path):
        with open(samples_with_medical_knowledge_path, 'rb') as f:
            samples_with_medical_knowledge = pickle.load(f)
        self.samples = samples_with_medical_knowledge
    # TODO：4.find similar patients

    # tokenize medical knowledges and medicine codes
    def tokenize_medical_code(self):
        for sample in self.samples: #sample:patients
            for visit_codes in sample['visit_sequence_codes']: #list[code]
                for code in visit_codes: #code
                    self.code_all_set.add(code)
        #construct map
        self.code2id = {c: i + 1 for i, c in enumerate(self.code_all_set)}
        #tokenize medicine code
        for sample in self.samples:
            visit_sequence_codes = []
            for visit_codes in sample['visit_sequence_codes']: #sample['visit_sequence_codes']: list[list[code]]; visit_codes: list[code]
                visit_ids = []
                for code in visit_codes: #code: code
                    # print("code: ", code, "code2id", self.code2id[code])
                    visit_ids.append(self.code2id[code])
                visit_sequence_codes.append(visit_ids)
            sample['visit_sequence_codes'] = visit_sequence_codes
        
        # print(self.code2id['V1582'])
        #test
        print("test tokenize_medical_code\n", self.samples[0]['visit_sequence_codes'][0])

    def tokenize_medical_knowledge(self, max_word_lens = 512):
        #add start tokens
        start_token = '[CLS]'
        for sample in self.samples:
            sample['medical_knowledge_dict']['medical_knowledge'] = start_token + ' ' + sample['medical_knowledge_dict']['medical_knowledge']

        #build dict
        med_knowledge = []
        for sample in self.samples:
            notes = sample['medical_knowledge_dict']['medical_knowledge']
            notes = notes.split(' ')
            med_knowledge.append(notes)

        notes_truncated = [notes[:max_word_lens] for notes in med_knowledge]
        print("test notes_truncated\n", notes_truncated[0])
        self.vocab = Vocab.build(notes_truncated, 50000, 2)

        #tokenize medicine knowledge
        notes_encoded, note_lens = self.vocab.src.encode_notes(notes_truncated)
        notes_encoded = np.array(notes_encoded)    #After indexing & padding
        note_lens = np.array(note_lens)
        for i, sample in enumerate(self.samples):
            sample['encoded_knowledge'] = notes_encoded[i]
            sample['encoded_knowledge_length'] = note_lens[i]

        #test
        print("test tokenize_medical_knowledge\n", self.samples[0]['encoded_knowledge'])

    def adapt(self):
        for sample in self.samples:
            #convert to relative times, step 1
            start_time = None
            time_steps = []
            visit_sequence_len_dim2_array = []
            for visit in sample['visit_sequence']:
                if start_time == None:
                    start_time = visit.encounter_time
                    time_steps.append(0)
                else:
                    time_steps.append((visit.encounter_time - start_time).days)
            
            final_time = time_steps[-1]
            final_time_steps = [final_time - visit_time for visit_time in time_steps]
            sample['time_step'] = final_time_steps
            for codes in sample['visit_sequence_codes']:
                visit_sequence_len_dim2_array.append(len(codes))
            sample['visit_sequence_len_dim2_array'] = visit_sequence_len_dim2_array

    def dump_dict(self, output_data_dir):
        with open(output_data_dir + "/code_note_dict.pkl", "wb") as f:
            pickle.dump(self.code2id, f)
            pickle.dump(self.vocab, f)

if __name__ == "__main__":
    from pyhealth.datasets import MIMIC3Dataset
    from pyhealth.tasks import mortality_prediction_time_series_mimic3_fn

    dataset = MIMIC3Dataset(
        root='/data/fy/fy/database/mimic/MIMICIII_data',
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping={
            # "ICD9CM": "CCSCM",
            # "ICD9PROC": "CCSPROC",
            "NDC": ("ATC", {"target_kwargs": {"level": 3}})
        },
        dev=True,
        refresh_cache=True
    )
    sample_dataset = dataset.set_task_ts(mortality_prediction_time_series_mimic3_fn)

    '''
    lame_dataset = LameDataset(samples= sample_dataset.samples,
                               code_vocs=sample_dataset.code_vocs,
                               dataset_name=sample_dataset.dataset_name,
                               task_name=sample_dataset.task_name)
    print(lame_dataset.task_name)
    lame_dataset.transform_codes2nl()
    lame_dataset.transform_nl2prompt()
    print(lame_dataset.samples[10]['prompt'])
    for sample in lame_dataset.samples:
        assert len(sample['visit_sequence']) == len(sample['visit_sequence_nl'])
    lame_dataset.generate_medical_knowledge(dump_samples=True)
    '''
    lame_dataset_test = LameDataset(samples= sample_dataset.samples,
                                    code_vocs=sample_dataset.code_vocs,
                                    dataset_name=sample_dataset.dataset_name,
                                    task_name=sample_dataset.task_name,
                                    load_medical_knowledge=True)
    print(lame_dataset_test.samples[0]['medical_knowledge_dict'])



