'''
    Define arguments for scripts
'''
import argparse

# Define mapping from llm_name to model path
LLM_PATH_MAP = {
    'qwen2-5-7b-instruct': '/data1/dhx/llmbase/hub/Qwen/Qwen2___5-7B-Instruct', # NOTE: You can modify the path to your own path
    'qwen2-5-1.5b-instruct': '/data1/xiaobei/llmbase/qwen2.5-1.5B-Instruct', # NOTE: You can modify the path to your own path
    'deepseek-r1-local-preview': None  # API model, no local path needed
    # NOTE:add more LLMs here!
}

# Define mapping from dataset to dataset path
DATASET_PATH_MAP = {
    'mimic3': '/data1/xiaobei/codebase/EHRPO/baseline/hitanet_modified/data/raw_data/MIMICIII_data', # NOTE: You can modify the path to your own path
    # NOTE: add more datasets here!
}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1128)

    '''
        Dataset settings
    '''
    parser.add_argument("--dataset", type=str, default="mimic3")
    parser.add_argument("--task_name", type=str, default="mortality_prediction")
    parser.add_argument("--dataset_path", type=str, default=None)
    
    '''
        EHR Model settings
    '''
    
    '''
        LLM settings
    '''
    parser.add_argument("--llm_name", type=str, default="qwen2-5-1.5b-instruct")
    # load local LLM
    parser.add_argument("--llm_local_path", type=str, default=None)
    # use API of LLM
    parser.add_argument("--is_api", type=bool, default=True)
    parser.add_argument("--api_key", type=str, default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjE3YzQzODY2LTE2YTctNGI1Ni05ZjFjLTk1ODIzODY0OTJmZCJ9.OusI4q1_w_wzFk46cE4hfV8t1tRGkwBlZoJG9UxZI7o")
    # inference method of LLM
    parser.add_argument("--inference_type", type=str, default="straight_forward")
    
    '''
        Experiment settings
    '''
    parser.add_argument("--EHR_model_prompt_injection", type=bool, default=False)
    
    args = parser.parse_args()

    '''
        Path mapping
    '''
    
    # Set llm_local_path based on llm_name if not provided
    if args.llm_local_path is None and args.llm_name in LLM_PATH_MAP:
        args.llm_local_path = LLM_PATH_MAP[args.llm_name]
    
    # Set dataset_path based on dataset if not provided
    if args.dataset_path is None and args.dataset in DATASET_PATH_MAP:
        args.dataset_path = DATASET_PATH_MAP[args.dataset]
    
    return args