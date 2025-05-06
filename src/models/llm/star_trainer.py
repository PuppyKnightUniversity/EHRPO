import os
import torch
import logging
import numpy as np
import json
import pickle
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from pyhealth.utils import load_pickle
from pyhealth.datasets import collate_fn_dict

from mcts import medical_mcts_search
from LLM_worker import LLM_worker

class STaRTrainer:
    """
    STaR: Bootstrapping Reasoning with Reasoning
    
    This class implements the STaR algorithm that iteratively improves a model's
    reasoning abilities by bootstrapping from its own successful reasoning paths.
    """
    def __init__(
        self,
        llm_worker,
        train_loader,
        test_loader,
        partial_test_loader,
        output_dir="./star_outputs",
        n_iterations=3,
        p_rationalization=1.0,
        mcts_rollouts=10,
        max_depth=5,
        device="cuda",
        train_batch_size=2,
        gradient_accumulation_steps=4
    ):
        self.llm_worker = llm_worker
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.partial_test_loader = partial_test_loader
        self.output_dir = output_dir
        self.n_iterations = n_iterations
        self.p_rationalization = p_rationalization
        self.mcts_rollouts = mcts_rollouts
        self.max_depth = max_depth
        self.device = device
        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        os.makedirs(f"{output_dir}/reasoning_data", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{output_dir}/datasets", exist_ok=True)
        
        # Checkpoint state
        self.checkpoint_state = {
            'current_iteration': 0,
            'reasoning_paths_progress': {'batch_idx': 0, 'paths': []},
            'rationalized_paths_progress': {'path_idx': 0, 'paths': []},
            'finetune_progress': {'epoch': 0, 'batch': 0},
            'successful_paths': [],
            'incorrect_paths': [],
            'rationalized_paths': [],
            'training_data': []
        }
        
    def save_checkpoint(self):
        """Save current checkpoint state to disk"""
        checkpoint_path = f"{self.output_dir}/checkpoints/checkpoint.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(self.checkpoint_state, f)
        
    def load_checkpoint(self):
        """Load checkpoint state from disk if exists"""
        checkpoint_path = f"{self.output_dir}/checkpoints/checkpoint.pkl"
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                self.checkpoint_state = pickle.load(f)
            print(f"Loaded checkpoint from {checkpoint_path}")
            return True
        return False
        
    def run(self):
        """Run the complete STaR training loop with checkpoint support"""
        # Try to load checkpoint first
        print("GPU count:", torch.cuda.device_count())
        print("Current GPU index:", torch.cuda.current_device())
        resumed = self.load_checkpoint()
        if resumed:
            print(f"Resuming from iteration {self.checkpoint_state['current_iteration']}")
            start_iteration = self.checkpoint_state['current_iteration']
        else:
            print(f"Starting STaR training for {self.n_iterations} iterations")
            start_iteration = 0
        
        # Main STaR loop
        for iteration in range(start_iteration, self.n_iterations):
            self.checkpoint_state['current_iteration'] = iteration
            self.save_checkpoint()
            
            print(f"Starting iteration {iteration+1}/{self.n_iterations}")
            
            # 1. Generate reasoning paths with MCTS
            if resumed and iteration == start_iteration and len(self.checkpoint_state['reasoning_paths_progress']['paths']) > 0:
                print(f"Resuming reasoning path generation from batch {self.checkpoint_state['reasoning_paths_progress']['batch_idx']}")
                reasoning_paths = self.generate_reasoning_paths(
                    iteration=iteration, 
                    loader=self.train_loader,
                    resume=True
                )
            else:
                reasoning_paths = self.generate_reasoning_paths(
                    iteration=iteration, 
                    loader=self.train_loader
                )
            
            # 2. Filter successful reasoning paths
            successful_paths, incorrect_paths = self.filter_reasoning_paths(reasoning_paths)
            self.checkpoint_state['successful_paths'] = successful_paths
            self.checkpoint_state['incorrect_paths'] = incorrect_paths
            self.save_checkpoint()
            
            # 3. Generate rationalized reasoning paths for incorrect answers
            if resumed and iteration == start_iteration and len(self.checkpoint_state['rationalized_paths_progress']['paths']) > 0:
                print(f"Resuming rationalization from path {self.checkpoint_state['rationalized_paths_progress']['path_idx']}")
                rationalized_paths = self.generate_rationalized_paths(incorrect_paths, resume=True)
            else:
                rationalized_paths = self.generate_rationalized_paths(incorrect_paths)
            
            self.checkpoint_state['rationalized_paths'] = rationalized_paths
            self.save_checkpoint()
            
            # 4. Combine successful paths for training
            training_data = self.prepare_training_data(successful_paths, rationalized_paths, iteration)
            self.checkpoint_state['training_data'] = training_data
            self.save_checkpoint()
            
            # 5. Finetune the model with successful reasoning paths
            if resumed and iteration == start_iteration and self.checkpoint_state['finetune_progress']['epoch'] > 0:
                print(f"Resuming finetuning from epoch {self.checkpoint_state['finetune_progress']['epoch']}, batch {self.checkpoint_state['finetune_progress']['batch']}")
                self.finetune_model(training_data, iteration, resume=True)
            else:
                self.finetune_model(training_data, iteration)
            
            # 6. Evaluate on validation set
            test_metrics = self.evaluate(self.partial_test_loader, f"iteration_{iteration+1}_test")
            print(f"Test metrics after iteration {iteration+1}: {test_metrics}")
            
            # Reset resume flag after completing the resumed iteration
            if resumed and iteration == start_iteration:
                resumed = False
            
            # Reset batch progress for the next iteration
            self.checkpoint_state['reasoning_paths_progress'] = {'batch_idx': 0, 'paths': []}
            self.checkpoint_state['rationalized_paths_progress'] = {'path_idx': 0, 'paths': []}
            self.checkpoint_state['finetune_progress'] = {'epoch': 0, 'batch': 0}
            self.save_checkpoint()
        
        return test_metrics
    
    def generate_reasoning_paths(self, iteration, loader, sample_size=None, resume=False):
        """Generate reasoning paths using MCTS"""
        print("Generating reasoning paths with MCTS")
        
        reasoning_paths = []
        start_batch_idx = 0
        
        # If resuming, load existing paths and start from the last batch
        if resume:
            reasoning_paths = self.checkpoint_state['reasoning_paths_progress']['paths']
            start_batch_idx = self.checkpoint_state['reasoning_paths_progress']['batch_idx']
            print(f"Resuming from batch {start_batch_idx} with {len(reasoning_paths)} existing paths")
        
        # Create a subset of the loader if sample_size is specified
        if sample_size is not None:
            samples = []
            for i, batch in enumerate(loader):
                if i < start_batch_idx:
                    continue
                samples.append(batch)
                if len(samples) * loader.batch_size >= sample_size:
                    break
            data_iterator = samples
            total_batches = len(samples)
        else:
            # Skip to the correct batch when resuming
            data_iterator = list(loader)[start_batch_idx:]
            total_batches = len(loader) - start_batch_idx
        
        for batch_idx, batch in enumerate(tqdm(data_iterator, desc="Generating reasoning paths", total=total_batches)):
            current_batch_idx = batch_idx + start_batch_idx
            
            batch_nl = self.llm_worker.transform_codes2nl(**batch)
            batch_prompts = self.llm_worker.prompt_generate(
                batch_nl=batch_nl,
                task_name=self.llm_worker.task_name,
                inference_type='mcts'
            )
            
            batch_y_true = self.llm_worker.prepare_labels(
                batch[self.llm_worker.label_key], 
                self.llm_worker.label_tokenizer
            ).cpu().numpy()
            
            batch_paths = []
            for prompt_idx, prompt in enumerate(tqdm(batch_prompts, desc="Processing cases", unit="case")):
                y_true = batch_y_true[prompt_idx]
                
                # Run MCTS search
                best_logits, trees, reasoning = medical_mcts_search(
                    prompt=prompt,
                    llm_model=self.llm_worker.model,
                    tokenizer=self.llm_worker.tokenizer,
                    task_name=self.llm_worker.task_name,
                    mcts_rollouts=self.mcts_rollouts,
                    max_depth=self.max_depth
                )
                
                # Get predicted label
                y_pred = 1 if best_logits >= 0.5 else 0
                
                batch_paths.append({
                    "batch_idx": current_batch_idx,
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "y_true": int(y_true),
                    "y_pred": y_pred,
                    "logits": float(best_logits),
                    "reasoning_path": reasoning,
                    "is_correct": y_pred == int(y_true)
                })
            
            reasoning_paths.extend(batch_paths)
            
            # Update checkpoint after each batch
            self.checkpoint_state['reasoning_paths_progress']['paths'] = reasoning_paths
            self.checkpoint_state['reasoning_paths_progress']['batch_idx'] = current_batch_idx + 1
            self.save_checkpoint()
        
        # Save the reasoning paths
        with open(f"{self.output_dir}/reasoning_data/paths_iteration_{iteration}.json", "w") as f:
            json.dump(reasoning_paths, f, indent=2)
        
        return reasoning_paths
    
    def filter_reasoning_paths(self, reasoning_paths):
        """Filter reasoning paths into successful and incorrect ones"""
        successful_paths = [path for path in reasoning_paths if path["is_correct"]]
        incorrect_paths = [path for path in reasoning_paths if not path["is_correct"]]
        
        print(f"Filtered {len(successful_paths)} successful and {len(incorrect_paths)} incorrect paths")
        
        return successful_paths, incorrect_paths
    
    def generate_rationalized_paths(self, incorrect_paths, resume=False):
        """Generate rationalized reasoning paths for incorrect answers using MCTS with hints"""
        print("Generating rationalized paths for incorrect examples")
        
        rationalized_paths = []
        start_path_idx = 0
        
        # If resuming, load existing rationalized paths and start from the last position
        if resume:
            rationalized_paths = self.checkpoint_state['rationalized_paths_progress']['paths']
            start_path_idx = self.checkpoint_state['rationalized_paths_progress']['path_idx']
            print(f"Resuming rationalization from index {start_path_idx} with {len(rationalized_paths)} existing paths")
        
        for path_idx, path in enumerate(tqdm(incorrect_paths[start_path_idx:], desc="Generating rationalized paths")):
            current_path_idx = path_idx + start_path_idx
            
            # Skip some paths based on p_rationalization
            if np.random.random() > self.p_rationalization:
                continue
                
            prompt = path["prompt"]
            y_true = path["y_true"]
            
            # Add correct answer as hint to prompt
            hint_answer = "A. Yes" if y_true == 1 else "B. No"
            hint_prompt = prompt + f"<Correct answer>: {hint_answer}"
            
            # Run MCTS with hint
            best_logits, trees, reasoning = medical_mcts_search(
                prompt=hint_prompt,
                llm_model=self.llm_worker.model,
                tokenizer=self.llm_worker.tokenizer,
                task_name=self.llm_worker.task_name,
                mcts_rollouts=self.mcts_rollouts,
                max_depth=self.max_depth
            )
            
            # Get predicted label with hint
            y_pred = 1 if best_logits >= 0.5 else 0
            
            # Check if rationalization was successful
            is_rationalized = y_pred == y_true
            
            if is_rationalized:
                rationalized_paths.append({
                    "original_prompt": prompt,
                    "hint_prompt": hint_prompt,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "logits": float(best_logits),
                    "reasoning_path": reasoning,
                    "is_rationalized": True
                })
            
            # Update checkpoint after processing each path
            self.checkpoint_state['rationalized_paths_progress']['paths'] = rationalized_paths
            self.checkpoint_state['rationalized_paths_progress']['path_idx'] = current_path_idx + 1
            self.save_checkpoint()
        
        print(f"Generated {len(rationalized_paths)} successful rationalized paths")
        
        return rationalized_paths
    
    def prepare_training_data(self, successful_paths, rationalized_paths, iteration):
        """Prepare training data from successful and rationalized paths and save it"""
        training_data = []
        
        # Add successful paths
        for path in successful_paths:
            training_data.append({
                "input": path["prompt"],
                "output": path["reasoning_path"],
                "source": "direct"
            })
        
        # Add rationalized paths
        for path in rationalized_paths:
            training_data.append({
                "input": path["original_prompt"],  # Use original prompt without hint
                "output": path["reasoning_path"],
                "source": "rationalized"
            })
        
        print(f"Prepared {len(training_data)} examples for training")
        
        # Save the training dataset for this iteration
        dataset_path = f"{self.output_dir}/datasets/training_data_iteration_{iteration}.json"
        with open(dataset_path, "w") as f:
            json.dump(training_data, f, indent=2)
        print(f"Saved training dataset to {dataset_path}")
        
        return training_data
    
    def finetune_model(self, training_data, iteration, resume=False):
        """Finetune the model on successful reasoning paths"""
        if not training_data:
            print("No training data available for finetuning")
            return
        
        print(f"Finetuning model on {len(training_data)} examples")

        torch.cuda.empty_cache()
        tokenizer = self.llm_worker.tokenizer
        
        # Prepare dataset for training
        train_texts = []
        for example in training_data:
            formatted_example = {
                "text": f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
            }
            train_texts.append(formatted_example)
        
        train_dataset = Dataset.from_list(train_texts)

        def tokenize_function(examples):
            return tokenizer(
                examples["text"], 
                truncation=True, 
                padding=False, 
                max_length=4096
            )

        tokenized_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=8,  
            remove_columns=["text"]
        )

        input_lengths = [len(x) for x in tokenized_dataset["input_ids"]]
        print(f"Max input length: {max(input_lengths)}, Avg: {sum(input_lengths)/len(input_lengths):.2f}")

        # Data collator 
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # We're doing CLM not MLM
        )
        torch.cuda.empty_cache()

        model = self.llm_worker.model
        model.gradient_checkpointing_enable()

        peft_config = LoraConfig(
            r=4,          
            lora_alpha=8,  
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            init_lora_weights="gaussian",
            modules_to_save=None
        )

        # Prepare the model for PEFT
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()
        torch.cuda.empty_cache()
        
        # Custom checkpoint callback
        class CheckpointCallback(TrainerCallback):
            def __init__(self, trainer_instance):
                self.trainer_instance = trainer_instance
            
            def on_train_begin(self, args, state, control, **kwargs):
                return control
                
            def on_step_end(self, args, state, control, **kwargs):
                # Update checkpoint state
                self.trainer_instance.checkpoint_state['finetune_progress']['epoch'] = state.epoch
                self.trainer_instance.checkpoint_state['finetune_progress']['batch'] = state.global_step
                self.trainer_instance.save_checkpoint()
                return control
        
        # Configure training arguments
        checkpoint_dir = f"{self.output_dir}/models/iteration_{iteration}"
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            num_train_epochs=1,
            per_device_train_batch_size=self.train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_ratio=0.1,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,
            save_strategy="steps",
            save_steps=50,  
            learning_rate=2e-5,
            fp16=True,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            dataloader_drop_last=True,
            dataloader_num_workers=1,
            weight_decay=0.01,
            group_by_length=True,  
            gradient_checkpointing_kwargs={"use_reentrant": False},
            torch_compile=False,
            max_grad_norm=0.5,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Add our custom callback
        trainer.add_callback(CheckpointCallback(self))
        torch.cuda.empty_cache()
        
        # Resume training if checkpoint exists
        if resume and os.path.exists(checkpoint_dir):
            # Find the latest checkpoint
            checkpoints = [str(x) for x in Path(checkpoint_dir).glob("checkpoint-*")]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
                print(f"Resuming training from checkpoint {latest_checkpoint}")
                trainer.train(resume_from_checkpoint=latest_checkpoint)
            else:
                print("No checkpoint found for resuming. Starting from beginning.")
                trainer.train()
        else:
            # Train the model
            trainer.train()

        torch.cuda.empty_cache()
        
        # Save the finetuned model
        model.save_pretrained(f"{self.output_dir}/models/iteration_{iteration}_final")
        tokenizer.save_pretrained(f"{self.output_dir}/models/iteration_{iteration}_final")
        
        # Update the model in llm_worker
        self.llm_worker.model = model
        torch.cuda.empty_cache()
    
    def evaluate(self, loader, tag):
        """Evaluate the model on a dataset"""
        print(f"Evaluating model on {tag}")
        
        results, scores = self.llm_worker.inference(loader)
        
        # Save metrics
        with open(f"{self.output_dir}/{tag}_metrics.json", "w") as f:
            json.dump(scores, f, indent=2)
        
        return scores