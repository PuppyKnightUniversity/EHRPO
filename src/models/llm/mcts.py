import math
import random
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Any
from models.llm.prompt import subquestion_prompts, usefulness_prompts, reward_prompts
import warnings
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn.functional as F
import re
import datetime

class MCTSNode(ABC):
    @abstractmethod
    def find_children(self):
        return set()

    @abstractmethod
    def find_one_child(self) -> 'MCTSNode':
        raise NotImplementedError

    @property
    @abstractmethod
    def is_terminal(self):
        return True

    @property
    @abstractmethod
    def reward(self):
        return 0

    @property
    @abstractmethod
    def visited(self):
        return 0


class MCTS:
    def __init__(self, w_exp=1, discount=1, prior=False, aggr_reward='sum', aggr_child='max'):
        self.Q: dict[MCTSNode, float] = defaultdict(lambda : 0.)
        self.N: dict[MCTSNode, int] = defaultdict(lambda : 0)
        self.M: dict[MCTSNode, float] = defaultdict(lambda : -math.inf)
        self.children = dict()
        self.w_exp = w_exp
        self.discount = discount
        self.prior = prior
        self.aggr_reward = aggr_reward
        self.aggr_child = aggr_child

    def rollout(self, node: MCTSNode):
        if self.prior:
            path = self._select_prior(node)
        else:
            path = self._select(node)
            self._expand(path[-1])
            self._simulate(path)
        self._back_propagate(path)
        torch.cuda.empty_cache()

    def _select_prior(self, node: MCTSNode):
        path = [node]
        while not node.is_terminal:
            self._expand(node)
            if len(self.children[node]) == 0:
                return path
            node = self._uct_select(node)
            path.append(node)
        self._expand(node)
        return path

    def _select(self, node: MCTSNode):
        path = []
        while True:
            path.append(node)
            if node not in self.children or node.is_terminal:
                return path
            for child in self.children[node]:
                if child not in self.children.keys():
                    path.append(child)
                    return path
            node = self._uct_select(node)

    def _expand(self, node: MCTSNode):
        if node not in self.children:
            self.children[node] = node.find_children()

    @staticmethod
    def _simulate(path: list[MCTSNode]):
        node = path[-1]
        while not node.is_terminal:
            node = node.find_one_child()
            if node:
                path.append(node)
            else:
                break

    def max_terminal(self, cur: MCTSNode):
        if cur.is_terminal:
            if cur.visited:
                return cur, cur.reward
            else:
                return cur, -math.inf
        if cur not in self.children:
            return cur, -math.inf
        max_n, max_r = max((self.max_terminal(child) for child in self.children[cur]), key=lambda x: x[1])
        return max_n, max_r + cur.reward

    def max_mean_terminal(self, cur: MCTSNode, sum=0., cnt=0):
        if cur.is_terminal:
            if cur.visited:
                return cur, (sum + cur.reward) / (cnt + 1)
            else:
                return cur, -math.inf
        if cur not in self.children or not self.children[cur]:
            return cur, -math.inf
        
        return max((self.max_mean_terminal(child, sum + cur.reward, cnt + 1) for child in self.children[cur]), key=lambda x: x[1])

    def _back_propagate(self, path: list[MCTSNode], reward=0.):
        coeff = 1
        for node in reversed(path):
            reward = reward * self.discount + node.reward
            coeff = coeff * self.discount + 1
            if self.aggr_reward == 'mean':
                c_reward = reward / coeff
            else:
                c_reward = reward
            if node not in self.N:
                self.Q[node] = c_reward
            else:
                self.Q[node] += c_reward
            self.N[node] += 1
            self.M[node] = max(self.M[node], c_reward)

    def _uct(self, node: MCTSNode, log_n_f: float):
        if self.prior and self.N[node] == 0:
            return node.reward + self.w_exp * math.sqrt(log_n_f)
        if self.aggr_child == 'max':
            return self.M[node] + self.w_exp * math.sqrt(log_n_f / self.N[node])
        elif self.aggr_child == 'mean':
            return self.Q[node] / self.N[node] + self.w_exp * math.sqrt(log_n_f / self.N[node])

    def _uct_select(self, node: MCTSNode):
        if self.prior and self.N[node] == 0:
            log_n = math.log(1)
        else:
            log_n = math.log(self.N[node])
        return max(self.children[node], key=lambda n: self._uct(n, log_n))


class MedicalMCTSNode(MCTSNode):
    @property
    def visited(self):
        return self._visited

    def __init__(self, prompt, llm_model, tokenizer, depth, subquestion_prompts, usefulness_prompts, reward_prompts, task_name, r1_default=1.0, r_alpha=0.4, 
                 parent: 'MedicalMCTSNode' = None, r0=0., max_depth=5):
        self._conf = None
        self.children = []
        self.prompt = prompt 
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.depth = depth
        self._r0 = r0
        self._r1 = self._r1_default = r1_default
        self._r_alpha = r_alpha
        self._visited = False
        self.parent = parent
        self._reasoning = None
        self.subquestion_prompts = subquestion_prompts 
        self.usefulness_prompts = usefulness_prompts  
        self.reward_prompts = reward_prompts
        self.max_depth = max_depth
        self.logits = None
        self.task_name = task_name
        self.subquestions = []
        self.node_id = f"{depth}.{random.randint(1000, 9999)}"  # Unique ID for node reference

    def _child_node(self, prompt, r0):
        return MedicalMCTSNode(self.prompt + prompt, self.llm_model, self.tokenizer, self.depth + 1,
                              self.subquestion_prompts, self.usefulness_prompts, self.reward_prompts, self.task_name, self._r1_default, 
                              self._r_alpha, parent=self, r0=r0)

    def _get_children(self):
        self._visited = True
        if self.parent is not None:
            self._calculate_reward()
        if self.is_terminal:
            return self.children
        
        questions, r0 = self._generate_subquestions()
        
        for question, r in zip(questions, r0):
            self.children.append(self._child_node(question, r))
        
        return self.children

    def _generate_subquestions(self):
        """Generate medical subquestions"""
        agent_input = self.subquestion_prompts["few_shot"] + self.prompt + self.subquestion_prompts["subquestion_prefix"].format(self.depth)
        
        # If at max depth, create the "Now we can answer the question" prompt
        if self.depth >= self.max_depth - 1:  
            overall_question_output = self.subquestion_prompts["overall_question_prefix"][self.task_name].format(
                self.depth)
            agent_output = [overall_question_output]
        else:
            # Generate subquestions
            text = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{self.subquestion_prompts['input']}<|im_end|>
<|im_start|>assistant
{agent_input.replace("<|im_end|>", "").strip()}"""
            
            inputs = self.tokenizer([text], return_tensors="pt").to(self.llm_model.device)
            
            # Generate 3 subquestions
            agent_outputs = []
            for i in range(3):  
                with torch.no_grad():
                    generated_ids = self.llm_model.generate(
                        **inputs,
                        max_new_tokens=64,
                        temperature=1.2,
                        top_p=0.8,
                        do_sample=True,
                    )
                
                generated_ids = generated_ids[0][inputs.input_ids.shape[1]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Extract only the first question
                question_mark_index = response.find('?') if response.find('?') != -1 else response.find('.')
                if question_mark_index != -1:
                    first_question = response[:question_mark_index + 1].strip()
                else:
                    first_question = response.strip()
 
                agent_outputs.append(self.subquestion_prompts["subquestion_prefix"].format(self.depth) + first_question)
            
            agent_output = agent_outputs
        
        questions = [o.split(self.subquestion_prompts["subquestion_prefix"].format(self.depth))[-1] for o in agent_output]
        self.subquestions = questions
        
        if self.depth >= self.max_depth - 1:
            r0_values = [1]   
        else:
            # Evaluate usefulness of each subquestion (r0)
            r0_values = self._evaluate_subquestion_usefulness(questions)
        
        return agent_output,r0_values

    def _evaluate_subquestion_usefulness(self, questions):
        """Estimate how useful a subquestion is for the medical prediction"""
        r0_values = []
        
        for i, q in enumerate(questions):
            if 'Now we can answer' in q:
                yes_prob = 1.000
            else:
                agent_input = self.usefulness_prompts["few_shot"] + self.prompt + self.usefulness_prompts["new_subquestion_prefix"].format(self.depth) + q + self.usefulness_prompts["useful_prefix"]
                
                text = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{self.usefulness_prompts["input"]}<|im_end|>
<|im_start|>assistant
{agent_input.replace("<|im_end|>", "").strip()}"""

                inputs = self.tokenizer([text], return_tensors="pt").to(self.llm_model.device)
            
                with torch.no_grad():
                    outputs = self.llm_model(**inputs)
                    next_token_logits = outputs.logits[:, -1, :]  
        
                yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
                no_token_id = self.tokenizer.convert_tokens_to_ids("No")
        
                yes_logit = next_token_logits[0, yes_token_id].item()
                no_logit = next_token_logits[0, no_token_id].item()
        
                yes_prob = torch.softmax(torch.tensor([yes_logit, no_logit]), dim=0)[0].item()

            print(f"Useful {self.depth}.{i + 1}: {yes_prob}")
            r0_values.append(yes_prob)

        return r0_values
        

    def _calculate_reward(self):
        """Calculate the reward value (r1) for this node"""         
        agent_input = self.subquestion_prompts["few_shot"] + self.prompt + self.subquestion_prompts["answer_prefix"].format(self.depth - 1)

        text = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{self.subquestion_prompts['input']}<|im_end|>
<|im_start|>assistant
{agent_input.replace("<|im_end|>", "").strip()}"""

        inputs = self.tokenizer([text], return_tensors="pt").to(self.llm_model.device)
        Answer = ""

        if self.is_terminal:
            with torch.no_grad():
                outputs = self.llm_model(**inputs)
                next_token_logits = outputs.logits[:, -1, :]  
    
            A_token_id = self.tokenizer.convert_tokens_to_ids("A")
            B_token_id = self.tokenizer.convert_tokens_to_ids("B")
    
            A_logit = next_token_logits[0, A_token_id].item()
            B_logit = next_token_logits[0, B_token_id].item()
    
            self.logits = torch.softmax(torch.tensor([A_logit, B_logit]), dim=0)[0].item()
            self._r1 = max(self.logits, 1 - self.logits)

            if self.logits >= 0.5:
                Answer = "A"
            else:
                Answer = "B"
            print(f"\nFinal logits: {self.logits}, answer: {Answer}")
        else:
            with torch.no_grad():
                generated_ids = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.9,
                    top_p=0.8,
                    do_sample=True
                )
            
            generated_ids = generated_ids[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Extract only the first question
            answer_mark_index = response.find('.')
            if answer_mark_index != -1:
                Answer = response[:answer_mark_index + 1].strip()
            else:
                Answer = response.strip()

            model_input = self.reward_prompts["few_shot"] + self.prompt + f"\n<Answer {self.depth - 1}>:" + Answer + self.reward_prompts["reward_prefix"]

            text = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{self.reward_prompts['input']}<|im_end|>
<|im_start|>assistant
{model_input.replace("<|im_end|>", "").strip()}"""

            inputs = self.tokenizer([text], return_tensors="pt").to(self.llm_model.device)

            with torch.no_grad():
                generated_ids = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=8,
                    temperature=0.7,
                    top_p=0.8,
                    do_sample=True
                )

            generated_ids = generated_ids[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Extract the first 0-100 number ending with "." from the response
            score_match = re.search(r'\b([0-9]|[1-9][0-9]|100)\b', response)
            if score_match:
                score = int(score_match.group(1))
                self._r1 = score / 100  
            else:
                self._r1 = 0.5
            
            print(f"\nReward score {self.depth - 1}: {self._r1*100:.0f}")

                
        if not Answer:
            self._r1 = 0
            self.prompt = self.prompt  # No change
            return
        
        # Update prompt with the answer
        self.prompt += f"\n<Answer {self.depth - 1}>: " + Answer
        self._reasoning = Answer

    def find_children(self):
        self.children = self.children or self._get_children()
        return self.children

    def find_one_child(self) -> MCTSNode:
        return random.choice(self.find_children()) if self.children else None

    def _is_terminal_question(self, prompt):
        """Check if this is a terminal question"""
        question = prompt.strip().split('\n')[-1]
        if 'Now we can answer' in question:
            return True          

        if "<Answer" in prompt:
            prompt = prompt[:prompt.rindex("<Answer")]
        else:
            return False
        question = prompt.strip().split('\n')[-1]
        if 'Now we can answer' in question:
            return True
        return False
    
    def _static_terminal(self):
        """Determine if node is in terminal state based on prompt content"""
        if self.depth >= self.max_depth:
            return True

        return self._is_terminal_question(self.prompt)
            

    @property
    def is_terminal(self):
        return self._static_terminal() or self.reward < -1

    @property
    def reward(self):
        if self._r0 < 0 or self._r1 < 0:
            return min(self._r0, self._r1)
        return self._r0 * self._r_alpha + self._r1 * (1 - self._r_alpha)

    def _generate_final_summary(self, mcts: MCTS):
        """
        New method to generate the final rollout summary to be printed only once
        at the end of each complete rollout.
        """
        output_buffer = []
        
        # Add header for the summary
        output_buffer.append("\n" + "="*80)
        output_buffer.append("MCTS ROLLOUT SUMMARY")
        output_buffer.append("="*80 + "\n")
        
        # Find the best terminal node in the tree
        best_terminal_node, best_reward = mcts.max_mean_terminal(self)
        
        # Reconstruct path from root to best terminal node
        selected_path = []
        current = best_terminal_node
        while current:
            selected_path.insert(0, current)
            current = current.parent
        
        # Print the selected path with its complete prompt
        output_buffer.append("SELECTED PATH DETAILS:")
        output_buffer.append("-"*80)
        
        for i, node in enumerate(selected_path):
            node_type = "ROOT" if i == 0 else f"LEVEL {i}"
            if node.is_terminal:
                node_type = "TERMINAL"
            
            output_buffer.append(f"{node_type} Node: {node.node_id} | Reward: {node.reward:.3f}")
            
            # For the terminal node, print decision and confidence
            if node.is_terminal and hasattr(node, 'logits') and node.logits:
                answer = "A" if node.logits >= 0.5 else "B"
                output_buffer.append(f"Final Answer: {answer} (Confidence: {node.logits:.3f})")
        
        # Print complete prompt of the selected path
        output_buffer.append("-"*80)
        output_buffer.append("COMPLETE PROMPT OF SELECTED PATH:")
        output_buffer.append("-"*80)
        output_buffer.append(best_terminal_node.prompt)
        output_buffer.append("-"*80)
        
        # Print tree structure summary
        output_buffer.append("\nTREE STRUCTURE SUMMARY:")
        output_buffer.append("-"*80)
        self._print_tree_structure(mcts, output_buffer)
        
        output_buffer.append("="*80)
        output_buffer.append("")
        
        return output_buffer

    def _print_tree_structure(self, mcts: MCTS, output_buffer, node=None, depth=0, prefix=""):
        """Helper method to print the tree structure in a readable format"""
        if node is None:
            node = self  # Start with root
            output_buffer.append("Tree depth: " + str(self._count_max_depth()))
            output_buffer.append("Total nodes: " + str(self._count_total_nodes()))
            output_buffer.append("Visited nodes: " + str(self._count_visited_nodes()))
            output_buffer.append("")
            
        indent = "  " * depth
        
        # Node representation
        node_info = f"{indent}{prefix}Node {node.node_id} | R: {node.reward:.3f} | r0: {node._r0:.3f} | r1: {node._r1:.3f} | N: {mcts.N[node]} | Visited: {node.visited}"
        
        if node.is_terminal:
            node_info += " [TERMINAL]"
            if hasattr(node, 'logits') and node.logits:
                answer = "A" if node.logits >= 0.5 else "B"
                node_info += f" Answer: {answer} ({max(node.logits, 1 - node.logits):.3f})"
                
        output_buffer.append(node_info)
        
        # Print children
        for i, child in enumerate(node.children):
            is_last = (i == len(node.children) - 1)
            new_prefix = "└── " if is_last else "├── "
            self._print_tree_structure(mcts, output_buffer, child, depth + 1, new_prefix)
        
    def _count_max_depth(self):
        """Helper method to count maximum depth of the tree"""
        if not self.children:
            return self.depth
        return max(child._count_max_depth() for child in self.children)
        
    def _count_total_nodes(self):
        """Helper method to count total nodes in the tree"""
        return 1 + sum(child._count_total_nodes() for child in self.children)
        
    def _count_visited_nodes(self):
        """Helper method to count visited nodes in the tree"""
        visited = 1 if self.visited else 0
        return visited + sum(child._count_visited_nodes() for child in self.children)

    def print(self, mcts: MCTS, file=None):
        """
        Modified print method to only print the final summary once
        after a complete rollout instead of all details.
        """
        # Only generate the full summary if this is the root node (depth == 1)
        if self.depth == 1:
            output_lines = self._generate_final_summary(mcts)
            
            if file is None:
                log_file = "mcts_log.txt"
                with open(log_file, 'a') as f:
                    for line in output_lines:
                        print(line, file=f)
                print(f"Tree summary written to {log_file}", flush=True)
            else:
                with open(file, 'a') as f:
                    for line in output_lines:
                        print(line, file=f)
            
    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self,'llm_model') or self.llm_model is None:
            warnings.warn('MCTSNode loaded from pickle is read-only; Do not further roll out the tree!')

    def __getstate__(self):
        state = self.__dict__.copy()
        state['llm_model'] = None
        state['tokenizer'] = None
        return state


def medical_mcts_search(prompt: str, 
                        llm_model, 
                        tokenizer,
                        task_name,
                        mcts_rollouts=10,
                        w_exp=1.0,
                        r_alpha=0.4,
                        r1_default=1.0,
                        max_depth=5):
    """
    Run MCTS search for medical prediction
    
    Args:
        prompt: The medical case prompt
        llm_model: The language model for reasoning
        tokenizer: The model's tokenizer
        mcts_rollouts: Number of MCTS rollouts
        w_exp: Exploration weight
        r_alpha: Reward alpha parameter (balance between r0 and r1)
        r1_default: Default r1 value
        
    Returns:
        Final answer, detailed path, and the search tree
    """
    log_file = "mcts_search_log2.txt"
    
    with open(log_file, 'a') as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write("\n\n" + "="*80 + "\n")
        f.write(f"NEW CASE - {timestamp}\n")
        f.write("="*80 + "\n\n")
    
    print(f"MCTS search logs will be written to {log_file}")
    
    # Prepare initial prompt
    input_prompts = "\n<Case 3>:\n" + prompt.strip()
    
    mcts = MCTS(w_exp=w_exp, prior=True, aggr_reward='mean', aggr_child='max')
    root = MedicalMCTSNode(input_prompts, llm_model, tokenizer, depth=1, 
                          subquestion_prompts=subquestion_prompts, usefulness_prompts=usefulness_prompts,reward_prompts=reward_prompts,
                          task_name=task_name, r1_default=r1_default, r_alpha=r_alpha, max_depth=max_depth)
    
    logits = []
    trees = []
    
    with tqdm(range(mcts_rollouts), desc="MCTS Rollouts", position=0, leave=True) as pbar:
        for i in pbar:
            print(f"\nStarting rollout {i+1}/{mcts_rollouts}", flush=True)
            
            mcts.rollout(root)
            
            # Get the terminal node with highest reward
            max_n, max_r = mcts.max_mean_terminal(root)
            logits.append(max_n.logits)
            
            # Copy the tree for later analysis
            tree_copy = deepcopy(root)
            tree_copy.Q = dict(mcts.Q)
            tree_copy.N = dict(mcts.N)
            tree_copy.M = dict(mcts.M)
            trees.append(tree_copy)
            
            # Update progress
            pbar.set_postfix({"Best reward": f"{max_r:.3f}"})
    
    root.print(mcts, file=log_file)
    # Extract final prediction from best trajectory
    best_logits = logits[-1]
    answer = ""
    
    if best_logits >= 0.5:
        answer = "A"
    else:
        answer = "B"
        
    with open(log_file, 'a') as f:
        f.write("\nFINAL ANSWER: " + answer + "\n")
        f.write("=" * 80 + "\n\n")
    
    print(f"Final answer: {answer}, logits: {best_logits}")
    
    return best_logits, trees