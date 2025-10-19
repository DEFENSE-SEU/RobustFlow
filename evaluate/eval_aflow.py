import ast
import re
import json
import sys
import os
from graph_evaluator import t_eval_nodes, t_eval_graph, SentenceTransformer
from openai import OpenAI
import yaml

config_path = "../../config/config2.yaml"

def load_config(config_path):
    """Load YAML configuration file and return configuration dictionary"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_openai_client(config_path, model_name=None):
    """Get OpenAI client and model configuration based on config file"""
    config = load_config(config_path)
    
    if not model_name:
        model_name = list(config['models'].keys())[0]
    
    if model_name not in config['models']:
        raise ValueError(f"Model '{model_name}' not found in configuration")
    
    model_config = config['models'][model_name]

    client = OpenAI(
        api_key=model_config['api_key'],
        base_url=model_config['base_url']
    )
    
    return client, model_config

client, model_config = get_openai_client(config_path)


def gen_prompt(graph_path, prompt_path):
    """Generate prompt for analyzing code structure and converting to graph structure"""
    with open(graph_path, 'r') as f:
        graph_content = f.read()

    with open(prompt_path, 'r') as f:
        prompt_content = f.read()
    

    PROMPT = f"""
The following is the content of `graph.py`:
```python\n{graph_content}\n```

The following is the content of `prompt.py`:
```python\n{prompt_content}\n```

Your task is to parse the above Python code and convert it into a **Python dictionary** representing a directed acyclic graph that satisfies **all** of the following requirements:
1. Each function call in the code must be represented as an **independent node**, even if the same function is called multiple times (e.g., three CODE_GENERATE calls must create three separate nodes).
2. Each node must include the complete Prompt content used for that function call. Use the following format for each node's description string:
"{{Operation Type}} using {{PROMPT_NAME}}: {{Prompt Content}}"
- The "Operation Type" is a symbolic label like `CODE_GENERATE`, `SC_ENSEMBLE`, `TEST`, etc.
- The PROMPT_NAME must match either:
(a) The parameter name in the function call (e.g., instruction or prompt) if the prompt is passed directly as a string,
OR
(b) The variable name from prompt.py (e.g., SC_ENSEMBLE_PROMPT), if the prompt is passed implicitly or by reference.
- The Prompt Content must be included verbatim and in full, whether it is extracted from prompt.py or from inline function arguments.
3. The graph should reflect execution flow using directed edges. Use **Python tuples** to represent edges between node indices, e.g., `(0, 1)` means an edge from node 0 to node 1. Example structure:
- `"START"` → all CODE_GENERATE nodes
- CODE_GENERATE nodes → SC_ENSEMBLE
- SC_ENSEMBLE → TEST_CASE_GEN → TEST
- TEST → FIX_CODE (only if test fails)
- All flows must eventually end at the `"END"` node.
4. The graph must be a **Directed Acyclic Graph (DAG)** — no cycles are allowed. Non-linear execution flow is allowed.
5. There must be a `"START"` node and an `"END"` node in the graph.
6. **Do not include comments** in the output graph structure.
7. The final output must be a **valid Python dictionary** enclosed in triple backticks, and must follow this format exactly:

Example format:
{{
"nodes": ["START", "Node1 description", "Node2 description", "END"],
"edges": [(0,1), (1,2), (2,3)]
}}
"""
    return PROMPT
    

def gen_answer(PROMPT):
    """Call GPT model to generate graph structure analysis results"""
    try:
        completion = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a Code Structure Analyst and Graph Modeling Expert. "
                        "Your task is to analyze Python code and convert the execution flow into "
                        "a **Python dictionary format Directed Acyclic Graph (DAG)** with precise node labeling "
                        "and edge dependencies.\n\n"
                        "RULES FOR OUTPUT:\n"
                        "- Each function call should be represented as a separate node, even if repeated.\n"
                        "- Node description format: '{Operation Type} using {PROMPT_NAME}: {PROMPT Content}'.\n"
                        "- Edges represent execution order; use indices (0, 1), (1, 2), etc., in Python tuple format.\n"
                        "- Always include 'START' and 'END' nodes as part of the graph structure.\n"
                        "- Output must be in **valid Python dictionary format**, enclosed in triple backticks (```), not JSON.\n"
                        "- Do not include any explanation outside of the code block.\n"
                        "- Ensure the graph is acyclic, reflecting conditional branches appropriately (if applicable).\n"
                    )
                },
                {
                    "role": "user", 
                    "content": PROMPT
                }
            ],
            temperature = 0.3,
        )
        answer = completion.choices[0].message.content
        return answer.strip()
    except Exception as e:
        return f"Error when calling LLM: {str(e)}"

def extract_graph_from_response(response):
    """Extract graph structure information from GPT response"""
    node_pattern = re.compile(r'"([^"]+)",')
    node_matches = node_pattern.findall(response)
    node_workflow = [match.strip() for match in node_matches]
    node_workflow.append("END")
    
    edge_pattern = re.compile(r'\(\s*(\d+|START)\s*,\s*(\d+|END)\s*\)')
    edge_matches = edge_pattern.findall(response)
    edge_workflow = []
    for i, match in enumerate(edge_matches):
        edge = list(match)
        if "START" in edge:
            edge[edge.index("START")] = "0"
        if "END" in edge:
            edge_num = len(node_workflow) - 1
            edge[edge.index("END")] = str(edge_num)
        edge = tuple(map(int, edge))
        edge_workflow.append(edge)
    workflow = {"nodes":node_workflow,"edges":edge_workflow}
    return workflow

def get_scoreflow(original_path, prompt_path):
    """Get ScoreFlow workflow structure"""
    prompt = gen_prompt(original_path, prompt_path)
    answer = gen_answer(prompt)
    workflow = extract_graph_from_response(answer)
    return workflow

def evaluate_variant_group(original_path, requirements_path, paraphrasing_path, light_noise_path, moderate_noise_path, heavy_noise_path, prompt_path):
    """Evaluate workflow structure similarity for a group of variant datasets"""
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # Get ScoreFlow workflows
    workflow_original = get_scoreflow(original_path, prompt_path)
    workflow_requirements = get_scoreflow(requirements_path, prompt_path)
    workflow_paraphrasing = get_scoreflow(paraphrasing_path, prompt_path)
    workflow_light_noise = get_scoreflow(light_noise_path, prompt_path)
    workflow_moderate_noise = get_scoreflow(moderate_noise_path, prompt_path)
    workflow_heavy_noise = get_scoreflow(heavy_noise_path, prompt_path)

    print("===========================================================")
    print(workflow_original)
    print("-----------------------------------------------------------")
    print(workflow_requirements)
    print("-----------------------------------------------------------")
    print(workflow_paraphrasing)
    print("-----------------------------------------------------------")
    print(workflow_light_noise)
    print("-----------------------------------------------------------")
    print(workflow_moderate_noise)
    print("-----------------------------------------------------------")
    print(workflow_heavy_noise)

    # Calculate node F1 scores
    original_nodes_f1 = t_eval_nodes(workflow_original, workflow_original, model)["f1_score"]
    requirements_nodes_f1 = t_eval_nodes(workflow_requirements, workflow_original, model)["f1_score"]
    paraphrasing_nodes_f1 = t_eval_nodes(workflow_paraphrasing, workflow_original, model)["f1_score"]
    light_noise_nodes_f1 = t_eval_nodes(workflow_light_noise, workflow_original, model)["f1_score"]
    moderate_noise_nodes_f1 = t_eval_nodes(workflow_moderate_noise, workflow_original, model)["f1_score"]
    heavy_noise_nodes_f1 = t_eval_nodes(workflow_heavy_noise, workflow_original, model)["f1_score"]

    # Calculate graph structure F1 scores
    original_graph_f1 = t_eval_graph(workflow_original, workflow_original, model)["f1_score"]
    requirements_graph_f1 = t_eval_graph(workflow_requirements, workflow_original, model)["f1_score"]
    paraphrasing_graph_f1 = t_eval_graph(workflow_paraphrasing, workflow_original, model)["f1_score"]
    light_noise_graph_f1 = t_eval_graph(workflow_light_noise, workflow_original, model)["f1_score"]
    moderate_noise_graph_f1 = t_eval_graph(workflow_moderate_noise, workflow_original, model)["f1_score"]
    heavy_noise_graph_f1 = t_eval_graph(workflow_heavy_noise, workflow_original, model)["f1_score"]

    scores = [
        original_nodes_f1,
        requirements_nodes_f1,
        paraphrasing_nodes_f1,
        light_noise_nodes_f1,
        moderate_noise_nodes_f1,
        heavy_noise_nodes_f1,
        original_graph_f1,
        requirements_graph_f1,
        paraphrasing_graph_f1,
        light_noise_graph_f1,
        moderate_noise_graph_f1,
        heavy_noise_graph_f1,
    ]

    ori_node_scores.append(scores[0])
    req_node_scores.append(scores[1])
    para_node_scores.append(scores[2])
    light_node_scores.append(scores[3])
    moderate_node_scores.append(scores[4])
    heavy_node_scores.append(scores[5])

    ori_graph_scores.append(scores[6])
    req_graph_scores.append(scores[7])
    para_graph_scores.append(scores[8])
    light_graph_scores.append(scores[9])
    moderate_graph_scores.append(scores[10])
    heavy_graph_scores.append(scores[11])

ori_node_scores = []
req_node_scores = []
para_node_scores = []
light_node_scores = []
moderate_node_scores = []
heavy_node_scores = []

ori_graph_scores = []
req_graph_scores = []
para_graph_scores = []
light_graph_scores = []
moderate_graph_scores = []
heavy_graph_scores = []

def parse_best_results(file_path):
    """Parse best.txt file and extract best result paths for each dataset"""
    datasets = {}
    current_dataset = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('[INFO] Entering task directory:'):
                dataset_name = line.split('\\')[-1]
                current_dataset = dataset_name
                datasets[current_dataset] = {}
                
            elif line.startswith('[BEST]'):
                parts = line.split('/round ')
                if len(parts) >= 2:
                    path_part = parts[0].replace('[BEST] ', '')
                    round_part = parts[1].split(' ')[0]
                    
                    graph_path = path_part.replace('\\results.json', f'\\round_{round_part}\\graph.py')
                    
                    path_segments = graph_path.split('\\')
                    noise_type = None
                    for segment in path_segments:
                        if 'original' in segment:
                            noise_type = 'original'
                            break
                        elif 'requirements' in segment:
                            noise_type = 'requirements'
                            break
                        elif 'paraphrasing' in segment:
                            noise_type = 'paraphrasing'
                            break
                        elif 'light_noise' in segment:
                            noise_type = 'light_noise'
                            break
                        elif 'moderate_noise' in segment:
                            noise_type = 'moderate_noise'
                            break
                        elif 'heavy_noise' in segment:
                            noise_type = 'heavy_noise'
                            break
                    
                    if noise_type and current_dataset:
                        datasets[current_dataset][noise_type] = graph_path
    
    return datasets

def clear_score_arrays():
    """Clear all score storage arrays"""
    global ori_node_scores, req_node_scores, para_node_scores, light_node_scores, moderate_node_scores, heavy_node_scores
    global ori_graph_scores, req_graph_scores, para_graph_scores, light_graph_scores, moderate_graph_scores, heavy_graph_scores
    
    ori_node_scores.clear()
    req_node_scores.clear()
    para_node_scores.clear()
    light_node_scores.clear()
    moderate_node_scores.clear()
    heavy_node_scores.clear()
    
    ori_graph_scores.clear()
    req_graph_scores.clear()
    para_graph_scores.clear()
    light_graph_scores.clear()
    moderate_graph_scores.clear()
    heavy_graph_scores.clear()

def write_dataset_scores(dataset_name, f):
    """Write dataset scores to file"""
    f.write(f"{dataset_name}\n")
    
    # Write node scores (calculate mean)
    if ori_node_scores:
        f.write("original_nodes_score: " + str(sum(ori_node_scores) / len(ori_node_scores)) + "\n")
    if req_node_scores:
        f.write("requirements_nodes_score: " + str(sum(req_node_scores) / len(req_node_scores)) + "\n")
    if para_node_scores:
        f.write("paraphrasing_nodes_score: " + str(sum(para_node_scores) / len(para_node_scores)) + "\n")
    if light_node_scores:
        f.write("light_noise_nodes_score: " + str(sum(light_node_scores) / len(light_node_scores)) + "\n")
    if moderate_node_scores:
        f.write("moderate_noise_nodes_score: " + str(sum(moderate_node_scores) / len(moderate_node_scores)) + "\n")
    if heavy_node_scores:
        f.write("heavy_noise_nodes_score: " + str(sum(heavy_node_scores) / len(heavy_node_scores)) + "\n")
    
    # Write graph structure scores (calculate mean)
    if ori_graph_scores:
        f.write("original_graph_score: " + str(sum(ori_graph_scores) / len(ori_graph_scores)) + "\n")
    if req_graph_scores:
        f.write("requirements_graph_score: " + str(sum(req_graph_scores) / len(req_graph_scores)) + "\n")
    if para_graph_scores:
        f.write("paraphrasing_graph_score: " + str(sum(para_graph_scores) / len(para_graph_scores)) + "\n")
    if light_graph_scores:
        f.write("light_noise_graph_score: " + str(sum(light_graph_scores) / len(light_graph_scores)) + "\n")
    if moderate_graph_scores:
        f.write("moderate_noise_graph_score: " + str(sum(moderate_graph_scores) / len(moderate_graph_scores)) + "\n")
    if heavy_graph_scores:
        f.write("heavy_noise_graph_score: " + str(sum(heavy_graph_scores) / len(heavy_graph_scores)) + "\n")

if __name__ == "__main__":
    """Main function: Read paths from best.txt and execute AFlow evaluation for all datasets"""
    prompt_path = "../AFlow/workspace/MBPP/workflows/template/op_prompt.py"
    
    # Parse best.txt file to get best result paths for all datasets
    datasets = parse_best_results("aflow_scripts/best.txt")
    
    # Process datasets sorted by name
    for dataset_name in sorted(datasets.keys()):
        print(f"Processing dataset: {dataset_name}")
        
        dataset_paths = datasets[dataset_name]
        
        # Check if all required paths are included
        required_types = ['original', 'requirements', 'paraphrasing', 'light_noise', 'moderate_noise', 'heavy_noise']
        if not all(t in dataset_paths for t in required_types):
            print(f"Warning: {dataset_name} is missing some required paths, skipping...")
            continue
        
        # Clear score arrays (prepare for current dataset)
        clear_score_arrays()
        
        # Call evaluation function
        evaluate_variant_group(
            dataset_paths['original'],
            dataset_paths['requirements'],
            dataset_paths['paraphrasing'],
            dataset_paths['light_noise'],
            dataset_paths['moderate_noise'],
            dataset_paths['heavy_noise'],
            prompt_path
        )
        
        # Write current dataset scores to file
        with open("aflow_score.txt", "a") as f:
            write_dataset_scores(dataset_name, f)
        
        print(f"Completed processing dataset: {dataset_name}")