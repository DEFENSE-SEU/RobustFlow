import pickle
import ast
import re
import json
import os
from graph_evaluator import t_eval_nodes, t_eval_graph, SentenceTransformer
from openai import OpenAI

client = OpenAI(
    api_key="sk-XilSri6YnWxgj96h3rvrHjAWl4Qly8GNu9ZWwfxpwdxN0plk",
    base_url="https://api.linkapi.org/v1",
)


def gen_prompt(graph_path, prompt_path, num):
    """第num个工作流"""
    with open(graph_path, 'rb') as f:
        graph_path_content = pickle.load(f)
    pattern = r'<graph>(.*?)</graph>'
    match = re.search(pattern, graph_path_content[num][1], re.DOTALL)
    graph_content = match.group(1).strip()

    with open(prompt_path, 'r') as f:
        prompt_content = f.read()
    prompt_content = prompt_content.strip()


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
    return PROMPT.strip()


def gen_answer(PROMPT):
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

def get_scoreflow(original_path, prompt_path, num):
    prompt = gen_prompt(original_path, prompt_path, num)
    answer = gen_answer(prompt)
    workflow = extract_graph_from_response(answer)
    return workflow
    

if __name__ == "__main__":
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    original_path = "../ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-original.pkl"
    requirements_path = "../ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-requirements.pkl"
    paraphrasing_path = "../ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-paraphrasing.pkl"
    light_noise_path = "../ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-light-noise.pkl"
    moderate_noise_path = "../ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-moderate-noise.pkl"
    heavy_noise_path = "../ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-heavy-noise.pkl"

    prompt_path = "../ScoreFlow/ScoreFlow/scripts/MBPP/op_prompt.py"
    original_nodes = []
    requirements_nodes = []
    paraphrasing_nodes = []
    light_noise_nodes = []
    moderate_noise_nodes = []
    heavy_noise_nodes = []
    original_graph = []
    requirements_graph = []
    paraphrasing_graph = []
    light_noise_graph = []
    moderate_noise_graph = []
    heavy_noise_graph = []

    with open(original_path, 'rb') as f:
        data = pickle.load(f)

    for num in range(len(data)):
        workflow_original = get_scoreflow(original_path, prompt_path, num)
        workflow_requirements = get_scoreflow(requirements_path, prompt_path, num)
        workflow_paraphrasing = get_scoreflow(paraphrasing_path, prompt_path, num)
        workflow_light_noise = get_scoreflow(light_noise_path, prompt_path, num)
        workflow_moderate_noise = get_scoreflow(moderate_noise_path, prompt_path, num)
        workflow_heavy_noise = get_scoreflow(heavy_noise_path, prompt_path, num)
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print(num)
        # print(workflow_original)
        # print(workflow_requirements)
        # print(workflow_paraphrasing)
        # print(workflow_noise)
        original_nodes.append(t_eval_nodes(workflow_original, workflow_original, model)["f1_score"])
        requirements_nodes.append(t_eval_nodes(workflow_requirements, workflow_original, model)["f1_score"])
        paraphrasing_nodes.append(t_eval_nodes(workflow_paraphrasing, workflow_original, model)["f1_score"])
        light_noise_nodes.append(t_eval_nodes(workflow_light_noise, workflow_original, model)["f1_score"])
        moderate_noise_nodes.append(t_eval_nodes(workflow_moderate_noise, workflow_original, model)["f1_score"])
        heavy_noise_nodes.append(t_eval_nodes(workflow_heavy_noise, workflow_original, model)["f1_score"])

        original_graph.append(t_eval_graph(workflow_original, workflow_original, model)["f1_score"])
        requirements_graph.append(t_eval_graph(workflow_requirements, workflow_original, model)["f1_score"])
        paraphrasing_graph.append(t_eval_graph(workflow_paraphrasing, workflow_original, model)["f1_score"])
        light_noise_graph.append(t_eval_graph(workflow_light_noise, workflow_original, model)["f1_score"])
        moderate_noise_graph.append(t_eval_graph(workflow_moderate_noise, workflow_original, model)["f1_score"])
        heavy_noise_graph.append(t_eval_graph(workflow_heavy_noise, workflow_original, model)["f1_score"])
        
    # print("===========================================================")
    # print("Nodes Score:")
    # print(original_nodes)
    # print(requirements_nodes)
    # print(paraphrasing_nodes)
    # print(noise_nodes)
    # print("Graph Score:")
    # print(original_graph)
    # print(requirements_graph)
    # print(paraphrasing_graph)
    # print(noise_graph)
    # print("===========================================================")
    # 将原来的print语句替换为文件写入
    with open("scoreflow_score.txt", "a") as f:
        f.write(f"original_nodes_score: {sum(original_nodes) / 5}\n")
        f.write(f"requirements_nodes_score: {sum(requirements_nodes) / 5}\n")
        f.write(f"paraphrasing_nodes_score: {sum(paraphrasing_nodes) / 5}\n")
        f.write(f"light_noise_nodes_score: {sum(light_noise_nodes) / 5}\n")
        f.write(f"moderate_noise_nodes_score: {sum(moderate_noise_nodes) / 5}\n")
        f.write(f"heavy_noise_nodes_score: {sum(heavy_noise_nodes) / 5}\n")
        f.write(f"original_graph_score: {sum(original_graph) / 5}\n")
        f.write(f"requirements_graph_score: {sum(requirements_graph) / 5}\n")
        f.write(f"paraphrasing_graph_score: {sum(paraphrasing_graph) / 5}\n")
        f.write(f"light_noise_graph_score: {sum(light_noise_graph) / 5}\n")
        f.write(f"moderate_noise_graph_score: {sum(moderate_noise_graph) / 5}\n")
        f.write(f"heavy_noise_graph_score: {sum(heavy_noise_graph) / 5}\n")