"""
AFlow评估器模块 (AFlow Evaluator Module)

该模块实现了对AFlow工作流代码的自动化评估功能，通过将Python代码转换为有向无环图(DAG)，
并使用大语言模型进行分析，最终计算不同变体工作流的相似度分数。

主要功能：
1. 配置管理 - 从YAML文件读取API配置参数
2. 代码解析 - 将Python工作流代码转换为图结构
3. 图生成 - 使用LLM生成标准化的DAG表示
4. 图提取 - 从LLM响应中提取图结构信息
5. 相似度评估 - 计算不同工作流变体之间的相似度分数
"""

import ast  # 用于Python代码的抽象语法树解析
import re  # 用于正则表达式匹配和文本处理
import json  # 用于JSON数据处理
import sys  # 用于系统相关操作
import os  # 用于操作系统相关操作
import yaml  # 用于YAML配置文件解析
from graph_evaluator import t_eval_nodes, t_eval_graph, SentenceTransformer  # 导入图评估器和语义模型
from openai import OpenAI  # 导入OpenAI API客户端


def load_config(config_path="config/config2.yaml"):
    """
    从YAML配置文件加载配置信息
    
    该函数负责读取和解析YAML格式的配置文件，配置文件包含大语言模型的API密钥、
    基础URL、温度参数等信息。通过集中管理配置信息，提高了代码的可维护性和安全性。
    
    参数:
        config_path (str): 配置文件路径，默认为"config/config2.yaml"
        
    返回:
        dict: 包含所有配置信息的字典，结构为：
              {
                  "models": {
                      "model_name": {
                          "api_type": "openai",
                          "base_url": "...",
                          "api_key": "...",
                          "temperature": 0.3
                      }
                  }
              }
    
    异常:
        FileNotFoundError: 当配置文件不存在时抛出
        ValueError: 当YAML格式错误时抛出
    
    示例:
        >>> config = load_config("config/config2.yaml")
        >>> print(config['models'].keys())
        ['gpt-4o-mini', 'gpt-4']
    """
    try:
        # 以UTF-8编码打开配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            # 使用yaml.safe_load安全地解析YAML内容，避免执行恶意代码
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        # 配置文件不存在时的错误处理
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    except yaml.YAMLError as e:
        # YAML格式错误时的错误处理
        raise ValueError(f"YAML配置文件解析错误: {e}")


def get_openai_client(config_path="config/config2.yaml", model_name=None):
    """
    根据配置文件初始化OpenAI客户端
    
    该函数从配置文件中读取指定模型的配置信息，创建并返回OpenAI API客户端。
    支持多个模型配置，可以根据需要选择不同的模型进行API调用。
    
    参数:
        config_path (str): 配置文件路径，默认为"config/config2.yaml"
        model_name (str): 配置中的模型名称。如果为None，则使用第一个可用模型
        
    返回:
        tuple: 包含两个元素的元组
            - OpenAI: 初始化完成的OpenAI客户端对象
            - dict: 指定模型的配置信息字典
    
    异常:
        ValueError: 当指定的模型不存在或API类型不支持时抛出
    
    示例:
        >>> client, config = get_openai_client("config/config2.yaml", "gpt-4o-mini")
        >>> print(config['temperature'])
        0.3
    """
    # 从配置文件加载配置信息
    config = load_config(config_path)
    
    if not model_name:
        # 如果没有指定模型名称，使用配置中的第一个可用模型
        model_name = list(config['models'].keys())[0]
    
    # 检查指定的模型是否存在于配置中
    if model_name not in config['models']:
        raise ValueError(f"配置中未找到模型 '{model_name}'")
    
    # 获取指定模型的配置信息
    model_config = config['models'][model_name]
    
    # 使用配置信息创建OpenAI客户端
    client = OpenAI(
        api_key=model_config['api_key'],  # 从配置中获取API密钥
        base_url=model_config['base_url']  # 从配置中获取基础URL
    )
    
    # 返回客户端对象和模型配置信息
    return client, model_config


# 从配置文件初始化OpenAI客户端
# 这里使用默认配置创建全局客户端实例，供后续函数使用
client, model_config = get_openai_client()


def gen_prompt(graph_path, prompt_path):
    """
    生成用于LLM的提示词，将Python工作流代码转换为图结构
    
    该函数读取两个Python文件的内容：graph.py（包含工作流代码）和prompt.py（包含提示模板），
    然后生成一个详细的提示词，指导大语言模型将Python代码转换为标准化的有向无环图(DAG)表示。
    
    参数:
        graph_path (str): graph.py文件的路径，包含待分析的工作流Python代码
        prompt_path (str): prompt.py文件的路径，包含提示模板和常量定义
        
    返回:
        str: 格式化后的提示词字符串，包含完整的指令和示例
        
    功能说明:
        1. 读取并解析Python工作流代码
        2. 提取提示模板信息
        3. 生成标准化的DAG转换指令
        4. 提供详细的输出格式要求
        
    示例:
        >>> prompt = gen_prompt("workflow/graph.py", "templates/prompt.py")
        >>> print(prompt[:100])
        The following is the content of `graph.py`:
        ```python
    """
    # 读取graph.py文件内容，包含待分析的工作流Python代码
    with open(graph_path, 'r') as f:
        graph_content = f.read()

    # 读取prompt.py文件内容，包含提示模板和常量定义
    with open(prompt_path, 'r') as f:
        prompt_content = f.read()
    
    # 构建完整的提示词，包含详细的转换指令和要求
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
    

def gen_answer(PROMPT, model_name=None, config_path="config/config2.yaml"):
    """
    使用OpenAI API生成答案，支持从YAML配置文件读取参数
    
    该函数将用户提供的提示词发送给大语言模型，并返回生成的响应。函数支持
    灵活的模型选择和配置管理，可以根据需要选择不同的模型和参数设置。
    
    参数:
        PROMPT (str): 发送给模型的提示词内容
        model_name (str): 配置中的模型名称。如果为None，使用第一个可用模型
        config_path (str): 配置文件路径，默认为"config/config2.yaml"
        
    返回:
        str: 模型生成的答案文本，如果出现错误则返回错误信息
        
    功能说明:
        1. 根据配置创建OpenAI客户端
        2. 构建包含系统指令和用户提示的消息
        3. 调用API生成响应
        4. 处理异常情况并返回错误信息
        
    示例:
        >>> answer = gen_answer("Convert this code to a graph: ...")
        >>> print(answer[:100])
        {
        "nodes": ["START", "CODE_GENERATE using instruction: ...", "END"],
    """
    try:
        # 根据指定配置获取OpenAI客户端和模型配置
        client, model_config = get_openai_client(config_path, model_name)
        
        # 调用OpenAI API生成回答
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # 使用的模型名称（可以根据需要配置）
            messages=[
                {
                    "role": "system",  # 系统角色消息，定义AI助手的行为
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
                    "role": "user",  # 用户角色消息，包含实际的提示词内容
                    "content": PROMPT
                }
            ],
            temperature=model_config.get('temperature', 0),  # 从配置获取温度参数，默认0
        )
        # 提取生成的回答内容并去除首尾空白字符
        answer = completion.choices[0].message.content
        return answer.strip()
    except Exception as e:
        # 捕获所有异常并返回错误信息
        return f"调用大语言模型时出错: {str(e)}"

def extract_graph_from_response(response):
    """
    从LLM响应中提取图结构信息
    
    该函数使用正则表达式从大语言模型的响应文本中提取节点和边的信息，
    并将其转换为标准的工作流字典格式。函数能够处理包含START和END
    特殊节点的图结构，并自动转换索引格式。
    
    参数:
        response (str): LLM返回的响应文本，包含图结构的字典表示
        
    返回:
        dict: 包含"nodes"和"edges"键的字典
            - "nodes": 节点列表，包含START、END和中间节点
            - "edges": 边列表，每个边为(node_index1, node_index2)的元组
    
    功能说明:
        1. 使用正则表达式提取节点名称
        2. 自动添加END节点
        3. 提取边信息并转换为索引格式
        4. 处理START和END节点的特殊索引
    
    示例:
        >>> response = '{"nodes": ["START", "Node1", "END"], "edges": [(0,1), (1,2)]}'
        >>> workflow = extract_graph_from_response(response)
        >>> print(workflow)
        {'nodes': ['START', 'Node1', 'END'], 'edges': [(0, 1), (1, 2)]}
    """
    # 定义正则表达式模式，用于匹配节点名称（双引号包围的字符串）
    node_pattern = re.compile(r'"([^"]+)",')
    # 在响应文本中查找所有匹配的节点名称
    node_matches = node_pattern.findall(response)
    # 清理节点名称（去除空白字符）并构建节点列表
    node_workflow = [match.strip() for match in node_matches]
    # 确保节点列表以END节点结尾
    node_workflow.append("END")
    
    # 定义正则表达式模式，用于匹配边信息（括号包围的节点索引对）
    edge_pattern = re.compile(r'\(\s*(\d+|START)\s*,\s*(\d+|END)\s*\)')
    # 在响应文本中查找所有匹配的边
    edge_matches = edge_pattern.findall(response)
    edge_workflow = []
    
    # 处理每个匹配的边
    for i, match in enumerate(edge_matches):
        edge = list(match)  # 将元组转换为列表以便修改
        
        # 如果边中包含START，将其转换为索引0
        if "START" in edge:
            edge[edge.index("START")] = "0"
        
        # 如果边中包含END，将其转换为最后一个节点的索引
        if "END" in edge:
            edge_num = len(node_workflow) - 1  # END节点的索引
            edge[edge.index("END")] = str(edge_num)
        
        # 将字符串索引转换为整数并转换为元组
        edge = tuple(map(int, edge))
        edge_workflow.append(edge)
    
    # 构建并返回完整的工作流字典
    workflow = {"nodes": node_workflow, "edges": edge_workflow}
    return workflow

def get_scoreflow(original_path, prompt_path, model_name=None, config_path="config/config2.yaml"):
    """
    通过分析原始图文件生成工作流结构
    
    该函数是工作流生成的核心函数，它将Python代码文件转换为标准化的图结构。
    整个流程包括：读取代码文件、生成提示词、调用LLM分析、提取图结构。
    
    参数:
        original_path (str): 原始图文件路径，包含待分析的Python工作流代码
        prompt_path (str): 提示模板文件路径，包含提示词模板和常量定义
        model_name (str): 配置中的模型名称。如果为None，使用第一个可用模型
        config_path (str): 配置文件路径，默认为"config/config2.yaml"
        
    返回:
        dict: 生成的工作流字典，包含"nodes"和"edges"键
            - "nodes": 节点列表，每个节点包含操作类型和提示内容
            - "edges": 边列表，表示节点间的执行依赖关系
    
    功能流程:
        1. 生成包含代码和模板的提示词
        2. 调用LLM分析代码并生成图结构
        3. 从LLM响应中提取标准化的图结构
        4. 返回完整的工作流字典
    
    示例:
        >>> workflow = get_scoreflow("workflow/graph.py", "templates/prompt.py")
        >>> print(workflow["nodes"])
        ["START", "CODE_GENERATE using instruction: ...", "END"]
    """
    # 第一步：生成包含代码分析和转换指令的提示词
    prompt = gen_prompt(original_path, prompt_path)
    
    # 第二步：调用大语言模型分析代码并生成图结构响应
    answer = gen_answer(prompt, model_name, config_path)
    
    # 第三步：从LLM响应中提取标准化的图结构信息
    workflow = extract_graph_from_response(answer)
    
    # 返回完整的工作流字典
    return workflow

def evaluate_variant_group(original_path, requirements_path, paraphrasing_path, light_noise_path, moderate_noise_path, heavy_noise_path, prompt_path, model_name=None, config_path="config/config2.yaml"):
    """
    评估不同变体工作流组的相似度
    
    该函数是评估系统的核心函数，它分析多个不同变体的工作流（原始版本、需求导向版本、
    释义版本、轻度噪声版本、中度噪声版本、重度噪声版本），并计算它们与原始工作流的相似度。
    
    参数:
        original_path (str): 原始工作流图文件路径
        requirements_path (str): 需求导向工作流图文件路径
        paraphrasing_path (str): 释义工作流图文件路径
        light_noise_path (str): 轻度噪声工作流图文件路径
        moderate_noise_path (str): 中度噪声工作流图文件路径
        heavy_noise_path (str): 重度噪声工作流图文件路径
        prompt_path (str): 提示模板文件路径
        model_name (str): 配置中的模型名称。如果为None，使用第一个可用模型
        config_path (str): 配置文件路径，默认为"config/config2.yaml"
    
    功能说明:
        1. 为所有变体工作流生成标准化的图结构
        2. 使用语义模型计算节点和图级别的相似度
        3. 计算精确率、召回率和F1分数
        4. 将结果存储到全局分数列表中
    
    评估指标:
        - 节点级F1分数：基于节点语义内容和序列匹配的相似度
        - 图级F1分数：基于图结构和可达性关系的相似度
    """
    # 初始化语义相似度模型，用于计算节点间的语义相似度
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # 为所有变体工作流生成标准化的图结构
    workflow_original = get_scoreflow(original_path, prompt_path, model_name, config_path)
    workflow_requirements = get_scoreflow(requirements_path, prompt_path, model_name, config_path)
    workflow_paraphrasing = get_scoreflow(paraphrasing_path, prompt_path, model_name, config_path)
    workflow_light_noise = get_scoreflow(light_noise_path, prompt_path, model_name, config_path)
    workflow_moderate_noise = get_scoreflow(moderate_noise_path, prompt_path, model_name, config_path)
    workflow_heavy_noise = get_scoreflow(heavy_noise_path, prompt_path, model_name, config_path)

    # 打印所有工作流的图结构信息，用于调试和验证
    print("===========================================================")
    print("原始工作流结构:")
    print(workflow_original)
    print("-----------------------------------------------------------")
    print("需求导向工作流结构:")
    print(workflow_requirements)
    print("-----------------------------------------------------------")
    print("释义工作流结构:")
    print(workflow_paraphrasing)
    print("-----------------------------------------------------------")
    print("轻度噪声工作流结构:")
    print(workflow_light_noise)
    print("-----------------------------------------------------------")
    print("中度噪声工作流结构:")
    print(workflow_moderate_noise)
    print("-----------------------------------------------------------")
    print("重度噪声工作流结构:")
    print(workflow_heavy_noise)

    # 计算所有变体工作流的节点级F1分数（以原始工作流为基准）
    original_nodes_f1 = t_eval_nodes(workflow_original, workflow_original, model)["f1_score"]
    requirements_nodes_f1 = t_eval_nodes(workflow_requirements, workflow_original, model)["f1_score"]
    paraphrasing_nodes_f1 = t_eval_nodes(workflow_paraphrasing, workflow_original, model)["f1_score"]
    light_noise_nodes_f1 = t_eval_nodes(workflow_light_noise, workflow_original, model)["f1_score"]
    moderate_noise_nodes_f1 = t_eval_nodes(workflow_moderate_noise, workflow_original, model)["f1_score"]
    heavy_noise_nodes_f1 = t_eval_nodes(workflow_heavy_noise, workflow_original, model)["f1_score"]

    # 计算所有变体工作流的图级F1分数（以原始工作流为基准）
    original_graph_f1 = t_eval_graph(workflow_original, workflow_original, model)["f1_score"]
    requirements_graph_f1 = t_eval_graph(workflow_requirements, workflow_original, model)["f1_score"]
    paraphrasing_graph_f1 = t_eval_graph(workflow_paraphrasing, workflow_original, model)["f1_score"]
    light_noise_graph_f1 = t_eval_graph(workflow_light_noise, workflow_original, model)["f1_score"]
    moderate_noise_graph_f1 = t_eval_graph(workflow_moderate_noise, workflow_original, model)["f1_score"]
    heavy_noise_graph_f1 = t_eval_graph(workflow_heavy_noise, workflow_original, model)["f1_score"]

    # 将所有评估分数组织成列表，便于后续处理和存储
    scores = [
        original_nodes_f1,      # 原始工作流节点级F1分数
        requirements_nodes_f1,  # 需求导向工作流节点级F1分数
        paraphrasing_nodes_f1,  # 释义工作流节点级F1分数
        light_noise_nodes_f1,   # 轻度噪声工作流节点级F1分数
        moderate_noise_nodes_f1, # 中度噪声工作流节点级F1分数
        heavy_noise_nodes_f1,   # 重度噪声工作流节点级F1分数
        original_graph_f1,      # 原始工作流图级F1分数
        requirements_graph_f1,  # 需求导向工作流图级F1分数
        paraphrasing_graph_f1,  # 释义工作流图级F1分数
        light_noise_graph_f1,   # 轻度噪声工作流图级F1分数
        moderate_noise_graph_f1, # 中度噪声工作流图级F1分数
        heavy_noise_graph_f1,   # 重度噪声工作流图级F1分数
    ]

    # 将节点级评估分数添加到对应的全局分数列表中
    ori_node_scores.append(scores[0])    # 原始工作流节点分数
    req_node_scores.append(scores[1])    # 需求导向工作流节点分数
    para_node_scores.append(scores[2])   # 释义工作流节点分数
    light_node_scores.append(scores[3])  # 轻度噪声工作流节点分数
    moderate_node_scores.append(scores[4]) # 中度噪声工作流节点分数
    heavy_node_scores.append(scores[5])  # 重度噪声工作流节点分数

    # 将图级评估分数添加到对应的全局分数列表中
    ori_graph_scores.append(scores[6])   # 原始工作流图分数
    req_graph_scores.append(scores[7])   # 需求导向工作流图分数
    para_graph_scores.append(scores[8])  # 释义工作流图分数
    light_graph_scores.append(scores[9]) # 轻度噪声工作流图分数
    moderate_graph_scores.append(scores[10]) # 中度噪声工作流图分数
    heavy_graph_scores.append(scores[11]) # 重度噪声工作流图分数

# =============== 全局分数存储列表 ===============
# 以下全局变量用于存储多次评估的分数结果，便于后续统计分析

# 节点级评估分数存储列表
ori_node_scores = []      # 原始工作流节点级F1分数列表
req_node_scores = []      # 需求导向工作流节点级F1分数列表
para_node_scores = []     # 释义工作流节点级F1分数列表
light_node_scores = []    # 轻度噪声工作流节点级F1分数列表
moderate_node_scores = [] # 中度噪声工作流节点级F1分数列表
heavy_node_scores = []    # 重度噪声工作流节点级F1分数列表

# 图级评估分数存储列表
ori_graph_scores = []     # 原始工作流图级F1分数列表
req_graph_scores = []     # 需求导向工作流图级F1分数列表
para_graph_scores = []    # 释义工作流图级F1分数列表
light_graph_scores = []   # 轻度噪声工作流图级F1分数列表
moderate_graph_scores = [] # 中度噪声工作流图级F1分数列表
heavy_graph_scores = []   # 重度噪声工作流图级F1分数列表

if __name__ == "__main__":
    """
    主函数：执行AFlow工作流评估的主程序
    
    该主函数负责设置评估参数、定义测试数据路径，并执行多个工作流变体的评估。
    评估过程包括：读取不同变体的工作流代码、生成图结构、计算相似度分数、保存结果。
    
    评估流程:
        1. 设置提示模板和配置文件路径
        2. 定义多个测试案例的工作流文件路径
        3. 对每个测试案例执行变体组评估
        4. 计算并保存评估结果到文件
    
    数据说明:
        - MBPP: Microsoft Big Python Problems数据集
        - 每个测试案例包含6种变体：原始、需求导向、释义、轻度噪声、中度噪声、重度噪声
        - 评估指标：节点级F1分数和图级F1分数
    """
    
    # =============== 配置参数设置 ===============
    # 提示模板文件路径，包含用于代码分析的提示词模板
    prompt_path = "../AFlow/workspace/MBPP/workflows/template/op_prompt.py"

    # =============== 测试案例1：MBPP问题1 ===============
    # 原始工作流文件路径
    original_path1 = "aflow_scripts/MBPP/mbpp_original_1/round_8/graph.py"
    # 以下路径被注释掉，表示这些变体暂时不可用或跳过
    # requirements_path1 = "aflow_scripts/MBPP/mbpp_requirements_1/round_3/graph.py"
    # paraphrasing_path1 = "aflow_scripts/MBPP/mbpp_paraphrasing_1/round_4/graph.py"
    
    # 可用的变体工作流文件路径
    light_noise_path1 = "aflow_scripts/MBPP/mbpp_light_noise_1/round_6/graph.py"
    moderate_noise_path1 = "aflow_scripts/MBPP/mbpp_moderate_noise_1/round_7/graph.py"
    # heavy_noise_path1 = "aflow_scripts/MBPP/mbpp_heavy_noise_1/round_3/graph.py"
    
    # 临时使用light_noise路径作为其他变体的占位符（实际使用时应该替换为真实路径）
    requirements_path1 = "aflow_scripts/MBPP/mbpp_light_noise_1/round_6/graph.py"
    paraphrasing_path1 = "aflow_scripts/MBPP/mbpp_light_noise_1/round_6/graph.py"
    heavy_noise_path1 = "aflow_scripts/MBPP/mbpp_light_noise_1/round_6/graph.py"
    
    # =============== 模型配置参数 ===============
    config_path = "config/config2.yaml"  # 配置文件路径
    model_name = None  # 使用配置中的第一个可用模型，或指定特定模型名称
    
    # 执行第一个测试案例的变体组评估
    evaluate_variant_group(original_path1, requirements_path1, paraphrasing_path1, 
                          light_noise_path1, moderate_noise_path1, heavy_noise_path1, 
                          prompt_path, model_name, config_path)

    # =============== 测试案例2：MBPP问题2 ===============
    # 原始工作流文件路径
    original_path2 = "aflow_scripts/MBPP/mbpp_original_2/round_14/graph.py"
    # 被注释的变体路径（暂时不可用）
    # requirements_path2 = "aflow_scripts/MBPP/mbpp_requirements_2/round_20/graph.py"
    # paraphrasing_path2 = "aflow_scripts/MBPP/mbpp_paraphrasing_2/round_7/graph.py"
    
    # 可用的变体工作流文件路径
    light_noise_path2 = "aflow_scripts/MBPP/mbpp_light_noise_2/round_2/graph.py"
    moderate_noise_path2 = "aflow_scripts/MBPP/mbpp_moderate_noise_2/round_14/graph.py"
    # heavy_noise_path2 = "aflow_scripts/MBPP/mbpp_heavy_noise_2/round_15/graph.py"
    
    # 临时占位符路径
    requirements_path2 = "aflow_scripts/MBPP/mbpp_light_noise_1/round_6/graph.py"
    paraphrasing_path2 = "aflow_scripts/MBPP/mbpp_light_noise_1/round_6/graph.py"
    heavy_noise_path2 = "aflow_scripts/MBPP/mbpp_light_noise_1/round_6/graph.py"
    
    # 执行第二个测试案例的变体组评估
    evaluate_variant_group(original_path2, requirements_path2, paraphrasing_path2, 
                          light_noise_path2, moderate_noise_path2, heavy_noise_path2, 
                          prompt_path, model_name, config_path)

    # =============== 测试案例3：MBPP问题3 ===============
    # 原始工作流文件路径
    original_path3 = "aflow_scripts/MBPP/mbpp_original_3/round_10/graph.py"
    # 被注释的变体路径（暂时不可用）
    # requirements_path3 = "aflow_scripts/MBPP/mbpp_requirements_3/round_10/graph.py"
    # paraphrasing_path3 = "aflow_scripts/MBPP/mbpp_paraphrasing_3/round_18/graph.py"
    
    # 可用的变体工作流文件路径
    light_noise_path3 = "aflow_scripts/MBPP/mbpp_light_noise_3/round_12/graph.py"
    moderate_noise_path3 = "aflow_scripts/MBPP/mbpp_moderate_noise_3/round_17/graph.py"
    # heavy_noise_path3 = "aflow_scripts/MBPP/mbpp_heavy_noise_3/round_5/graph.py"
    
    # 临时占位符路径
    requirements_path3 = "aflow_scripts/MBPP/mbpp_light_noise_1/round_6/graph.py"
    paraphrasing_path3 = "aflow_scripts/MBPP/mbpp_light_noise_1/round_6/graph.py"
    heavy_noise_path3 = "aflow_scripts/MBPP/mbpp_light_noise_1/round_6/graph.py"
    
    # 执行第三个测试案例的变体组评估
    evaluate_variant_group(original_path3, requirements_path3, paraphrasing_path3, 
                          light_noise_path3, moderate_noise_path3, heavy_noise_path3, 
                          prompt_path, model_name, config_path)

    # =============== 测试案例4：MBPP问题4 ===============
    # 原始工作流文件路径
    original_path4 = "aflow_scripts/MBPP/mbpp_original_4/round_1/graph.py"
    # 被注释的变体路径（暂时不可用）
    # requirements_path4 = "aflow_scripts/MBPP/mbpp_requirements_4/round_2/graph.py"
    # paraphrasing_path4 = "aflow_scripts/MBPP/mbpp_paraphrasing_4/round_4/graph.py"
    
    # 可用的变体工作流文件路径
    light_noise_path4 = "aflow_scripts/MBPP/mbpp_light_noise_4/round_8/graph.py"
    moderate_noise_path4 = "aflow_scripts/MBPP/mbpp_moderate_noise_4/round_8/graph.py"
    # heavy_noise_path4 = "aflow_scripts/MBPP/mbpp_heavy_noise_4/round_3/graph.py"
    
    # 临时占位符路径
    requirements_path4 = "aflow_scripts/MBPP/mbpp_light_noise_1/round_6/graph.py"
    paraphrasing_path4 = "aflow_scripts/MBPP/mbpp_light_noise_1/round_6/graph.py"
    heavy_noise_path4 = "aflow_scripts/MBPP/mbpp_light_noise_1/round_6/graph.py"
    
    # 执行第四个测试案例的变体组评估
    evaluate_variant_group(original_path4, requirements_path4, paraphrasing_path4, 
                          light_noise_path4, moderate_noise_path4, heavy_noise_path4, 
                          prompt_path, model_name, config_path)

    # =============== 测试案例5（已注释，暂时不执行） ===============
    # 第五个测试案例的路径定义（被注释掉，表示暂时不执行此测试案例）
    # original_path5 = "aflow_scripts/MBPP/mbpp_original_5/round_5/graph.py"
    # requirements_path5 = "aflow_scripts/MBPP/mbpp_requirements_5/round_6/graph.py"
    # paraphrasing_path5 = "aflow_scripts/MBPP/mbpp_paraphrasing_5/round_8/graph.py"
    # light_noise_path5 = "aflow_scripts/MBPP/mbpp_light_noise_5/round_7/graph.py"
    # moderate_noise_path5 = "aflow_scripts/MBPP/mbpp_moderate_noise_5/round_6/graph.py"
    # heavy_noise_path5 = "aflow_scripts/MBPP/mbpp_heavy_noise_5/round_4/graph.py"
    # evaluate_variant_group(original_path5, requirements_path5, paraphrasing_path5, light_noise_path5, moderate_noise_path5, heavy_noise_path5, prompt_path)

    # =============== 结果保存和统计 ===============
    # 将评估结果保存到文本文件中，用于后续分析和报告
    with open("aflow_score.txt", "a") as f:
        # 写入数据集标识
        f.write("MBPP1\n")
        
        # 写入节点级评估结果（只保存可用的变体）
        # 注释掉的行表示对应的变体暂时不可用或跳过
        # f.write("original_nodes_score: " + str(sum(ori_node_scores) / len(ori_node_scores)) + "\n")
        # f.write("requirements_nodes_score: " + str(sum(req_node_scores) / len(req_node_scores)) + "\n")
        # f.write("paraphrasing_nodes_score: " + str(sum(para_node_scores) / len(para_node_scores)) + "\n")
        
        # 计算并保存轻度噪声变体的平均节点级F1分数
        f.write("light_noise_nodes_score: " + str(sum(light_node_scores) / len(light_node_scores)) + "\n")
        # 计算并保存中度噪声变体的平均节点级F1分数
        f.write("moderate_noise_nodes_score: " + str(sum(moderate_node_scores) / len(moderate_node_scores)) + "\n")
        # f.write("heavy_noise_nodes_score: " + str(sum(heavy_node_scores) / len(heavy_node_scores)) + "\n")
        
        # 写入图级评估结果（只保存可用的变体）
        # f.write("original_graph_score: " + str(sum(ori_graph_scores) / len(ori_graph_scores)) + "\n")
        # f.write("requirements_graph_score: " + str(sum(req_graph_scores) / len(req_graph_scores)) + "\n")
        # f.write("paraphrasing_graph_score: " + str(sum(para_graph_scores) / len(para_graph_scores)) + "\n")
        
        # 计算并保存轻度噪声变体的平均图级F1分数
        f.write("light_noise_graph_score: " + str(sum(light_graph_scores) / len(light_graph_scores)) + "\n")
        # 计算并保存中度噪声变体的平均图级F1分数
        f.write("moderate_noise_graph_score: " + str(sum(moderate_graph_scores) / len(moderate_graph_scores)) + "\n")
        # f.write("heavy_noise_graph_score: " + str(sum(heavy_graph_scores) / len(heavy_graph_scores)) + "\n")