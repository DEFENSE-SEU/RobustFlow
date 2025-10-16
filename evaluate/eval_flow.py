from __future__ import annotations
import argparse, json
from collections import defaultdict, deque
from graph_evaluator import t_eval_nodes, t_eval_graph, SentenceTransformer
from pathlib import Path
from typing import Dict, Any, List, Tuple

def lowercase_first_alpha(s: str) -> str:
    """只把字符串中的第一个字母小写，其余不变。"""
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.lower() + s[i+1:]
    return s

def load_workflow_json(path: str) -> Dict[str, Any]:
    """读取 .json；兼容顶层为 {'tasks': {...}} 或直接为任务映射 dict。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "tasks" in data and isinstance(data["tasks"], dict):
        return {"tasks": data["tasks"]}
    if isinstance(data, dict):
        return {"tasks": data}
    raise ValueError("JSON 格式不符合预期：需要 {'tasks': {...}} 或直接为任务映射的 dict。")

def topo_order(tasks: Dict[str, Dict[str, Any]]) -> List[str]:
    """基于 next 做拓扑排序（零入度按 task id 排序，保证确定性）。"""
    succ = defaultdict(list)
    indeg = defaultdict(int)

    for tid, t in tasks.items():
        for nxt in t.get("next", []) or []:
            if nxt not in tasks:
                raise ValueError(f"{tid} -> {nxt} 不存在于 tasks")
            succ[tid].append(nxt)
            indeg[nxt] += 1
        indeg.setdefault(tid, 0)

    q = deque(sorted([tid for tid, d in indeg.items() if d == 0]))
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(order) != len(tasks):
        raise ValueError("任务图存在环或缺失，无法拓扑排序。")
    return order

def build_nodes(tasks: Dict[str, Dict[str, Any]], order: List[str]) -> List[str]:
    """按拓扑顺序构造节点文本：'<agent> agent to <objective(首字母小写)>'，并包上 START/END。"""
    nodes = ["START"]
    for tid in order:
        t = tasks[tid]
        agent = t.get("agent", t.get("agent_id", "Agent"))
        obj = t.get("objective", tid) or ""
        nodes.append(f"{agent} agent to {lowercase_first_alpha(obj)}")
    nodes.append("END")
    return nodes

def make_chain_edges(n_nodes: int):
    """按 nodes 的顺序构造链式边：(0,1),(1,2),...,(N-2,N-1)。"""
    return [(i, i + 1) for i in range(n_nodes - 1)]

def print_python_literal(nodes: List[str], edges: List[tuple[int, int]]) -> None:
    """
    以“Python字面量”打印（nodes 用 JSON 转义确保引号安全；edges 用元组）。
    不是 JSON 文件，是可读的 Python 风格文本。
    """
    nodes_lit = "[" + ", ".join(json.dumps(s, ensure_ascii=False) for s in nodes) + "]"
    edges_lit = "[" + ", ".join(f"({u}, {v})" for (u, v) in edges) + "]"
    out = "{\n  \"nodes\": " + nodes_lit + ",\n  \"edges\": " + edges_lit + "\n}"
    print(out)


def get_scoreflow(path: str) -> Dict[str, Any]:
    """
    读取 workflow.json → 生成链式工作流表示并返回：
    {
      "nodes": ["START", "...", "END"],
      "edges": [(0,1), (1,2), ..., (N-2,N-1)]
    }
    依赖以下已定义函数：load_workflow_json, topo_order, build_nodes, make_chain_edges
    """
    wf = load_workflow_json(path)        # -> {"tasks": {...}}
    tasks = wf["tasks"]
    order: List[str] = topo_order(tasks) # 拓扑序（确定性）
    nodes: List[str] = build_nodes(tasks, order)  # 包含 START/END
    edges: List[Tuple[int, int]] = make_chain_edges(len(nodes))
    return {"nodes": nodes, "edges": edges}

def evaluate_variant_group(original_path, requirements_path, paraphrasing_path, noise_path):
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # 获取 ScoreFlow 工作流
    workflow_original = get_scoreflow(original_path)
    workflow_requirements = get_scoreflow(requirements_path)
    workflow_paraphrasing = get_scoreflow(paraphrasing_path)
    workflow_noise = get_scoreflow(noise_path)

    print("===========================================================")
    print(workflow_original)
    print("-----------------------------------------------------------")
    print(workflow_requirements)
    print("-----------------------------------------------------------")
    print(workflow_paraphrasing)
    print("-----------------------------------------------------------")
    print(workflow_noise)

    # 计算节点 F1 分数
    original_nodes_f1 = t_eval_nodes(workflow_original, workflow_original, model)["f1_score"]
    requirements_nodes_f1 = t_eval_nodes(workflow_requirements, workflow_original, model)["f1_score"]
    paraphrasing_nodes_f1 = t_eval_nodes(workflow_paraphrasing, workflow_original, model)["f1_score"]
    noise_nodes_f1 = t_eval_nodes(workflow_noise, workflow_original, model)["f1_score"]

    # 计算图结构 F1 分数
    original_graph_f1 = t_eval_graph(workflow_original, workflow_original, model)["f1_score"]
    requirements_graph_f1 = t_eval_graph(workflow_requirements, workflow_original, model)["f1_score"]
    paraphrasing_graph_f1 = t_eval_graph(workflow_paraphrasing, workflow_original, model)["f1_score"]
    noise_graph_f1 = t_eval_graph(workflow_noise, workflow_original, model)["f1_score"]

    scores = [
        original_nodes_f1,
        requirements_nodes_f1,
        paraphrasing_nodes_f1,
        noise_nodes_f1,
        original_graph_f1,
        requirements_graph_f1,
        paraphrasing_graph_f1,
        noise_graph_f1,
    ]

    ori_node_scores.append(scores[0])
    req_node_scores.append(scores[1])
    para_node_scores.append(scores[2])
    noise_node_scores.append(scores[3])

    ori_graph_scores.append(scores[4])
    req_graph_scores.append(scores[5])
    para_graph_scores.append(scores[6])
    noise_graph_scores.append(scores[7])

ori_node_scores = []
req_node_scores = []
para_node_scores = []
noise_node_scores = []

ori_graph_scores = []
req_graph_scores = []
para_graph_scores = []
noise_graph_scores = []

if __name__ == "__main__":
    original_path1 = "flow_scripts/mbpp_workflow1_original.json"
    requirements_path1 = "flow_scripts/mbpp_workflow1_light_noise.json"
    paraphrasing_path1 = "flow_scripts/mbpp_workflow1_moderate_noise.json"
    noise_path1 = "flow_scripts/mbpp_workflow1_heavy_noise.json"
    evaluate_variant_group(original_path1, requirements_path1, paraphrasing_path1, noise_path1)

    original_path2 = "flow_scripts/mbpp_workflow2_original.json"
    requirements_path2 = "flow_scripts/mbpp_workflow2_light_noise.json"
    paraphrasing_path2 = "flow_scripts/mbpp_workflow2_moderate_noise.json"
    noise_path2 = "flow_scripts/mbpp_workflow2_heavy_noise.json"
    evaluate_variant_group(original_path2, requirements_path2, paraphrasing_path2, noise_path2)

    original_path3 = "flow_scripts/mbpp_workflow3_original.json"
    requirements_path3 = "flow_scripts/mbpp_workflow3_light_noise.json"
    paraphrasing_path3 = "flow_scripts/mbpp_workflow3_moderate_noise.json"
    noise_path3 = "flow_scripts/mbpp_workflow3_heavy_noise.json"
    evaluate_variant_group(original_path3, requirements_path3, paraphrasing_path3, noise_path3)

    original_path4 = "flow_scripts/mbpp_workflow4_original.json"
    requirements_path4 = "flow_scripts/mbpp_workflow4_light_noise.json"
    paraphrasing_path4 = "flow_scripts/mbpp_workflow4_moderate_noise.json"
    noise_path4 = "flow_scripts/mbpp_workflow4_heavy_noise.json"
    evaluate_variant_group(original_path4, requirements_path4, paraphrasing_path4, noise_path4)

    original_path5 = "flow_scripts/mbpp_workflow5_original.json"
    requirements_path5 = "flow_scripts/mbpp_workflow5_light_noise.json"
    paraphrasing_path5 = "flow_scripts/mbpp_workflow5_moderate_noise.json"
    noise_path5 = "flow_scripts/mbpp_workflow5_heavy_noise.json"
    evaluate_variant_group(original_path5, requirements_path5, paraphrasing_path5, noise_path5)

    with open("flow_score.txt", "a") as f:
        f.write("TEMP\n")
        f.write("original_nodes_score: " + str(sum(ori_node_scores) / len(ori_node_scores)) + "\n")
        f.write("requirements_nodes_score: " + str(sum(req_node_scores) / len(req_node_scores)) + "\n")
        f.write("paraphrasing_nodes_score: " + str(sum(para_node_scores) / len(para_node_scores)) + "\n")
        f.write("noise_nodes_score: " + str(sum(noise_node_scores) / len(noise_node_scores)) + "\n")
        f.write("original_graph_score: " + str(sum(ori_graph_scores) / len(ori_graph_scores)) + "\n")
        f.write("requirements_graph_score: " + str(sum(req_graph_scores) / len(req_graph_scores)) + "\n")
        f.write("paraphrasing_graph_score: " + str(sum(para_graph_scores) / len(para_graph_scores)) + "\n")
        f.write("noise_graph_score: " + str(sum(noise_graph_scores) / len(noise_graph_scores)) + "\n")
    