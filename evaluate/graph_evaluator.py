"""
Graph Evaluator Module

This module implements various evaluation methods for directed graph structures, 
mainly used for evaluating workflow graphs, action sequence graphs, etc.
Supports node-level evaluation, graph-level evaluation, and plan sequence evaluation, 
using semantic similarity for node matching.

Main Features:
1. Topological Sort Calculation - Calculate all possible topological sorts of directed graphs
2. Connectivity Analysis - Calculate the largest connected component of graphs
3. Node Semantic Matching - Node semantic similarity matching based on BERT models
4. Graph-level Evaluation - Graph structure evaluation based on reachability closure
5. Node-level Evaluation - Node sequence evaluation based on topological sorting
6. Plan Evaluation - Precision and recall evaluation of action sequences
"""

import networkx as nx  # For graph algorithms and data structure operations
import numpy as np  # For numerical computation and array operations
from sentence_transformers import SentenceTransformer, util  # For semantic similarity computation
from typing import List, Dict  # Type hints

def all_topological_sorts(graph: Dict[str, List[str]]) -> List[List[str]]:
    """
    Calculate all possible topological sorts for a given directed graph (excluding START and END nodes)
    
    A topological sort is a linear ordering of nodes in a Directed Acyclic Graph (DAG) such that 
    for every directed edge (u,v), node u appears before node v in the ordering. This function 
    generates all possible valid sorting sequences.
    
    Parameters:
        graph (Dict[str, List[str]]): Dictionary containing nodes and edges, format: {
            "nodes": ["START", "A", "B", "C", "END"],
            "edges": [(0, 1), (1, 2), (2, 3), (3, 4)]  # Edges use node indices
        }
    
    Returns:
        List[List[str]]: List of all possible topological sort sequences, each sequence is 
                         a list of nodes excluding START and END
        
    Algorithm Description:
        1. If node count >= 12, return original node list directly to avoid excessive computation
        2. Use NetworkX to build directed graph and calculate all topological sorts
        3. Filter out START and END special nodes
        4. Return at most first 20 results to avoid excessive combinations
        
    Time Complexity: O(n! * m), where n is the number of nodes, m is the number of edges
    Space Complexity: O(n! * n)
    
    Example:
        >>> graph = {"nodes": ["START", "A", "B", "C", "END"], "edges": [(0,1), (1,2), (2,3), (3,4)]}
        >>> sorts = all_topological_sorts(graph)
        >>> print(sorts)  # [["A", "B", "C"]]
    """
    # Performance optimization: If the number of nodes in the graph (excluding START and END) 
    # is greater than or equal to 12, return the original node list directly to avoid 
    # excessive computation overhead from too many topological sort combinations
    if len(graph["nodes"]) >= 12:
        # Filter out START and END nodes and return
        filtered = [node for node in graph["nodes"] if node not in ["START", "END"]]
        return [filtered]

    # Create an empty directed graph object for storing graph topology
    G = nx.DiGraph()

    # Save the original node name list, where index represents each node's number
    # Example: ["START", "A", "B", "C", "END"] -> indices 0,1,2,3,4
    original_nodes = graph["nodes"]

    # Add all nodes to the graph by index number, such as 0, 1, 2, ..., n-1
    # This makes it convenient to represent edge relationships using indices
    G.add_nodes_from(range(len(original_nodes)))

    # Convert edges to index form and add to the graph
    # Example: (0,1) represents an edge from START to A
    edges_with_indices = [(u, v) for u, v in graph["edges"]]
    G.add_edges_from(edges_with_indices)

    # Use NetworkX to generate all valid topological sort results (returns index number lists)
    # Example: [[0,1,2,3,4], [0,2,1,3,4]] etc.
    all_sorts = list(nx.all_topological_sorts(G))

    # Iterate through all topological sorts, convert indices back to original node names, and filter START and END
    filtered_sorts = []
    for sort in all_sorts:
        # Restore node names based on indices while removing special marker nodes
        # Example: [0,1,2,3,4] -> ["START","A","B","C","END"] -> ["A","B","C"]
        filtered_sort = [original_nodes[i] for i in sort if original_nodes[i] not in ["START", "END"]]
        filtered_sorts.append(filtered_sort)

    # Return at most the first 20 topological sort results to avoid performance impact from too many combinations
    return filtered_sorts[:20]


def all_topological_sorts_test():
    """
    Test cases for the all_topological_sorts function
    Contains three graph structure tests of different complexities to verify the correctness of topological sorting functionality
    
    Test Case Description:
    1. graph1: Simple linear graph structure
    2. graph2: Medium complexity branching graph structure
    3. graph3: Complex multi-branch converging graph structure
    """
    # Test case 1: Simple linear graph structure
    # Structure: START -> A,B -> C -> END
    graph1 = {
        "nodes": ["START", "A", "B", "C", "END"],
        "edges": [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]  # START->A,B; A,B->C; C->END
    }
    
    # Test case 2: Medium complexity branching graph structure
    # Contains multiple branches and convergence points
    graph2 = {
        "nodes": ["START", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "END"],
        "edges": [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6), (6, 7), (7, 8), (8, 9), (8, 10),
                  (10, 11), (11, 12)]
    }
    
    # Test case 3: Complex multi-branch converging graph structure
    # Contains multiple branches converging to the same node
    graph3 = {
        "nodes": ["START", "A", "B", "C", "D", "E", "F", "G", "H", "I", "END"],
        "edges": [(0, 1), (1, 2), (1, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 8), (7, 8), (8, 9), (9, 10)]
    }
    
    # Execute tests and print results
    print("Test case 1 result:", all_topological_sorts(graph1))
    print("Test case 2 result:", all_topological_sorts(graph2))
    print("Test case 3 result:", all_topological_sorts(graph3))
    

def largest_connected_component(nodes, edges):
    """
    Calculate the number of nodes in the largest connected component of an undirected graph 
    constructed from specified nodes and edges
    
    A connected component is a set of nodes where there exists a path between any two nodes.
    This function is used to analyze graph connectivity and find the size of the largest connected subgraph.
    
    Parameters:
        nodes (List): List of nodes, can be string or numeric identifiers
        edges (List[Tuple]): List of edges, each edge is a tuple of (node1, node2)
    
    Returns:
        int: Number of nodes in the largest connected component
        
    Algorithm Description:
        1. Build undirected graph: Use NetworkX to create undirected graph object
        2. Calculate all connected components: Find all connected node sets in the graph
        3. Return the node count of the largest connected component: Select the component with the most nodes
        
    Time Complexity: O(n + m), where n is the number of nodes, m is the number of edges
    Space Complexity: O(n + m)
    
    Example:
        >>> nodes = [0, 1, 2, 3, 4, 5]
        >>> edges = [(0, 1), (1, 2), (3, 4)]
        >>> result = largest_connected_component(nodes, edges)
        >>> print(result)  # 3 (connected component {0,1,2} is the largest)
    """
    # Create an undirected graph object G for storing graph topology
    G = nx.Graph()

    # Add all nodes and edges to the graph to build the complete graph structure
    G.add_nodes_from(nodes)  # Add all nodes to the graph
    G.add_edges_from(edges)  # Add all edges to the graph

    # Calculate all connected subgraphs in the graph (each subgraph is a set of nodes)
    # Returns a generator where each element is a node set of a connected component
    connected_components = nx.connected_components(G)

    # Find the largest connected subgraph (the subgraph with the most nodes)
    # Use max function with key=len parameter to find the connected component with the largest length
    largest_component = max(connected_components, key=len)

    # Return the number of nodes in the largest connected subgraph
    return len(largest_component)


def largest_connected_component_test():
    """
    Test cases for the largest_connected_component function
    Verify the correctness of connected component calculation functionality
    """
    # Test case 1: Simple disconnected graph
    # Contains two connected components: {0,1,2} and {3,4}, largest connected component size is 3
    nodes = [0, 1, 2, 3, 4, 5]
    edges = [(0, 1), (1, 2), (3, 4)]  # 0-1-2 connected, 3-4 connected, 5 isolated
    print("Test case 1 - Largest connected component size:", largest_connected_component(nodes, edges))
    
    # Test case 2: Complex workflow graph
    # All nodes are connected, largest connected component size is 11
    nodes = ["START", "A", "B", "C", "D", "E", "F", "G", "H", "I", "END"]
    edges = [(0, 1), (1, 2), (1, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 8), (7, 8), (8, 9), (9, 10)]
    print("Test case 2 - Largest connected component size:", largest_connected_component(nodes, edges))


def match_node(pred_nodes: List[str], gt_nodes: List[str], sentence_model: object, match_threshold=0.6) -> dict:
    """
    Perform optimal one-to-one matching between predicted nodes pred_nodes and ground truth nodes gt_nodes based on semantic similarity
    
    This function implements a two-stage matching strategy:
    1. Exact text matching: Prioritize matching completely identical text
    2. Semantic similarity matching: Use BERT model to calculate semantic similarity, find optimal pairing through maximum weight matching algorithm
    
    Parameters:
        pred_nodes (List[str]): List of predicted nodes that need to be matched
        gt_nodes (List[str]): List of ground truth nodes as matching targets
        sentence_model (object): Sentence encoding model (e.g., SentenceTransformer) for calculating semantic similarity
        match_threshold (float): Semantic matching threshold, default 0.6, nodes with similarity below this value will not be matched
    
    Returns:
        dict: Dictionary mapping predicted node indices to ground truth node indices
              Format: {pred_index: gt_index, ...}, unmatched predicted nodes are mapped to -1
    
    Algorithm Description:
        1. Greedy matching: First perform greedy pairing with completely identical text while maintaining order
        2. Semantic matching: Use BERT cosine similarity + maximum weight matching for remaining nodes
        3. Position bias: Prioritize matching nodes with close indices (resolve remaining semantic ambiguity)
        
    Time Complexity: O(n² * d), where n is the number of nodes, d is BERT embedding dimension
    Space Complexity: O(n²)
    
    Example:
        >>> pred = ["go to kitchen", "pick up sponge"]
        >>> gt = ["go to the kitchen", "grab the sponge"]
        >>> model = SentenceTransformer("all-mpnet-base-v2")
        >>> mapping = match_node(pred, gt, model)
        >>> print(mapping)  # {0: 0, 1: 1}
    """
    # Get the number of predicted nodes and ground truth nodes for subsequent index operations
    len_pred, len_gt = len(pred_nodes), len(gt_nodes)

    # ---------- Phase 1: Greedy matching of completely identical text ----------
    pred_to_gt_mapping: Dict[int, int] = {}  # Store mapping from predicted node indices to ground truth node indices
    used_gt = set()  # Record already matched ground truth node indices to avoid duplicate matching
    
    # Iterate through all predicted nodes, looking for completely identical text matches
    for p_idx, p_txt in enumerate(pred_nodes):
        for g_idx, g_txt in enumerate(gt_nodes):
            if g_idx in used_gt:  # If this ground truth node is already used, skip
                continue
            if p_txt == g_txt:          # Text is completely identical, perform matching
                pred_to_gt_mapping[p_idx] = g_idx  # Establish mapping relationship
                used_gt.add(g_idx)  # Mark this ground truth node as used
                break  # Break out of inner loop after finding a match, process next predicted node

    # If all nodes are matched, return result directly
    if len(pred_to_gt_mapping) == len_pred and len(pred_to_gt_mapping) == len_gt:
        return pred_to_gt_mapping

    # ---------- Phase 2: Semantic matching for remaining nodes ----------
    # Get indices of unmatched predicted nodes and ground truth nodes
    remain_pred = [i for i in range(len_pred) if i not in pred_to_gt_mapping]
    remain_gt   = [j for j in range(len_gt)   if j not in used_gt]

    # If there are still unmatched nodes, perform semantic similarity matching
    if remain_pred and remain_gt:
        # Use BERT model to calculate semantic embedding vectors for remaining nodes
        node_pred_emb = sentence_model.encode([pred_nodes[i] for i in remain_pred], convert_to_tensor=True)
        node_gt_emb   = sentence_model.encode([gt_nodes[j]  for j in remain_gt],  convert_to_tensor=True)
        
        # Calculate cosine similarity matrix and ensure similarity is non-negative
        sim_matrix = np.maximum(util.cos_sim(node_pred_emb, node_gt_emb).cpu().numpy(), 0)

        # Build bipartite graph for maximum weight matching
        # Use predicted indices on left, (len_pred + gt_idx) on right to avoid index conflicts
        G = nx.Graph()
        for a, p_idx in enumerate(remain_pred):
            for b, g_idx in enumerate(remain_gt):
                score = sim_matrix[a, b]  # Get semantic similarity score
                if score >= match_threshold:  # Only consider matches above the threshold
                    # Add position bias, prioritize matching nodes with close indices (resolve semantic ambiguity)
                    score_adj = float(score) - 1e-4 * abs(p_idx - g_idx)
                    G.add_edge(p_idx, len_pred + g_idx, weight=score_adj)

        # Use NetworkX's maximum weight matching algorithm to find optimal matching
        mwm = nx.max_weight_matching(G)

        # Parse matching results, convert bipartite graph edges to predicted-to-ground-truth mapping
        for u, v in mwm:
            if u < len_pred:          # u is predicted node index
                pred_to_gt_mapping[u] = v - len_pred  # v - len_pred gives ground truth node index
            else:                     # v is predicted node index
                pred_to_gt_mapping[v] = u - len_pred  # u - len_pred gives ground truth node index

    # ---------- Phase 3: Handle unmatched predicted nodes ----------
    # Map unmatched predicted nodes to -1, indicating no corresponding ground truth node
    for i in range(len_pred):
        if i not in pred_to_gt_mapping:
            pred_to_gt_mapping[i] = -1

    return pred_to_gt_mapping


def match_node_test():
    """
    Test cases for the match_node function
    Verify node matching functionality based on semantic similarity
    """
    # Test case 1: Node matching with semantic similarity but not completely identical text
    # Test whether BERT model can correctly match semantically similar nodes
    pred_nodes = ["go to kitchen", "pick up sponge", "wash sponge", "listen to the radio"]
    gt_nodes = ["go to kitchen", "grab the sponge", "clean the sponge", "listen to music"]
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    mapping = match_node(pred_nodes, gt_nodes, model)
    print("Test case 1 - Semantic matching result:", mapping)
    
    # Test case 2: Completely identical node matching
    # Test exact text matching functionality
    pred_nodes = ["go to kitchen", "go to kitchen", "go to kitchen", "pick up sponge", "pick up sponge", "pick up sponge", "wash sponge", "listen to the radio"]
    gt_nodes = ["go to kitchen", "go to kitchen", "go to kitchen", "pick up sponge", "pick up sponge", "pick up sponge", "wash sponge", "listen to the radio"]
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    mapping = match_node(pred_nodes, gt_nodes, model)
    print("Test case 2 - Exact matching result:", mapping)



def t_eval_graph(pred_graph: Dict[str, List[str]], gt_graph: Dict[str, List[str]], sentence_model: object) -> Dict[
    str, float]:
    """
    Graph-level evaluation function: Edge-level relaxed evaluation based on reachability closure
    
    This function implements graph reachability-based evaluation by comparing all reachable node pairs 
    in predicted and ground truth graphs to calculate precision, recall, and F1 score. This method 
    is more robust to local graph changes (such as branch count, intermediate steps) and focuses 
    on overall graph structure similarity.
    
    Parameters:
        pred_graph (Dict[str, List[str]]): Predicted graph, dictionary containing nodes and edges
            Format: {"nodes": [...], "edges": [...]}
        gt_graph (Dict[str, List[str]]): Ground truth graph, dictionary containing nodes and edges
            Format: {"nodes": [...], "edges": [...]}
        sentence_model (object): Sentence encoding model for node semantic matching
    
    Returns:
        Dict[str, float]: Dictionary containing evaluation results with precision, recall, f1_score
            - precision: Precision, proportion of correct predicted reachable pairs
            - recall: Recall, proportion of correctly predicted ground truth reachable pairs
            - f1_score: F1 score, harmonic mean of precision and recall
    
    Algorithm Description:
        1. Node matching: Use semantic similarity to match predicted and ground truth nodes
        2. Build mapped graph: Map predicted graph edges to ground truth graph node space
        3. Generate reachable pair sets: Calculate all ordered pairs "u can reach v" in both graphs
        4. Compare reachability: Calculate intersection of predicted and ground truth reachable pairs, compute precision, recall, and F1 score
        
    Features:
        - Edge-level relaxed evaluation: Compare all ordered pairs "u can reach v" in both graphs
        - More robust to branch count and intermediate step insertion
        - Use reachability closure instead of direct edge comparison, focus more on overall graph structure
        
    Time Complexity: O(n³ + m), where n is the number of nodes, m is the number of edges
    Space Complexity: O(n²)
    
    Example:
        >>> pred = {"nodes": ["START", "A", "B", "END"], "edges": [(0,1), (1,2), (2,3)]}
        >>> gt = {"nodes": ["START", "A", "B", "END"], "edges": [(0,1), (1,2), (2,3)]}
        >>> model = SentenceTransformer("all-mpnet-base-v2")
        >>> result = t_eval_graph(pred, gt, model)
        >>> print(result)  # {"precision": 1.0, "recall": 1.0, "f1_score": 1.0}
    """
    # Extract node and edge information from predicted and ground truth graphs
    pred_nodes, gt_nodes = pred_graph["nodes"], gt_graph["nodes"]
    pred_edges, gt_edges = pred_graph["edges"], gt_graph["edges"]

    # Boundary case handling: If either graph is empty, return zero score
    if not pred_nodes or not gt_nodes:
        return {"precision": 0, "recall": 0, "f1_score": 0}

    # ---------- Phase 1: Node matching ----------
    # Use semantic similarity to map predicted nodes to ground truth nodes
    mapping = match_node(pred_nodes, gt_nodes, sentence_model)

    # ---------- Phase 2: Build mapped graph ----------
    # Map predicted graph edges to ground truth graph node space, establish unified comparison baseline
    mapped_edges = []
    for u, v in pred_edges:
        if u in mapping and v in mapping:  # Ensure both ends of the edge have mappings
            gu, gv = mapping[u], mapping[v]  # Get mapped ground truth node indices
            if gu != -1 and gv != -1:  # Ensure mapping is valid (not unmatched -1)
                mapped_edges.append((gu, gv))  # Add mapped edges

    # ---------- Phase 3: Generate reachable pair sets ----------
    def reach_pairs(n_nodes, edges):
        """
        Calculate all reachable ordered pair sets in the graph (reachability closure)
        
        Reachability closure refers to the set of all ordered pairs (u,v) where "node u can reach node v" in the graph.
        This includes direct edges and all node pairs reachable indirectly through multi-hop paths.
        
        Parameters:
            n_nodes (int): Number of nodes
            edges (List[Tuple]): List of edges, each edge is a tuple (u,v)
            
        Returns:
            set: All reachable ordered pair sets, such as {(0,1), (0,2), (1,2)}
        """
        # Create directed graph object
        G = nx.DiGraph()
        G.add_nodes_from(range(n_nodes))  # Add all nodes
        G.add_edges_from(edges)  # Add all edges
        
        # Calculate reachability closure
        reach = set()
        for s in G.nodes:  # Iterate through each starting node
            # Get all descendant nodes reachable from node s
            for t in nx.descendants(G, s):
                reach.add((s, t))  # Add reachable pair (s,t)
        return reach

    n_gt = len(gt_nodes)  # Number of nodes in ground truth graph
    R_pred = reach_pairs(n_gt, mapped_edges)  # Set of reachable pairs in predicted graph
    R_gt   = reach_pairs(n_gt, gt_edges)      # Set of reachable pairs in ground truth graph

    # Special case handling: If both graphs have no reachable pairs, consider them completely matched
    if not R_pred and not R_gt:
        return {"precision": 1, "recall": 1, "f1_score": 1}

    # ---------- Phase 4: Calculate evaluation metrics ----------
    # Calculate intersection of predicted and ground truth reachable pairs
    inter = R_pred & R_gt
    
    # Calculate precision: correctly predicted reachable pairs / total predicted reachable pairs
    precision = len(inter) / len(R_pred) if R_pred else 0
    
    # Calculate recall: correctly predicted reachable pairs / total ground truth reachable pairs
    recall    = len(inter) / len(R_gt)   if R_gt   else 0
    
    # Calculate F1 score: harmonic mean of precision and recall
    f1 = 0 if precision == 0 or recall == 0 else 2*precision*recall/(precision+recall)

    return {"precision": precision, "recall": recall, "f1_score": f1}


def t_eval_graph_test():
    """
    Test cases for the t_eval_graph function
    Contains multiple graph structure tests of different complexities to verify the correctness of graph-level evaluation functionality
    """
    # Test case 1: Workflow graph with semantic similarity but different text
    # Test semantic matching and reachability evaluation functionality
    pred_graph1 = {
        "nodes": ["START", "go to kitchen", "take the dish sponge", "walk to sink",
                  "clean the sponge at sink", "return to shelf", "place sponge on shelf", "END"],
        "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]  # Linear structure
    }

    gt_graph1 = {
        "nodes": ["START", "go to the kitchen", "grab sponge from table", "move to sink area",
                  "rinse sponge with water", "go back to cabinet", "put sponge into drawer", "END"],
        "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]  # Same linear structure
    }

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    result1 = t_eval_graph(pred_graph1, gt_graph1, model)
    print("Test case 1 - Semantic similarity workflow graph evaluation result:", result1)

    # Test case 2: Complex branching converging graph structure
    # Test graph structure evaluation with multiple branches and convergence points
    gt_graph2 = {
        "nodes": ["START", "prepare workstation", "sanitize hands", "wear gloves", "collect equipment",
                  "draw blood sample", "label sample", "dispose needle", "remove gloves", "END"],
        "edges": [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]
    }

    pred_graph2 = {
        "nodes": ["START", "setup workstation", "clean hands", "put on gloves", "collect equipment",
                  "extract blood sample", "label blood tube", "dispose of needle", "take off gloves", "END"],
        "edges": [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]
    }

    result2 = t_eval_graph(pred_graph2, gt_graph2, model)
    print("Test case 2 - Complex branching graph evaluation result:", result2)

    # Test case 3: Completely identical graph structure
    # Test evaluation results under ideal conditions (should get full score)
    pred_graph3 = {
        "nodes": ["START", "A", "B", "C", "D", "E", "F", "G", "H", "I", "END"],
        "edges": [(0, 1), (1, 2), (1, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 8), (7, 8), (8, 9), (9, 10)]
    }
    result3 = t_eval_graph(pred_graph3, pred_graph3, model)
    print("Test case 3 - Perfect match graph evaluation result:", result3)


def t_eval_plan(pred_plan:List[str],gt_plan:List[str], eval_model:object,order:bool = True) -> dict:
    """
    Compare the similarity between predicted action sequence pred_plan and ground truth action sequence gt_plan
    
    This function implements action sequence evaluation, supporting two evaluation modes:
    1. Order-sensitive mode: Consider action execution order, use Longest Increasing Subsequence (LIS) algorithm
    2. Order-insensitive mode: Only focus on action content, ignore execution order
    
    Parameters:
        pred_plan (List[str]): Predicted action sequence, list of actions to be evaluated
        gt_plan (List[str]): Ground truth action sequence, list of actions as evaluation standard
        eval_model (object): Sentence encoding model for action semantic matching
        order (bool): Whether to consider action order, default True
            - True: Consider action execution order, use LIS algorithm
            - False: Ignore order, only compare action content
    
    Returns:
        dict: Dictionary containing evaluation results with precision, recall, f1_score
            - precision: Precision, number of correctly predicted actions / total predicted actions
            - recall: Recall, number of correctly predicted actions / total ground truth actions
            - f1_score: F1 score, harmonic mean of precision and recall
    
    Algorithm Description:
        1. Use match_node function to get matching mapping between predicted and ground truth actions
        2. If order is considered (order=True):
           - Use Longest Increasing Subsequence (LIS) algorithm to determine number of order-correct matches
           - Dynamic programming to calculate longest matching sequence length ending with each action
        3. If order is not considered (order=False):
           - Only calculate number of correct matches
           - Count unmatched items in ground truth sequence
           
    Time Complexity: O(n²) when order=True, O(n) when order=False
    Space Complexity: O(n)
    
    Example:
        >>> pred = ["go to kitchen", "pick up sponge", "wash sponge"]
        >>> gt = ["go to kitchen", "grab sponge", "clean sponge"]
        >>> model = SentenceTransformer("all-mpnet-base-v2")
        >>> result = t_eval_plan(pred, gt, model, order=True)
        >>> print(result)  # {"precision": 0.67, "recall": 0.67, "f1_score": 0.67}
    """
    # Get matching mapping between predicted and ground truth actions
    # Use semantic similarity to map predicted actions to ground truth actions
    pred_to_gt_mapping = match_node(pred_plan,gt_plan,eval_model)

    # Get the number of predicted and ground truth actions
    len_pred = len(pred_plan)
    len_gt = len(gt_plan)

    # Choose different evaluation strategies based on whether to consider action order
    if order:
        # ---------- Order-sensitive mode: Use Longest Increasing Subsequence (LIS) algorithm ----------
        # Initialize dynamic programming array, dp[i] represents the longest matching sequence length ending with the i-th predicted action
        dp = np.ones(len_pred)  # Initialize to 1, each action can form at least a sequence of length 1
        
        # Dynamic programming to calculate longest increasing subsequence
        for i in range(len_pred):
            for j in range(i):  # Check all actions before i
                # Ignore unmatched nodes (nodes mapped to -1)
                if pred_to_gt_mapping[i] == -1 or pred_to_gt_mapping[j] == -1:
                    continue
                # If the ground truth action index mapped by pred[i] is greater than the index mapped by pred[j],
                # it indicates consistent order and can extend the subsequence
                if pred_to_gt_mapping[i] > pred_to_gt_mapping[j]:
                    dp[i] = max(dp[i], dp[j] + 1)  # Update the longest sequence length ending with i

        # Maximum number of order-correct matches (length of longest increasing subsequence)
        correct_count = int(max(dp))

        # Calculate evaluation metrics
        precision = correct_count / len_pred  # Precision: correct predictions / total predictions
        recall = correct_count / len_gt      # Recall: correct predictions / total ground truth
        f1_score = 2 * recall * precision / (recall + precision)  # F1 score

        result = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    else:
        # ---------- Order-insensitive mode: Only calculate action content matching ----------
        # Count the number of correctly matched actions (ignoring order)
        correct_count = 0
        for i in range(len_pred):
            if pred_to_gt_mapping[i] != -1:  # If the predicted action has a corresponding ground truth action
                correct_count += 1

        # Count the number of unmatched actions in the ground truth sequence for recall calculation
        fail_recall = 0
        for i in range(len_gt):
            if i not in pred_to_gt_mapping.values():  # If the ground truth action is not matched by any predicted action
                fail_recall += 1
        
        # Calculate evaluation metrics
        recall = (len_gt - fail_recall)/len_gt  # Recall: matched ground truth actions / total ground truth actions
        precision = correct_count / len_pred   # Precision: correct matches / total predicted actions

        # Prevent division by zero
        if correct_count == 0:
            f1_score = 0
        else:
            f1_score = 2 * recall * precision / (recall + precision)  # F1 score

        result = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    return result


def t_eval_nodes(pred_graph:Dict[str,List[str]],gt_graph:Dict[str,List[str]],sentence_model:object)-> Dict[str,float]:
    """
    Node-level evaluation function: Node sequence evaluation based on topological sort sequences
    
    This function implements node-level evaluation by comparing the node sequence of the predicted graph 
    with all possible topological sorts of the ground truth graph to calculate the best matching result. 
    This method considers the execution order of nodes in directed graphs and is suitable for evaluating 
    whether the node arrangement in workflows or task sequences is reasonable.
    
    Parameters:
        pred_graph (Dict[str,List[str]]): Predicted graph, dictionary containing nodes and edges
            Format: {"nodes": [...], "edges": [...]}
        gt_graph (Dict[str,List[str]]): Ground truth graph, dictionary containing nodes and edges
            Format: {"nodes": [...], "edges": [...]}
        sentence_model (object): Sentence encoding model for node semantic matching
    
    Returns:
        Dict[str, float]: Dictionary containing evaluation results with precision, recall, f1_score
            - precision: Precision, number of correctly predicted nodes / total predicted nodes
            - recall: Recall, number of correctly predicted nodes / total ground truth nodes
            - f1_score: F1 score, harmonic mean of precision and recall
    
    Algorithm Description:
        1. Get all topological sort sequences of the ground truth graph
        2. Remove START and END special nodes from the predicted graph, keep only intermediate action nodes
        3. Iterate through all topological sort sequences of the ground truth graph, evaluate semantic sequence matching with predicted nodes
        4. Return the best matching result (evaluation result corresponding to the topological sequence with highest F1 score)
        
    Features:
        - Node-level evaluation: Focus on node semantic content and order
        - Multi-sequence comparison: Consider all possible topological sorts of the ground truth graph
        - Maximum F1 strategy: Select the topological sort that best matches the predicted node sequence for evaluation
        - Strong robustness: Can handle complex graph structures with multiple legal execution orders
        
    Time Complexity: O(k * n²), where k is the number of topological sorts, n is the number of nodes
    Space Complexity: O(k * n)
    
    Example:
        >>> pred = {"nodes": ["START", "A", "B", "C", "END"], "edges": [(0,1), (1,2), (2,3), (3,4)]}
        >>> gt = {"nodes": ["START", "A", "B", "C", "END"], "edges": [(0,1), (1,2), (2,3), (3,4)]}
        >>> model = SentenceTransformer("all-mpnet-base-v2")
        >>> result = t_eval_nodes(pred, gt, model)
        >>> print(result)  # {"precision": 1.0, "recall": 1.0, "f1_score": 1.0}
    """
    # Initialize best evaluation result to record the highest F1 score among all topological sorts
    max_f1 = {
        'precision': 0,
        'recall': 0,
        'f1_score': 0
    }

    try:
        # Get all topological sort sequences of the ground truth graph
        # These sequences represent all possible legal execution orders in the ground truth graph
        all_gold_trajects = all_topological_sorts(gt_graph)
    except Exception as e:
        # If topological sorting process fails (e.g., cycle in graph), print exception info and return zero score
        print(f"Topological sorting error: {e}")
        print(f"Original graph structure: gt_graph")
        return max_f1

    # Remove START and END special nodes from predicted graph, keep only intermediate action nodes
    # The reason is that START and END nodes are usually just markers and don't participate in actual semantic evaluation
    pred_nodes = [pred_node for pred_node in pred_graph["nodes"] if pred_node not in ["START","END"]]

    # Iterate through all topological sort sequences of the ground truth graph, evaluate semantic sequence matching with predicted nodes
    # Select the topological sort that best matches the predicted node sequence as the final evaluation result
    for gold_nodes in all_gold_trajects:
        # Use t_eval_plan function to evaluate the matching degree between current topological sequence and predicted nodes
        result = t_eval_plan(pred_nodes,gold_nodes,sentence_model)
        # If current topological sequence has higher F1-score, update best result
        if result["f1_score"] > max_f1["f1_score"]:
            max_f1 = result
    
    # Return best matching result (evaluation result corresponding to the topological sequence with highest F1 score)
    return max_f1



if __name__ == "__main__":
    """
    Main function: Contains various test cases and examples
    
    This main function demonstrates various functionalities of the graph evaluator, including:
    1. Topological sort functionality test - Verify correctness of directed graph topological sort calculation
    2. Largest connected component calculation test - Verify accuracy of undirected graph connectivity analysis
    3. Node matching functionality test - Verify node matching algorithm based on semantic similarity
    4. Graph-level evaluation test - Verify graph structure evaluation based on reachability closure
    5. Node-level evaluation test - Verify node sequence evaluation based on topological sorting
    6. Workflow graph evaluation comparison test - Demonstrate evaluation results of different types of workflow graphs
    
    Usage:
        python graph_evaluator.py
    """
    # Execute graph-level evaluation functionality tests
    print("=== Executing Graph-level Evaluation Tests ===")
    t_eval_graph_test()
    
    # Commented code contains some additional test cases and examples
    # Can be uncommented as needed to run these tests
    
    # Initialize BERT model for semantic similarity computation
    # Use pre-trained model from sentence-transformers library
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    
    # The following are some additional test cases that can be enabled as needed:
    # 
    # Test case 1: Simple legal calculation workflow
    # pred_graph = {
    #     "nodes": ["START","Find the statutory formula Calculate the K value.", "Compute each parent's monthly net disposable income.","END"],
    #     "edges": [(0,1),(1,2),(2,3)]
    # }
    # 
    # Test case 2: Sponge cleaning workflow
    # gt_graph ={
    #     'nodes': ['START', 'go to toilet', 'take dishsponge from toilet', 'go to sinkbasin', 'clean dishsponge with sinkbasin', 'go to toilet', 'put dishsponge in/on toilet.', 'END'], 
    #     'edges': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    # }
    # 
    # Evaluation result examples:
    # print(t_eval_nodes(pred_graph, gt_graph, model)["f1_score"])  # Node-level evaluation F1 score
    # print(t_eval_graph(pred_graph, gt_graph, model)["f1_score"])  # Graph-level evaluation F1 score
    # print(t_eval_nodes(pred_graph, pred_graph, model)["f1_score"])  # Perfect match node-level evaluation
    # print(t_eval_graph(pred_graph, pred_graph, model)["f1_score"])  # Perfect match graph-level evaluation

    # ==================== Workflow Graph Evaluation Comparison Test ====================
    print("\n=== Workflow Graph Evaluation Comparison Test ===")
    print("This test compares the similarity between different types of workflow graphs and the original workflow")
    print("=" * 60)
    
    # Define four different types of workflow graphs for evaluation comparison
    
    # Original workflow: Contains detailed code generation and testing process
    workflow_original = {
        'nodes': ['START', 'Logical correctness\\', 'Consideration of all situations\\', 'Potential misunderstanding of the problem\\', '.', 'Logical correctness\\', 'Consideration of all situations\\', 'Potential misunderstanding of the problem\\', '.', 'Logical correctness\\', 'Consideration of all situations\\', 'Potential misunderstanding of the problem\\', '.', 'field.', 'TEST using REFLECTION_ON_PUBLIC_TEST_PROMPT: Given a code problem and a python code solution which failed to pass test or execute, you need to analyze the reason for the failure and propose a better code solution.: ### problem {problem} ### Code Solution {solution} ### Execution Result {exec_pass} #### Failed Test Case {test_fail} Please provide a reflection on the failed test cases and code solution, followed by a better code solution without any additional text or test cases. Remember to keep the entry_point function name: {entry_point}. You MUST NOT give a code with dead loop!', 'END'], 
        'edges': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]  # Linear structure
    }
    
    # Requirements-oriented workflow: Emphasizes code generation requirements
    workflow_requirements = {
        'nodes': ['START', 'CODE_GENERATE using instruction: Can you generate a Python function called min_Jumps that determines the minimum number of jumps required to reach a point from the origin in a 2D plane, with each jump having a fixed length?', 'CODE_GENERATE using instruction: Can you generate a Python function called min_Jumps that determines the minimum number of jumps required to reach a point from the origin in a 2D plane, with each jump having a fixed length?', 'CODE_GENERATE using instruction: Can you generate a Python function called min_Jumps that determines the minimum number of jumps required to reach a point from the origin in a 2D plane, with each jump having a fixed length?', 'field.\\n', 'Please provide a reflection on the failed test cases and code solution, followed by a better code solution without any additional text or test cases. Remember to keep the entry_point function name: {entry_point}. You MUST NOT give a code with dead loop!', 'END'], 
        'edges': [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4), (4, 5), (5, 6)]  # Branching converging structure
    }
    
    # Paraphrasing workflow: Simplified code generation instructions
    workflow_paraphrasing = {
        'nodes': ['START', 'CODE_GENERATE using instruction: Can you generate the code to calculate the number of fixed-length jumps to reach a point?', 'CODE_GENERATE using instruction: Can you generate the code to calculate the number of fixed-length jumps to reach a point?', 'CODE_GENERATE using instruction: Can you generate the code to calculate the number of fixed-length jumps to reach a point?', 'field.\\n', 'TEST using REFLECTION_ON_PUBLIC_TEST_PROMPT: \\nGiven a code problem and a python code solution which failed to pass test or execute, you need to analyze the reason for the failure and propose a better code solution.: \\n### problem\\n{problem}\\n\\n### Code Solution\\n{solution}\\n\\n### Execution Result\\n{exec_pass}\\n\\n#### Failed Test Case\\n{test_fail}\\n\\nPlease provide a reflection on the failed test cases and code solution, followed by a better code solution without any additional text or test cases. Remember to keep the entry_point function name: {entry_point}. You MUST NOT give a code with dead loop!\\n', 'END'], 
        'edges': [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4), (4, 5), (5, 6)]  # Same branching converging structure
    }
    
    # Noise workflow: Contains noise and simplified testing process
    workflow_noise = {
        'nodes': ['START', 'CODE_GENERATE using instruction: Can you analyze the problem and generate the code?', 'CODE_GENERATE using instruction: Can you analyze the problem and generate the code?', 'CODE_GENERATE using instruction: Can you analyze the problem and generate the code?', 'field.\\n', 'TEST using TEST_PROMPT: None', 'END'], 
        'edges': [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4), (4, 5)]  # Simplified structure
    }
    
    # Execute node-level evaluation comparison
    print("Node-level evaluation results (F1 scores):")
    print(f"Original workflow vs Original workflow: {t_eval_nodes(workflow_original, workflow_original, model)['f1_score']:.4f}")
    print(f"Requirements-oriented workflow vs Original workflow: {t_eval_nodes(workflow_requirements, workflow_original, model)['f1_score']:.4f}")
    print(f"Paraphrasing workflow vs Original workflow: {t_eval_nodes(workflow_paraphrasing, workflow_original, model)['f1_score']:.4f}")
    print(f"Noise workflow vs Original workflow: {t_eval_nodes(workflow_noise, workflow_original, model)['f1_score']:.4f}")
    
    print("\nGraph-level evaluation results (F1 scores):")
    print(f"Original workflow vs Original workflow: {t_eval_graph(workflow_original, workflow_original, model)['f1_score']:.4f}")
    print(f"Requirements-oriented workflow vs Original workflow: {t_eval_graph(workflow_requirements, workflow_original, model)['f1_score']:.4f}")
    print(f"Paraphrasing workflow vs Original workflow: {t_eval_graph(workflow_paraphrasing, workflow_original, model)['f1_score']:.4f}")
    print(f"Noise workflow vs Original workflow: {t_eval_graph(workflow_noise, workflow_original, model)['f1_score']:.4f}")
    
    print("\n=== Test Completed ===")
    print("Evaluation result explanation:")
    print("- Node-level evaluation: Focus on node semantic content and sequence matching")
    print("- Graph-level evaluation: Focus on overall graph structure and reachability relationships")
    print("- Higher F1 score indicates higher similarity, 1.0 indicates perfect match")