<div align="center">

# RobustFlow: Towards Robust Agentic Workflow Generation

[![Arxiv](https://img.shields.io/badge/arXiv-RobustFlow-b31b1b)](https://arxiv.org/abs/2509.21834)
[![PR Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](https://github.com/DEFENSE-SEU/RobustFlow/pulls)
![Github stars](https://img.shields.io/github/stars/DEFENSE-SEU/RobustFlow.svg)

[Shengxiang Xu (徐圣翔)*](https://xushengxianggg.github.io/)<img src="assets/SEU.png" alt="Logo" width="20">, &nbsp; &nbsp; 
[Jiayi Zhang (张佳钇)*](https://didiforgithub.github.io/)<img src="assets/HKUST.png" alt="Logo" width="20">, &nbsp; &nbsp; 
[Shimin Di (邸世民)](https://sdiaa.github.io/)✉<img src="assets/SEU.png" alt="Logo" width="20">, &nbsp; &nbsp; 

[Yuyu Luo (骆昱宇)](https://luoyuyu.vip/)✉<img src="assets/HKUST.png" alt="Logo" width="20">, &nbsp; &nbsp; 
[Liang Yao (姚亮)](https://1e12leon.top/)<img src="assets/HHU.jpg" alt="Logo" width="20">, &nbsp; &nbsp; 
[Hanmo Liu (刘翰墨)](https://liuhanmo321.github.io/)<img src="assets/HKUST.png" alt="Logo" width="20">, &nbsp; &nbsp; 

[Jia Zhu (朱佳)](https://www.scholat.com/javaelf)<img src="assets/ZJNU.jpg" alt="Logo" width="25">, &nbsp; &nbsp; 
[Fan Liu (刘凡)](https://multimodality.group/author/%E5%88%98%E5%87%A1/)<img src="assets/HHU.jpg" alt="Logo" width="20">, &nbsp; &nbsp; 
[Min-Ling Zhang (张敏灵)](https://palm.seu.edu.cn/zhangml/)<img src="assets/SEU.png" alt="Logo" width="20">, &nbsp; &nbsp; 

\* *Equal Contribution*
✉ *Corresponding Author*

</div>

> If you encounter any difficulties in using or reproducing the code, please get in touch with me directly (Email: xushx@seu.edu.cn, Wechat: 13270628738).

## Introduction

Welcome to the official repository of our paper "RobustFlow: Towards Robust Agentic Workflow Generation"!

The automated generation of agentic workflows is a promising frontier for enabling large language models (LLMs) to solve complex tasks. However, our investigation reveals that the robustness of agentic workflow remains a critical, unaddressed challenge. Current methods often generate wildly inconsistent workflows when provided with instructions that are semantically identical but differently phrased. This brittleness severely undermines their reliability and trustworthiness for real-world applications. 

To quantitatively diagnose this instability, we propose metrics based on nodal and topological similarity to evaluate workflow consistency against common semantic variations such as paraphrasing and noise injection. 

![robust_evaluation_metric](assets/robust_evaluation_metric.png)

Subsequently, we further propose a novel training framework, RobustFlow, that leverages preference optimization to teach models invariance to instruction variations.

![method_overview](assets/method_overview.png)

By training on sets of synonymous task descriptions, RobustFlow boosts workflow robustness scores to 70% - 90%, which is a substantial improvement over existing approaches.

![experiments](assets/experiments.png)

## Quick Start

1. Setup the Python environment:
   
    ```python
    conda create -n robustflow python=3.9
    pip install -r requirements.txt
    ```

## Citation
If you use RobustFlow in your research, please cite our paper:
```
@misc{xu2025robustflowrobustagenticworkflow,
      title={RobustFlow: Towards Robust Agentic Workflow Generation}, 
      author={Shengxiang Xu and Jiayi Zhang and Shimin Di and Yuyu Luo and Liang Yao and Hanmo Liu and Jia Zhu and Fan Liu and Min-Ling Zhang},
      year={2025},
      eprint={2509.21834},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2509.21834}, 
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=DEFENSE-SEU/RobustFlow&type=Date)](https://www.star-history.com/#DEFENSE-SEU/RobustFlow&Date)
