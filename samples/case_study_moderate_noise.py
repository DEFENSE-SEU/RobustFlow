from typing import Literal
import workspace.MBPP.workflows.template.operator as operator
import workspace.MBPP.workflows.round_1.prompt as prompt_custom
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.test = operator.Test(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        prompts = [
            "Reason step by step and implement a deterministic function with clear types.",
            "Think step by step and keep the code readable and compact with explicit branches.",
            "Proceed step by step, handle boundary inputs conservatively, and avoid hidden state."
        ]
        pool = []
        for i in range(3):
            r = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompts[i])
            pool.append(r["response"])

        chosen = await self.sc_ensemble(solutions=pool, problem=problem)
        solution = (await self.custom(
            input=chosen["response"],
            instruction=(
                "Review step by step to refine naming, add a brief docstring, and ensure the entry point is defined. "
                "Do not alter the overall behavior."
            )
        ))["response"]

        t1 = await self.test(problem=problem, solution=solution, entry_point=entry_point)
        if not t1["result"]:
            backup = await self.custom_code_generate(
                problem=problem,
                entry_point=entry_point,
                instruction=(
                    "Proceed step by step to produce a robust, dependency-free implementation. "
                    "Ensure deterministic returns and strict entry point compliance."
                )
            )
            mix = await self.sc_ensemble(solutions=[t1["solution"], backup["response"]], problem=problem)
            solution = mix["response"]
            t1 = await self.test(problem=problem, solution=solution, entry_point=entry_point)

        return solution, self.llm.get_usage_summary()["total_cost"]
