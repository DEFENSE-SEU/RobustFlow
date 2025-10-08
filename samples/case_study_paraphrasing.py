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
        seeds = [
            "Think step by step and implement a deterministic solution with clear structure and type hints.",
            "Reason step by step, prefer explicit branches, avoid magic numbers, and define the entry point strictly.",
            "Proceed step by step, handle typical boundary inputs conservatively, and keep the code dependency-free."
        ]
        cands = []
        for i in range(3):
            r = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=seeds[i])
            cands.append(r["response"])

        chosen = await self.sc_ensemble(solutions=cands, problem=problem)
        solution = chosen["response"]

        solution = (await self.custom(
            input=solution,
            instruction=(
                "Review step by step and improve docstring, identifiers, and light comments. "
                "Keep functionality the same."
            )
        ))["response"]

        solution = (await self.custom(
            input=solution,
            instruction=(
                "Go through branches step by step and add minimal validation for boundary conditions. "
                "Ensure return values remain deterministic."
            )
        ))["response"]

        tested = await self.test(problem=problem, solution=solution, entry_point=entry_point)
        if not tested["result"]:
            patch = await self.custom(
                input=tested["solution"],
                instruction=(
                    "Analyze failure modes step by step and correct logic precisely. "
                    "Preserve the function signature and external behavior."
                )
            )
            solution = patch["response"]
            tested = await self.test(problem=problem, solution=solution, entry_point=entry_point)

        return solution, self.llm.get_usage_summary()["total_cost"]
