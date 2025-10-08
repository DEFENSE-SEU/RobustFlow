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
        gens = [
            "Reason step by step to implement a clear, deterministic function. "
            "Ensure the declared entry point exists and add minimal type hints.",
            "Think step by step and write straightforward logic with small helpers if needed. "
            "Prefer explicit control flow and keep the interface stable.",
            "Proceed step by step, validate inputs conservatively, and make returns predictable. "
            "Keep the code dependency-free and test-friendly."
        ]
        candidates = []
        for i in range(3):
            sol = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=gens[i])
            candidates.append(sol["response"])

        chosen = await self.sc_ensemble(solutions=candidates, problem=problem)
        solution = chosen["response"]

        refinement = await self.custom(
            input=solution,
            instruction=(
                "Review the code step by step. Improve naming and docstring briefly, "
                "ensure the entry point is implemented, and keep semantics unchanged."
            )
        )
        solution = refinement["response"]

        tested = await self.test(problem=problem, solution=solution, entry_point=entry_point)
        if not tested["result"]:
            patch = await self.custom(
                input=tested["solution"],
                instruction=(
                    "Analyze the logic step by step and fix subtle boundary conditions. "
                    "Preserve the function signature and deterministic behavior."
                )
            )
            solution = patch["response"]
            tested = await self.test(problem=problem, solution=solution, entry_point=entry_point)

        return solution, self.llm.get_usage_summary()["total_cost"]