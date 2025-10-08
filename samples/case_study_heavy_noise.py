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
        a = await self.custom_code_generate(
            problem=problem, entry_point=entry_point,
            instruction=(
                "Step by step, implement a clean and typed solution. "
                "Use clear control flow and keep the function deterministic."
            )
        )

        b = await self.custom_code_generate(
            problem=problem, entry_point=entry_point,
            instruction=(
                "Think step by step and provide a readable implementation with minimal validation. "
                "Ensure the entry point exists and returns are predictable."
            )
        )

        chosen = await self.sc_ensemble(solutions=[a["response"], b["response"]], problem=problem)
        solution = (await self.custom(
            input=chosen["response"],
            instruction=(
                "Review the code step by step and streamline conditionals. "
                "Keep names clear and behavior unchanged."
            )
        ))["response"]

        t = await self.test(problem=problem, solution=solution, entry_point=entry_point)
        if not t["result"]:
            fix = await self.custom(
                input=t["solution"],
                instruction=(
                    "Reason through the failing paths step by step and correct them with minimal edits. "
                    "Do not modify the external interface."
                )
            )
            solution = fix["response"]
            t = await self.test(problem=problem, solution=solution, entry_point=entry_point)

        return solution, self.llm.get_usage_summary()["total_cost"]