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
        g1 = await self.custom_code_generate(
            problem=problem, entry_point=entry_point,
            instruction=(
                "Step by step, implement a concise and correct function. "
                "Make control flow explicit and ensure the entry point is present."
            )
        )

        g2 = await self.custom_code_generate(
            problem=problem, entry_point=entry_point,
            instruction=(
                "Proceed step by step to write readable code with minimal validation and type hints. "
                "Keep returns deterministic and avoid side effects."
            )
        )
        
        chosen = await self.sc_ensemble(solutions=[g1["response"], g2["response"]], problem=problem)
        solution = chosen["response"]

        solution = (await self.custom(
            input=solution,
            instruction=(
                "Walk through the code step by step and polish comments, simplify branches, "
                "and keep behavior identical."
            )
        ))["response"]

        tested = await self.test(problem=problem, solution=solution, entry_point=entry_point)
        if not tested["result"]:
            fix = await self.custom(
                input=tested["solution"],
                instruction=(
                    "Reason step by step about potential edge cases and correct them with minimal edits. "
                    "Do not change the public interface."
                )
            )
            solution = fix["response"]
            tested = await self.test(problem=problem, solution=solution, entry_point=entry_point)

        return solution, self.llm.get_usage_summary()["total_cost"]