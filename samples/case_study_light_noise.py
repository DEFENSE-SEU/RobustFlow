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
        focus = [
            "Step by step, write a compact and clear implementation. Ensure the entry point is defined.",
            "Proceed step by step with emphasis on readability and explicit error handling where appropriate.",
            "Reason step by step and keep logic deterministic with simple, composable helpers."
        ]
        raw = []
        for i in range(3):
            s = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=focus[i])
            raw.append(s["response"])

        tested_candidates = []
        for sol in raw:
            t = await self.test(problem=problem, solution=sol, entry_point=entry_point)
            tested_candidates.append(t["solution"])

        chosen = await self.sc_ensemble(solutions=tested_candidates, problem=problem)
        solution = chosen["response"]

        solution = (await self.custom(
            input=solution,
            instruction=(
                "Review the code step by step, simplify conditionals, and ensure minimal yet clear comments. "
                "Keep semantics and interface unchanged."
            )
        ))["response"]

        final = await self.test(problem=problem, solution=solution, entry_point=entry_point)
        if not final["result"]:
            fix = await self.custom(
                input=final["solution"],
                instruction=(
                    "Trace execution step by step and correct logic precisely. "
                    "Maintain deterministic outputs and the same signature."
                )
            )
            solution = fix["response"]
            final = await self.test(problem=problem, solution=solution, entry_point=entry_point)

        return solution, self.llm.get_usage_summary()["total_cost"]