from typing import Literal
import workspace.DROP.workflows.template.operator as operator
import workspace.DROP.workflows.round_6.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble()  # Added ScEnsemble operator

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate multiple step-by-step breakdowns of the answer
        breakdowns = await self.answer_generate(input=problem)
        # Use the breakdowns to create comprehensive solutions
        solutions = []
        for breakdown in breakdowns['thoughts']:  # Assuming 'thoughts' is a list of breakdowns
            solution = await self.custom(input=problem + breakdown, instruction="")
            solutions.append(solution['response'])
        # Use ScEnsemble to select the best solution
        final_solution = await self.sc_ensemble(solutions=solutions)
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]
