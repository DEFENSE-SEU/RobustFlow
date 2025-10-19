from typing import Literal
import workspace.DROP.workflows.template.operator as operator
import workspace.DROP.workflows.round_10.prompt as prompt_custom
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

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate a step-by-step breakdown of the problem
        breakdown = await self.answer_generate(input=problem)
        # Use the breakdown to generate the final solution
        solution = await self.custom(input=problem + breakdown['thought'], instruction="")
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
