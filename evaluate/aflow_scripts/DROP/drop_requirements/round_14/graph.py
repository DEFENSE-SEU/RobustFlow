from typing import Literal
import workspace.DROP.workflows.template.operator as operator
import workspace.DROP.workflows.round_14.prompt as prompt_custom
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

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate multiple solutions using the custom method
        solutions = []
        for _ in range(3):  # Generate three different outputs
            response = await self.custom(input=problem, instruction="")
            solutions.append(response['response'])
        
        # Use ScEnsemble to select the best solution
        ensemble_response = await operator.ScEnsemble(solutions=solutions)
        return ensemble_response['response'], self.llm.get_usage_summary()["total_cost"]
