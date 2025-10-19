from typing import Literal
import workspace.DROP.workflows.template.operator as operator
import workspace.DROP.workflows.round_4.prompt as prompt_custom
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
        self.ensemble = operator.ScEnsemble()  # Added ensemble operator

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate multiple responses
        responses = await self.custom(input=problem, instruction="")
        # Use ensemble to select the best response
        final_solution = await self.ensemble(solutions=[response['response'] for response in responses])
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]
