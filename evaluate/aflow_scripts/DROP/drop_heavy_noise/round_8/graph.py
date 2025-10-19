from typing import Literal
import workspace.DROP.workflows.template.operator as operator
import workspace.DROP.workflows.round_8.prompt as prompt_custom
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
        # Generate multiple answers to improve selection
        breakdown = await self.answer_generate(input=problem)
        response1 = await self.custom(input=problem + breakdown['thought'], instruction="")
        response2 = await self.custom(input=problem + breakdown['thought'], instruction="")
        
        # Use ScEnsemble to select the best response
        ensemble_response = await self.sc_ensemble(solutions=[response1['response'], response2['response']])
        return ensemble_response['response'], self.llm.get_usage_summary()["total_cost"]
