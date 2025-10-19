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
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate multiple responses for better accuracy
        responses = []
        for _ in range(3):  # Generate three different responses
            breakdown = await self.answer_generate(input=problem)
            response = await self.custom(input=problem + breakdown['thought'], instruction="")
            responses.append(response['response'])
        
        # Use self-consistency to select the best response
        final_solution = await self.sc_ensemble(solutions=responses)
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]
