from typing import Literal
import workspace.DROP.workflows.template.operator as operator
import workspace.DROP.workflows.round_3.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)  # Added ScEnsemble operator

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate initial response using custom method
        response = await self.custom(input=problem, instruction="")
        
        # Generate a step-by-step answer using AnswerGenerate
        answer_response = await self.answer_generate(input=problem)
        
        # Combine responses for ensemble selection
        ensemble_response = await self.sc_ensemble(solutions=[response['response'], answer_response['answer']])
        
        # Final output
        final_response = ensemble_response['response']
        
        return final_response, self.llm.get_usage_summary()["total_cost"]
