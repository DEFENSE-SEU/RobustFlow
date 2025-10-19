from typing import Literal
import workspace.DROP.workflows.template.operator as operator
import workspace.DROP.workflows.round_2.prompt as prompt_custom
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
        self.answer_generate = operator.AnswerGenerate(self.llm)  # Added AnswerGenerate operator

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate initial response using custom method
        response = await self.custom(input=problem, instruction="")
        
        # Generate a step-by-step answer using AnswerGenerate
        answer_response = await self.answer_generate(input=problem)
        
        # Combine responses for final output
        final_response = f"{response['response']} | {answer_response['answer']}"
        
        return final_response, self.llm.get_usage_summary()["total_cost"]
