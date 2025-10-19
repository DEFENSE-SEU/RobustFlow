from typing import Literal
import workspace.DROP.workflows.template.operator as operator
import workspace.DROP.workflows.round_16.prompt as prompt_custom
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
        # Generate a step-by-step thought process
        thought_process = await self.answer_generate(input=problem)
        
        # Validate the thought process before generating the final answer
        validation_response = await self.custom(input=problem + thought_process['thought'], instruction="Validate the thought process.")
        
        # Use the thought process to generate the final answer
        solution = await self.custom(input=problem + thought_process['thought'], instruction="")
        
        # Collect multiple solutions for ensemble
        ensemble_solutions = [solution['response'], thought_process['answer']]
        final_solution = await self.sc_ensemble(solutions=ensemble_solutions)
        
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]
