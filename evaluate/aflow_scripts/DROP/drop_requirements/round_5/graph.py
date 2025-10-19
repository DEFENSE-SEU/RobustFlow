from typing import Literal
import workspace.DROP.workflows.template.operator as operator
import workspace.DROP.workflows.round_5.prompt as prompt_custom
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
        step_by_step = await self.answer_generate(input=problem)
        
        # Use the generated thought process to create a comprehensive input
        combined_input = f"{step_by_step['thought']} {problem}"
        
        # Generate the final solution using the custom method
        solution = await self.custom(input=combined_input, instruction="")
        
        # Use self-consistency to select the best solution
        final_solution = await self.sc_ensemble(solutions=[solution['response'], step_by_step['answer']])
        
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]
