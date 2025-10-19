from typing import Literal
import workspace.DROP.workflows.template.operator as operator
import workspace.DROP.workflows.round_11.prompt as prompt_custom
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
        
        # Generate multiple initial solutions using the custom method
        initial_solutions = await self.custom(input=combined_input, instruction="")
        initial_solutions_list = [initial_solutions['response'], step_by_step['answer']]
        
        # Review each initial solution for accuracy
        review_results = []
        for solution in initial_solutions_list:
            review_input = f"Review the following solution: {solution}. Is it correct?"
            review_result = await self.custom(input=review_input, instruction="")
            review_results.append(review_result['response'])
        
        # Use self-consistency to select the best solution based on review
        final_solution = await self.sc_ensemble(solutions=review_results)
        
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]
