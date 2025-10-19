from typing import Literal
import workspace.DROP.workflows.template.operator as operator
import workspace.DROP.workflows.round_21.prompt as prompt_custom
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
        initial_solutions = []
        for _ in range(3):  # Generate three solutions
            solution = await self.custom(input=combined_input, instruction="")
            initial_solutions.append(solution['response'])
        
        # Review the initial solutions for accuracy
        review_input = f"Review the following solutions: {', '.join(initial_solutions)}. Are they correct?"
        review_result = await self.custom(input=review_input, instruction="")
        
        # If the review indicates the solutions are not satisfactory, generate additional solutions
        if review_result['response'] == "No":
            for _ in range(2):  # Generate two more solutions
                further_solutions = await self.custom(input=combined_input, instruction="")
                initial_solutions.append(further_solutions['response'])
            review_input += f" and {', '.join(initial_solutions[-2:])}. Are they correct?"
            review_result = await self.custom(input=review_input, instruction="")
        
        # Use self-consistency to select the best solution based on review
        final_solution = await self.sc_ensemble(solutions=initial_solutions + [review_result['response']])
        
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]
