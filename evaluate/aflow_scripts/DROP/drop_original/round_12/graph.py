from typing import Literal
import workspace.DROP.workflows.template.operator as operator
import workspace.DROP.workflows.round_12.prompt as prompt_custom
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
        # Generate a summary of the question for context
        question_summary = await self.custom(input=problem, instruction="Summarize the question for clarity.")
        
        # Generate a step-by-step breakdown of the solution
        step_by_step = await self.answer_generate(input=problem)
        
        # Use the custom method to generate multiple answers
        multiple_answers = await self.custom(input=problem + step_by_step['thought'] + question_summary['response'], instruction="")
        
        # Review the generated answers for quality
        review = await self.custom(input=multiple_answers['response'], instruction="Review the answers and provide feedback.")
        
        # Use self-consistency to select the best solution
        final_solution = await self.sc_ensemble(solutions=[review['response'], step_by_step['answer']])
        
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]
