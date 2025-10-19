from typing import Literal
import workspace.DROP.workflows.template.operator as operator
import workspace.DROP.workflows.round_14.prompt as prompt_custom
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
        
        # Self-ask to generate additional context
        self_ask = await self.custom(input=problem + question_summary['response'], instruction="Generate additional context or questions related to the problem.")
        
        # Generate a step-by-step breakdown of the solution
        step_by_step = await self.answer_generate(input=problem)
        
        # Check for self-consistency before generating multiple answers
        consistency_check = await self.custom(input=problem + step_by_step['thought'] + question_summary['response'], instruction="Check for consistency.")
        
        # Use the custom method to generate multiple answers
        multiple_answers = await self.custom(input=problem + step_by_step['thought'] + question_summary['response'] + consistency_check['response'] + self_ask['response'], instruction="")
        
        # Use self-consistency to select the best solution
        final_solution = await self.sc_ensemble(solutions=[multiple_answers['response'], step_by_step['answer']])
        
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]
