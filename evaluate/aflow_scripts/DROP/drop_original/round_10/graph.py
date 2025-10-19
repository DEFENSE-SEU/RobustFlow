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
        # Generate a step-by-step breakdown of the solution
        step_by_step = await self.answer_generate(input=problem)
        
        # Use the custom method to generate multiple answers
        multiple_answers = await self.custom(input=problem + step_by_step['thought'], instruction="")
        
        # Generate a summary of the multiple answers for context
        summary = await self.custom(input=multiple_answers['response'], instruction="Provide a summary of the answers.")
        
        # Use self-consistency to select the best solution
        final_solution = await self.sc_ensemble(solutions=[multiple_answers['response'], step_by_step['answer'], summary['response']])
        
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]
