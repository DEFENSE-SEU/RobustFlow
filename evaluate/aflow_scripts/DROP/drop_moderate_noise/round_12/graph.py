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
        # Generate a step-by-step thought process
        thought_process = await self.answer_generate(input=problem)
        
        # Self-ask to refine the thought process
        refined_thought = await self.custom(input=thought_process['thought'], instruction="Refine this thought process.")
        
        # Generate multiple solutions using the refined thought process
        multiple_solutions = await self.answer_generate(input=problem + refined_thought['response'])
        
        # Collect solutions for ensemble
        solutions = [multiple_solutions['answer'], thought_process['answer']]
        
        # Review step to evaluate solutions before final selection
        reviewed_solutions = await self.custom(input="Evaluate these solutions: " + ", ".join(solutions), instruction="Select the best solution based on relevance and accuracy.")
        
        # Use self-consistency to select the best solution
        final_solution = await self.sc_ensemble(solutions=[reviewed_solutions['response']])
        
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]
