from typing import Literal
import workspace.DROP.workflows.template.operator as operator
import workspace.DROP.workflows.round_20.prompt as prompt_custom
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
        # Preliminary analysis to summarize key data points
        analysis = await self.custom(input=problem, instruction="Summarize the key data points from the problem statement.")
        
        # Generate a step-by-step thought process
        thought_process = await self.answer_generate(input=problem + analysis['response'])
        
        # Use the thought process to generate the final answer
        solution = await self.custom(input=problem + thought_process['thought'], instruction="")
        
        # Collect multiple solutions for ensemble
        ensemble_solutions = [solution['response'], thought_process['answer']]
        
        # Self-consistency check with a similarity threshold
        if abs(float(solution['response']) - float(thought_process['answer'])) > 0.1:
            ensemble_solutions.append(thought_process['answer'])
        
        final_solution = await self.sc_ensemble(solutions=ensemble_solutions)
        
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]
