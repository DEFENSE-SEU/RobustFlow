from typing import Literal
import workspace.DROP.workflows.template.operator as operator
import workspace.DROP.workflows.round_9.prompt as prompt_custom
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
        # Generate multiple responses for better accuracy
        responses = []
        self_ask_variation = problem + " What are the key points to consider?"
        thought_process = await self.answer_generate(input=self_ask_variation)
        
        for variation in [problem, problem + " Please summarize.", problem + " Can you explain it simply?"]:  # Different variations
            response = await self.custom(input=variation + thought_process['thought'], instruction="")
            responses.append(response['response'])
        
        # Use self-consistency to select the best solution
        final_solution = await self.sc_ensemble(solutions=responses)
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]
