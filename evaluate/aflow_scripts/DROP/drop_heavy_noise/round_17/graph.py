from typing import Literal
import workspace.DROP.workflows.template.operator as operator
import workspace.DROP.workflows.round_17.prompt as prompt_custom
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
        variations = [problem, problem.replace("?", "??"), problem.lower(), 
                      problem.replace("how many", "what is the total"), 
                      problem.replace("how many", "count of")]  # Expanded variations of the question
        for variation in variations:  # Generate responses for each variation
            breakdown = await self.answer_generate(input=variation)
            response = await self.custom(input=variation + breakdown['thought'], instruction="")
            responses.append(response['response'])
        
        # Review responses for consistency and relevance
        reviewed_responses = [response for response in responses if "error" not in response.lower()]
        
        # Use self-consistency to select the best response
        final_solution = await self.sc_ensemble(solutions=reviewed_responses)
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]
