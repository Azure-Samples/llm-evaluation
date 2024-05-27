import os
from openai import AzureOpenAI
from collections import defaultdict

from azure.ai.generative.synthetic.qa import QADataGenerator
from azure.ai.generative.synthetic.qa import QAType
from azure.ai.resources.client import AIClient 
from azure.identity import DefaultAzureCredential


class SyntheticDataGenerator:
    def __init__(self):
        # Initialize the OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
            api_key=os.getenv("AZURE_OPENAI_KEY"),  
            api_version="2024-02-15-preview"
            )
        
        # Initialize the AI Studio client (for evaluation purposes)
        self.eval_client = AIClient(subscription_id=os.getenv("SUBSCRIPTION_ID"),
                                    resource_group_name=os.getenv("RESOURCE_GROUP"),
                                    project_name=os.getenv("PROJECT_NAME"),
                                    credential=DefaultAzureCredential())
        
        aoai_connection = self.eval_client.get_default_aoai_connection()
        aoai_connection.set_current_environment()
        
        self.model_config = dict(deployment=os.getenv("MODEL_NAME"),
                            model=os.getenv("MODEL_NAME"),
                            max_tokens=2000)

        self.qa_generator = QADataGenerator(model_config=self.model_config)
        self.qa_type = QAType.LONG_ANSWER


    def generate_dataset(self, system_prompt, user_prompt):
        message_text = [{"role":"system",
                        "content": system_prompt},
                        {"role":"user",
                        "content": user_prompt}]

        # Call the completion endpoint to generate the dataset. We use GPT-3.5-turbo with 16k tokens for this task.
        completion = self.client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_MOVIES_MODEL"),
            messages = message_text,
            temperature=0,
            max_tokens=16000,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
            )
        
        return completion.choices[0].message.content
    
    def generate_qa_evaluation_dataset(self, data, field_name, num_questions=3):
        data_dict = defaultdict(list)

        # Generate questions and answers for the content
        for item in data:
            text = item[field_name]
            result = self.qa_generator.generate(text=text,
                                                qa_type=self.qa_type,
                                                num_questions=num_questions)

            for question, answer in result["question_answers"]:
                print(f"Q: {question}")
                print(f"A: {answer}")
                
                data_dict["question"].append(question)  # Consider generated answer as the ground truth
                data_dict["ground_truth"].append(answer)  # Consider generated answer as the ground truth

                print(f"Tokens used: {result['token_usage']}")
                print("\n")

        return data_dict