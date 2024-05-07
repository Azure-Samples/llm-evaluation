import os
from openai import AzureOpenAI
import pandas as pd

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
    
    def generate_qa_evaluation_dataset():
        pass
