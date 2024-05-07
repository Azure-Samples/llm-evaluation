# Evalute your LLMs using Synthetic Data Generation

This project is a framework to generate synthetic data and evaluate your LLMs. It is based on the [Azure AI SDK synthetic data generation](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/generate-data-qa) and helps you to evaluate your LLMs on the generated data. This data can then be used for various purposes like unit testing for your LLM lookup, evaluation and iteration of retrieval augmented generation (RAG) flows, and prompt tuning.

We will use the dataset to create an evaluation project on Azure AI Studio.

Azure AI Studio provides a versatile hub for evaluating AI models. You can create an evaluation run from a test dataset (for example generated using this repo) or flow with built-in evaluation metrics from Azure AI Studio UI, or establish a custom evaluation flow and employ the custom evaluation feature.

We also provide a process to generate synthetic sample data (not only for evaluation purposes). In this example shown below, we will generate synthetic data for a *Movies assistant* application. The data contains synopses of movies and the corresponding genres. The data is generated using [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/) and a GPT-4 model.

## Getting Started

- The file [`simulate_movies_data.py`](simulate_movies_data.py) file contains the code to generate synthetic data for a Movies assistant application.
It executes the following steps:
  - Create an Azure AI Index
  - Generate synthetic data (movie synopses and genres) using Azure OpenAI (GPT-4 model)
  - Transform synopses and genres into a Vector representation
  - Ingest the data into the Azure AI Index

- The file [`simulate_qa_evaluation.py`](simulate_qa_evaluation.py) file contains the code to generate a synthetic dataset for evaluation purposes. It uses the movies index created in the previous step.

The processed datasets are available in the [`data`](./data/) folder.

### Prerequisites

- You must have a Pay-As-You-Go Azure account with administrator — or contributor-level access to your subscription. If you don’t have an account, you can sign up for an account following the instructions.
- Get Access to Azure OpenAI
- Once got approved create an Azure OpenAI in you Azure’s subcription.
- You must have a Workspace of [Azure AI Studio](https://azure.microsoft.com/en-gb/products/ai-studio/)
- You must have the [Azure AI Search](https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search) service deployed in your Azure subscription.
- Python 3.11

### Required Libraries

- openai
- python-dotenv
- azure-search-documents==11.4.0b6
- azure-identity
- azure-ai-generative
- pandas
- tenacity

## Resources

- [How to generate question and answer pairs from your source dataset](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/generate-data-qa)
- [Evaluation of generative AI applications
](https://learn.microsoft.com/en-us/azure/ai-studio/concepts/evaluation-approach-gen-ai)
