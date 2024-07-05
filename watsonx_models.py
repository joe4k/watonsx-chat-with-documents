import os
from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
# watsonx.ai python SDK
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watsonx_ai.foundation_models import ModelInference

from llama_index.llms.ibm import WatsonxLLM


# Read creadentials from local .env file in the same directory as this script
def get_credentials():

    load_dotenv()

    # Update the global variables that will be used for authentication in another function
    watsonx_creds = {}
    watsonx_creds["api_key"] = os.getenv("api_key", None)
    watsonx_creds["project_id"] = os.getenv("project_id", None)
    watsonx_creds["url"] = os.getenv("url", None)
    
    return watsonx_creds


# Get watsonx.ai Python client using defined credentials
def get_watsonx_ai_client(url,api_key):
    credentials = Credentials(
                   url = url,
                   api_key = api_key
                  )
    client = APIClient(credentials)
    return client


# The get_model function creates an LLM model object with the specified parameters
def get_model(url,api_key,project_id,model_type,max_tokens,min_tokens,decoding,stop_sequences,temperature):

    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.STOP_SEQUENCES:stop_sequences
    }
    
    temperature = 0.5
    max_new_tokens = 200
    additional_params = {
        "decoding_method": decoding,
        "min_new_tokens": min_tokens,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_k": 50,
        "top_p": 1,
    }
    
    watsonx_llm = WatsonxLLM(
        model_id=model_type,
        apikey=api_key,
        url=url,
        project_id=project_id,
        additional_params=additional_params,
    )

    return watsonx_llm




# Create a generic prompt to answer the user question
def get_prompt(question, selected_model):

    # Prompts are passed to LLMs as one string. We are building it out as separate strings for ease of understanding
    # Instruction
    #instruction = "Follow examples and answer the question briefly."
    instruction = "You are a helpful AI assistant. Answer the question below. " 
    # Examples to help the model set the context
    ##examples = "\n\nQuestion: What is the capital of Germany\nAnswer: Berlin\n\nQuestion: What year was George Washington born?\nAnswer: 1732\n\nQuestion: What are the main micro nutrients in food?\nAnswer: Protein, carbohydrates, and fat\n\nQuestion: What language is spoken in Brazil?\nAnswer: Portuguese \n\nQuestion: "
    examples = ""
    # Question entered in the UI
    your_prompt = question
    # Since LLMs want to "complete a document", we're are giving it a "pattern to complete" - provide the answer
    # This format works for all models with the exception of llama
    end_prompt = "\nAnswer:"

    final_prompt = instruction + examples + your_prompt + end_prompt

    return final_prompt


def answer_questions(user_question, selected_model):

    # Get the prompt
    final_prompt = get_prompt(user_question, selected_model)
    
    # Display our complete prompt - for debugging/understanding
    print("***final prompt***")
    print(final_prompt)
    print("***end of final prompt***")

    # Look up parameters in documentation:
    # https://ibm.github.io/watson-machine-learning-sdk/foundation_models.html#
    model_type = selected_model
    max_tokens = 300
    min_tokens = 50
    decoding = DecodingMethods.GREEDY
    stop_sequences = ['.', '\n']

    # Get the model
    model = get_model(model_type, max_tokens, min_tokens, decoding, stop_sequences)

    # Generate response
    generated_response = model.generate(prompt=final_prompt)
    model_output = generated_response['results'][0]['generated_text']
    # For debugging
    print("Answer: " + model_output)

    return model_output



# List all available and supported Foundation Models in watsonx.ai
#def list_models():
#    modelList = watsonx_client.foundation_models.TextModels
#    models = [member.value for member in modelList]
#    return models