import os
import shutil
import streamlit as st

from watsonx_models import *
from llamaindex_ingestDocs import *
from llm_eval import *


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if 'models' not in st.session_state:
    st.session_state["models"] = None

if 'watsonx_creds' not in st.session_state:
    st.session_state["watsonx_creds"] = None

if 'watsonx_client' not in st.session_state:
    st.session_state["watsonx_client"] = None

if 'setupDone' not in st.session_state:
    st.session_state["setupDone"] = False

if 'dirpath' not in st.session_state:
    st.session_state["dirpath"] = None

if 'watsonx_llm' not in st.session_state:
    st.session_state["watsonx_llm"] = None

if 'eval_llm' not in st.session_state:
    st.session_state["eval_llm"] = None

if 'embed_model' not in st.session_state:
    st.session_state["embed_model"] = None

if 'chat_engine' not in st.session_state:
    st.session_state["chat_engine"] = None

if 'custom_engine' not in st.session_state:
    st.session_state["custom_engine"] = None

if 'vector_engine' not in st.session_state:
    st.session_state["vector_engine"] = None

if 'keyword_engine' not in st.session_state:
    st.session_state["keyword_engine"] = None

if 'sample_questions' not in st.session_state:
    st.session_state["sample_questions"] = None

# Setup sidebar for the streamlit app with the list of watsonx.ai models to choose from
def setupSideBar():
    with st.sidebar:
        print("session dir path: ", st.session_state["dirpath"])
        if st.session_state["dirpath"] is None:
            dirpath = f"{os.getcwd()}/.tmpdir"
            st.session_state["dirpath"] = dirpath
    
            if os.path.exists(dirpath):
                # removing directory
                shutil.rmtree(dirpath, ignore_errors=True)
        
            # Make dir
            print("making dir: ", dirpath)
            os.mkdir(dirpath)
    
        uploaded_files = st.file_uploader("Choose files to upload", accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                bytes_data = uploaded_file.read()
                st.write("filename:", uploaded_file.name)
                dpath = st.session_state["dirpath"]
                outfile = f"{dpath}/{uploaded_file.name}"
                print("outfile: ", outfile)
                with open(outfile, 'wb') as f: 
                    f.write(bytes_data)
            files = os.listdir(st.session_state["dirpath"])
            print('files: ', files)
            if len(files) > 0:
                return True
            else:
                return False
        return False

model_id = "ibm/granite-20b-multilingual"
max_tokens = 400
min_tokens = 20
decoding = "greedy"
temperature = 0.5
stop_sequences = []

eval_model_id = "meta-llama/llama-3-70b-instruct"


def format_responses(custom_response,vector_response,keyword_response):
    response = ""
    response = f"{response}R1:\n{custom_response}\nR2:\n{vector_response}\nR3:\n{keyword_response}"
    return response

chunk_size = 1024
chunk_overlap = 160
top_k = 3

# Main function for the application
def main():
    st.set_page_config(
    page_title='chat with docs',
    layout='wide',
    page_icon=':rocket:'
    )

    # Set the api key and project id global variables
    st.session_state["watsonx_creds"] = get_credentials()

    # Web app UI - title and input box for the question
    st.title('watsonx.ai chat with Documents')

    # Get list of supported models in watsonx.ai
    if st.session_state['watsonx_creds'] != None:
        watsonx_creds = st.session_state["watsonx_creds"]
        url = watsonx_creds["url"]
        api_key = watsonx_creds["api_key"]
        project_id = watsonx_creds["project_id"]
        st.session_state["watsonx_client"] = get_watsonx_ai_client(url,api_key)
        st.session_state["watsonx_llm"] = get_model(url,api_key,project_id,model_id,max_tokens,min_tokens,decoding,stop_sequences,temperature)
        st.session_state["eval_llm"] = get_model(url,api_key,project_id,eval_model_id,max_tokens,min_tokens,decoding,stop_sequences,temperature)
        st.session_state["embed_model"] = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")



    with st.sidebar:
        print("session dir path: ", st.session_state["dirpath"])
        if st.session_state["dirpath"] is None:
            dirpath = f"{os.getcwd()}/.tmpdir"
            st.session_state["dirpath"] = dirpath
    
            if os.path.exists(dirpath):
                # removing directory
                shutil.rmtree(dirpath, ignore_errors=True)
        
            # Make dir
            print("making dir: ", dirpath)
            os.mkdir(dirpath)
    
        uploaded_files = st.file_uploader("Choose files to upload", accept_multiple_files=True)
        if uploaded_files and st.session_state["setupDone"] == False:
            for uploaded_file in uploaded_files:
                bytes_data = uploaded_file.read()
                #st.write("filename:", uploaded_file.name)
                dpath = st.session_state["dirpath"]
                outfile = f"{dpath}/{uploaded_file.name}"
                #print("outfile: ", outfile)
                with open(outfile, 'wb') as f: 
                    f.write(bytes_data)
            files = os.listdir(st.session_state["dirpath"])
            print('files: ', files)
            if len(files) > 0:
                st.session_state["setupDone"] = True
        
            with st.spinner("Preparing Knowledge Base"):
                ##[custom_engine, vector_engine, keyword_engine] = ingestDocs_hybrid(st.session_state["dirpath"],st.session_state["watsonx_llm"],1024,160,3)
                ##st.session_state["custom_engine"] = custom_engine
                ##st.session_state["vector_engine"] = vector_engine
                ##st.session_state["keyword_engine"] = keyword_engine
                chat_engine,sample_questions = ingestDocs_chat(st.session_state["dirpath"],st.session_state["watsonx_llm"],chunk_size,chunk_overlap,top_k)
                st.session_state["chat_engine"] = chat_engine
                if len(sample_questions) > 0:
                    st.session_state["sample_questions"] = sample_questions
                else:
                    st.session_state["sample_questions"] = None


    
    #print("setup done: ", st.session_state["setupDone"])
    #if st.session_state["setupDone"] == False:
    #    print("calling setup side bar")
    #    if setupSideBar():
    #        print("setupsidebar is true")
    #        st.session_state["setupDone"] = True
    #        #print("setup done: ", st.session_state["setupDone"])
    #        print("Ingesting documents")
    #        with st.spinner("Preparing Knowledge Base"):
    #            [custom_engine, vector_engine, keyword_engine] = ingestDocs(st.session_state["dirpath"],st.session_state["watsonx_llm"],1024,160)
    #            st.session_state["custom_engine"] = custom_engine
    #            st.session_state["vector_engine"] = vector_engine
    #            st.session_state["keyword_engine"] = keyword_engine
    #    else:
    #        print("setupSideBar is false")    
    
    if st.session_state["chat_engine"] != None:
        chat_engine = st.session_state["chat_engine"]
        #custom_engine = st.session_state["custom_engine"]
        #vector_engine = st.session_state["vector_engine"]
        #keyword_engine = st.session_state["keyword_engine"]
        
        ##q = "what is IBM's main business"
        ##response = kb_engine.query(q)
        ##print("question: ", q)
        ##print("res: ", response)
    
    
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
        # React to user input
        if st.session_state["sample_questions"]:
            init_query = st.selectbox("Sample questions",sample_questions)
        else:
            init_query = []
        #st.chat_input()
        #st.text_input()
        if user_question := st.chat_input('Ask a question'):
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(user_question)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_question})
    
            #response = f"Echo: {prompt}"
            #response = answer_questions(kb,user_question,llm)

            chat_response = chat_engine.chat(user_question)
            response = chat_response

            faithfulness,relevancy = eval(st.session_state["eval_llm"],
                                          st.session_state["embed_model"],
                                          user_question,
                                          response)
            print("faithfulness: ", faithfulness)
            print("relevancy: ", relevancy)
            

            #custom_response = custom_engine.query(user_question)
            #print("customo response: ", custom_response)
            #vector_response = vector_engine.query(user_question)
            #print("\n\n vector response: ", vector_response)
            #keyword_response = keyword_engine.query(user_question)
            #print("\n\n keyword response: ", keyword_response)
    
            #response = format_responses(custom_response,vector_response,keyword_response)
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
                st.markdown("")
                fmkdown = f"Faithfulness: {faithfulness}"
                rmkdown = f"Relevancy: {relevancy}"
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(fmkdown,help="metric to report how faithful is generated answer to the retrieved context from the documents")
                with col2: 
                    st.markdown(rmkdown,help="metric to report how relevant is generated answer to the input question")


            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()