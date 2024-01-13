import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

openai.api_key = st.secrets.openai_key
st.header("Chat with Medicine-GPT 💬💊")
st.error(
    """
    THIS CHATBOT IS DEMO AND NOT INTENDED OR AUTHORIZED TO PROVIDE MEDICAL ADVICE, DIAGNOSIS OR TREATMENT. 
"""
)
st.write(
    """
    **Medicine-GPT** is an AI assistant trained on medical documents to that able answer questions about medicine, their usage, and their side effects
"""
)
st.info(
    """
Due to free-compute cost limitation, it only list 500 medicine/drugs listed [here](https://raw.githubusercontent.com/fahmirzldi/medicine-gpt-llamaindex/main/data/drugs_side_effects_drugs_com.csv)

Example of available medicine: **doxycycline**, **corticotropin**, or **Tenormin**
"""
)

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about your Medicine! Try: What do you know about doxycyline?"}
    ]

system_prompt="""
You are an expert Pharmacist and your job is to answer basic question on medical drugs. 
Assume that all questions are related to medicine. Assume that you will answer to common patients. Keep your answers based on document and based on facts and do not hallucinate features or information that are not present in the documents.

Prioritize to tell what the medicine is used for. Only tell about common side effect first!
Answer in pretty and easy to read format. 
When listing side effects, list each point on a new line for better readability.

if there is no information available, tell user there is no medicine in database and ask to consult professional healthcare"""

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Medicine-GPT is Loading"):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-16k-0613", temperature=0.5, system_prompt=system_prompt))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
