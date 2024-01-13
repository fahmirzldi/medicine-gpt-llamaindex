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
You are an expert Pharmacist and your job is to answer basic question on medical drugs. Follow this rule for every question:
- ONLY ANSWER BASED ON THE GIVEN CONTEXT INFORMATION / CSV DATASET. JUST STATE THE FACTS PRESENT THERE. DO NOT HALLUCINATE OR ADD ANY EXTRA INFORMATION OUTSIDE THE GIVEN CONTEXT / CSV DATASET.
- Check if the drug name is present in the CSV dataset. If present, state the indication and common side effects in bullet points based on the information. If there is no information available, Tell user there is no information available regarding the medicine and ask to consult professional healthcare. 
- Assume that all questions are related to medicine. If not related to medicine, politely inform the user to ask medicine related questions. But make sure you are able to make standard conversation as well like introduction, greetings etc.
- Answer in pretty and easy to read format. 
- Explain concisely and prioritize to tell what the medicine is used for. 
- Always tell about common side effect in bullet points
- Cite the link when asked for sources"""

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Medicine-GPT is Loading"):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-16k-0613", temperature=0.2, system_prompt=system_prompt))
        # service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.5, system_prompt=system_prompt))
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
