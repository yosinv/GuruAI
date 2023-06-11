"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from utilLLM.utilllm import llm_doc_search


# From here down is all the StreamLit UI.
st.set_page_config(page_title="Guru QuestionAnswer Bot", page_icon=":robot:")
st.header("Guru QA Bot")
with st.sidebar:
    st.title("guru QA")
    st.markdown(
        """
        ##About LLm chatbot built using:
        -[langchain]
        -[pinecone]
        -[OpenAI]
        -[streamlit]
        """
    )
    add_vertical_space(5)
    st.write("created by Yosi Navaro")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text(instruction: str = "You: "):
    input_text = st.text_input(instruction, "", key=f"input-{instruction}")
    return input_text


input_container = st.container()
with input_container:
    user_input = get_text()

response_container = st.container()
with response_container:
    if user_input:
        with st.spinner("Response..."):
            result = llm_doc_search(query=user_input, chat_history=st.session_state['chat_history'])
            st.session_state.past.append(user_input)
            st.session_state.generated.append(result['answer'])
            st.session_state["chat_history"].append((user_input, result["answer"]))

    if st.session_state["generated"]:

        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))
