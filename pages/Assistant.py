import streamlit as st
import openai
import os
from streamlit_chatbox import *
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from web_reader import *
from llms import *

openai.api_key = os.environ['OPENAI_API_KEY']
with st.sidebar:
    st.subheader('Start to chat with ReadingExpert!')
    in_expander = st.checkbox('show messages in expander', True)
    show_history = st.checkbox('show history', False)
    option = st.selectbox(
        "Please choose the mode:",
        ("Crypto Slang Dictionary", "Crypto Expert", "Learning Consultant", "Your Crypto Library"),
        index=None,
        placeholder="Select the mode...",
        )
    st.divider()
    btns = st.container()
    OPENAI_API_KEY = st.text_area(label = 'OPENAI_API_KEY', placeholder = 'Please enter your OpenAI API key...')
    st.session_state['OPENAI_API_KEY'] = OPENAI_API_KEY

if 'library' not in st.session_state:
    st.session_state['library'] = []

if 'last_mode_library' not in st.session_state:
    st.session_state['last_mode_library'] = False

if 'OPENAI_API_KEY' not in st.session_state:
    st.error("Please input your API Key in the sidebar")
    st.stop()

else:
    st.title("Web 3 Expert")
    if option == 'Your Crypto Library':
        st.session_state['last_mode_library'] = True
        for i in st.session_state['library']:
            chat_box = ChatBox()
            chat_box.init_session(clear = True)
            chat_box.output_messages()
            chat_box.ai_say(
                [
                    Markdown(i, in_expander=in_expander,
                                expanded=True, state='complete', title="Knowledge Card"),
                ]
            )
    else:
        chat_box = ChatBox()
        if st.session_state['last_mode_library']:
            chat_box.init_session(clear = True)
            st.session_state['last_mode_library'] = False
        chat_box.init_session()
        chat_box.output_messages()
        if query := st.chat_input('Chat with Web3 AI...'):
            chat_box.user_say(query)
            if option == 'Crypto Slang Dictionary':
                response = webThreeHelper(query, 'slang')
            elif option == 'Crypto Expert':
                hrefs, response = webThreeHelper(query, 'langchain')
                hrefs = [f'[{i+1}] {link}' for i, link in enumerate(hrefs)]
                response += '<br>Sources:<br>'+'<br>'.join(hrefs)
            elif option == 'Learning Consultant':
                plan = studyPlanning(query, 'Green hand with 0 knowledge in Web 3 and block chain.')
                answer = json.loads(plan)
                steps = answer["number of steps"]
                Hrefs = []
                response = ''
                for i in range(1, steps+1):
                    detail = answer[f"Step {i}"]
                    hrefs = webSearch(detail)
                    hrefs = [f'[{i+1}] {link}' for i, link in enumerate(hrefs)]
                    hrefs = '<br>Sources:<br>'+'<br>'.join(hrefs) + '<br>'
                    response += f"Step {i}:<br>" + detail + hrefs
                # response = answer[0]
                # hrefs = answer[1]
            knowledge_card = query + '<br>' + response
            st.session_state['library'].append(knowledge_card)
            chat_box.ai_say(
                [
                    Markdown(response, in_expander=in_expander,
                                expanded=True, state='complete', title="Web3Expert"),
                ]
            )

    if btns.button("Clear history"):
        chat_box.init_session(clear=True)
        st.rerun()

    if show_history:
        st.write(chat_box.history)