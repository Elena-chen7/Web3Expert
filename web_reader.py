import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import openai
from langchain_openai import ChatOpenAI
import getpass
import os
import json
from llms import *
from tools import perplexity, simpleMultiPagesReader, singlePageReader, webSearch, complexMultiPagesReader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

openai.api_key = os.environ['OPENAI_API_KEY']

#slang
def slangSearch(prompt:str)->str:
    loader = WebBaseLoader(
        web_paths=("https://medium.com/@RongHui_Academy/%E5%B0%8F%E7%99%BD%E8%BF%9B%E9%98%B6%E4%B9%8B%E8%B7%AF-web3%E8%A1%8C%E4%B8%9A%E6%9C%AF%E8%AF%AD%E5%A4%A7%E5%85%A8-ea29163ef175",))

    docs = loader.load()
    answer = chatGPT(prompt, docs)
    return answer

def webThreeHelper(prompt:str, mode = 'perplexity', href = 'https://en.wikipedia.org/wiki/Web3', level = 'entry')->str:
    if mode == 'slang':
        answer = slangSearch(prompt) 
    elif mode == 'langchain':
        hrefs = webSearch(prompt)
        answer = simpleMultiPagesReader(prompt, hrefs)
        return hrefs, answer
    elif mode == 'web_reading':
        answer = singlePageReader(prompt, href)
    elif mode == 'study_plan':
        plan = studyPlanning(prompt, 'Green hand with 0 knowledge in Web 3 and block chain.')
        answer = json.loads(plan)
        steps = answer["number of steps"]
        response = ''
        for i in range(1, steps+1):
            detail = answer[f"Step {i}"]
            hrefs = webSearch(detail)
            hrefs = [f'[{i+1}] {link}' for i, link in enumerate(hrefs)]
            hrefs = '\nSources:\n'+'\n'.join(hrefs) + '\n'
            response += f"Step {i}:\n" + detail + hrefs
        return response
    return answer

# prompt = 'AMA是什么的缩写?'
# answer = webThreeHelper(prompt, 'slang')
# print(answer)

# prompt = 'I want to know how bitcoin works and why it is important.'
# level = 'Green hand with 0 knowledge in Web 3 and block chain.'
# answers = webThreeHelper(prompt, mode = 'study_plan', level=level)
# print(answers)

