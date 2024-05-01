import bs4
import os
import openai
from openai import OpenAI
from duckduckgo_search import DDGS
from langchain_community.document_loaders import WebBaseLoader
#from rag_assistant import *
from transformers import AutoTokenizer
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever

def tokenCount(prompt: str)->int:
    """
    count the number of tokens in a given prompt.
    :param prompt: user input like 'Give me a breif introduction of Dune 2'.
    :return: the len of the prompt.
    """
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return len(tokens)

def chatGPT(prompt: str, document: str)->str:
    """
    define an agent to read the document and complete the conversation or answer the question.
    :param prompt: user input like 'Give me a breif introduction of Dune 2'.
    :param document: the document that includes the necessary information to complete the conversation or answer the question .
    :return: the answer.
    """
    completion = openai.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": f"You will be given a document which consists of many movie related information. You need to use these information to finish the conversation or answer the question."},
        {"role": "system", "content": f"This is the document: {document}"},
        {"role": "user", "content": f"{prompt}"}
    ],
    temperature = 0.1
    )
    answer = completion.choices[0].message.content
    return answer

def summaryAssistant(prompt: str, document: list)->str:
    """
    define an agent to summarize the given content.
    :param prompt: user input like 'Give me a breif introduction of Dune 2'.
    :param document: the document that includes the necessary information to complete the conversation or answer the question .
    :return: the answer.
    """
    completion = openai.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": f"You will be given a document which contains the contents from different web pages about a specific prompt. Please give a summary of these contents."},
        {"role": "system", "content": f"This is the document: {document}"},
        {"role": "user", "content": f"{prompt}"}
    ],
    temperature = 1
    )
    answer = completion.choices[0].message.content
    return answer

def webSearch(prompt: str)->list:
    """
    use the API of Duckduckgo to search online
    :param prompt: user input like 'Give me a breif introduction of Dune 2'.
    :return: a list that contains the link of the 5 most related web pages.
    """
    results = DDGS().text(prompt, max_results=5)
    hrefs = [i['href'] for i in results]
    return hrefs

def singlePageReader(prompt: str, href: str)->str:
    """
    use LangChain to read the content on a given web page
    :param prompt: user input like 'Give me a breif introduction of Dune 2'.
    :href: the web link that we want the agent to read
    :return: answer of the prompt based on the web page.
    """
    loader = WebBaseLoader(href)
    docs = (loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=splits, embedding=embedding)
    llm = ChatOpenAI(temperature=1)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(), llm=llm
    )
    unique_docs = retriever_from_llm.get_relevant_documents(query=prompt)
    answer = chatGPT(prompt, unique_docs)
    return answer

def simpleMultiPagesReader(prompt: str, hrefs: list)->str:
    '''
    use LangChain to read the content on multiple web pages
    :param prompt: user input like 'Give me a breif introduction of Dune 2'.
    :href: the web link that we want the agent to read
    :return: answer of the prompt based on the web page.
    '''
    answer = ''
    for i, href in enumerate(hrefs[:2]):
        try:
          answer += f'Content of web page {i+1}: ' + singlePageReader(prompt, href)
        except:
          continue
    answer = summaryAssistant(prompt, answer)
    return answer

def complexMultiPagesReader(prompt: str, hrefs: list)->str:
    answer = ''
    doc = ''
    Metadata = []
    for href in hrefs:
        loader = WebBaseLoader(href)
        docs = (loader.load())
        text = ' '.join(docs[0].page_content.split())
        metadata = docs[0].metadata
        Metadata.append(metadata)
        doc += text
    answer = summaryAssistant(prompt, doc)
    return answer, Metadata
