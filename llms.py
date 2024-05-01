import os
import openai

def chatGPT(prompt:str, document:str)->str:
    completion = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": f"You will be given a document which consists of many abbreviations of Web3 and block chain. You need to explain the meaning of an abbreviation to the use."},
        {"role": "system", "content": f"This is the document: {document}"},
        {"role": "user", "content": f"{prompt}"}
    ],
    temperature = 0.1
    )
    response = completion.choices[0].message.content
    return response

def studyPlanning(prompt:str, level:str)->str:
    answer_format = '''
    {number of steps(n): <your answer>,
    Step 1: <your answer>,
    Step 2: <your answer>,
    ...
    Step n: <your answer>}
    '''
    completion = openai.chat.completions.create(
    model="gpt-4-turbo",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": f"You are a fintech professor which is teaching a course about Web 3 and blockchain. Your student is asking you for a study plan that she can follow to learn by herself. Please give her a study plan based on the field she wants to learn, which includes what she needs to do in each step. Your answer must strictly follow the Json format:{answer_format}."},
        {"role": "system", "content": f"The student's experience level with Web 3: {level}"},
        {"role": "user", "content": f"{prompt}"}
    ],
    temperature = 0.1
    )
    response = completion.choices[0].message.content
    return response

