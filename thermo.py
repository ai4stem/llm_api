import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime
import time
from openai import OpenAI
import re

# .env 파일 불러오기
load_dotenv()

context = []

# ChatGPT 메시지 가져오는 함수
def single_ask(message):
    message = message.strip()

    # ChatGPT 대화를 위한 메시지 형태 설정하기
    if len(context) == 0:
        context.append({"role": "system", "content": "You are a helpful assistant."})
        context.append({"role": "user", "content": message})
    else:
        context.append({"role": "user", "content": message})

    client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(model='gpt-4o-mini', messages=context)
    answer = response.choices[0].message.content
    
    context.append({'role': 'assistant', 'content': answer})

    return answer

# Streamlit app
st.title("ChatGPT 연습")

text = st.text_input(
    label='질문', 
    placeholder='질문을 입력해 주세요'
)

if text != "":
    st.write(f'질문: :violet[{text}]')
    st.write(f'대답: :red[{single_ask(text)}]')