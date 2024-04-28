from flask import (
    Flask, request, render_template, url_for, redirect, flash, session, jsonify)
from flask.json import JSONEncoder
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from werkzeug.utils import secure_filename
from flask_cors import CORS  # CORS 라이브러리 임포트
from flask_moment import Moment
from datetime import timedelta
from datetime import datetime
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.schema import AIMessage, HumanMessage
from langchain.chains import ConversationalRetrievalChain
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from openai._client import OpenAI #ChatGPT API 이용을 위한 라이브러리
import pandas as pd
import os
import joblib
import markdown
import nltk
import os

app = Flask(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL = 'gpt-3.5-turbo-0125'
TEMPERATURE = 0.0
MAX_TOKENS = 16000  # GPT-3.5의 최대 토큰 길이임(입력, 출력 동일함)

nltk.download('punkt')  # 토크나이저에 필요한 데이터를 다운로드합니다.

# Personalized AI
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=OPENAI_API_KEY, max_retries=20)
knowledge_model = ChatOpenAI(temperature=0, model_name=MODEL)

# If you want to use your own dataset, just change this paremeter 'persist_directory'.
folder_path = 'wos_gifted_db' #영재교육 관련 정보 Vector DB
vector_gift = Chroma(persist_directory=folder_path, embedding_function=embeddings)

qa_gift = ConversationalRetrievalChain.from_llm(knowledge_model, vector_gift.as_retriever(search_kwargs={"k": 3}), return_source_documents=True)

client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

app.config['SESSION_COOKIE_MAX_SIZE'] = 4096  # 세션의 크기에 대한 문제
app.config['SECRET_KEY'] = 'your_secret_key_here'  # 보안을 위한 시크릿 키 설정

# 파일 업로드를 위한 클래스임
class UploadForm(FlaskForm):
    file = FileField('파일 업로드', validators=[
        FileRequired(),
        FileAllowed(['xlsx', 'csv'], '허용되는 파일 형식: xlsx, csv')
    ])

CORS(app)  # CORS 설정 추가
moment = Moment(app)

UPLOAD_FOLDER = './uploads' #데이터셋을 업로드하면 저장하는 폴더
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print('Initialization complete.')

# Basic function
# 기본 함수: context의 메시지가 최대 길이를 초과했는지 확인하는 코드
def check_tokens(items):
    cnt = 0

    if items is None:
        return cnt

    for item in items:
        cnt += len(item['content'])

    return cnt

# 기본 함수: QA Model에서의 질의 응답에 대한 History 관리
def manage_history(lists):    
    tot_size = 0
    if len(lists) >= 1:
        for item in lists:
            tot_size += len(item['content'])

        if tot_size >= 3000:
            lists = lists[2:]
        
        # Human / AI 메시지 형태로 변환
        results = []
        for item in lists:
            if item['type'] == 'human':
                results.append(HumanMessage(content=item['content'], additional_kwargs=item['additional_kwargs'], example=item['example']))
            else:
                results.append(AIMessage(content=item['content'], additional_kwargs=item['additional_kwargs'], example=item['example'])) 
            
        return results
    else:
        return lists
        
# 기본 함수: 응답이 오면 해답 메시지와 출처 메시지를 구분한다.
def questioning(model, lists, query):
    results = manage_history(lists)  # manage_history 함수가 리스트를 반환하도록 수정 필요
    result = model({"question": query, "chat_history": results})
    print(result['answer']) # 응답 출력
    refs = []
    print('Reference:')
    for item in result['source_documents']:
        filename_with_extension = os.path.basename(item.metadata['source'])
        filename = os.path.splitext(filename_with_extension)[0]
        print(filename)
        refs.append(filename)
    
    print(refs)
    
    lists.append({'type':'human', 'content':query, 'additional_kwargs':{}, 'example':False})
    lists.append({'type':'ai', 'content':result['answer'], 'additional_kwargs':{'source':refs}, 'example':False})
            
    return lists, refs


@app.route('/')
def hello():
    return 'It is really drizzle and windy!!'


@app.route('/retrieveai', methods=['GET', 'POST'])
def retrieveai():
    if request.method == 'GET':
        sel_lang = request.args.get('lang')
        print(sel_lang)

        if (sel_lang is not None):
            session['lang'] = sel_lang
            
        return render_template('retrieveai.html', lang=session.get('lang', ''))
    else:
        message = request.form.get('text')
        print(message)
        chat_history = session.get('gift_history', [])
        print(chat_history)
        chat_history, refs = questioning(qa_gift, chat_history, message)
        
        ref_text = ''
        
        for item in refs:
            ref_text = ref_text + ' - ' + str(item) + '\n'
            
        answer = chat_history[-1]['content']
        result = markdown.markdown(answer, extensions=['md_in_html'])
        ref_text = markdown.markdown(ref_text, extensions=['md_in_html'])
        result = result + '<br><b>Reference</b>' + ref_text
        session['gift_history'] = chat_history
        
        return jsonify({'data': result})


@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    form = UploadForm()
    if request.method == 'POST':
        print('uploading')
        file = request.files['file']
        if file:
            # 파일을 원하는 위치에 저장하고 분석 작업 수행, 분석 결과를 얻은 뒤 클라이언트에게 전달
            filename = file.filename
            ext = os.path.splitext(filename)[1].lower()[1:]
            
            if ext != 'csv' and ext != 'xlsx' and ext != 'xls':
                msg = '데이터프레임 형식의 파일(csv, xlsx)만 업로드해 주세요.'
                print(msg)
                return {'result':'error', 'message':msg}
            
            if ext == 'xls' or ext == 'xlsx':
                df = pd.read_excel(file, index_col=0)
            else:
                df = pd.read_csv(file, index_col=0)
            
            filename = get_unique_filename(filename)
            joblib.dump(df, os.path.join(app.config['UPLOAD_FOLDER'], filename + '.pkl'))
            session['df_name'] = filename
            
            # 임시 폴더에 파일 저장하기
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            base_text = f"데이터셋의 행:{df.shape[0]}, 데이터셋의 열:{df.shape[1]}<br>데이터의 종류(변수):{' '.join(df.columns.tolist())}"
              
            return {'result':'success', 'analysis_result': base_text}
        else:
            print('no file at all')
            return {'result':'error', 'message':'No file at all.'}
    else:
        return render_template('upload.html', form=form)

def get_unique_filename(filename):
    """
    만약 동일한 이름의 파일이 존재하면, 파일 이름 뒤에 숫자를 붙여서 고유한 이름을 생성합니다.
    예) file.csv -> file_1.csv, file_2.csv, ...
    """
    name, ext = os.path.splitext(filename)
    
    print('upload folder:', app.config['UPLOAD_FOLDER'])
    
    counter = 1
    while os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
        filename = f"{name}_{counter}{ext}"
        counter += 1
    
    return filename


@app.route("/ask", methods=["POST"])
def ask():
    print('submitted')
    question = request.form.get('text')
    print(question)
    
    df_name = session.get('df_name', None)
    print(df_name)
    
    if df_name is None:
        return jsonify({"result": "fail", "answer": "데이터셋이 정상적으로 업로드되지 않았습니다."})
    else:
        df = joblib.load(os.path.join(app.config['UPLOAD_FOLDER'], df_name + '.pkl'))
        print(df.head())
        agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106"), df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
        answer = agent.run(question)
    
        print(answer)
        
        return jsonify({"result": "success", "answer": answer})


@app.route('/review', methods=['GET', 'POST'])
def review():
    if request.method == 'GET':
        sel_lang = request.args.get('lang')
        print(sel_lang)

        if (sel_lang is not None):
            session['lang'] = sel_lang
            
        return render_template('review.html', lang=session.get('lang', ''))
    else:
        message = request.form.get('text')
        message = message.strip()

        tokens = word_tokenize(message)
        print('total tokens:', len(tokens))
        
        if len(tokens) >= 4000:
            print('it has too many tokens.')
            message = ' '.join(tokens[:4000])

        query = '보기:\n'
        query += message
        query += '\n'
        query += '보기의 글을 전체적으로 확인해서 교정하고, 더 나은 글을 쓸 수 있도록 유용한 피드백을 함께 제공해 줘.'
        query += '결과는 다음과 같은 양식에 맞게 출력해 줘.\n'
        query += '[교정]:{결과}\n'
        query += '[피드백]:{결과}'
        
        print(query)
        
        msg = []
        msg.append({"role": "system", "content": "You are a kind and thoughtful assistant for writing."})
        msg.append({"role": "user", "content": query})

        response = client.chat.completions.create(model=MODEL, messages=msg, temperature=TEMPERATURE)
        answer = response.choices[0].message.content
        print(answer)
        
        pos1 = answer.find('[교정]:')
        pos2 = answer.find('[피드백]:')
        
        if pos1 < pos2:
            print('upright')
            result = answer[pos1+5:pos2].strip()
            feedback = answer[pos2+6:].strip()
        else:
            print('reverse')
            result = answer[pos1+5:].strip()
            feedback = answer[pos2+6:pos1].strip()
                
        return {'result': result, 'feedback': feedback}


if __name__ == '__main__':
    session['lang'] = 'ko'
    session['gift_history'] = []
    app.run(host='0.0.0.0', port=5000, debug=True)
