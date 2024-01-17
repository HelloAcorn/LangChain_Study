import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from st_chat_message import message
from PIL import Image
import base64
import io
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import json

# 이미지 파일 로드
image_path = "/Users/icarus/Library/CloudStorage/OneDrive-koreatech.ac.kr/Python/solo_study/project/grad_sup/ph.jpeg"

# 이미지 로드 및 인코딩 함수
def load_and_encode_image(image_path):
    with Image.open(image_path) as image:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

# UI 설정
image_str = load_and_encode_image(image_path=image_path)
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 50px;
    }
    </style>
    <div class="title">💬 Koreatech_GPT</div>
    """, unsafe_allow_html=True)
st.write("---")
img_html = f'<div style="text-align: center;"><img src="data:image/jpeg;base64,{image_str}" style="width:auto;"/></div>'
st.markdown(img_html, unsafe_allow_html=True)


# 환경 변수 로드
load_dotenv()
persist_directory = 'db'
embedding = OpenAIEmbeddings()

def load_and_split_documents(file_path):
    try:
        # PDF 파일 로드
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader()
            documents = loader.load(file_path)
        
        # JSON 파일 로드
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                # JSON 파일에서 'content' 키의 값을 추출하여 리스트로 변환
                documents = [json_data['content']]
        
        # 일반 텍스트 파일 로드
        else:
            loader = DirectoryLoader('./test', glob="*.txt", loader_cls=TextLoader)
            documents = loader.load()
        
        # 문서 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        return texts
    
    except Exception as e:
        st.error(f"문서 로딩 중 오류 발생: {e}")
        return []

texts = load_and_split_documents('/Users/icarus/Library/CloudStorage/OneDrive-koreatech.ac.kr/Python/solo_study/project/grad_sup/test')  # 문서 경로 지정

# 벡터 데이터베이스 구성
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding,
    persist_directory=persist_directory)

vectordb = Chroma(
    persist_directory = persist_directory, 
    embedding_function=embedding)

# 검색 기능 설정
retriever = vectordb.as_retriever(search_kwargs={"k":2})

prompt_template = """

Of course! The modified prompt template would be:

You are a knowledgeable and real-time AI chatbot assistant for Koreatech. Your primary role is to deliver precise and useful information about the university's educational programs, policies, campus life, and various services. It's crucial to formulate your responses to be thorough, well-researched, and considerate. If uncertain about an answer, candidly admit your lack of information and recommend contacting the relevant university department for further clarification. Remember, it's 2023 November 28th.

Now, with this context, answer the following query in Korean: {context}

Question: {question}

Answer in Korean:


"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

llm = ChatOpenAI(
    temperature=1,              
    max_tokens=2048,            
    model_name='gpt-4-0613' 
)

# RetrievalQA 체인 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
)

def process_llm_response(llm_response):
    if isinstance(llm_response, dict) and "source_documents" in llm_response:
        sources = "\n".join([source.metadata['source'] for source in llm_response["source_documents"]])
        return f"{llm_response.get('result', '죄송합니다. 답변을 생성할 수 없습니다.')}\n\n관련 소스:\n{sources}"
    else:
        return "죄송합니다. 답변을 생성할 수 없습니다."


st.header("Koreatech GPT에게 질문해보세요!!")

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful AI assistant for Koreatech")
    ]
    st.session_state.messages.append(AIMessage(content="안녕하세요, Koreatech-GPT입니다. 무엇을 도와드릴까요?"))

# 사용자 입력 필드
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''

user_input = st.text_input("질문을 입력하세요: ", key="user_input")

# 사용자 입력 처리
if user_input:
    # 사용자 메시지를 세션 상태에 추가
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # 챗봇 응답 생성
    with st.spinner("답변 생성 중..."):
        result = qa_chain({"query": user_input})
        response_t = process_llm_response(result)
    
    # 챗봇 응답을 세션 상태에 추가
    st.session_state.messages.append(AIMessage(content=response_t))

# 대화 이력 표시
for i, msg in enumerate(st.session_state.messages):
    if isinstance(msg, HumanMessage):
        message(msg.content, is_user=True, key=f"message_{i}_user")
    elif isinstance(msg, AIMessage):
        message(msg.content, is_user=False, key=f"message_{i}_ai")