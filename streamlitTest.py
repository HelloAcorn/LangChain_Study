import streamlit as st

# 애플리케이션 제목 설정
st.title('Streamlit 예제')

# 사용자 입력을 받는 텍스트 박스 생성
user_input = st.text_input("이름을 입력하세요:")

# 사용자 입력을 받는 텍스트 박스 생성
user_age = st.text_input("나이을 입력하세요:")

# 버튼을 추가하고, 클릭 시 동작 정의
if st.button('인사하기'):
    st.write(f'Hello {user_input}!')
    st.write(f'Hello {user_age}0 !')

# 스크립트를 실행하기 위해 터미널에서 'streamlit run your_script.py' 명령어를 사용하세요.
