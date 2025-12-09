import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import requests
import os
import tempfile
from pathlib import Path

# --- 페이지 설정 ---
st.set_page_config(
    page_title="🦄 유니코 AI ", 
    layout="wide", 
    page_icon="🌽",
    initial_sidebar_state="expanded"
)

# --- CSS 스타일 ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3e9d0 100%);
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d5016 0%, #3d6526 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4 {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.15);
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #a5d6a7;
    }
    
    h1 {
        color: #2d5016;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        padding: 20px 0;
        border-bottom: 3px solid #7cb342;
        margin-bottom: 30px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #7cb342 0%, #689f38 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #689f38 0%, #558b2f 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .info-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 5px solid #66bb6a;
        padding: 15px;
        margin: 15px 0;
        border-radius: 10px;
    }
    
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    [data-testid="metric-container"] {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #7cb342;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- API 키 확인 ---
if 'gemini' not in st.secrets or 'api_key' not in st.secrets['gemini']:
    st.error("❌ API 키가 필요합니다.")
    with st.container():
        st.markdown("""
        <div class="info-box">
        <h4>🔑 API 키 설정 방법</h4>
        <p>1. .streamlit/secrets.toml 파일 생성</p>
        <p>2. 아래 내용 추가:</p>
        <pre>
[gemini]
api_key = "your-gemini-api-key"

[telegram]
bot_token = "your-telegram-bot-token"
        </pre>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

GOOGLE_API_KEY = st.secrets['gemini']['api_key']
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- Telegram 봇 토큰 가져오기 ---
@st.cache_resource
def get_telegram_token():
    """Telegram 봇 토큰 가져오기"""
    try:
        if 'telegram' in st.secrets and 'bot_token' in st.secrets['telegram']:
            return st.secrets['telegram']['bot_token']
    except:
        pass
    return None

telegram_token = get_telegram_token()

# --- Telegram 메시지 전송 함수 ---
def send_telegram_message(chat_id, message):
    """텔레그램 메시지 전송"""
    try:
        if not telegram_token:
            return False, "Telegram 봇 토큰이 필요합니다"
        
        if not chat_id:
            return False, "Chat ID를 입력하세요"
        
        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            message_id = response.json().get('result', {}).get('message_id', 'Unknown')
            return True, message_id
        else:
            error_msg = response.json().get('description', 'Unknown error')
            return False, error_msg
            
    except Exception as e:
        return False, str(e)

# --- 모델 초기화 ---
@st.cache_resource
def init_models():
    """LLM과 임베딩 모델 초기화"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
        convert_system_message_to_human=True,
        max_output_tokens=2048
    )
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    return llm, embeddings

llm, embeddings = init_models()

# --- Session State 초기화 ---
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'full_text' not in st.session_state:
    st.session_state.full_text = ""
if 'search_k' not in st.session_state:
    st.session_state.search_k = 5
if 'pdf_pages' not in st.session_state:
    st.session_state.pdf_pages = 0
if 'num_chunks' not in st.session_state:
    st.session_state.num_chunks = 0
if 'user_telegram_id' not in st.session_state:
    st.session_state.user_telegram_id = ""
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'auto_loaded' not in st.session_state:
    st.session_state.auto_loaded = False

# --- 자동 PDF 로드 함수 ---
@st.cache_resource
def auto_load_pdf():
    """앱 시작 시 자동으로 PDF 로드"""
    fixed_pdf_dir = Path("fixed_pdfs")
    fixed_pdf_dir.mkdir(exist_ok=True)
    
    fixed_files = sorted([p for p in fixed_pdf_dir.glob("*.pdf")])
    
    if fixed_files:
        first_pdf = fixed_files[0]
        return first_pdf
    return None

# --- PDF 처리 함수 ---
def process_pdf(uploaded_file, embeddings):
    """PDF를 처리하고 벡터 DB 생성"""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        with st.spinner("🌾 PDF 내용을 수확하는 중..."):
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            if not documents:
                st.error("❌ PDF에서 텍스트를 추출할 수 없습니다.")
                return None, 0, "", 0
            
            pdf_pages = len(documents)
            
            total_text = ""
            for doc in documents:
                page_text = doc.page_content.strip()
                if page_text:
                    total_text += f"\n[페이지 {doc.metadata.get('page', 'Unknown')}]\n{page_text}\n"
            
            if len(total_text.strip()) < 50:
                st.error("❌ PDF에 충분한 텍스트가 없습니다.")
                return None, 0, "", 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 페이지", f"{pdf_pages}장", delta="수확 완료")
        with col2:
            st.metric("📝 글자 수", f"{len(total_text):,}자", delta="분석 준비")
        with col3:
            st.metric("✅ 상태", "추출 성공", delta="100%")
        
        with st.expander("🌱 추출된 텍스트 미리보기", expanded=False):
            st.text_area("PDF 내용", total_text[:5000], height=400, disabled=True)
            if len(total_text) > 5000:
                st.info(f"🌾 전체 {len(total_text):,}글자 중 처음 5000자만 표시")
        
        with st.spinner("🚜 문서를 분석 단위로 경작 중..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                length_function=len
            )
            splits = text_splitter.split_documents(documents)
            splits = [doc for doc in splits if len(doc.page_content.strip()) > 50]
        
        st.success(f"🌾 {len(splits)}개의 지식 단위로 분할 완료!")
        
        with st.spinner("🌻 지식 데이터베이스 파종 중..."):
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                collection_name="farm_docs"
            )
        
        st.balloons()
        st.success("🎊 문서 분석 준비 완료! 이제 질문해주세요.")
        return vectorstore, len(splits), total_text, pdf_pages
        
    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")
        return None, 0, "", 0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- RAG 체인 생성 ---
def create_rag_chain(vectorstore, llm, search_k=5):
    """농업 전문 RAG 체인 생성"""
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": search_k,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )
    
    template = """당신은 농업 및 스마트팜 전문 AI 조언자입니다. 🌱
주어진 문서를 깊이 이해하고 실용적인 농업 인사이트를 제공합니다.

📄 참고 문서:
{context}

❓ 질문: {question}

답변 지침:
1. 🌾 농업 실무에 도움이 되는 구체적인 조언 제공
2. 📊 수치, 데이터, 과학적 근거 명확히 제시
3. 🚜 실제 적용 가능한 방법 설명
4. 💡 문서 내용 + AI의 농업 지식 결합
5. ⚠️ 주의사항이나 팁도 함께 제공

답변:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        formatted = ""
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get('page', 'Unknown')
            formatted += f"\n[참고 {i} - {page}페이지]\n{doc.page_content}\n"
            formatted += "=" * 50
        return formatted
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

# --- 헤더 ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <h1 style='text-align: center;'>
        🦄 유니코 AI
    </h1>
    """, unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; padding: 10px; background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); border-radius: 15px; margin-bottom: 20px;'>
    <h3 style='color: #2d5016; margin: 0;'>🚜 농업 문서를 AI로 스마트하게 분석하세요 🌻</h3>
    <p style='color: #558b2f; margin: 5px 0;'>재배, 스마트팜, 농업 기술 문서를 깊이 있게 분석합니다</p>
</div>
""", unsafe_allow_html=True)

# --- 사이드바 ---
with st.sidebar:
    st.markdown("""
    <h2 style='color: white; text-align: center;'>
        🌿 농업 AI 컨트롤 센터
    </h2>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.expander("📱 텔레그램 설정", expanded=False):
        st.markdown("""
        <div style='background: rgba(52,152,219,0.2); padding: 10px; border-radius: 10px; margin: 10px 0;'>
            <p style='color: #2980b9; margin: 5px 0;'>
            💡 Chat ID 확인 방법:<br>
            1. Telegram에서 @userinfobot 검색<br>
            2. /start 입력<br>
            3. 표시되는 숫자가 Chat ID
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.user_telegram_id = st.text_input(
            "🤖 텔레그램 Chat ID",
            value=st.session_state.user_telegram_id,
            placeholder="7078646539",
            help="@userinfobot 에게 /start를 입력해서 Chat ID 확인"
        )
        
        if st.session_state.user_telegram_id:
            if not st.session_state.user_telegram_id.isdigit():
                st.warning("⚠️ Chat ID는 숫자만 입력하세요")
            elif len(st.session_state.user_telegram_id) < 5:
                st.warning("⚠️ Chat ID가 너무 짧습니다")
            else:
                st.success("✅ 유효한 Chat ID")
        
        if st.button("📤 테스트 메시지 전송", use_container_width=True):
            if not st.session_state.user_telegram_id:
                st.error("❌ Chat ID를 입력하세요")
            else:
                test_msg = f"🧪 유니코 농업 AI 테스트 메시지입니다.\n현재 시간: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                success, result = send_telegram_message(st.session_state.user_telegram_id, test_msg)
                if success:
                    st.success("✅ 테스트 메시지 전송 완료!")
                else:
                    st.error(f"❌ 메시지 전송 실패: {result}")
    
    st.markdown("---")
    
    with st.expander("⚙️ AI 분석 설정", expanded=True):
        st.session_state.search_k = st.slider(
            "🔍 참고 문서 깊이", 
            min_value=3, 
            max_value=10, 
            value=st.session_state.search_k,
            help="더 많은 문서를 참고하면 더 깊은 분석이 가능합니다"
        )
        
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 10px; border-radius: 10px; margin-top: 10px;'>
            <small style='color: white;'>
            💡 복잡한 농업 기술 문서는 7-10으로 설정하세요
            </small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <h3 style='color: white; text-align: center;'>
        📤 농업 문서 업로드
    </h3>
    """, unsafe_allow_html=True)
    
    fixed_pdf_dir = Path("fixed_pdfs")
    fixed_pdf_dir.mkdir(exist_ok=True)
    fixed_files = sorted([p.name for p in fixed_pdf_dir.glob("*.pdf")])
    
    if fixed_files:
        st.info(f"✅ {len(fixed_files)}개의 고정 PDF를 찾았습니다")
        
        auto_pdf_path = auto_load_pdf()
        if auto_pdf_path and not st.session_state.auto_loaded:
            st.session_state.auto_loaded = True
            data = auto_pdf_path.read_bytes()
            
            class FixedFile:
                def __init__(self, name, data):
                    self.name = name
                    self._data = data
                def getvalue(self):
                    return self._data
            
            with st.spinner(f"🚀 자동으로 '{auto_pdf_path.name}' 로드 중..."):
                st.session_state.vectorstore, st.session_state.num_chunks, st.session_state.full_text, st.session_state.pdf_pages = process_pdf(
                    FixedFile(auto_pdf_path.name, data), 
                    embeddings
                )
        
        with st.expander("🔄 다른 PDF 선택", expanded=False):
            selected_pdf = st.selectbox("📄 PDF 선택", options=fixed_files, key="pdf_selector")
            if selected_pdf:
                fp = fixed_pdf_dir / selected_pdf
                size_kb = fp.stat().st_size / 1024
                st.markdown(f"""
                <div style='background: rgba(124,179,66,0.2); padding: 15px; border-radius: 10px; margin: 10px 0;'>
                    <p style='color: white; margin: 0;'>📁 <b>{selected_pdf}</b></p>
                    <p style='color: #a5d6a7; margin: 5px 0;'>📊 크기: {size_kb:.1f} KB</p>
                </div>
                """, unsafe_allow_html=True)

                if st.button('🌾 이 PDF 로드', type='primary', use_container_width=True):
                    data = fp.read_bytes()
                    class FixedFile:
                        def __init__(self, name, data):
                            self.name = name
                            self._data = data
                        def getvalue(self):
                            return self._data

                    st.session_state.vectorstore, st.session_state.num_chunks, st.session_state.full_text, st.session_state.pdf_pages = process_pdf(
                        FixedFile(selected_pdf, data), 
                        embeddings
                    )
                    st.rerun()
    else:
        st.warning("⚠️ fixed_pdfs 폴더에 PDF 파일이 없습니다")
        st.info("📌 프로젝트 루트에 fixed_pdfs 폴더를 만들고 PDF를 넣으세요")
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "📤 다른 문서 업로드 (선택)",
        type=['pdf'],
        help="다른 농업 관련 PDF 파일을 업로드할 수 있습니다"
    )

    if uploaded_file:
        file_size = len(uploaded_file.getvalue()) / 1024
        st.markdown(f"""
        <div style='background: rgba(124,179,66,0.2); padding: 15px; border-radius: 10px; margin: 10px 0;'>
            <p style='color: white; margin: 0;'>📁 <b>{uploaded_file.name}</b></p>
            <p style='color: #a5d6a7; margin: 5px 0;'>📊 크기: {file_size:.1f} KB</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button('🌾 이 파일 분석', type='primary', use_container_width=True):
            st.session_state.vectorstore, st.session_state.num_chunks, st.session_state.full_text, st.session_state.pdf_pages = process_pdf(
                uploaded_file, 
                embeddings
            )
            st.rerun()
    
    if st.button('🔄 시스템 초기화', use_container_width=True):
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.session_state.full_text = ""
        st.session_state.pdf_pages = 0
        st.session_state.num_chunks = 0
        st.session_state.auto_loaded = False
        st.rerun()
    
    st.markdown("---")
    if st.session_state.vectorstore:
        st.markdown("""
        <div style='background: rgba(76,175,80,0.3); padding: 15px; border-radius: 10px; text-align: center;'>
            <h4 style='color: white; margin: 0;'>✅ 시스템 준비 완료</h4>
            <p style='color: #c8e6c9; margin: 5px 0;'>질문을 입력하세요</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.full_text:
            word_count = len(st.session_state.full_text.split())
            char_count = len(st.session_state.full_text)
            
            st.markdown("""
            <div style='background: rgba(255,255,255,0.2); padding: 12px; border-radius: 10px; margin-top: 10px;'>
                <p style='color: white; margin: 3px 0; font-size: 14px;'><b>📄 분석된 문서 정보</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📑 페이지", f"{st.session_state.pdf_pages}")
                st.metric("📝 문자", f"{char_count:,}")
            with col2:
                st.metric("🔍 청크", f"{st.session_state.num_chunks}")
                st.metric("📊 단어", f"{word_count:,}")
    else:
        st.markdown("""
        <div style='background: rgba(255,193,7,0.2); padding: 15px; border-radius: 10px; text-align: center;'>
            <h4 style='color: white; margin: 0;'>⏳ PDF 준비 중...</h4>
            <p style='color: #fff3cd; margin: 5px 0;'>파일을 로드하는 중입니다</p>
        </div>
        """, unsafe_allow_html=True)

# --- 메인 화면 ---
if not st.session_state.vectorstore:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3 style='color: #2d5016; text-align: center;'>🌱 스마트 농업</h3>
            <hr style='border-color: #7cb342;'>
            <ul style='color: #558b2f;'>
                <li>작물 재배 매뉴얼</li>
                <li>스마트팜 운영 가이드</li>
                <li>병충해 방제 문서</li>
                <li>토양 관리 지침서</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3 style='color: #2d5016; text-align: center;'>🚜 주요 기능</h3>
            <hr style='border-color: #7cb342;'>
            <ul style='color: #558b2f;'>
                <li>PDF 완벽 분석</li>
                <li>농업 전문 AI 답변</li>
                <li>실시간 데이터 처리</li>
                <li>텔레그램 알림</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3 style='color: #2d5016; text-align: center;'>💡 활용 예시</h3>
            <hr style='border-color: #7cb342;'>
            <ul style='color: #558b2f;'>
                <li>적정 재배 온도는?</li>
                <li>수확 시기 판단법</li>
                <li>영양분 관리 방법</li>
                <li>수익성 분석</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #fff9c4 0%, #fff59d 100%); padding: 30px; border-radius: 20px; margin: 20px 0; text-align: center;'>
        <h2 style='color: #f57c00;'>🌻 시작하기</h2>
        <p style='color: #e65100; font-size: 18px;'>
            1️⃣ fixed_pdfs 폴더에 농업 PDF 추가<br>
            2️⃣ Streamlit 앱 시작하면 자동 로드<br>
            3️⃣ AI에게 농업 관련 질문하기
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    if st.session_state.full_text:
        with st.expander("📄 전체 문서 내용 (복사 가능)", expanded=False):
            st.text_area(
                "전체 텍스트", 
                st.session_state.full_text, 
                height=300,
                help="Ctrl+A로 전체 선택 후 복사 가능합니다"
            )
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); padding: 20px; border-radius: 15px; margin: 20px 0;'>
        <h3 style='color: #2d5016; text-align: center; margin: 0 0 15px 0;'>🚀 빠른 농업 분석</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    user_question = None
    
    with col1:
        if st.button("🌾 재배법 요약", use_container_width=True):
            user_question = "이 문서의 핵심 재배 방법을 단계별로 요약해주세요."
    
    with col2:
        if st.button("🌡️ 환경 조건", use_container_width=True):
            user_question = "최적 재배 환경 조건 (온도, 습도, 광량 등)을 정리해주세요."
    
    with col3:
        if st.button("🐛 병충해 관리", use_container_width=True):
            user_question = "병충해 예방 및 방제 방법을 상세히 설명해주세요."
    
    with col4:
        if st.button("💰 수익성 분석", use_container_width=True):
            user_question = "재배 비용과 예상 수익, 경제성을 분석해주세요."
    
    st.markdown("---")
    
    for question, answer in st.session_state.chat_history:
        with st.chat_message("user", avatar="👨‍🌾"):
            st.write(question)
        with st.chat_message("assistant", avatar="🌱"):
            st.write(answer)
    
    user_input = st.chat_input("🌾 농업 관련 질문을 입력하세요...")
    
    if user_input or user_question:
        st.session_state.current_question = user_input if user_input else user_question
    
    if 'current_question' in st.session_state and st.session_state.current_question:
        question_to_process = st.session_state.current_question
        
        if any(q == question_to_process for q, _ in st.session_state.chat_history):
            question_to_process = None
    else:
        question_to_process = None
    
    if question_to_process:
        
        with st.chat_message("user", avatar="👨‍🌾"):
            st.write(question_to_process)
        
        with st.chat_message("assistant", avatar="🌱"):
            with st.spinner("🚜 문서를 분석하고 답변 생성 중..."):
                try:
                    rag_chain, retriever = create_rag_chain(
                        st.session_state.vectorstore, 
                        llm,
                        st.session_state.search_k
                    )
                    
                    response = rag_chain.invoke(question_to_process)
                    st.write(response)
                    
                    st.session_state.chat_history.append((question_to_process, response))
                    
                    with st.expander(f"🔍 참고한 문서 부분 ({st.session_state.search_k}개)"):
                        docs = retriever.invoke(question_to_process)
                        for i, doc in enumerate(docs, 1):
                            page_num = doc.metadata.get('page', '?')
                            st.markdown(f"""
                            <div style='background: #f1f8e9; padding: 10px; border-radius: 10px; margin: 10px 0;'>
                                <h4 style='color: #33691e;'>[참고 {i}] 📄 {page_num}페이지</h4>
                                <p style='color: #558b2f;'>{doc.page_content[:500]}...</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ 오류: {str(e)}")
                    st.info("💡 다른 질문을 시도해보세요.")
    
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); padding: 20px; border-radius: 15px;'>
            <h3 style='color: #2d5016; text-align: center;'>📱 AI 답변을 텔레그램으로 전송</h3>
            <p style='color: #558b2f; text-align: center;'>마지막 답변을 텔레그램으로 보내세요</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            send_telegram_chat = st.text_input(
                "🤖 텔레그램 Chat ID",
                placeholder="123456789",
                help="@userinfobot 에게 /start 입력하면 Chat ID 확인",
                key="send_telegram_id"
            )
        
        with col2:
            if st.button("📤 텔레그램으로 전송", type='primary', use_container_width=True):
                if not send_telegram_chat:
                    st.error("❌ Chat ID를 입력하세요")
                elif not send_telegram_chat.isdigit():
                    st.error("❌ Chat ID는 숫자만 입력하세요")
                else:
                    if st.session_state.chat_history:
                        last_question, last_answer = st.session_state.chat_history[-1]
                        
                        message = f"""<b>🌾 농업 AI 답변</b>

<b>❓ 질문:</b>
{last_question}

<b>💡 답변:</b>
{last_answer}"""
                        
                        success, result = send_telegram_message(send_telegram_chat, message)
                        
                        if success:
                            st.success("✅ 텔레그램으로 전송 완료! 📱")
                            st.balloons()
                        else:
                            st.error(f"❌ 전송 실패: {result}")
                    else:
                        st.error("❌ 답변이 없습니다")

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%); border-radius: 15px;'>
    <h4 style='color: #1b5e20; margin: 0;'>🦄 UNICO AI</h4>
    <p style='color: #2e7d32; margin: 5px 0;'>Powered by Google Gemini 2.0, LangChain & Telegram</p>
    <p style='color: #388e3c; margin: 5px 0;'>🚜 농업의 디지털 혁신을 선도합니다</p>
</div>
""", unsafe_allow_html=True)
