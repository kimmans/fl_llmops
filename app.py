import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os
from typing import Dict, Any, Optional
import requests
import time
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# API 클라이언트 import
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# 페이지 설정
st.set_page_config(
    page_title="프롬프트 LLMOps Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .result-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 20px 0;
    }
    
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# 모델별 가격 정보 (USD per 1M tokens)
MODEL_PRICING = {
    # OpenAI
    "GPT-4o": {"input": 2.50, "output": 10.00},
    "GPT-4o-mini": {"input": 0.15, "output": 0.60},
    
    # Claude (Anthropic)
    "Claude Sonnet 4": {"input": 3.00, "output": 15.00},
    "Claude Sonnet 3.5": {"input": 3.00, "output": 15.00},
    "Claude Haiku 3.5": {"input": 0.25, "output": 1.25},
    
    # Perplexity
    "Perplexity Sonar": {"input": 0.20, "output": 0.20},
    "Perplexity Sonar Pro": {"input": 1.00, "output": 1.00},
    
    # Gemini
    "Gemini 1.5 Flash": {"input": 0.075, "output": 0.30},
    "Gemini 1.5 Pro": {"input": 1.25, "output": 5.00},
    "Gemini 2.0 Flash": {"input": 0.10, "output": 0.40}
}

def truncate_text(text: str, max_tokens: int = 100000) -> str:
    """텍스트를 토큰 한도에 맞춰 자르기 (대략적 계산)"""
    # 대략적으로 1토큰 = 4문자로 계산
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    
    # 텍스트를 자르되, 문장 단위로 자르려고 시도
    truncated = text[:max_chars]
    
    # 마지막 완전한 문장까지만 포함
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n')
    
    cut_point = max(last_period, last_newline)
    if cut_point > max_chars * 0.8:  # 80% 이상이면 문장 단위로 자름
        truncated = truncated[:cut_point + 1]
    
    return truncated + "\n\n[... 텍스트가 토큰 한도로 인해 잘렸습니다 ...]"

def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """모델별 비용 계산"""
    if model_name not in MODEL_PRICING:
        return 0.0
    
    pricing = MODEL_PRICING[model_name]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost

# 세션 상태 초기화
if 'results_history' not in st.session_state:
    st.session_state.results_history = []

if 'current_result' not in st.session_state:
    st.session_state.current_result = None

# API 키 확인 및 클라이언트 초기화
def check_api_keys():
    """API 키 존재 여부 확인"""
    keys_status = {
        "OpenAI": bool(os.getenv("OPENAI_API_KEY")),
        "Anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "Gemini": bool(os.getenv("GEMINI_API_KEY")),
        "Perplexity": bool(os.getenv("PERPLEXITY_API_KEY"))
    }
    return keys_status

def init_clients():
    """API 클라이언트 초기화"""
    clients = {}
    
    # OpenAI 클라이언트
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            clients['openai'] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            st.error(f"OpenAI 클라이언트 초기화 실패: {e}")
    
    # Anthropic 클라이언트
    if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
        try:
            clients['anthropic'] = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        except Exception as e:
            st.error(f"Anthropic 클라이언트 초기화 실패: {e}")
    
    # Google 클라이언트
    if GOOGLE_AVAILABLE and os.getenv("GEMINI_API_KEY"):
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            clients['google'] = genai.GenerativeModel('gemini-1.5-flash')  # 기본 모델 변경
        except Exception as e:
            st.error(f"Gemini 클라이언트 초기화 실패: {e}")
    
    return clients

# API 클라이언트 초기화
clients = init_clients()

# 실제 API 호출 함수들
def call_openai(prompt: str, data: str, temperature: float, model_name: str) -> Dict[str, Any]:
    """OpenAI GPT API 실제 호출"""
    if 'openai' not in clients:
        return {
            "model": model_name,
            "response": "❌ OpenAI API 키가 설정되지 않았거나 openai 패키지가 설치되지 않았습니다.",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temperature,
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "error": True
        }
    
    try:
        # 모델명 매핑 (GPT-4.1 제거)
        model_mapping = {
            "GPT-4o": "gpt-4o",                # GPT-4o 모델
            "GPT-4o-mini": "gpt-4o-mini"       # GPT-4o-mini 모델
        }
        
        api_model = model_mapping.get(model_name, "gpt-4o-mini")
        
        # 모델별 토큰 한도 설정
        model_token_limits = {
            "gpt-4o": 128000,          # 128k 토큰 컨텍스트  
            "gpt-4o-mini": 128000      # 128k 토큰 컨텍스트
        }
        
        max_input_tokens = model_token_limits.get(api_model, 128000) - 4096  # 출력용 토큰 예약
        
        # 데이터 크기 확인 및 자르기
        data_truncated = truncate_text(data, max_input_tokens // 2)  # 절반은 데이터, 절반은 프롬프트+응답용
        prompt_truncated = truncate_text(prompt, 4000)  # 프롬프트는 4000토큰으로 제한
        
        # 프롬프트와 데이터 결합
        full_prompt = f"데이터: {data_truncated}\n\n요청: {prompt_truncated}"
        
        # 입력 토큰 수 대략 계산
        estimated_input_tokens = len(full_prompt) // 4
        
        if estimated_input_tokens > max_input_tokens:
            return {
                "model": model_name,
                "response": f"❌ 입력 데이터가 너무 큽니다. 예상 토큰: {estimated_input_tokens:,}, 한도: {max_input_tokens:,}. 더 작은 파일을 사용하거나 데이터를 요약해주세요.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "temperature": temperature,
                "tokens_used": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
                "error": True
            }
        
        response = clients['openai'].chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=temperature,
            max_tokens=min(int(os.getenv("DEFAULT_MAX_TOKENS", 2000)), 4000),  # 출력 토큰 제한
            timeout=60  # 60초 타임아웃 설정
        )
        
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        cost = calculate_cost(model_name, input_tokens, output_tokens)
        
        response_text = response.choices[0].message.content
        
        # 데이터가 잘렸으면 알림 추가
        if len(data) != len(data_truncated):
            response_text = f"⚠️ 입력 데이터가 토큰 한도로 인해 잘렸습니다.\n\n{response_text}"
        
        return {
            "model": model_name,
            "response": response_text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temperature,
            "tokens_used": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "error": False
        }
        
    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower():
            error_msg = f"⏱️ 요청 시간 초과: {error_msg}\n\n💡 해결 방법:\n- 잠시 후 다시 시도\n- 입력 데이터 크기 줄이기\n- 다른 모델 사용해보기"
        
        return {
            "model": model_name,
            "response": f"❌ API 호출 오류: {error_msg}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temperature,
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "error": True
        }

def call_claude(prompt: str, data: str, temperature: float, model_name: str) -> Dict[str, Any]:
    """Claude API 실제 호출"""
    if 'anthropic' not in clients:
        return {
            "model": model_name,
            "response": "❌ Anthropic API 키가 설정되지 않았거나 anthropic 패키지가 설치되지 않았습니다.",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temperature,
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "error": True
        }
    
    try:
        # 모델명 매핑
        model_mapping = {
            "Claude Sonnet 4": "claude-3-5-sonnet-20241022",
            "Claude Sonnet 3.5": "claude-3-5-sonnet-20240620",
            "Claude Haiku 3.5": "claude-3-5-haiku-20241022"
        }
        
        api_model = model_mapping.get(model_name, "claude-3-5-sonnet-20241022")
        
        full_prompt = f"데이터: {data}\n\n요청: {prompt}"
        
        response = clients['anthropic'].messages.create(
            model=api_model,
            max_tokens=int(os.getenv("DEFAULT_MAX_TOKENS", 2000)),
            temperature=temperature,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens
        cost = calculate_cost(model_name, input_tokens, output_tokens)
        
        return {
            "model": model_name,
            "response": response.content[0].text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temperature,
            "tokens_used": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "error": False
        }
        
    except Exception as e:
        return {
            "model": model_name,
            "response": f"❌ API 호출 오류: {str(e)}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temperature,
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "error": True
        }

def call_perplexity(prompt: str, data: str, temperature: float, model_name: str) -> Dict[str, Any]:
    """Perplexity API 실제 호출"""
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return {
            "model": model_name,
            "response": "❌ Perplexity API 키가 설정되지 않았습니다.",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temperature,
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "error": True
        }
    
    try:
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 모델명 매핑
        model_mapping = {
            "Perplexity Sonar": "llama-3.1-sonar-small-128k-online",
            "Perplexity Sonar Pro": "llama-3.1-sonar-large-128k-online"
        }
        
        api_model = model_mapping.get(model_name, "llama-3.1-sonar-small-128k-online")
        
        full_prompt = f"데이터: {data}\n\n요청: {prompt}"
        
        payload = {
            "model": api_model,
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "temperature": temperature,
            "max_tokens": int(os.getenv("DEFAULT_MAX_TOKENS", 2000))
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=60)  # 60초로 증가
        response.raise_for_status()
        
        result = response.json()
        
        input_tokens = result.get('usage', {}).get('prompt_tokens', 0)
        output_tokens = result.get('usage', {}).get('completion_tokens', 0)
        total_tokens = result.get('usage', {}).get('total_tokens', input_tokens + output_tokens)
        cost = calculate_cost(model_name, input_tokens, output_tokens)
        
        return {
            "model": model_name,
            "response": result['choices'][0]['message']['content'],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temperature,
            "tokens_used": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "error": False
        }
        
    except Exception as e:
        return {
            "model": model_name,
            "response": f"❌ API 호출 오류: {str(e)}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temperature,
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "error": True
        }

def call_gemini(prompt: str, data: str, temperature: float, model_name: str) -> Dict[str, Any]:
    """Gemini API 실제 호출"""
    if 'google' not in clients:
        return {
            "model": model_name,
            "response": "❌ Gemini API 키가 설정되지 않았거나 google-generativeai 패키지가 설치되지 않았습니다.",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temperature,
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "error": True
        }
    
    try:
        # 모델명 매핑
        model_mapping = {
            "Gemini 1.5 Flash": "gemini-1.5-flash",
            "Gemini 1.5 Pro": "gemini-1.5-pro",
            "Gemini 2.0 Flash": "gemini-2.0-flash-exp"
        }
        
        api_model = model_mapping.get(model_name, "gemini-1.5-flash")
        
        # 새로운 모델 클라이언트 생성
        try:
            model_client = genai.GenerativeModel(api_model)
        except Exception:
            # 모델이 존재하지 않으면 기본 모델 사용
            model_client = genai.GenerativeModel("gemini-1.5-flash")
        
        full_prompt = f"데이터: {data}\n\n요청: {prompt}"
        
        # Generation config 설정
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=int(os.getenv("DEFAULT_MAX_TOKENS", 2000))
        )
        
        response = model_client.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        # 응답 텍스트 확인
        response_text = ""
        if response and hasattr(response, 'text') and response.text:
            response_text = response.text
        elif response and hasattr(response, 'candidates') and response.candidates:
            # candidates에서 텍스트 추출 시도
            try:
                response_text = response.candidates[0].content.parts[0].text
            except (IndexError, AttributeError):
                response_text = "응답을 생성할 수 없습니다."
        else:
            response_text = "응답을 생성할 수 없습니다."
        
        # 토큰 사용량 계산 (근사치)
        input_tokens = len(full_prompt.split()) * 1.3  # 단어 수 × 1.3 (근사치)
        output_tokens = len(response_text.split()) * 1.3 if response_text else 0
        total_tokens = int(input_tokens + output_tokens)
        cost = calculate_cost(model_name, int(input_tokens), int(output_tokens))
        
        return {
            "model": model_name,
            "response": response_text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temperature,
            "tokens_used": total_tokens,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "cost": cost,
            "error": False
        }
        
    except Exception as e:
        return {
            "model": model_name,
            "response": f"❌ API 호출 오류: {str(e)}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temperature,
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "error": True
        }

# 메인 헤더
st.markdown('<h1 class="main-header">🚀 프롬프트 LLMOps Dashboard</h1>', unsafe_allow_html=True)

# API 키 상태 확인
api_status = check_api_keys()

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 모델 설정")
    
    # 사용 가능한 모델만 선택지에 표시 (GPT-4.1 제거)
    available_models = []
    if api_status["OpenAI"]:
        available_models.extend(["GPT-4o", "GPT-4o-mini"])
    if api_status["Anthropic"]:
        available_models.extend(["Claude Sonnet 4", "Claude Sonnet 3.5", "Claude Haiku 3.5"])
    if api_status["Perplexity"]:
        available_models.extend(["Perplexity Sonar", "Perplexity Sonar Pro"])
    if api_status["Gemini"]:
        available_models.extend(["Gemini 1.5 Flash", "Gemini 1.5 Pro", "Gemini 2.0 Flash"])
    
    if not available_models:
        available_models = ["데모 모드"]
        st.error("사용 가능한 모델이 없습니다. API 키를 확인해주세요.")
    
    # 모델 선택
    model_choice = st.selectbox(
        "🤖 AI 모델 선택",
        available_models,
        help="사용할 AI 모델을 선택하세요"
    )
    
    # 템퍼처 조정
    temperature = st.slider(
        "🌡️ Temperature (창의성)",
        min_value=0.0,
        max_value=1.0,
        value=float(os.getenv("DEFAULT_TEMPERATURE", 0.7)),
        step=0.1,
        help="낮을수록 일관된 답변, 높을수록 창의적 답변"
    )
    
    # 현재 설정 표시
    st.markdown("---")
    st.markdown("### 📊 현재 설정")
    st.info(f"**모델**: {model_choice}\n**창의성**: {temperature}")
    
    # 통계
    if st.session_state.results_history:
        st.markdown("### 📈 세션 통계")
        total_queries = len(st.session_state.results_history)
        st.metric("총 질의 수", total_queries)
        
        models_used = [r['model'] for r in st.session_state.results_history]
        most_used = max(set(models_used), key=models_used.count)
        st.metric("최다 사용 모델", most_used)
        
        # 성공/실패 통계
        successful_queries = len([r for r in st.session_state.results_history if not r.get('error', False)])
        success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
        st.metric("성공률", f"{success_rate:.1f}%")

# 메인 콘텐츠 영역
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 입력 섹션")
    
    # 데이터 입력 방식 선택
    input_method = st.radio(
        "데이터 입력 방식",
        ["텍스트 직접 입력", "파일 업로드"],
        horizontal=True
    )
    
    data_input = ""
    
    if input_method == "텍스트 직접 입력":
        data_input = st.text_area(
            "📄 데이터 입력",
            height=300,  # 150 → 300으로 증가
            placeholder="분석하고 싶은 데이터나 텍스트를 입력하세요..."
        )
    else:
        allowed_types = os.getenv("ALLOWED_FILE_TYPES", "txt,csv,json,md").split(",")
        uploaded_file = st.file_uploader(
            "📎 파일 업로드",
            type=allowed_types,
            help=f"지원 파일 형식: {', '.join(allowed_types)}"
        )
        
        if uploaded_file is not None:
            try:
                # 파일 크기 확인
                max_size = int(os.getenv("MAX_FILE_SIZE_MB", 10)) * 1024 * 1024
                if uploaded_file.size > max_size:
                    st.error(f"❌ 파일 크기가 {os.getenv('MAX_FILE_SIZE_MB', 10)}MB를 초과합니다.")
                    data_input = ""
                else:
                    # 파일 포인터를 처음으로 되돌리기
                    uploaded_file.seek(0)
                    
                    # 파일 읽기 시도
                    file_content = None
                    
                    # 파일 확장자 기반 처리
                    file_extension = uploaded_file.name.lower().split('.')[-1]
                    
                    if file_extension == 'txt' or uploaded_file.type == "text/plain":
                        try:
                            bytes_data = uploaded_file.getvalue()
                            file_content = bytes_data.decode('utf-8')
                        except UnicodeDecodeError:
                            try:
                                file_content = bytes_data.decode('utf-8', errors='replace')
                            except:
                                file_content = bytes_data.decode('latin-1', errors='ignore')
                                
                    elif file_extension == 'csv' or uploaded_file.type == "text/csv":
                        try:
                            # CSV 파일을 DataFrame으로 읽기
                            df = pd.read_csv(uploaded_file, encoding='utf-8')
                            file_content = df.to_string(index=False)
                        except UnicodeDecodeError:
                            uploaded_file.seek(0)
                            df = pd.read_csv(uploaded_file, encoding='latin-1')
                            file_content = df.to_string(index=False)
                        except Exception:
                            uploaded_file.seek(0)
                            try:
                                df = pd.read_csv(uploaded_file, encoding='cp949')  # 한글 인코딩
                                file_content = df.to_string(index=False)
                            except:
                                st.error("❌ CSV 파일을 읽을 수 없습니다. 파일 형식을 확인해주세요.")
                                file_content = None
                                
                    elif file_extension == 'json' or uploaded_file.type == "application/json":
                        try:
                            bytes_data = uploaded_file.getvalue()
                            json_str = bytes_data.decode('utf-8')
                            json_data = json.loads(json_str)
                            file_content = json.dumps(json_data, indent=2, ensure_ascii=False)
                        except UnicodeDecodeError:
                            try:
                                json_str = bytes_data.decode('utf-8', errors='replace')
                                json_data = json.loads(json_str)
                                file_content = json.dumps(json_data, indent=2, ensure_ascii=False)
                            except:
                                st.error("❌ JSON 파일을 읽을 수 없습니다.")
                                file_content = None
                        except json.JSONDecodeError:
                            st.error("❌ 유효하지 않은 JSON 형식입니다.")
                            file_content = None
                            
                    elif file_extension == 'md':
                        try:
                            bytes_data = uploaded_file.getvalue()
                            file_content = bytes_data.decode('utf-8')
                        except UnicodeDecodeError:
                            try:
                                file_content = bytes_data.decode('utf-8', errors='replace')
                            except:
                                file_content = bytes_data.decode('latin-1', errors='ignore')
                                
                    else:
                        # 기본적으로 텍스트로 읽기 시도
                        try:
                            bytes_data = uploaded_file.getvalue()
                            file_content = bytes_data.decode('utf-8')
                        except UnicodeDecodeError:
                            try:
                                file_content = bytes_data.decode('utf-8', errors='replace')
                            except:
                                file_content = bytes_data.decode('latin-1', errors='ignore')
                    
                    # 결과 처리
                    if file_content and file_content.strip():
                        data_input = file_content
                        st.success(f"✅ {uploaded_file.name} 파일이 성공적으로 로드되었습니다!")
                        st.info(f"📊 파일 크기: {len(data_input):,} 문자")
                        
                        # 파일 내용 미리보기
                        with st.expander("📄 파일 내용 미리보기"):
                            preview_text = data_input[:1000] + ("..." if len(data_input) > 1000 else "")
                            st.text_area(
                                "파일 내용",
                                value=preview_text,
                                height=300,  # 200 → 300으로 증가
                                disabled=True,
                                key="file_preview"
                            )
                            
                        # 토큰 사용량 경고
                        estimated_tokens = len(data_input) // 4
                        if estimated_tokens > 100000:
                            st.warning(f"⚠️ 큰 파일입니다! 예상 토큰: {estimated_tokens:,}개. API 한도 초과 시 자동으로 잘립니다.")
                        elif estimated_tokens > 50000:
                            st.info(f"📊 중간 크기 파일입니다. 예상 토큰: {estimated_tokens:,}개")
                    else:
                        st.error("❌ 파일 내용이 비어있거나 읽을 수 없습니다.")
                        data_input = ""
                    
            except Exception as e:
                st.error(f"❌ 파일 읽기 오류: {str(e)}")
                st.error(f"🔍 디버그 정보: 파일명={uploaded_file.name}, 크기={uploaded_file.size}, 타입={uploaded_file.type}")
                data_input = ""
    
    # 프롬프트 입력
    prompt_input = st.text_area(
        "💡 프롬프트 입력",
        height=400,  
        placeholder="AI에게 요청할 작업을 구체적으로 입력하세요..."
    )
    
    # 실행 버튼
    can_execute = (
        bool(data_input and data_input.strip()) and 
        bool(prompt_input and prompt_input.strip()) and 
        model_choice != "데모 모드" and
        any(api_status.values())
    )
    
    # 디버그 정보 (개발용 - 나중에 제거 가능)
    if os.getenv("DEBUG_MODE", "False").lower() == "true":
        st.write(f"Debug - data_input 길이: {len(data_input) if data_input else 0}")
        st.write(f"Debug - prompt_input 길이: {len(prompt_input) if prompt_input else 0}")
        st.write(f"Debug - model_choice: {model_choice}")
        st.write(f"Debug - can_execute: {can_execute}")
    
    execute_button = st.button(
        "🚀 프롬프트 실행",
        type="primary",
        use_container_width=True,
        disabled=not can_execute
    )
    
    # 버튼이 비활성화된 이유 표시
    if not can_execute:
        if not data_input or not data_input.strip():
            st.warning("⚠️ 데이터를 입력하거나 파일을 업로드해주세요.")
        elif not prompt_input or not prompt_input.strip():
            st.warning("⚠️ 프롬프트를 입력해주세요.")
        elif model_choice == "데모 모드":
            st.warning("⚠️ 사용 가능한 모델이 없습니다.")
        elif not any(api_status.values()):
            st.warning("⚠️ API 키를 설정해주세요.")

with col2:
    st.header("📊 결과 섹션")
    
    if execute_button and can_execute:
        with st.spinner("AI가 응답을 생성 중입니다..."):
            # 모델별 API 호출 (GPT-4.1 제거)
            model_functions = {
                "GPT-4o": lambda p, d, t: call_openai(p, d, t, "GPT-4o"),
                "GPT-4o-mini": lambda p, d, t: call_openai(p, d, t, "GPT-4o-mini"),
                "Claude Sonnet 4": lambda p, d, t: call_claude(p, d, t, "Claude Sonnet 4"),
                "Claude Sonnet 3.5": lambda p, d, t: call_claude(p, d, t, "Claude Sonnet 3.5"),
                "Claude Haiku 3.5": lambda p, d, t: call_claude(p, d, t, "Claude Haiku 3.5"),
                "Perplexity Sonar": lambda p, d, t: call_perplexity(p, d, t, "Perplexity Sonar"),
                "Perplexity Sonar Pro": lambda p, d, t: call_perplexity(p, d, t, "Perplexity Sonar Pro"),
                "Gemini 1.5 Flash": lambda p, d, t: call_gemini(p, d, t, "Gemini 1.5 Flash"),
                "Gemini 1.5 Pro": lambda p, d, t: call_gemini(p, d, t, "Gemini 1.5 Pro"),
                "Gemini 2.0 Flash": lambda p, d, t: call_gemini(p, d, t, "Gemini 2.0 Flash")
            }
            
            # 실제 API 호출
            result = model_functions[model_choice](prompt_input, data_input, temperature)
            result['prompt'] = prompt_input
            result['data_preview'] = data_input[:200] + "..." if len(data_input) > 200 else data_input
            
            # 현재 결과 저장
            st.session_state.current_result = result
            
            # 히스토리에 추가
            st.session_state.results_history.append(result)
    
    # 현재 결과 표시
    if st.session_state.current_result:
        result = st.session_state.current_result
        
        # 에러 여부에 따른 스타일링
        if result.get('error', False):
            container_class = "error-box"
        else:
            container_class = "result-container"
        
        # 결과 컨테이너
        if result.get('error', False):
            st.markdown(f"""
            <div class="{container_class}">
                <h3>❌ 오류 발생</h3>
                <p><strong>모델:</strong> {result['model']}</p>
                <p><strong>시간:</strong> {result['timestamp']}</p>
                <p><strong>온도:</strong> {result['temperature']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="{container_class}">
                <h3>🎯 최신 결과</h3>
                <p><strong>모델:</strong> {result['model']}</p>
                <p><strong>시간:</strong> {result['timestamp']}</p>
                <p><strong>온도:</strong> {result['temperature']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 응답 내용
        st.markdown("### 💬 AI 응답")
        if result.get('error', False):
            st.error(result['response'])
        else:
            # 응답을 더 보기 좋게 표시
            st.text_area(
                "응답 내용",
                value=result['response'],
                height=400,
                disabled=True,
                key="main_response"
            )
        
        # 메트릭 표시
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        
        with col_metric1:
            st.metric(
                "토큰 사용량",
                result.get('tokens_used', 0),
                delta=None
            )
        
        with col_metric2:
            st.metric(
                "응답 길이",
                len(result['response']),
                delta=None
            )
        
        with col_metric3:
            cost = result.get('cost', 0.0)
            st.metric(
                "예상 비용",
                f"${cost:.6f}" if cost > 0 else "$0.000000",
                delta=None
            )

# 하단 섹션 - 누적 기록
st.markdown("---")
st.header("📚 누적 기록")

if st.session_state.results_history:
    # 기록 필터링 옵션
    col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)
    
    with col_filter1:
        filter_model = st.selectbox(
            "모델 필터",
            ["전체"] + list(set([r['model'] for r in st.session_state.results_history]))
        )
    
    with col_filter2:
        show_count = st.number_input(
            "표시할 기록 수",
            min_value=1,
            max_value=len(st.session_state.results_history),
            value=min(5, len(st.session_state.results_history))
        )
    
    with col_filter3:
        # CSV 다운로드 버튼
        df_export = pd.DataFrame(st.session_state.results_history)
        
        # 내보낼 데이터 정리
        export_data = []
        for record in st.session_state.results_history:
            export_row = {
                '시간': record.get('timestamp', ''),
                '모델': record.get('model', ''),
                '프롬프트': record.get('prompt', ''),
                '응답': record.get('response', ''),
                '온도': record.get('temperature', 0),
                '토큰_사용량': record.get('tokens_used', 0),
                '입력_토큰': record.get('input_tokens', 0),
                '출력_토큰': record.get('output_tokens', 0),
                '예상_비용_USD': record.get('cost', 0),
                '상태': '오류' if record.get('error', False) else '성공',
                '데이터_미리보기': record.get('data_preview', '')
            }
            export_data.append(export_row)
        
        df_export = pd.DataFrame(export_data)
        
        # 여러 확실한 다운로드 옵션 제공
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            # 1. JSON 다운로드 (가장 확실)
            json_data = df_export.to_json(orient='records', force_ascii=False, indent=2)
            
            st.download_button(
                label="📄 JSON 다운로드",
                data=json_data,
                file_name=f"llmops_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="JSON 형식 - 메모장에서 열어서 확인 가능"
            )
            
            # 2. 파이프 구분 TXT (Excel 가져오기용)
            pipe_data = df_export.to_csv(index=False, sep='|', encoding='utf-8')
            
            st.download_button(
                label="📊 파이프 구분 TXT",
                data=pipe_data,
                file_name=f"llmops_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}_pipe.txt",
                mime="text/plain",
                help="파이프(|)로 구분 - Excel에서 '데이터 가져오기' 사용"
            )
        
        with col_download2:
            # 3. HTML 다운로드 (브라우저에서 복사 붙여넣기용)
            html_simple = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LLMOps History</title>
</head>
<body>
    <h1>LLMOps Dashboard 기록</h1>
    <p>생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    {df_export.to_html(index=False, escape=False)}
</body>
</html>"""
            
            st.download_button(
                label="🌐 HTML 다운로드",
                data=html_simple,
                file_name=f"llmops_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                help="HTML 파일 - 브라우저에서 열어서 표를 복사 가능"
            )
            
            # 4. 이메일 친화적 텍스트
            email_text = f"""LLMOps Dashboard 기록
생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

"""
            for i, row in df_export.iterrows():
                email_text += f"[기록 {i+1}]\n"
                for col, value in row.items():
                    email_text += f"• {col}: {value}\n"
                email_text += "\n" + "-"*40 + "\n\n"
            
            st.download_button(
                label="📧 이메일용 TXT",
                data=email_text,
                file_name=f"llmops_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}_email.txt",
                mime="text/plain",
                help="이메일 본문에 붙여넣기 가능한 형식"
            )
        
        # 사용법 안내
        st.markdown("---")
        st.markdown("### 💡 **추천 사용법**")
        st.info("""
**1. JSON 다운로드** → 메모장으로 열어서 내용 확인 ✅

**2. HTML 다운로드** → 크롬에서 열어서 표를 드래그로 선택 → 복사 → Excel에 붙여넣기 ✅

**3. 파이프 구분 TXT** → Excel에서 '데이터' → '텍스트/CSV에서' → 구분기호를 파이프(|)로 설정 ✅

**4. 이메일용 TXT** → 메모장에서 열어서 읽기 편한 형식 ✅
        """)
        
        # 브라우저에서 바로 보기 (복사용)
        with st.expander("🖥️ 브라우저에서 바로 보기 (복사용)"):
            st.dataframe(df_export, use_container_width=True)
            st.markdown("**위 표를 드래그로 선택 → 복사 → Excel에 붙여넣기**")
    
    with col_filter4:
        if st.button("🗑️ 기록 초기화"):
            st.session_state.results_history = []
            st.session_state.current_result = None
            st.rerun()
    
    # 필터링된 기록 표시
    filtered_history = st.session_state.results_history
    if filter_model != "전체":
        filtered_history = [r for r in filtered_history if r['model'] == filter_model]
    
    # 최신 기록부터 표시
    filtered_history = filtered_history[-show_count:][::-1]
    
    # 기록을 탭으로 표시
    if filtered_history:
        for i, record in enumerate(filtered_history):
            status_icon = "❌" if record.get('error', False) else "✅"
            with st.expander(f"{status_icon} {record['timestamp']} - {record['model']}", expanded=(i==0)):
                col_record1, col_record2 = st.columns([2, 1])
                
                with col_record1:
                    st.markdown("**프롬프트:**")
                    st.text(record.get('prompt', 'N/A'))
                    
                    st.markdown("**응답:**")
                    if record.get('error', False):
                        st.error(record['response'])
                    else:
                        st.text_area(
                            "응답 내용",
                            value=record['response'],
                            height=200,
                            key=f"response_{i}",
                            disabled=True
                        )
                
                with col_record2:
                    st.markdown("**설정 정보**")
                    record_info = {
                        "모델": record['model'],
                        "온도": record['temperature'],
                        "토큰": record.get('tokens_used', 0),
                        "시간": record['timestamp'],
                        "상태": "오류" if record.get('error', False) else "성공"
                    }
                    
                    # 비용 정보 추가
                    if record.get('cost', 0) > 0:
                        record_info["예상 비용"] = f"${record.get('cost', 0):.6f}"
                    
                    st.json(record_info)
    
    # 통계 요약
    if len(st.session_state.results_history) > 1:
        st.markdown("### 📈 세션 요약")
        
        # 데이터프레임으로 변환
        df_history = pd.DataFrame(st.session_state.results_history)
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("총 질의", len(df_history))
        
        with col_stat2:
            avg_temp = df_history['temperature'].mean()
            st.metric("평균 온도", f"{avg_temp:.2f}")
        
        with col_stat3:
            total_tokens = df_history['tokens_used'].sum()
            st.metric("총 토큰", total_tokens)
        
        with col_stat4:
            success_rate = len([r for r in st.session_state.results_history if not r.get('error', False)]) / len(st.session_state.results_history) * 100
            st.metric("성공률", f"{success_rate:.1f}%")
        
        # 총 비용 계산 및 표시
        if st.session_state.results_history:
            total_cost = sum([r.get('cost', 0) for r in st.session_state.results_history])
            if total_cost > 0:
                st.markdown("### 💰 비용 정보")
                col_cost1, col_cost2, col_cost3 = st.columns(3)
                with col_cost1:
                    st.metric("총 사용 비용", f"${total_cost:.6f}")
                with col_cost2:
                    avg_cost = total_cost / len(st.session_state.results_history)
                    st.metric("평균 질의 비용", f"${avg_cost:.6f}")
                with col_cost3:
                    # 비용 요약 다운로드 (간단한 텍스트)
                    cost_summary_text = f"""LLMOps 비용 요약
생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*40}

총 질의수: {len(st.session_state.results_history)}
총 비용: ${total_cost:.6f}
평균 질의 비용: ${avg_cost:.6f}
총 토큰: {sum([r.get('tokens_used', 0) for r in st.session_state.results_history]):,}
성공률: {len([r for r in st.session_state.results_history if not r.get('error', False)]) / len(st.session_state.results_history) * 100:.1f}%
"""
                    
                    st.download_button(
                        label="📊 비용 요약 다운로드",
                        data=cost_summary_text,
                        file_name=f"cost_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        help="비용 요약 정보를 텍스트 파일로 다운로드합니다"
                    )

else:
    st.info("📋 아직 실행 기록이 없습니다. 프롬프트를 실행해보세요!")

# 푸터
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🤖 Prompt LLMOps Dashboard v2.0 | "
    "Built with Streamlit | Real API Integration"
    "</div>",
    unsafe_allow_html=True
)