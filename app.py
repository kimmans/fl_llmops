import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os
from typing import Dict, Any
import requests
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# API 클라이언트 import
try:
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

# PDF 처리
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# 페이지 설정
st.set_page_config(page_title="프롬프트 LLMOps Dashboard", page_icon="🤖", layout="wide")

# 모델별 가격 정보 (USD per 1M tokens)
MODEL_PRICING = {
    "GPT-4o": {"input": 2.50, "output": 10.00},
    "GPT-4o-mini": {"input": 0.15, "output": 0.60},
    "Claude Sonnet 4": {"input": 3.00, "output": 15.00},
    "Claude Sonnet 3.7": {"input": 3.00, "output": 15.00},
    "Perplexity Sonar": {"input": 0.20, "output": 0.20},
    "Perplexity Sonar Pro": {"input": 1.00, "output": 1.00},
    "Gemini 1.5 Flash": {"input": 0.075, "output": 0.30},
    "Gemini 1.5 Pro": {"input": 1.25, "output": 5.00},
    "Gemini 2.0 Flash": {"input": 0.10, "output": 0.40}
}

def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    if model_name not in MODEL_PRICING:
        return 0.0
    pricing = MODEL_PRICING[model_name]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost

# 세션 상태 초기화
if 'results_history' not in st.session_state:
    st.session_state.results_history = []

def check_api_keys():
    return {
        "OpenAI": bool(os.getenv("OPENAI_API_KEY")),
        "Anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "Gemini": bool(os.getenv("GEMINI_API_KEY")),
        "Perplexity": bool(os.getenv("PERPLEXITY_API_KEY"))
    }

def init_clients():
    clients = {}
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        clients['openai'] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
        clients['anthropic'] = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    if GOOGLE_AVAILABLE and os.getenv("GEMINI_API_KEY"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        clients['google'] = genai.GenerativeModel('gemini-1.5-flash')
    return clients

clients = init_clients()

def call_openai(prompt: str, data: str, temperature: float, model_name: str) -> Dict[str, Any]:
    if 'openai' not in clients:
        return {"model": model_name, "response": "OpenAI API 키 오류", "error": True}
    
    try:
        model_mapping = {"GPT-4o": "gpt-4o", "GPT-4o-mini": "gpt-4o-mini"}
        api_model = model_mapping.get(model_name, "gpt-4o-mini")
        
        if data and data.strip():
            full_prompt = f"데이터: {data}\n\n요청: {prompt}"
        else:
            full_prompt = prompt
        
        response = clients['openai'].chat.completions.create(
            model=api_model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=temperature,
            max_tokens=2000
        )
        
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = calculate_cost(model_name, input_tokens, output_tokens)
        
        return {
            "model": model_name,
            "response": response.choices[0].message.content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temperature,
            "tokens_used": input_tokens + output_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "error": False
        }
    except Exception as e:
        return {"model": model_name, "response": f"API 오류: {str(e)}", "error": True}

def call_claude(prompt: str, data: str, temperature: float, model_name: str) -> Dict[str, Any]:
    if 'anthropic' not in clients:
        return {"model": model_name, "response": "Anthropic API 키 오류", "error": True}
    
    try:
        model_mapping = {
            "Claude Sonnet 4": "claude-3-5-sonnet-20241022",
            "Claude Sonnet 3.7": "claude-3-5-sonnet-20241022"
        }
        api_model = model_mapping.get(model_name, "claude-3-5-sonnet-20241022")
        
        if data and data.strip():
            full_prompt = f"데이터: {data}\n\n요청: {prompt}"
        else:
            full_prompt = prompt
        
        response = clients['anthropic'].messages.create(
            model=api_model,
            max_tokens=2000,
            temperature=temperature,
            messages=[{"role": "user", "content": full_prompt}]
        )
        
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = calculate_cost(model_name, input_tokens, output_tokens)
        
        return {
            "model": model_name,
            "response": response.content[0].text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temperature,
            "tokens_used": input_tokens + output_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "error": False
        }
    except Exception as e:
        return {"model": model_name, "response": f"API 오류: {str(e)}", "error": True}

def call_perplexity(prompt: str, data: str, temperature: float, model_name: str) -> Dict[str, Any]:
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return {"model": model_name, "response": "Perplexity API 키 오류", "error": True}
    
    try:
        url = "https://api.perplexity.ai/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        model_mapping = {
            "Perplexity Sonar": "llama-3.1-sonar-small-128k-online",
            "Perplexity Sonar Pro": "llama-3.1-sonar-large-128k-online"
        }
        api_model = model_mapping.get(model_name, "llama-3.1-sonar-small-128k-online")
        
        if data and data.strip():
            full_prompt = f"데이터: {data}\n\n요청: {prompt}"
        else:
            full_prompt = prompt
        
        payload = {
            "model": api_model,
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": temperature,
            "max_tokens": 2000
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        input_tokens = result.get('usage', {}).get('prompt_tokens', 0)
        output_tokens = result.get('usage', {}).get('completion_tokens', 0)
        cost = calculate_cost(model_name, input_tokens, output_tokens)
        
        return {
            "model": model_name,
            "response": result['choices'][0]['message']['content'],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temperature,
            "tokens_used": input_tokens + output_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "error": False
        }
    except Exception as e:
        return {"model": model_name, "response": f"API 오류: {str(e)}", "error": True}

def call_gemini(prompt: str, data: str, temperature: float, model_name: str) -> Dict[str, Any]:
    if 'google' not in clients:
        return {"model": model_name, "response": "Gemini API 키 오류", "error": True}
    
    try:
        model_mapping = {
            "Gemini 1.5 Flash": "gemini-1.5-flash",
            "Gemini 1.5 Pro": "gemini-1.5-pro",
            "Gemini 2.0 Flash": "gemini-2.0-flash-exp"
        }
        api_model = model_mapping.get(model_name, "gemini-1.5-flash")
        
        try:
            model_client = genai.GenerativeModel(api_model)
        except:
            model_client = genai.GenerativeModel("gemini-1.5-flash")
        
        if data and data.strip():
            full_prompt = f"데이터: {data}\n\n요청: {prompt}"
        else:
            full_prompt = prompt
        
        generation_config = genai.types.GenerationConfig(temperature=temperature, max_output_tokens=2000)
        response = model_client.generate_content(full_prompt, generation_config=generation_config)
        response_text = response.text if hasattr(response, 'text') and response.text else "응답 생성 실패"
        
        input_tokens = int(len(full_prompt.split()) * 1.3)
        output_tokens = int(len(response_text.split()) * 1.3)
        cost = calculate_cost(model_name, input_tokens, output_tokens)
        
        return {
            "model": model_name,
            "response": response_text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temperature,
            "tokens_used": input_tokens + output_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "error": False
        }
    except Exception as e:
        return {"model": model_name, "response": f"API 오류: {str(e)}", "error": True}

def read_pdf(uploaded_file):
    if not PDF_AVAILABLE:
        return "PDF 라이브러리 없음"
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"PDF 읽기 오류: {str(e)}"

# UI
st.title("🚀 프롬프트 LLMOps Dashboard")

api_status = check_api_keys()

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 모드 선택
    execution_mode = st.radio("실행 모드", ["단일 모델", "다중 모델 비교"], horizontal=True)
    
    available_models = []
    if api_status["OpenAI"]:
        available_models.extend(["GPT-4o", "GPT-4o-mini"])
    if api_status["Anthropic"]:
        available_models.extend(["Claude Sonnet 4", "Claude Sonnet 3.7"])
    if api_status["Perplexity"]:
        available_models.extend(["Perplexity Sonar", "Perplexity Sonar Pro"])
    if api_status["Gemini"]:
        available_models.extend(["Gemini 1.5 Flash", "Gemini 1.5 Pro", "Gemini 2.0 Flash"])
    
    if not available_models:
        available_models = ["API 키 없음"]
    
    if execution_mode == "단일 모델":
        model_choice = st.selectbox("🤖 AI 모델 선택", available_models)
        selected_models = [model_choice] if model_choice != "API 키 없음" else []
    else:
        selected_models = st.multiselect(
            "🤖 AI 모델 선택 (최대 4개)",
            available_models,
            default=[available_models[0]] if available_models and available_models[0] != "API 키 없음" else [],
            max_selections=4
        )
    
    temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.7, 0.1)
    
    st.markdown("### Made by: KIM JINMAN")
    st.markdown("### Last Update: 2025-06-06")
    st.markdown("### Version: 1.0.1")

# 메인 영역
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 입력")
    
    # 데이터 입력 (선택사항)
    input_method = st.radio("데이터 입력 방식", ["없음", "텍스트 입력", "파일 업로드"])
    
    data_input = ""
    
    if input_method == "텍스트 입력":
        data_input = st.text_area("📄 데이터 입력", height=200, placeholder="분석할 데이터를 입력하세요 (선택사항)")
    
    elif input_method == "파일 업로드":
        file_types = ["txt", "csv", "json", "md"]
        if PDF_AVAILABLE:
            file_types.append("pdf")
        
        uploaded_file = st.file_uploader("📎 파일 업로드", type=file_types)
        
        if uploaded_file is not None:
            try:
                file_extension = uploaded_file.name.lower().split('.')[-1]
                
                if file_extension == 'pdf':
                    data_input = read_pdf(uploaded_file)
                elif file_extension == 'txt':
                    data_input = uploaded_file.getvalue().decode('utf-8')
                elif file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                    data_input = df.to_string(index=False)
                elif file_extension == 'json':
                    json_data = json.loads(uploaded_file.getvalue().decode('utf-8'))
                    data_input = json.dumps(json_data, indent=2, ensure_ascii=False)
                elif file_extension == 'md':
                    data_input = uploaded_file.getvalue().decode('utf-8')
                
                if data_input and not data_input.startswith(("PDF 라이브러리 없음", "PDF 읽기 오류")):
                    st.success(f"✅ {uploaded_file.name} 파일 로드 완료")
                    with st.expander("파일 내용 미리보기"):
                        st.text_area("", value=data_input[:500] + "..." if len(data_input) > 500 else data_input, height=200, disabled=True)
                else:
                    st.error("파일을 읽을 수 없습니다.")
                    
            except Exception as e:
                st.error(f"파일 읽기 오류: {str(e)}")
                data_input = ""
    
    # 프롬프트 입력 (필수)
    prompt_input = st.text_area("💡 프롬프트 입력", height=200, placeholder="AI에게 요청할 작업을 입력하세요")
    
    # 실행 버튼 - 프롬프트만 있으면 실행 가능
    can_execute = bool(prompt_input and prompt_input.strip()) and bool(selected_models) and "API 키 없음" not in selected_models
    
    if st.button("🚀 프롬프트 실행", type="primary", disabled=not can_execute):
        if can_execute:
            with st.spinner(f"AI가 응답을 생성 중입니다... ({len(selected_models)}개 모델)"):
                model_functions = {
                    "GPT-4o": lambda p, d, t: call_openai(p, d, t, "GPT-4o"),
                    "GPT-4o-mini": lambda p, d, t: call_openai(p, d, t, "GPT-4o-mini"),
                    "Claude Sonnet 4": lambda p, d, t: call_claude(p, d, t, "Claude Sonnet 4"),
                    "Claude Sonnet 3.7": lambda p, d, t: call_claude(p, d, t, "Claude Sonnet 3.7"),
                    "Perplexity Sonar": lambda p, d, t: call_perplexity(p, d, t, "Perplexity Sonar"),
                    "Perplexity Sonar Pro": lambda p, d, t: call_perplexity(p, d, t, "Perplexity Sonar Pro"),
                    "Gemini 1.5 Flash": lambda p, d, t: call_gemini(p, d, t, "Gemini 1.5 Flash"),
                    "Gemini 1.5 Pro": lambda p, d, t: call_gemini(p, d, t, "Gemini 1.5 Pro"),
                    "Gemini 2.0 Flash": lambda p, d, t: call_gemini(p, d, t, "Gemini 2.0 Flash")
                }
                
                # 선택된 모델들에 대해 순차적으로 실행
                results = []
                for model in selected_models:
                    result = model_functions[model](prompt_input, data_input, temperature)
                    result['prompt'] = prompt_input
                    result['data_preview'] = data_input[:200] + "..." if len(data_input) > 200 else data_input
                    results.append(result)
                    st.session_state.results_history.append(result)
                
                # 다중 모델 결과를 세션에 저장
                if len(selected_models) > 1:
                    st.session_state.multi_results = results
                
                st.rerun()

with col2:
    st.header("📊 결과")
    
    # 다중 모델 결과 표시
    if hasattr(st.session_state, 'multi_results') and st.session_state.multi_results:
        st.subheader("🔄 다중 모델 비교 결과")
        
        # 탭으로 각 모델 결과 표시
        if len(st.session_state.multi_results) > 1:
            tabs = st.tabs([result['model'] for result in st.session_state.multi_results])
            
            for i, (tab, result) in enumerate(zip(tabs, st.session_state.multi_results)):
                with tab:
                    if result.get('error', False):
                        st.error(f"❌ 오류: {result['response']}")
                    else:
                        st.success(f"✅ 응답 완료")
                        st.text_area("AI 응답", value=result['response'], height=300, disabled=True, key=f"multi_response_{i}")
                        
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("토큰", result.get('tokens_used', 0))
                        with col_m2:
                            st.metric("응답 길이", len(result['response']))
                        with col_m3:
                            cost = result.get('cost', 0.0)
                            st.metric("비용", f"${cost:.6f}")
            
            # 비교 요약
            st.subheader("📈 비교 요약")
            comparison_data = []
            for result in st.session_state.multi_results:
                comparison_data.append({
                    '모델': result['model'],
                    '상태': '성공' if not result.get('error', False) else '오류',
                    '토큰': result.get('tokens_used', 0),
                    '비용': f"${result.get('cost', 0):.6f}",
                    '응답 길이': len(result['response'])
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
    
    # 단일 모델 결과 표시
    elif st.session_state.results_history:
        latest_result = st.session_state.results_history[-1]
        
        if latest_result.get('error', False):
            st.error(f"❌ {latest_result['model']}: {latest_result['response']}")
        else:
            st.success(f"✅ {latest_result['model']} 응답 완료")
            st.text_area("AI 응답", value=latest_result['response'], height=400, disabled=True)
            
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.metric("토큰", latest_result.get('tokens_used', 0))
            with col_metric2:
                st.metric("응답 길이", len(latest_result['response']))
            with col_metric3:
                cost = latest_result.get('cost', 0.0)
                st.metric("비용", f"${cost:.6f}")

# 기록 섹션
if st.session_state.results_history:
    st.markdown("---")
    st.header("📚 기록")
    
    col_clear, col_download = st.columns([1, 1])
    with col_clear:
        if st.button("🗑️ 기록 초기화"):
            st.session_state.results_history = []
            st.rerun()
    
    with col_download:
        export_data = []
        for record in st.session_state.results_history:
            export_data.append({
                '시간': record.get('timestamp', ''),
                '모델': record.get('model', ''),
                '프롬프트': record.get('prompt', ''),
                '응답': record.get('response', ''),
                '토큰': record.get('tokens_used', 0),
                '비용': record.get('cost', 0),
                '상태': '오류' if record.get('error', False) else '성공'
            })
        
        df_export = pd.DataFrame(export_data)
        
        # TXT 파일 다운로드
        txt_data = f"""LLMOps Dashboard 기록
생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

"""
        for i, record in enumerate(st.session_state.results_history, 1):
            txt_data += f"[기록 {i}]\n"
            txt_data += f"시간: {record.get('timestamp', '')}\n"
            txt_data += f"모델: {record.get('model', '')}\n"
            txt_data += f"프롬프트: {record.get('prompt', '')}\n"
            txt_data += f"응답: {record.get('response', '')}\n"
            txt_data += f"토큰: {record.get('tokens_used', 0)}\n"
            txt_data += f"비용: ${record.get('cost', 0):.6f}\n"
            txt_data += f"상태: {'오류' if record.get('error', False) else '성공'}\n"
            txt_data += "-" * 50 + "\n\n"
        
        st.download_button(
            "📄 TXT 다운로드",
            data=txt_data,
            file_name=f"llmops_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    # 웹에서 보기
    with st.expander("🌐 웹에서 보기 (복사용)"):
        st.dataframe(df_export, use_container_width=True)
        st.caption("위 표를 선택하여 복사할 수 있습니다")
    
    # 최근 5개 기록 표시
    recent_history = st.session_state.results_history[-5:][::-1]
    
    for i, record in enumerate(recent_history):
        status_icon = "❌" if record.get('error', False) else "✅"
        with st.expander(f"{status_icon} {record['timestamp']} - {record['model']}", expanded=(i==0)):
            st.text(f"프롬프트: {record.get('prompt', 'N/A')}")
            if record.get('error', False):
                st.error(record['response'])
            else:
                st.text_area("응답", value=record['response'], height=150, key=f"history_{i}", disabled=True)
                st.caption(f"토큰: {record.get('tokens_used', 0)} | 비용: ${record.get('cost', 0):.6f}")