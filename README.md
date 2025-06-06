# 🚀 프롬프트 LLMOps Dashboard

여러 AI 모델의 성능을 비교하고 모니터링할 수 있는 통합 대시보드입니다. OpenAI GPT, Claude, Gemini, Perplexity 모델을 지원하며, 실시간으로 응답 시간, 토큰 사용량, 비용을 추적할 수 있습니다.

Made by: KIM JINMAN

## ✨ 주요 기능

### 🤖 지원 모델
- **OpenAI**: GPT-4o, GPT-4o-mini
- **Anthropic**: Claude Sonnet 4, Claude Sonnet 3.7
- **Google**: Gemini 1.5 Flash, Gemini 1.5 Pro, Gemini 2.0 Flash
- **Perplexity**: Sonar, Sonar Pro

### 📊 모니터링 기능
- **응답 시간 측정**: 밀리초 단위 정확한 측정
- **토큰 사용량 추적**: 입력/출력 토큰 별도 집계
- **비용 계산**: 모델별 실시간 비용 추정
- **성공률 통계**: 요청 성공/실패 모니터링

### 📁 데이터 입력 지원
- **텍스트 직접 입력**: 복사-붙여넣기 방식
- **파일 업로드**: TXT, CSV, JSON, MD 형식 지원
- **대용량 처리**: 자동 토큰 한도 관리 및 텍스트 자르기

### 📈 결과 관리
- **실시간 결과 표시**: 응답과 동시에 메트릭 표시
- **히스토리 관리**: 세션 내 모든 요청 기록 보관
- **다양한 내보내기**: JSON, HTML, CSV, 이메일용 텍스트 지원

## 🛠️ 설치 및 설정

### 1. 필수 패키지 설치

```bash
pip install streamlit pandas python-dotenv requests
```

### 2. AI 모델별 패키지 설치

```bash
# OpenAI (선택사항)
pip install openai

# Anthropic Claude (선택사항)  
pip install anthropic

# Google Gemini (선택사항)
pip install google-generativeai
```

### 3. 환경변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 사용할 API 키를 설정하세요:

```env
# OpenAI API 키 (선택사항)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API 키 (선택사항)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google Gemini API 키 (선택사항)
GEMINI_API_KEY=your_gemini_api_key_here

# Perplexity API 키 (선택사항)
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# 선택적 설정
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=2000
MAX_FILE_SIZE_MB=10
ALLOWED_FILE_TYPES=txt,csv,json,md
DEBUG_MODE=False
```

## 🚀 사용법

### 1. 애플리케이션 실행

```bash
streamlit run app.py
```

### 2. API 키 설정 확인
- 사이드바에서 설정된 API 키 상태를 확인하세요
- 최소 하나 이상의 API 키가 필요합니다

### 3. 데이터 입력
- **텍스트 직접 입력**: 분석할 데이터를 텍스트 영역에 붙여넣기
- **파일 업로드**: 지원 형식(TXT, CSV, JSON, MD) 파일 선택

### 4. 프롬프트 작성
```
# 예시 프롬프트
이 데이터를 분석하여 주요 트렌드 3가지를 찾아주세요.
각 트렌드에 대해 구체적인 근거와 함께 설명해주세요.
```

### 5. 모델 설정 및 실행
- 사이드바에서 원하는 AI 모델 선택
- Temperature 값 조정 (0.0: 일관성 높음, 1.0: 창의성 높음)
- "🚀 프롬프트 실행" 버튼 클릭

## 📊 결과 해석

### 메트릭 설명
- **토큰 사용량**: 입력 + 출력 토큰의 총합
- **응답 시간**: API 호출부터 응답 완료까지의 시간
- **예상 비용**: 모델별 요금제 기준 USD 단위 비용
- **응답 길이**: 생성된 텍스트의 문자 수

### 응답 시간 해석
- `ms`: 밀리초 (1초 미만)
- `s`: 초 단위
- `m s`: 분과 초 단위

## 📁 데이터 내보내기

### 지원 형식
1. **JSON**: 구조화된 데이터, 메모장으로 열기 가능
2. **HTML**: 브라우저에서 열어서 표 복사 가능
3. **파이프 구분 TXT**: Excel 가져오기용
4. **이메일용 TXT**: 읽기 쉬운 텍스트 형식

### 권장 사용법
```
JSON 다운로드 → 메모장에서 확인 ✅
HTML 다운로드 → 브라우저에서 열기 → 표 복사 → Excel 붙여넣기 ✅
```

## ⚙️ 고급 설정

### 환경변수 상세 설정

```env
# 토큰 제한 (기본값: 2000)
DEFAULT_MAX_TOKENS=4000

# 파일 크기 제한 (MB, 기본값: 10)
MAX_FILE_SIZE_MB=50

# 허용 파일 형식 (기본값: txt,csv,json,md)
ALLOWED_FILE_TYPES=txt,csv,json,md,xlsx

# 디버그 모드 (기본값: False)
DEBUG_MODE=True
```

### 모델별 특징

| 모델 | 강점 | 적합한 용도 | 예상 비용 |
|------|------|-------------|-----------|
| GPT-4o-mini | 빠른 속도, 저비용 | 간단한 분석, 요약 | $ |
| GPT-4o | 높은 품질 | 복잡한 추론, 창작 | $$$ |
| Claude Haiku 3.5 | 속도, 효율성 | 빠른 처리 | $ |
| Claude Sonnet 3.5/4 | 균형잡힌 성능 | 일반적 작업 | $$ |
| Gemini Flash | 빠른 속도 | 실시간 분석 | $ |
| Perplexity | 실시간 정보 | 최신 정보 검색 | $ |

## 🔧 문제 해결

### 자주 발생하는 문제

#### 1. API 키 오류
```
❌ API 키가 설정되지 않았습니다.
```
**해결책**: `.env` 파일에 올바른 API 키 설정 확인

#### 2. 토큰 한도 초과
```
❌ 입력 데이터가 너무 큽니다.
```
**해결책**: 
- 더 작은 파일 사용
- 데이터를 여러 번에 나누어 처리
- 요약된 데이터 사용

#### 3. 응답 시간 초과
```
⏱️ 요청 시간 초과
```
**해결책**:
- 잠시 후 재시도
- 입력 데이터 크기 줄이기
- 다른 모델 사용

#### 4. 패키지 설치 오류
```
ImportError: No module named 'openai'
```
**해결책**: 필요한 패키지 설치
```bash
pip install openai anthropic google-generativeai
```

### 성능 최적화 팁

1. **작은 모델부터 시작**: GPT-4o-mini나 Gemini Flash로 테스트
2. **적절한 Temperature**: 분석 작업은 0.3-0.5, 창작은 0.7-0.9
3. **배치 처리**: 큰 데이터는 여러 번에 나누어 처리
4. **모델별 특성 활용**: 용도에 맞는 모델 선택

## 📝 사용 예시

### 데이터 분석 예시
```
프롬프트: "이 판매 데이터에서 다음을 분석해주세요:
1. 월별 매출 트렌드
2. 가장 잘 팔리는 상품 카테고리
3. 개선이 필요한 영역"

권장 모델: Claude Sonnet 3.5, GPT-4o
Temperature: 0.3-0.5
```

### 창작 작업 예시
```
프롬프트: "이 고객 인터뷰 데이터를 바탕으로 
페르소나 3개를 만들고, 각각에 대한 
스토리텔링을 작성해주세요."

권장 모델: GPT-4o, Claude Sonnet 4
Temperature: 0.7-0.9
```

## 🤝 기여하기

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 있습니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## ⚠️ 주의사항

- API 키는 절대 공개 저장소에 커밋하지 마세요
- 각 AI 서비스의 이용약관을 준수하세요
- 비용이 발생할 수 있으니 사용량을 모니터링하세요
- 민감한 데이터 처리 시 보안에 주의하세요

## 🔗 관련 링크

- [OpenAI API 문서](https://platform.openai.com/docs)
- [Anthropic API 문서](https://docs.anthropic.com/)
- [Google AI Studio](https://aistudio.google.com/)
- [Perplexity API 문서](https://docs.perplexity.ai/)
- [Streamlit 문서](https://docs.streamlit.io/)

---

**📧 문의사항이나 버그 리포트는 Issues 탭을 이용해주세요.**