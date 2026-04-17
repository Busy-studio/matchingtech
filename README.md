# 부산대 수요기술-연구자 매칭 시스템 (Streamlit)

기존 Gradio 코드를 GitHub + Streamlit 배포용으로 변환한 프로젝트입니다.

## 파일 구성
- `app.py` : Streamlit 실행 파일
- `requirements.txt` : 의존성 목록

## 로컬 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 환경변수 설정
### 로컬
```bash
export GEMINI_API_KEY="your_gemini_api_key"
export OPENALEX_API_KEY="your_openalex_api_key"
```

Windows PowerShell:
```powershell
$env:GEMINI_API_KEY="your_gemini_api_key"
$env:OPENALEX_API_KEY="your_openalex_api_key"
```

## Streamlit Community Cloud 배포
1. 이 폴더 내용을 GitHub 저장소에 업로드
2. Streamlit Community Cloud에서 저장소 연결
3. Main file path를 `app.py`로 지정
4. Settings > Secrets에 아래처럼 입력

```toml
GEMINI_API_KEY = "your_gemini_api_key"
OPENALEX_API_KEY = "your_openalex_api_key"
```

## 주의사항
- 기존 코드에 하드코딩돼 있던 API 키는 제거했습니다.
- Gemini 모델명은 배포 안정성을 위해 `gemini-2.5-flash` / `gemini-2.5-flash-lite` 기준으로 정리했습니다.
- 저자 검색 JSON 파싱 실패 시, 교수 매칭 결과가 비어 보일 수 있습니다.
