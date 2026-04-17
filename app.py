import os
import json
import time
import requests
import pdfplumber
import streamlit as st
from docx import Document
from google import genai
from google.genai import types

st.set_page_config(page_title="부산대 수요기술-연구자 매칭 시스템", layout="wide")


def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


GEMINI_API_KEY = get_env("GEMINI_API_KEY")
OPENALEX_API_KEY = get_env("OPENALEX_API_KEY")


def init_client():
    if not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)


client = init_client()


def safe_gemini_call(prompt, config=None, retries=3):
    if client is None:
        raise RuntimeError("GEMINI_API_KEY가 설정되어 있지 않습니다.")

    primary_model = "gemini-2.5-flash"
    fallback_model = "gemini-2.5-flash-lite"

    last_error = None
    for attempt in range(retries):
        try:
            return client.models.generate_content(
                model=primary_model,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            last_error = e
            if ("503" in str(e) or "429" in str(e)) and attempt < retries - 1:
                time.sleep(2)
                continue
            break

    try:
        return client.models.generate_content(
            model=fallback_model,
            contents=prompt,
            config=config,
        )
    except Exception as e:
        raise RuntimeError(f"Gemini 호출 실패: {e} / 이전 오류: {last_error}")



def extract_text_from_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return ""

    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    text = ""

    try:
        if file_ext == ".pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
        elif file_ext == ".docx":
            doc = Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception:
        return ""

    return text



def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return "초록 정보가 없습니다."

    word_index = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_index.append((pos, word))
    word_index.sort()
    return " ".join([word for pos, word in word_index])



def unified_analyze(uploaded_file, manual_text):
    file_text = extract_text_from_uploaded_file(uploaded_file) if uploaded_file else ""
    query_text = file_text if file_text.strip() else manual_text

    if not query_text or len(query_text.strip()) < 5:
        return "분석할 내용이 없습니다. 파일을 업로드하거나 내용을 입력해주세요."

    keyword_prompt = f"""
    당신은 기술이전 전문가입니다. 아래 내용에서 논문 검색을 위한 핵심 기술 키워드 3~4개를 추출하세요.
    - 핵심 규칙: 반드시 해당 기술이 쓰이는 산업 도메인(예: Ship, Marine, Vessel, Medical 등)을 포함하세요.
    - 단순히 'Navigation'이 아니라 'Autonomous Ship Navigation'처럼 구체적으로 뽑아야 합니다.
    내용: {query_text[:3000]}
    결과는 반드시 콤마(,)로만 구분된 영어 키워드만 출력하세요.
    """

    try:
        kw_res = safe_gemini_call(prompt=keyword_prompt)
        raw_keywords = kw_res.text.strip()
        keywords = ", ".join([k.strip() for k in raw_keywords.split(",") if k.strip()])
        if not keywords:
            keywords = raw_keywords.strip()
    except Exception as e:
        return f"키워드 추출 오류: {str(e)}"

    url = "https://api.openalex.org/works"
    params = {
        "search": f"{keywords} Pusan National University",
        "sort": "publication_date:desc",
        "per_page": 50,
        "select": "title,authorships,publication_date,abstract_inverted_index,primary_location",
    }
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY

    try:
        res = requests.get(url, params=params, timeout=30)
    except Exception as e:
        return f"OpenAlex 요청 오류: {e}"

    if res.status_code != 200:
        return f"OpenAlex 검색 오류: HTTP {res.status_code}"

    raw_papers = res.json().get("results", [])

    valid_papers, unique_pnu_authors = [], set()
    for p in raw_papers:
        p_authors_info, is_pnu_paper = [], False
        for authorship in p.get("authorships", []):
            name = authorship.get("author", {}).get("display_name", "Unknown")
            insts = [inst.get("display_name", "").lower() for inst in authorship.get("institutions", [])]
            raw_affil = (authorship.get("raw_affiliation_string", "") or "").lower()
            combined = " ".join(insts) + " " + raw_affil

            if any(k in combined for k in ["pusan national", "busan national", "부산대"]):
                p_authors_info.append((name, True))
                unique_pnu_authors.add(name)
                is_pnu_paper = True
            else:
                p_authors_info.append((name, False))

        if is_pnu_paper:
            p["raw_authors_info"] = p_authors_info
            loc = p.get("primary_location")
            p["venue"] = loc["source"]["display_name"] if loc and loc.get("source") else "게재처 미상"
            valid_papers.append(p)

        if len(valid_papers) >= 10:
            break

    author_db = {}
    if unique_pnu_authors:
        paper_titles = [p.get("title") for p in valid_papers]
        author_prompt = f"""
        당신은 부산대학교 전문 리서처입니다. 아래 저자들의 정보를 구글 검색으로 확인하세요.

        [동명이인 혼동 방지 규칙 - 필수]
        1. 학과-논문 정합성 검토: 검색된 교수의 학과가 논문 주제({keywords})와 상식적으로 연결되는지 검증하세요.
        2. 로봇/선박 논문인데 관련 없는 학과가 검색되면 동명이인이므로 제외하세요.
        3. 영문 이름을 한글 이름으로 변환해 부산대 교수진과 대조하세요.

        대상 저자: {list(unique_pnu_authors)}
        참고 논문들: {paper_titles}

        결과는 JSON 형식으로만 출력:
        {{
          "영문이름": {{
            "korean_name": "성함",
            "department": "학과",
            "field": "주요연구분야",
            "link": "홈페이지URL",
            "is_active": "Yes/No",
            "is_relevant": "Yes/No"
          }}
        }}
        """
        try:
            auth_res = safe_gemini_call(
                prompt=author_prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.1,
                ),
            )
            clean_json = auth_res.text.replace("```json", "").replace("```", "").strip()
            author_db = json.loads(clean_json)
        except Exception:
            author_db = {}

    abstracts_to_sum = ""
    for i, p in enumerate(valid_papers):
        abs_text = reconstruct_abstract(p.get("abstract_inverted_index"))
        abstracts_to_sum += f"[{i+1}] Title: {p.get('title')}\nAbstract: {abs_text[:500]}\n"

    try:
        sum_prompt = (
            "각 논문의 1) 한국어 번역제목 2) 요약을 작성하세요.\n"
            "형식: [번호] 번역제목 | 요약내용\n"
            f"{abstracts_to_sum}"
        )
        sum_res = safe_gemini_call(prompt=sum_prompt)
    except Exception as e:
        return f"논문 요약 오류: {e}"

    parsed_results = {}
    for line in sum_res.text.split("\n"):
        if "|" in line and "[" in line and "]" in line:
            parts = line.split("]", 1)
            num = parts[0].replace("[", "").strip()
            details = parts[1].split("|", 1)
            if len(details) == 2:
                parsed_results[num] = {
                    "title": details[0].strip(),
                    "sum": details[1].strip(),
                }

    final_output = f"### 🔍 분석 키워드: **{keywords}**\n\n---\n"
    professor_map = {}

    for i, p in enumerate(valid_papers):
        idx = str(i + 1)
        info = parsed_results.get(idx, {"title": p.get("title"), "sum": "요약 실패"})
        paper_obj = {
            "title": p.get("title"),
            "k_title": info["title"],
            "summary": info["sum"],
            "date": p.get("publication_date", "날짜 미상"),
            "venue": p.get("venue"),
        }

        for name, is_pnu in p.get("raw_authors_info", []):
            if is_pnu:
                db = author_db.get(name, {})
                if db.get("is_active") == "Yes" and db.get("is_relevant") == "Yes":
                    if name not in professor_map:
                        professor_map[name] = {
                            "k_name": db.get("korean_name"),
                            "dept": db.get("department"),
                            "field": db.get("field"),
                            "link": db.get("link"),
                            "papers": [],
                        }
                    if paper_obj not in professor_map[name]["papers"]:
                        professor_map[name]["papers"].append(paper_obj)

    if not professor_map:
        return final_output + "전공 적합성을 충족하는 부산대 교수진 매칭 결과를 찾지 못했습니다. 키워드를 더 구체화해 보세요."

    for eng_name, data in professor_map.items():
        final_output += f"## 🏫 {data['dept']} | {data['k_name']} 교수 ({eng_name})\n"
        final_output += f"- **연구분야:** {data['field']}\n"
        final_output += f"- **홈페이지:** [링크 바로가기]({data['link']})\n\n"
        final_output += "#### 📄 관련 연구 논문 내역\n"
        for i, paper in enumerate(data["papers"]):
            final_output += f"{i+1}. **{paper['k_title']}**\n"
            final_output += f"   - *요약: {paper['summary']} ({paper['date']}, {paper['venue']})*\n"
        final_output += "\n---\n"

    return final_output


st.title("🎓 부산대 수요기술-연구자 매칭 시스템")
st.caption("전공 정합성 강화 버전 · Streamlit 배포용")

with st.sidebar:
    st.header("설정 안내")
    st.markdown(
        """
- `GEMINI_API_KEY` 환경변수 필요
- `OPENALEX_API_KEY`는 선택사항
- Streamlit Cloud에서는 **Secrets**에 등록하세요
        """
    )

uploaded_file = st.file_uploader(
    "1. 수요기술조사서 업로드 (선택)", type=["pdf", "docx"]
)
manual_text = st.text_area(
    "2. 또는 기술 내용 직접 입력",
    placeholder="내용을 입력하면 관련 교수님과 논문을 찾아드립니다.",
    height=220,
)

if st.button("매칭 리스트 생성", type="primary"):
    with st.spinner("분석 중입니다..."):
        result = unified_analyze(uploaded_file, manual_text)
    st.markdown(result)
