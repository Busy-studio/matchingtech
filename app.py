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



def normalize_yes_no(value, default="Unknown"):
    value = str(value or "").strip().lower()
    if value in {"yes", "y", "true", "confirmed", "active", "재직", "맞음"}:
        return "Yes"
    if value in {"no", "n", "false", "inactive", "퇴직", "아님"}:
        return "No"
    return default



def normalize_relevance(value):
    value = str(value or "").strip().lower()
    mapping = {
        "high": "High",
        "medium": "Medium",
        "med": "Medium",
        "mid": "Medium",
        "low": "Low",
        "yes": "High",
        "no": "Low",
    }
    return mapping.get(value, "Unknown")



def parse_json_safely(text):
    clean = (text or "").replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except Exception:
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(clean[start:end + 1])
    return {}



def build_search_query(keywords: str) -> str:
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
    if not kw_list:
        return keywords
    primary = kw_list[:3]
    return " ".join(primary)



def unified_analyze(uploaded_file, manual_text):
    file_text = extract_text_from_uploaded_file(uploaded_file) if uploaded_file else ""
    query_text = file_text if file_text.strip() else manual_text

    if not query_text or len(query_text.strip()) < 5:
        return "분석할 내용이 없습니다. 파일을 업로드하거나 내용을 입력해주세요."

    keyword_prompt = f"""
    당신은 기술이전 전문가입니다. 아래 내용에서 논문 검색용 영어 키워드 3~5개를 추출하세요.
    규칙:
    1. 반드시 기술 핵심어 + 적용 도메인을 함께 반영하세요.
    2. 너무 긴 문장 대신 검색 가능한 짧은 키워드구로 만드세요.
    3. 결과는 콤마(,)로만 구분된 영어 키워드만 출력하세요.

    내용:
    {query_text[:3000]}
    """

    try:
        kw_res = safe_gemini_call(prompt=keyword_prompt)
        raw_keywords = kw_res.text.strip()
        keyword_list = [k.strip() for k in raw_keywords.split(",") if k.strip()]
        keywords = ", ".join(keyword_list[:5]) if keyword_list else raw_keywords.strip()
    except Exception as e:
        return f"키워드 추출 오류: {str(e)}"

    search_query = build_search_query(keywords)

    url = "https://api.openalex.org/works"
    params = {
        "search": search_query,
        "sort": "publication_date:desc",
        "per_page": 80,
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

        if len(valid_papers) >= 15:
            break

    if not valid_papers:
        return f"### 🔍 분석 키워드: **{keywords}**\n\nOpenAlex에서 부산대 저자가 포함된 관련 논문을 찾지 못했습니다."

    author_db = {}
    if unique_pnu_authors:
        paper_titles = [p.get("title") for p in valid_papers[:10]]
        author_prompt = f"""
        당신은 부산대학교 연구자 매칭 담당자입니다. 아래 저자들이 현재 부산대학교 전임교원인지, 그리고 입력 기술과 연구 적합성이 있는지 웹검색으로 판별하세요.

        입력 기술 키워드: {keywords}
        참고 논문 제목: {paper_titles}
        대상 저자: {list(unique_pnu_authors)}

        판정 규칙:
        1. 현재 부산대학교 전임교원으로 확인되면 is_active를 Yes, 아니면 No, 불확실하면 Unknown
        2. 연구 적합성은 High / Medium / Low / Unknown 중 하나로 표시
        3. High: 논문 주제와 학과/연구분야가 명확히 일치
        4. Medium: 직접 일치까지는 아니지만 인접 전공 또는 연관 연구실적 확인
        5. Low: 학과/전공이 명확히 무관
        6. department, field, link는 가능한 경우만 채우고 없으면 빈 문자열
        7. 결과는 반드시 JSON만 출력

        형식:
        {{
          "영문이름": {{
            "korean_name": "성함",
            "department": "학과",
            "field": "주요연구분야",
            "link": "홈페이지URL",
            "is_active": "Yes/No/Unknown",
            "relevance": "High/Medium/Low/Unknown"
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
            author_db = parse_json_safely(auth_res.text)
        except Exception:
            author_db = {}

    abstracts_to_sum = ""
    for i, p in enumerate(valid_papers):
        abs_text = reconstruct_abstract(p.get("abstract_inverted_index"))
        abstracts_to_sum += f"[{i+1}] Title: {p.get('title')}\nAbstract: {abs_text[:700]}\n"

    try:
        sum_prompt = (
            "각 논문의 1) 한국어 번역제목 2) 2문장 이내 요약을 작성하세요.\n"
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

    final_output = f"### 🔍 분석 키워드: **{keywords}**\n\n"
    final_output += f"- OpenAlex 검색어: `{search_query}`\n"
    final_output += f"- 부산대 저자 포함 논문 수(검토대상): {len(valid_papers)}\n"
    final_output += f"- 검토 대상 저자 수: {len(unique_pnu_authors)}\n\n---\n"

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
            if not is_pnu:
                continue

            db = author_db.get(name, {}) if isinstance(author_db, dict) else {}
            active = normalize_yes_no(db.get("is_active"), default="Unknown")
            relevance = normalize_relevance(db.get("relevance"))

            # 1차: 현직 + 적합도 High/Medium
            passed = active == "Yes" and relevance in {"High", "Medium"}

            # 2차 완화: 현직 Yes인데 relevance가 비어 있으면,
            # 해당 저자가 실제 잡힌 논문이 이미 검색 키워드로 수집된 부산대 논문이므로 검토후보로 포함
            if not passed and active == "Yes" and relevance == "Unknown":
                passed = True
                relevance = "Medium"

            if not passed:
                continue

            if name not in professor_map:
                professor_map[name] = {
                    "k_name": db.get("korean_name") or "이름 확인 필요",
                    "dept": db.get("department") or "학과 확인 필요",
                    "field": db.get("field") or "연구분야 확인 필요",
                    "link": db.get("link") or "",
                    "relevance": relevance,
                    "papers": [],
                }

            if paper_obj not in professor_map[name]["papers"]:
                professor_map[name]["papers"].append(paper_obj)

    if not professor_map:
        final_output += "현재 재직 전임교원 + 연관 분야 기준을 만족하는 교수를 찾지 못했습니다.\n\n"
        final_output += "가능한 원인\n"
        final_output += "- 웹검색 단계에서 현직 여부 판별 실패\n"
        final_output += "- relevance가 과도하게 Low/Unknown으로 분류됨\n"
        final_output += "- 검색 키워드가 너무 좁거나 부정확함\n"
        return final_output

    sorted_professors = sorted(
        professor_map.items(),
        key=lambda x: (
            0 if x[1]["relevance"] == "High" else 1,
            -len(x[1]["papers"]),
            x[0],
        ),
    )

    for eng_name, data in sorted_professors:
        final_output += f"## 🏫 {data['dept']} | {data['k_name']} 교수 ({eng_name})\n"
        final_output += f"- **적합도:** {data['relevance']}\n"
        final_output += f"- **연구분야:** {data['field']}\n"
        if data["link"]:
            final_output += f"- **홈페이지:** [링크 바로가기]({data['link']})\n\n"
        else:
            final_output += "- **홈페이지:** 확인 필요\n\n"
        final_output += "#### 📄 관련 연구 논문 내역\n"
        for i, paper in enumerate(sorted(data["papers"], key=lambda x: x["date"], reverse=True)):
            final_output += f"{i+1}. **{paper['k_title']}**\n"
            final_output += f"   - *요약: {paper['summary']} ({paper['date']}, {paper['venue']})*\n"
        final_output += "\n---\n"

    return final_output


st.title("🎓 부산대 수요기술-연구자 매칭 시스템")
st.caption("현직 전임교원 중심 · 완화된 판정 버전")

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
