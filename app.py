import os
import json
import time
from typing import Dict, List, Tuple

import requests
import pdfplumber
import streamlit as st
from docx import Document
from google import genai
from google.genai import types

st.set_page_config(page_title="부산대 수요기술-연구자 매칭 시스템", layout="wide")

OPENALEX_URL = "https://api.openalex.org/works"
MAX_PAPERS = 20
MAX_AUTHORS_FOR_ENRICH = 20
MIN_RELEVANT_PAPERS = 3


# -----------------------------
# Environment / Client
# -----------------------------
def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


GEMINI_API_KEY = get_env("GEMINI_API_KEY")
OPENALEX_API_KEY = get_env("OPENALEX_API_KEY")


def init_client():
    if not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)


client = init_client()


# -----------------------------
# Utilities
# -----------------------------
def safe_gemini_call(prompt: str, config=None, retries: int = 3):
    if client is None:
        raise RuntimeError("GEMINI_API_KEY가 설정되어 있지 않습니다.")

    models = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    last_error = None

    for model_name in models:
        for attempt in range(retries):
            try:
                return client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
            except Exception as e:
                last_error = e
                if any(code in str(e) for code in ["429", "503"]) and attempt < retries - 1:
                    time.sleep(2)
                    continue
                break

    raise RuntimeError(f"Gemini 호출 실패: {last_error}")


@st.cache_data(show_spinner=False)
def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return "초록 정보가 없습니다."

    word_index = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_index.append((pos, word))
    word_index.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_index)


@st.cache_data(show_spinner=False)
def normalize_yes_no(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"yes", "y", "true", "재직", "현직", "예"}:
        return "Yes"
    if v in {"no", "n", "false", "비재직", "아니오"}:
        return "No"
    return "Unknown"


@st.cache_data(show_spinner=False)
def extract_json_object(text: str) -> Dict:
    text = (text or "").strip()
    if not text:
        return {}

    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = cleaned[start:end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return {}
    return {}


def compact_text(text: str, limit: int = 4000) -> str:
    text = (text or "").strip()
    return text[:limit]


# -----------------------------
# File extraction
# -----------------------------
def extract_text_from_uploaded_file(uploaded_file) -> str:
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
        elif file_ext in {".txt", ".md"}:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

    return text


# -----------------------------
# Step 1. Input parsing / keyword extraction
# -----------------------------
def extract_request_metadata(query_text: str) -> Dict[str, str]:
    company_name = "미확인"
    tech_summary = "입력된 수요기술 설명을 바탕으로 연구자 매칭을 수행했습니다."

    prompt = f"""
당신은 대학 기술이전 실무용 입력정보 정리기입니다.
아래 텍스트에서 다음 두 항목만 추출하세요.

규칙:
1. 기업명은 명확히 보일 때만 추출, 없으면 "미확인"
2. 수요기술 요약은 한국어 1~2문장, 너무 길지 않게
3. 과장 없이 소재, 공정, 장치, 성능, 적용처 중심으로 요약
4. 출력은 JSON만 반환

형식:
{{
  "company_name": "기업명 또는 미확인",
  "tech_summary": "수요기술 요약"
}}

입력 텍스트:
{compact_text(query_text)}
"""

    try:
        res = safe_gemini_call(prompt)
        data = extract_json_object(getattr(res, "text", ""))
        if isinstance(data, dict):
            company_name = str(data.get("company_name", company_name)).strip() or company_name
            tech_summary = str(data.get("tech_summary", tech_summary)).strip() or tech_summary
    except Exception:
        pass

    return {
        "company_name": company_name,
        "tech_summary": tech_summary,
    }


def extract_search_profile(query_text: str) -> Dict:
    prompt = f"""
당신은 대학 산학협력용 논문 검색 프로파일 설계기입니다.
아래 수요기술 설명을 읽고 OpenAlex 검색에 유리한 구조화 결과를 JSON으로 작성하세요.

반드시 포함할 항목:
1. core_tech: 핵심 기술 2~4개 (영어)
2. materials_or_methods: 재료/공정/방식 2~4개 (영어)
3. properties: 요구 성능/특성 2~4개 (영어)
4. applications: 적용 산업/제품/도메인 1~3개 (영어)
5. search_keywords: 실제 검색에 바로 쓸 핵심 키워드 4~6개 (영어, 짧은 구)
6. exclude_keywords: 가능하면 배제하고 싶은 비관련 분야 0~4개 (영어)
7. korean_summary: 한국어 한두 문장 요약

규칙:
- 반드시 영어 키워드 중심
- 너무 긴 문장 금지
- adhesive, coating, battery, marine, medical device처럼 명확한 기술명 우선
- 적용 산업이 보이면 반드시 applications에 반영
- 출력은 JSON만

형식:
{{
  "core_tech": ["..."],
  "materials_or_methods": ["..."],
  "properties": ["..."],
  "applications": ["..."],
  "search_keywords": ["..."],
  "exclude_keywords": ["..."],
  "korean_summary": "..."
}}

입력 텍스트:
{compact_text(query_text)}
"""

    fallback_tokens = []
    for token in [x.strip(",.()[]{}") for x in query_text.replace("\n", " ").split() if x.strip()]:
        if len(token) >= 4:
            fallback_tokens.append(token)
        if len(fallback_tokens) >= 5:
            break

    try:
        res = safe_gemini_call(prompt)
        data = extract_json_object(getattr(res, "text", ""))
        if isinstance(data, dict):
            for key in [
                "core_tech",
                "materials_or_methods",
                "properties",
                "applications",
                "search_keywords",
                "exclude_keywords",
            ]:
                data[key] = [str(x).strip() for x in data.get(key, []) if str(x).strip()]
            data["korean_summary"] = str(data.get("korean_summary", "")).strip()
            if data.get("search_keywords"):
                return data
    except Exception:
        pass

    fallback = [t for t in fallback_tokens if len(t) >= 4][:5]
    return {
        "core_tech": fallback[:2],
        "materials_or_methods": fallback[2:4],
        "properties": [],
        "applications": [],
        "search_keywords": fallback or ["Pusan National University"],
        "exclude_keywords": [],
        "korean_summary": "입력된 수요기술 설명을 바탕으로 검색용 키워드를 구성했습니다.",
    }


def format_keyword_text(search_profile: Dict) -> str:
    keywords = search_profile.get("search_keywords", [])
    return ", ".join(keywords)


# -----------------------------
# Step 2. OpenAlex search
# -----------------------------
@st.cache_data(show_spinner=False)
def search_openalex(search_keywords: Tuple[str, ...], applications: Tuple[str, ...], core_tech: Tuple[str, ...]) -> List[Dict]:
    search_keywords = list(search_keywords)
    applications = list(applications)
    core_tech = list(core_tech)

    base = search_keywords[:4] if search_keywords else core_tech[:3]
    queries = []
    if base:
        queries.append(" ".join(base[:3]) + " Pusan National University")
        queries.append(" OR ".join(base[:3]) + " Pusan National University")
        queries.append(" ".join(base[:2]))
    if core_tech and applications:
        queries.append(f"{' '.join(core_tech[:2])} {' '.join(applications[:2])} Pusan National University")
    if core_tech:
        queries.append(" OR ".join(core_tech[:3]))

    seen_ids = set()
    collected = []

    for q in queries:
        params = {
            "search": q,
            "sort": "publication_date:desc",
            "per_page": 50,
            "select": "id,title,authorships,publication_date,abstract_inverted_index,primary_location",
        }
        if OPENALEX_API_KEY:
            params["api_key"] = OPENALEX_API_KEY

        try:
            res = requests.get(OPENALEX_URL, params=params, timeout=30)
            if res.status_code != 200:
                continue
            items = res.json().get("results", [])
        except Exception:
            continue

        for item in items:
            item_id = item.get("id") or item.get("title")
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            collected.append(item)

    return collected


@st.cache_data(show_spinner=False)
def filter_pnu_papers(raw_papers: List[Dict]) -> Tuple[List[Dict], List[str]]:
    valid_papers = []
    unique_pnu_authors = []
    seen_authors = set()

    for p in raw_papers:
        p_authors_info = []
        is_pnu_paper = False

        for authorship in p.get("authorships", []):
            name = authorship.get("author", {}).get("display_name", "Unknown")
            insts = [inst.get("display_name", "").lower() for inst in authorship.get("institutions", [])]
            raw_affil = (authorship.get("raw_affiliation_string", "") or "").lower()
            combined = " ".join(insts) + " " + raw_affil

            is_pnu = any(k in combined for k in ["pusan national", "busan national", "부산대"])
            p_authors_info.append((name, is_pnu))
            if is_pnu:
                is_pnu_paper = True
                if name not in seen_authors:
                    seen_authors.add(name)
                    unique_pnu_authors.append(name)

        if is_pnu_paper:
            loc = p.get("primary_location") or {}
            source = loc.get("source") or {}
            p["venue"] = source.get("display_name", "게재처 미상")
            p["raw_authors_info"] = p_authors_info
            valid_papers.append(p)

        if len(valid_papers) >= MAX_PAPERS:
            break

    return valid_papers, unique_pnu_authors[:MAX_AUTHORS_FOR_ENRICH]


# -----------------------------
# Step 3. Paper relevance scoring
# -----------------------------
def score_paper_relevance(valid_papers: List[Dict], search_profile: Dict, tech_summary: str) -> Dict[str, Dict]:
    if not valid_papers:
        return {}

    paper_blocks = []
    for i, p in enumerate(valid_papers, start=1):
        abs_text = reconstruct_abstract(p.get("abstract_inverted_index"))
        paper_blocks.append(
            f"[{i}] Title: {p.get('title', '')}\nAbstract: {abs_text[:900]}\n"
        )

    prompt = f"""
당신은 대학 산학협력용 논문 적합성 평가기입니다.
아래 수요기술과 논문 목록을 비교하여, 각 논문이 수요기술과 실질적으로 얼마나 맞는지 평가하세요.

수요기술 요약:
{tech_summary}

핵심 기술 프로파일:
- core_tech: {search_profile.get('core_tech', [])}
- materials_or_methods: {search_profile.get('materials_or_methods', [])}
- properties: {search_profile.get('properties', [])}
- applications: {search_profile.get('applications', [])}
- exclude_keywords: {search_profile.get('exclude_keywords', [])}

판정 기준:
- High: 수요기술과 직접적으로 매우 관련
- Medium: 인접 분야 또는 일부 핵심 요소가 일치
- Low: 표면적으로만 비슷하거나 핵심이 다름
- Exclude: 비관련 분야

중요 규칙:
1. 제목과 초록 기준으로 판단
2. 재료/공정/성능/적용처 중 2개 이상 맞으면 Medium 이상 가능
3. 적용처가 다르더라도 핵심 소재/공정/기술이 맞으면 Medium 가능
4. 의료/생명/순수이론 등 수요와 명확히 벗어나면 Exclude
5. 출력은 JSON만

출력 형식:
{{
  "1": {{"relevance": "High/Medium/Low/Exclude", "score": 0-100, "reason": "한 줄 근거"}},
  "2": {{"relevance": "High/Medium/Low/Exclude", "score": 0-100, "reason": "한 줄 근거"}}
}}

논문 목록:
{chr(10).join(paper_blocks)}
"""

    try:
        res = safe_gemini_call(prompt)
        data = extract_json_object(getattr(res, "text", ""))
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    fallback = {}
    for i, _ in enumerate(valid_papers, start=1):
        fallback[str(i)] = {
            "relevance": "Medium",
            "score": 60,
            "reason": "자동 적합성 평가 실패로 기본값 적용",
        }
    return fallback


def select_relevant_papers(valid_papers: List[Dict], relevance_map: Dict[str, Dict]) -> List[Dict]:
    selected = []
    medium_candidates = []

    for i, p in enumerate(valid_papers, start=1):
        rel = relevance_map.get(str(i), {}) if isinstance(relevance_map, dict) else {}
        label = str(rel.get("relevance", "Medium")).strip()
        try:
            score = int(rel.get("score", 60))
        except Exception:
            score = 60
        reason = str(rel.get("reason", "")).strip()
        p["paper_relevance"] = label
        p["paper_score"] = score
        p["paper_reason"] = reason

        if label == "High":
            selected.append(p)
        elif label == "Medium":
            medium_candidates.append(p)

    selected.extend(sorted(medium_candidates, key=lambda x: x.get("paper_score", 0), reverse=True))

    if len(selected) < MIN_RELEVANT_PAPERS:
        low_pool = []
        for i, p in enumerate(valid_papers, start=1):
            if p in selected:
                continue
            label = p.get("paper_relevance", "")
            if label == "Low":
                low_pool.append(p)
        selected.extend(sorted(low_pool, key=lambda x: x.get("paper_score", 0), reverse=True)[: MIN_RELEVANT_PAPERS - len(selected)])

    return selected[:MAX_PAPERS]


# -----------------------------
# Step 4. Author enrichment
# -----------------------------
def enrich_authors_with_gemini(author_names: List[str], search_profile: Dict, paper_titles: List[str]) -> Dict:
    if not author_names:
        return {}

    prompt = f"""
당신은 부산대학교 교수 검색 보조 시스템입니다.
아래 영문 저자명이 현재 부산대학교 전임교원인지 확인하고, 맞다면 학과/연구분야/홈페이지를 찾아주세요.

중요 규칙:
1. 반드시 '현재 부산대학교 재직 전임교원'인 경우만 is_active를 Yes로 표시
2. 확실하지 않으면 No가 아니라 Unknown으로 표시
3. 논문 주제와 학과가 어느 정도 연결되면 relevance를 Medium 이상으로 판단
4. 너무 엄격하게 자르지 말고, 기계/전기/재료/조선/의생명처럼 인접 분야면 Medium 가능
5. 출력은 JSON만 반환

relevance 기준:
- High: 논문 주제와 교수 연구분야가 직접 일치
- Medium: 인접 분야로 충분히 연관
- Low: 연관성이 약함
- Unknown: 정보 부족

대상 저자:
{author_names}

수요기술 프로파일:
- core_tech: {search_profile.get('core_tech', [])}
- materials_or_methods: {search_profile.get('materials_or_methods', [])}
- properties: {search_profile.get('properties', [])}
- applications: {search_profile.get('applications', [])}

참고 논문 제목:
{paper_titles[:20]}

출력 형식:
{{
  "영문이름": {{
    "korean_name": "성함",
    "department": "학과",
    "field": "주요 연구분야",
    "link": "홈페이지URL",
    "is_active": "Yes/No/Unknown",
    "relevance": "High/Medium/Low/Unknown",
    "note": "판단 근거 한 줄"
  }}
}}
"""

    try:
        res = safe_gemini_call(
            prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.1,
            ),
        )
        return extract_json_object(getattr(res, "text", ""))
    except Exception:
        return {}


# -----------------------------
# Step 5. Paper summary
# -----------------------------
def summarize_papers(valid_papers: List[Dict]) -> Dict[str, Dict[str, str]]:
    if not valid_papers:
        return {}

    abstracts_to_sum = ""
    for i, p in enumerate(valid_papers, start=1):
        abs_text = reconstruct_abstract(p.get("abstract_inverted_index"))
        abstracts_to_sum += f"[{i}] Title: {p.get('title')}\nAbstract: {abs_text[:700]}\n\n"

    prompt = f"""
아래 논문들에 대해 각 번호별로
1) 한국어 번역 제목
2) 기술 핵심 요약 한 줄
을 작성하세요.

출력 형식은 각 줄마다 정확히 다음처럼 작성:
[번호] 번역제목 | 요약내용

{abstracts_to_sum}
"""

    try:
        res = safe_gemini_call(prompt)
        parsed = {}
        for line in getattr(res, "text", "").split("\n"):
            if "|" in line and line.strip().startswith("[") and "]" in line:
                parts = line.split("]", 1)
                idx = parts[0].replace("[", "").strip()
                detail_parts = parts[1].split("|", 1)
                if len(detail_parts) == 2:
                    parsed[idx] = {
                        "title": detail_parts[0].strip(),
                        "sum": detail_parts[1].strip(),
                    }
        return parsed
    except Exception:
        return {}


# -----------------------------
# Step 6. Assemble result
# -----------------------------
def build_professor_map(valid_papers: List[Dict], author_db: Dict, parsed_results: Dict) -> Dict:
    professor_map = {}

    for i, p in enumerate(valid_papers, start=1):
        idx = str(i)
        info = parsed_results.get(idx, {})
        paper_obj = {
            "title": p.get("title", "제목 미상"),
            "k_title": info.get("title", p.get("title", "제목 미상")),
            "summary": info.get("sum", "요약 없음"),
            "date": p.get("publication_date", "날짜 미상"),
            "venue": p.get("venue", "게재처 미상"),
            "paper_relevance": p.get("paper_relevance", "Unknown"),
            "paper_score": p.get("paper_score", 0),
            "paper_reason": p.get("paper_reason", ""),
        }

        for name, is_pnu in p.get("raw_authors_info", []):
            if not is_pnu:
                continue

            db = author_db.get(name, {}) if isinstance(author_db, dict) else {}
            active = normalize_yes_no(db.get("is_active", "Unknown"))
            relevance = (db.get("relevance") or "Unknown").strip()

            if active == "No":
                continue
            if active != "Yes":
                continue
            if relevance == "Low":
                continue

            if name not in professor_map:
                professor_map[name] = {
                    "k_name": db.get("korean_name", "확인안됨"),
                    "dept": db.get("department", "확인안됨"),
                    "field": db.get("field", "확인안됨"),
                    "link": db.get("link", "#"),
                    "relevance": relevance,
                    "note": db.get("note", ""),
                    "papers": [],
                }

            if paper_obj not in professor_map[name]["papers"]:
                professor_map[name]["papers"].append(paper_obj)

    for _, data in professor_map.items():
        data["papers"] = sorted(
            data["papers"],
            key=lambda x: (0 if x.get("paper_relevance") == "High" else 1, -int(x.get("paper_score", 0))),
        )

    return professor_map


def unified_analyze(uploaded_file, manual_text: str, progress_callback=None) -> str:
    def report(step: int, total: int, label: str, detail: str = ""):
        if progress_callback:
            progress_callback(step, total, label, detail)

    total_steps = 8
    report(0, total_steps, "입력 확인", "파일 또는 직접 입력 내용을 점검하는 중입니다.")

    file_text = extract_text_from_uploaded_file(uploaded_file) if uploaded_file else ""
    query_text = file_text.strip() if file_text.strip() else (manual_text or "").strip()

    if len(query_text) < 5:
        return "분석할 내용이 없습니다. 파일을 업로드하거나 내용을 입력해주세요."

    report(1, total_steps, "기본 정보 추출", "기업명과 수요기술 요약을 정리하는 중입니다.")
    request_meta = extract_request_metadata(query_text)

    report(2, total_steps, "기술 프로파일 생성", "검색용 키워드와 기술 구조를 도출하는 중입니다.")
    search_profile = extract_search_profile(query_text)
    if not request_meta.get("tech_summary") and search_profile.get("korean_summary"):
        request_meta["tech_summary"] = search_profile.get("korean_summary")

    keywords_text = format_keyword_text(search_profile)

    report(3, total_steps, "논문 검색", "OpenAlex에서 부산대 관련 논문을 수집하는 중입니다.")
    raw_papers = search_openalex(
        tuple(search_profile.get("search_keywords", [])),
        tuple(search_profile.get("applications", [])),
        tuple(search_profile.get("core_tech", [])),
    )

    report(4, total_steps, "부산대 논문 필터링", f"수집 논문 {len(raw_papers)}건에서 부산대 소속 저자를 확인하는 중입니다.")
    pnu_papers, unique_pnu_authors = filter_pnu_papers(raw_papers)

    if not pnu_papers:
        report(total_steps, total_steps, "분석 완료", "부산대 소속 논문을 찾지 못했습니다.")
        company_name = request_meta.get("company_name", "미확인")
        tech_summary = request_meta.get("tech_summary", "입력된 수요기술 설명을 바탕으로 연구자 매칭을 수행했습니다.")
        return (
            f"### 🏢 기업명: **{company_name}**\n\n"
            f"### 📝 수요기술 요약\n{tech_summary}\n\n"
            f"### 🔍 분석 키워드: **{keywords_text}**\n\n"
            "- OpenAlex에서 부산대 소속 논문을 찾지 못했습니다.\n"
            "- 기술 설명을 더 구체적으로 입력하거나, 영문 기술명/응용 분야를 함께 넣어보세요."
        )

    report(5, total_steps, "논문 적합성 검증", f"부산대 논문 {len(pnu_papers)}건이 수요기술과 실제로 맞는지 평가하는 중입니다.")
    relevance_map = score_paper_relevance(pnu_papers, search_profile, request_meta.get("tech_summary", ""))
    valid_papers = select_relevant_papers(pnu_papers, relevance_map)

    if not valid_papers:
        valid_papers = pnu_papers[:MIN_RELEVANT_PAPERS]

    filtered_authors = []
    seen_filtered_authors = set()
    for p in valid_papers:
        for name, is_pnu in p.get("raw_authors_info", []):
            if is_pnu and name not in seen_filtered_authors:
                seen_filtered_authors.add(name)
                filtered_authors.append(name)

    report(6, total_steps, "교수 정보 확인", f"교수 후보 {len(filtered_authors[:MAX_AUTHORS_FOR_ENRICH])}명의 현직 여부와 전공 정보를 확인하는 중입니다.")
    paper_titles = [p.get("title", "") for p in valid_papers]
    author_db = enrich_authors_with_gemini(filtered_authors[:MAX_AUTHORS_FOR_ENRICH], search_profile, paper_titles)

    report(7, total_steps, "논문 요약 및 결과 정리", f"적합 논문 {len(valid_papers)}건을 요약하고 최종 결과를 구성하는 중입니다.")
    parsed_results = summarize_papers(valid_papers)
    professor_map = build_professor_map(valid_papers, author_db, parsed_results)

    high_count = sum(1 for p in valid_papers if p.get("paper_relevance") == "High")
    medium_count = sum(1 for p in valid_papers if p.get("paper_relevance") == "Medium")

    final_output = []
    final_output.append(f"### 🏢 기업명: **{request_meta.get('company_name', '미확인')}**")
    final_output.append("")
    final_output.append("### 📝 수요기술 요약")
    final_output.append(request_meta.get("tech_summary", "입력된 수요기술 설명을 바탕으로 연구자 매칭을 수행했습니다."))
    final_output.append("")
    final_output.append(f"### 🔍 분석 키워드: **{keywords_text}**")
    final_output.append("")
    final_output.append(f"- 검토 논문 수: **{len(pnu_papers)}건**")
    final_output.append(f"- 적합성 통과 논문 수: **{len(valid_papers)}건** (High {high_count}건 / Medium {medium_count}건)")
    final_output.append(f"- 검토 부산대 저자 수: **{len(filtered_authors)}명**")
    final_output.append(f"- 최종 매칭 교수 수: **{len(professor_map)}명**")
    final_output.append("")
    final_output.append("---")

    if not professor_map:
        report(total_steps, total_steps, "분석 완료", "교수 정보 확인까지 마쳤지만 최종 매칭 교수는 없었습니다.")
        final_output.append("현재 부산대학교 재직 전임교원으로 확인된 후보가 없거나, 교수 정보 확인이 충분하지 않았습니다.")
        final_output.append("")
        final_output.append("#### 점검 포인트")
        final_output.append("- 논문 적합성 검증 후 교수 후보 풀이 좁아진 경우")
        final_output.append("- OpenAlex에 교수명이 아니라 학생/연구원 이름 위주로 잡힌 경우")
        final_output.append("- Gemini 검색에서 교수 정보 확인이 충분히 되지 않은 경우")
        final_output.append("- 입력 기술 설명이 너무 짧거나 일반적이라 연관 논문이 넓게 잡힌 경우")
        return "\n".join(final_output)

    sorted_professors = sorted(
        professor_map.items(),
        key=lambda x: (
            0 if x[1].get("relevance") == "High" else 1,
            -sum(int(p.get("paper_score", 0)) for p in x[1].get("papers", [])),
        ),
    )

    for eng_name, data in sorted_professors:
        final_output.append(f"## 🏫 {data['dept']} | {data['k_name']} 교수 ({eng_name})")
        final_output.append(f"- **주요 연구분야:** {data['field']}")
        final_output.append(f"- **기술 연관성:** {data.get('relevance', 'Unknown')}")
        if data.get("note"):
            final_output.append(f"- **판단 메모:** {data['note']}")
        final_output.append(f"- **학과/연구실 홈페이지:** [링크 바로가기]({data['link']})")
        final_output.append("")
        final_output.append("#### 📄 관련 연구 논문 내역")
        for idx, paper in enumerate(data["papers"], start=1):
            final_output.append(f"{idx}. **{paper['k_title']}**")
            final_output.append(f"   - 원제: {paper['title']}")
            final_output.append(f"   - 논문 적합도: {paper.get('paper_relevance', 'Unknown')} ({paper.get('paper_score', 0)}점)")
            if paper.get("paper_reason"):
                final_output.append(f"   - 적합성 근거: {paper['paper_reason']}")
            final_output.append(f"   - 요약: {paper['summary']} ({paper['date']}, {paper['venue']})")
        final_output.append("")
        final_output.append("---")

    report(total_steps, total_steps, "분석 완료", f"최종 매칭 교수 {len(professor_map)}명을 정리했습니다.")
    return "\n".join(final_output)


# -----------------------------
# UI
# -----------------------------
st.title("🎓 PNU 수요기술-연구자 매칭 시스템")
st.caption("수요기술을 기반으로 부산대학교 연구자를 매칭합니다")

with st.sidebar:
    st.header("설정 안내")
    st.markdown(
        """
- PDF, DOCX, TXT, MD 업로드 가능
        """
    )

uploaded_file = st.file_uploader(
    "1. 수요기술조사서 업로드", type=["pdf", "docx", "txt", "md"]
)
manual_text = st.text_area(
    "2. 또는 기술 내용 직접 입력 (선택)",
    placeholder="기술 설명, 적용 분야, 핵심 성능, 장치/공정/소재 정보를 넣으면 연구자 매칭 정확도가 올라갑니다.",
    height=240,
)

if st.button("연구자 매칭 리스트 생성", type="primary"):
    status_box = st.status("분석 준비 중입니다...", expanded=True)
    progress_bar = st.progress(0)
    step_placeholder = st.empty()

    def update_progress(step, total, label, detail=""):
        ratio = 0 if total == 0 else min(max(step / total, 0), 1)
        progress_bar.progress(ratio)
        status_box.update(label=label, state="running", expanded=True)
        message = f"**진행 단계:** {label}"
        if detail:
            message += f"  \n- {detail}"
        step_placeholder.markdown(message)

    try:
        result = unified_analyze(uploaded_file, manual_text, progress_callback=update_progress)
        progress_bar.progress(1.0)
        status_box.update(label="분석 완료", state="complete", expanded=False)
        step_placeholder.success("분석이 완료되었습니다. 아래 결과를 확인하세요.")
        st.markdown(result)
    except Exception as e:
        status_box.update(label="분석 중 오류 발생", state="error", expanded=True)
        progress_bar.progress(0)
        step_placeholder.error(f"오류가 발생했습니다: {e}")
