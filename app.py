import os
import re
import json
import time
import html
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import requests
import pdfplumber
import streamlit as st
from docx import Document
from google import genai

st.set_page_config(page_title="부산대 수요기술-연구자 증거형 매칭 시스템", layout="wide")

OPENALEX_URL = "https://api.openalex.org/works"
KIPRIS_BASE_URL = "https://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice"
MAX_PAPERS = 20
MAX_PATENTS = 20
MAX_AUTHORS = 30
MIN_RELEVANT_PAPERS = 3
MIN_RELEVANT_PATENTS = 3
USER_AGENT = "Mozilla/5.0 (EvidenceOnlyPNUMatcher/1.1)"
REQUEST_TIMEOUT = 20
OFFICIAL_DOMAINS = [
    "pusan.ac.kr",
    "pnu.edu",
]
PNU_IUCF_APPLICANT_KR = "부산대학교 산학협력단"
PNU_IUCF_APPLICANT_EN_HINTS = [
    "industry-university cooperation foundation",
    "industry academic cooperation foundation",
    "industry university cooperation foundation",
]


# -----------------------------
# Environment / Client
# -----------------------------
def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


GEMINI_API_KEY = get_env("GEMINI_API_KEY")
OPENALEX_API_KEY = get_env("OPENALEX_API_KEY")
KIPRIS_API_KEY = get_env("KIPRIS_API_KEY")


@st.cache_resource(show_spinner=False)
def init_client():
    if not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)


client = init_client()
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


# -----------------------------
# Utilities
# -----------------------------
def compact_text(text: str, limit: int = 5000) -> str:
    return (text or "").strip()[:limit]


@st.cache_data(show_spinner=False)
def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""
    word_index = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_index.append((pos, word))
    word_index.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_index)


@st.cache_data(show_spinner=False)
def extract_json_object(text: str) -> Dict:
    raw = (text or "").strip()
    if not raw:
        return {}
    cleaned = raw.replace("```json", "").replace("```", "").strip()
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


def safe_gemini_json(prompt: str, retries: int = 2) -> Dict:
    if client is None:
        return {}
    for model_name in ["gemini-2.5-flash", "gemini-2.5-flash-lite"]:
        for attempt in range(retries):
            try:
                res = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return extract_json_object(getattr(res, "text", ""))
            except Exception as e:
                if any(code in str(e) for code in ["429", "503"]) and attempt < retries - 1:
                    time.sleep(2)
                    continue
                break
    return {}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


@st.cache_data(show_spinner=False)
def file_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    text = ""
    try:
        if ext == ".pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
        elif ext == ".docx":
            doc = Document(uploaded_file)
            text = "\n".join(p.text for p in doc.paragraphs)
        elif ext in {".txt", ".md"}:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""
    return text


def clamp_score(value, default=60):
    try:
        score = int(value)
    except Exception:
        return default
    return max(0, min(100, score))


def split_people_text(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[;,/]|\s{2,}|\n|\||·|ㆍ", text)
    cleaned = []
    seen = set()
    for part in parts:
        p = normalize_space(part)
        p = re.sub(r"\(.*?\)", "", p).strip()
        if not p or len(p) < 2:
            continue
        if p not in seen:
            seen.add(p)
            cleaned.append(p)
    return cleaned


def safe_lower(x) -> str:
    return str(x or "").lower()


def escape_xml_text(text: str) -> str:
    return html.escape(str(text or ""), quote=True)


def extract_texts_by_tag(root: ET.Element, tag_name: str) -> List[str]:
    values = []
    for elem in root.iter():
        if elem.tag.split('}')[-1] == tag_name:
            val = normalize_space(elem.text or "")
            if val:
                values.append(val)
    return values


def first_text_by_tags(root: ET.Element, candidates: List[str]) -> str:
    for name in candidates:
        vals = extract_texts_by_tag(root, name)
        if vals:
            return vals[0]
    return ""


def unique_keep_order(items: List[str]) -> List[str]:
    out, seen = [], set()
    for item in items:
        key = normalize_space(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def strip_tags(raw_html: str) -> str:
    if not raw_html:
        return ""
    text = re.sub(r"<script[\s\S]*?</script>", " ", raw_html, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return normalize_space(text)


def normalize_name_for_match(name: str) -> str:
    s = normalize_space(name).lower()
    s = s.replace("–", "-").replace("—", "-").replace("‑", "-")
    s = re.sub(r"[^a-z0-9가-힣]", "", s)
    return s


def build_name_variants(name: str) -> List[str]:
    base = normalize_space(name)
    if not base:
        return []
    variants = [base]
    if "-" in base or "–" in base or "—" in base or "‑" in base:
        variants.append(base.replace("–", " ").replace("—", " ").replace("‑", " ").replace("-", " "))
        variants.append(base.replace("–", "").replace("—", "").replace("‑", "").replace("-", ""))
    if " " in base:
        variants.append(base.replace(" ", ""))
        variants.append("-".join(base.split()))
    return unique_keep_order(variants)


def extract_duckduckgo_results(raw_html: str, max_results: int = 8) -> List[Dict[str, str]]:
    results = []
    seen = set()
    patterns = [
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        r'<a[^>]+href="([^"]+)"[^>]*class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>',
    ]
    for pattern in patterns:
        for href, title_html in re.findall(pattern, raw_html, flags=re.IGNORECASE | re.DOTALL):
            title = strip_tags(title_html)
            href = html.unescape(href).strip()
            if not href or not title or href in seen:
                continue
            seen.add(href)
            results.append({"title": title, "url": href})
            if len(results) >= max_results:
                return results
    return results


# -----------------------------
# Input parsing
# -----------------------------
def extract_request_metadata(query_text: str) -> Dict[str, str]:
    prompt = f"""
당신은 대학 산학협력 실무용 입력정보 정리기입니다.
아래 텍스트에서 기업명과 수요기술 요약만 JSON으로 추출하세요.

규칙:
- 기업명이 명확하지 않으면 \"미확인\"
- 수요기술 요약은 한국어 1~2문장
- 과장 없이 핵심 기술/성능/적용처 중심
- 출력은 JSON만

형식:
{{
  \"company_name\": \"기업명 또는 미확인\",
  \"tech_summary\": \"수요기술 요약\"
}}

입력:
{compact_text(query_text)}
"""
    data = safe_gemini_json(prompt)
    return {
        "company_name": str(data.get("company_name") or "미확인").strip() or "미확인",
        "tech_summary": str(data.get("tech_summary") or "입력된 수요기술 설명을 바탕으로 연구자 매칭을 수행했습니다.").strip(),
    }


def extract_search_profile(query_text: str) -> Dict:
    fallback_tokens = []
    for token in [x.strip(",.()[]{}") for x in query_text.replace("\n", " ").split() if x.strip()]:
        if len(token) >= 4:
            fallback_tokens.append(token)
        if len(fallback_tokens) >= 6:
            break

    prompt = f"""
당신은 대학 산학협력용 검색 프로파일 설계기입니다.
아래 수요기술 설명을 바탕으로 논문/특허 검색용 키워드 JSON을 작성하세요.

반드시 포함할 항목:
- core_tech: 핵심 기술 2~4개 (영어)
- materials_or_methods: 재료/방법 2~4개 (영어)
- properties: 요구 특성 1~4개 (영어)
- applications: 적용처 1~3개 (영어)
- search_keywords: 검색용 핵심 키워드 4~6개 (영어 짧은 구)
- korean_patent_keywords: 특허 검색용 한국어 핵심 키워드 4~8개
- exclude_keywords: 배제 키워드 0~4개 (영어)
- korean_summary: 한국어 한두 문장

출력은 JSON만 반환.

입력:
{compact_text(query_text)}
"""
    data = safe_gemini_json(prompt)
    if isinstance(data, dict) and data.get("search_keywords"):
        for key in [
            "core_tech",
            "materials_or_methods",
            "properties",
            "applications",
            "search_keywords",
            "exclude_keywords",
            "korean_patent_keywords",
        ]:
            data[key] = [str(x).strip() for x in data.get(key, []) if str(x).strip()]
        data["korean_summary"] = str(data.get("korean_summary", "")).strip()
        return data

    fallback = [t for t in fallback_tokens if len(t) >= 2][:6]
    return {
        "core_tech": fallback[:2],
        "materials_or_methods": fallback[2:4],
        "properties": [],
        "applications": [],
        "search_keywords": fallback or ["Pusan National University"],
        "korean_patent_keywords": fallback or ["부산대학교"],
        "exclude_keywords": [],
        "korean_summary": "입력된 수요기술 설명을 바탕으로 검색 키워드를 구성했습니다.",
    }


def format_keyword_text(profile: Dict) -> str:
    return ", ".join(profile.get("search_keywords", []))


def format_patent_keyword_text(profile: Dict) -> str:
    return ", ".join(profile.get("korean_patent_keywords", []))


# -----------------------------
# OpenAlex search / relevance
# -----------------------------
@st.cache_data(show_spinner=False)
def search_openalex(search_keywords: Tuple[str, ...], applications: Tuple[str, ...], core_tech: Tuple[str, ...]) -> List[Dict]:
    keywords = list(search_keywords)
    apps = list(applications)
    techs = list(core_tech)

    queries = []
    base = keywords[:4] if keywords else techs[:3]
    if base:
        queries.append(" ".join(base[:3]) + " Pusan National University")
        queries.append(" OR ".join(base[:3]) + " Pusan National University")
        queries.append(" ".join(base[:2]))
    if techs and apps:
        queries.append(f"{' '.join(techs[:2])} {' '.join(apps[:2])} Pusan National University")
    if techs:
        queries.append(" OR ".join(techs[:3]))

    seen = set()
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
            r = SESSION.get(OPENALEX_URL, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                continue
            items = r.json().get("results", [])
        except Exception:
            continue

        for item in items:
            item_id = item.get("id") or item.get("title")
            if item_id in seen:
                continue
            seen.add(item_id)
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

        for authorship in p.get("authorships") or []:
            authorship = authorship or {}
            name = str((authorship.get("author") or {}).get("display_name") or "Unknown")
            institutions = authorship.get("institutions") or []
            inst_names = [safe_lower((inst or {}).get("display_name")) for inst in institutions]
            raw_aff = safe_lower(authorship.get("raw_affiliation_string"))
            combined = " ".join(inst_names) + " " + raw_aff
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
            p["venue"] = str(source.get("display_name") or "게재처 미상")
            p["raw_authors_info"] = p_authors_info
            valid_papers.append(p)
        if len(valid_papers) >= MAX_PAPERS:
            break

    return valid_papers, unique_pnu_authors[:MAX_AUTHORS]


def score_paper_relevance(valid_papers: List[Dict], profile: Dict, tech_summary: str) -> Dict[str, Dict]:
    if not valid_papers:
        return {}
    blocks = []
    for i, p in enumerate(valid_papers, start=1):
        abs_text = reconstruct_abstract(p.get("abstract_inverted_index"))
        blocks.append(f"[{i}] Title: {p.get('title', '')}\nAbstract: {abs_text[:900]}\n")

    prompt = f"""
당신은 대학 산학협력용 논문 적합성 평가기입니다.
아래 수요기술과 논문 목록을 비교해 각 논문의 적합도를 JSON으로 반환하세요.

수요기술 요약:
{tech_summary}

프로파일:
- core_tech: {profile.get('core_tech', [])}
- materials_or_methods: {profile.get('materials_or_methods', [])}
- properties: {profile.get('properties', [])}
- applications: {profile.get('applications', [])}
- exclude_keywords: {profile.get('exclude_keywords', [])}

등급:
- High: 직접 관련
- Medium: 인접 관련
- Low: 부분적 관련
- Exclude: 비관련

출력 형식:
{{
  "1": {{"relevance": "High/Medium/Low/Exclude", "score": 0-100, "reason": "한 줄 근거"}}
}}

논문 목록:
{chr(10).join(blocks)}
"""
    data = safe_gemini_json(prompt)
    if isinstance(data, dict) and data:
        return data
    fallback = {}
    for i, _ in enumerate(valid_papers, start=1):
        fallback[str(i)] = {"relevance": "Medium", "score": 60, "reason": "적합성 평가 실패로 기본값 적용"}
    return fallback


def select_relevant_papers(valid_papers: List[Dict], relevance_map: Dict[str, Dict]) -> List[Dict]:
    selected = []
    medium = []
    low = []
    for i, p in enumerate(valid_papers, start=1):
        rel = relevance_map.get(str(i), {}) if isinstance(relevance_map, dict) else {}
        label = str(rel.get("relevance", "Medium")).strip()
        score = clamp_score(rel.get("score", 60), 60)
        reason = str(rel.get("reason", "")).strip()
        p["paper_relevance"] = label
        p["paper_score"] = score
        p["paper_reason"] = reason
        if label == "High":
            selected.append(p)
        elif label == "Medium":
            medium.append(p)
        elif label == "Low":
            low.append(p)
    selected.extend(sorted(medium, key=lambda x: x.get("paper_score", 0), reverse=True))
    if len(selected) < MIN_RELEVANT_PAPERS:
        selected.extend(sorted(low, key=lambda x: x.get("paper_score", 0), reverse=True)[:MIN_RELEVANT_PAPERS - len(selected)])
    return selected[:MAX_PAPERS]


# -----------------------------
# KIPRIS patent search / detail
# -----------------------------
def kipris_enabled() -> bool:
    return bool(KIPRIS_API_KEY)


@st.cache_data(show_spinner=False)
def kipris_call(operation: str, params: Tuple[Tuple[str, str], ...]) -> Optional[ET.Element]:
    if not KIPRIS_API_KEY:
        return None
    final_params = {k: v for k, v in params if v not in [None, ""]}
    final_params["ServiceKey"] = KIPRIS_API_KEY
    url = f"{KIPRIS_BASE_URL}/{operation}"
    try:
        r = SESSION.get(url, params=final_params, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None
        return ET.fromstring(r.text)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def search_kipris_patents(korean_keywords: Tuple[str, ...]) -> List[Dict]:
    if not KIPRIS_API_KEY:
        return []

    queries = []
    words = [x for x in korean_keywords if x]
    if words:
        queries.append(" ".join(words[:3]))
        queries.append(" ".join(words[:2]))
        for kw in words[:5]:
            queries.append(kw)

    collected = []
    seen = set()
    applicant_queries = [PNU_IUCF_APPLICANT_KR, "부산대학교"]

    for q in queries[:7]:
        for applicant in applicant_queries:
            root = kipris_call(
                "getAdvancedSearch",
                tuple(sorted({
                    "word": q,
                    "applicant": applicant,
                    "patent": "true",
                    "utility": "false",
                    "lastvalue": "R",
                    "sortSpec": "AD",
                    "descSort": "true",
                    "pageNo": "1",
                    "numOfRows": "40",
                }.items()))
            )
            if root is None:
                continue
            items = [elem for elem in root.iter() if elem.tag.split('}')[-1] == 'items']
            if not items:
                continue
            for item in items[0]:
                app_no = first_text_by_tags(item, ["applicationNumber", "ApplicationNumber"])
                title = first_text_by_tags(item, ["inventionTitle", "InventionTitle"])
                abstract = first_text_by_tags(item, ["astrtCont", "astrtContent", "AstrtCont"])
                applicant_name = first_text_by_tags(item, ["applicantName", "ApplicantName"])
                application_date = first_text_by_tags(item, ["applicationDate", "ApplicationDate"])
                register_number = first_text_by_tags(item, ["registerNumber", "RegisterNumber"])
                register_date = first_text_by_tags(item, ["registerDate", "RegisterDate"])
                register_status = first_text_by_tags(item, ["registerStatus", "RegisterStatus"])
                key = app_no or f"{title}|{application_date}"
                if not key or key in seen:
                    continue
                seen.add(key)
                collected.append({
                    "application_number": app_no,
                    "title": title or "발명의 명칭 미상",
                    "abstract": abstract,
                    "application_date": application_date,
                    "register_number": register_number,
                    "register_date": register_date,
                    "register_status": register_status,
                    "applicant_name_summary": applicant_name,
                    "search_query": q,
                })
                if len(collected) >= 60:
                    return collected
    return collected


@st.cache_data(show_spinner=False)
def get_kipris_bibliography_detail(application_number: str) -> Optional[Dict]:
    if not application_number:
        return None
    root = kipris_call(
        "getBibliographyDetailInfoSearch",
        tuple(sorted({"applicationNumber": application_number}.items()))
    )
    if root is None:
        return None

    applicant_names = unique_keep_order(extract_texts_by_tag(root, "name"))
    applicant_eng_names = unique_keep_order(extract_texts_by_tag(root, "engName"))
    inventor_names = unique_keep_order(extract_texts_by_tag(root, "name"))

    # name 태그가 applicant / inventor 양쪽에서 반복되므로 배열 단위로 다시 한 번 분리 추출
    def extract_nested_names(array_tag: str) -> List[str]:
        vals = []
        for elem in root.iter():
            if elem.tag.split('}')[-1] == array_tag:
                for sub in elem.iter():
                    if sub.tag.split('}')[-1] == 'name':
                        val = normalize_space(sub.text or '')
                        if val:
                            vals.append(val)
        return unique_keep_order(vals)

    applicant_names = extract_nested_names("applicantInfoArray") or applicant_names[:5]
    inventor_names = extract_nested_names("inventorInfoArray") or inventor_names[:10]

    detail = {
        "application_number": first_text_by_tags(root, ["applicationNumber"]),
        "application_date": first_text_by_tags(root, ["applicationDate"]),
        "title": first_text_by_tags(root, ["inventionTitle"]),
        "title_eng": first_text_by_tags(root, ["inventionTitleEng"]),
        "abstract": first_text_by_tags(root, ["astrtCont"]),
        "register_number": first_text_by_tags(root, ["registerNumber"]),
        "register_date": first_text_by_tags(root, ["registerDate"]),
        "register_status": first_text_by_tags(root, ["registerStatus"]),
        "applicant_names": applicant_names,
        "applicant_eng_names": applicant_eng_names,
        "inventor_names": inventor_names,
        "ipc_numbers": unique_keep_order(extract_texts_by_tag(root, "ipcNumber"))[:8],
    }
    return detail


def is_pnu_iucf_included(detail: Dict) -> bool:
    applicants = [normalize_space(x) for x in detail.get("applicant_names", []) if normalize_space(x)]
    if not applicants:
        return False
    for name in applicants:
        if name == PNU_IUCF_APPLICANT_KR:
            return True
        low = name.lower()
        if ("pusan national university" in low or "busan national university" in low) and any(h in low for h in PNU_IUCF_APPLICANT_EN_HINTS):
            return True
    return False


@st.cache_data(show_spinner=False)
def enrich_and_filter_pnu_iucf_patents(raw_patents: List[Dict]) -> Tuple[List[Dict], List[str]]:
    valid = []
    inventors = []
    seen_people = set()
    for item in raw_patents:
        app_no = item.get("application_number", "")
        if not app_no:
            continue
        detail = get_kipris_bibliography_detail(app_no)
        if not detail:
            continue
        if not is_pnu_iucf_included(detail):
            continue
        merged = {**item, **detail}
        valid.append(merged)
        for name in detail.get("inventor_names", []):
            if name not in seen_people:
                seen_people.add(name)
                inventors.append(name)
        if len(valid) >= MAX_PATENTS:
            break
    return valid, inventors[:MAX_AUTHORS]


def score_patent_relevance(valid_patents: List[Dict], profile: Dict, tech_summary: str) -> Dict[str, Dict]:
    if not valid_patents:
        return {}
    blocks = []
    for i, p in enumerate(valid_patents, start=1):
        blocks.append(
            f"[{i}] Title: {p.get('title', '')}\n"
            f"Abstract: {compact_text(p.get('abstract', ''), 1200)}\n"
            f"IPC: {', '.join(p.get('ipc_numbers', []))}\n"
        )
    prompt = f"""
당신은 대학 산학협력용 특허 적합성 평가기입니다.
아래 수요기술과 특허 목록을 비교해 각 특허의 적합도를 JSON으로 반환하세요.

수요기술 요약:
{tech_summary}

프로파일:
- core_tech: {profile.get('core_tech', [])}
- materials_or_methods: {profile.get('materials_or_methods', [])}
- properties: {profile.get('properties', [])}
- applications: {profile.get('applications', [])}
- korean_patent_keywords: {profile.get('korean_patent_keywords', [])}

등급:
- High: 직접 관련
- Medium: 인접 관련
- Low: 부분적 관련
- Exclude: 비관련

출력 형식:
{{
  "1": {{"relevance": "High/Medium/Low/Exclude", "score": 0-100, "reason": "한 줄 근거"}}
}}

특허 목록:
{chr(10).join(blocks)}
"""
    data = safe_gemini_json(prompt)
    if isinstance(data, dict) and data:
        return data
    fallback = {}
    for i, _ in enumerate(valid_patents, start=1):
        fallback[str(i)] = {"relevance": "Medium", "score": 60, "reason": "적합성 평가 실패로 기본값 적용"}
    return fallback


def select_relevant_patents(valid_patents: List[Dict], relevance_map: Dict[str, Dict]) -> List[Dict]:
    selected, medium, low = [], [], []
    for i, p in enumerate(valid_patents, start=1):
        rel = relevance_map.get(str(i), {}) if isinstance(relevance_map, dict) else {}
        label = str(rel.get("relevance", "Medium")).strip()
        score = clamp_score(rel.get("score", 60), 60)
        reason = str(rel.get("reason", "")).strip()
        p["patent_relevance"] = label
        p["patent_score"] = score
        p["patent_reason"] = reason
        if label == "High":
            selected.append(p)
        elif label == "Medium":
            medium.append(p)
        elif label == "Low":
            low.append(p)
    selected.extend(sorted(medium, key=lambda x: x.get("patent_score", 0), reverse=True))
    if len(selected) < MIN_RELEVANT_PATENTS:
        selected.extend(sorted(low, key=lambda x: x.get("patent_score", 0), reverse=True)[:MIN_RELEVANT_PATENTS - len(selected)])
    return selected[:MAX_PATENTS]


def summarize_patents(valid_patents: List[Dict]) -> Dict[str, Dict[str, str]]:
    if not valid_patents:
        return {}
    blocks = []
    for i, p in enumerate(valid_patents, start=1):
        blocks.append(
            f"[{i}] Title: {p.get('title', '')}\n"
            f"Abstract: {compact_text(p.get('abstract', ''), 1000)}\n"
        )
    prompt = f"""
아래 특허들에 대해 각 번호별로
1) 한국어 정리 제목
2) 기술 핵심 요약 한 줄
을 작성하세요.

출력 형식:
[번호] 제목 | 요약내용

{chr(10).join(blocks)}
"""
    if client is None:
        return {}
    try:
        res = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        parsed = {}
        for line in getattr(res, "text", "").split("\n"):
            if "|" in line and line.strip().startswith("[") and "]" in line:
                parts = line.split("]", 1)
                idx = parts[0].replace("[", "").strip()
                title_part, sum_part = parts[1].split("|", 1)
                parsed[idx] = {
                    "title": title_part.strip(),
                    "sum": sum_part.strip(),
                }
        return parsed
    except Exception:
        return {}


# -----------------------------
# Evidence-only professor verification
# -----------------------------
def is_official_domain(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return False
    return any(netloc.endswith(domain) for domain in OFFICIAL_DOMAINS)


@st.cache_data(show_spinner=False)
def duckduckgo_search(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    url = "https://html.duckduckgo.com/html/"
    try:
        r = SESSION.post(url, data={"q": query}, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []
        return extract_duckduckgo_results(r.text, max_results=max_results)
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def fetch_page_text(url: str) -> str:
    try:
        r = SESSION.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return ""
        return strip_tags(r.text)[:15000]
    except Exception:
        return ""


def compute_verification_score(page_text: str, candidate_names: List[str], url: str = "") -> Tuple[int, str]:
    low_text = page_text.lower()
    norm_page = normalize_name_for_match(page_text)
    low_url = (url or "").lower()

    professor_keywords = [
        "professor", "assistant professor", "associate professor", "full professor",
        "faculty", "faculty member", "researcher", "교수", "조교수", "부교수", "정교수", "교원", "전임교원", "연구자"
    ]
    department_keywords = [
        "department", "school", "college", "faculty", "lab", "laboratory",
        "학과", "학부", "대학", "대학원", "전공", "연구실", "scholar"
    ]
    research_keywords = [
        "research interests", "research areas", "publications", "patents",
        "연구분야", "주요 연구분야", "논문", "특허", "profile", "faculty profile"
    ]

    matched_name = ""
    score = 0
    for variant in candidate_names:
        norm_variant = normalize_name_for_match(variant)
        if norm_variant and norm_variant in norm_page:
            matched_name = variant
            score += 4
            break
    if not matched_name:
        return 0, ""

    if any(k in low_text for k in professor_keywords):
        score += 3
    if any(k in low_text for k in department_keywords):
        score += 2
    if any(k in low_text for k in research_keywords):
        score += 1
    if "scholar.pusan.ac.kr" in low_url:
        score += 2
    if any(k in low_url for k in ["subview.do", "faculty", "professor", "research", "lab"]):
        score += 1
    return score, matched_name


def normalize_name_for_match(text: str) -> str:
    s = str(text or "").lower()
    s = s.replace("–", "-").replace("—", "-").replace("-", "-")
    s = re.sub(r"[^a-z0-9가-힣]", "", s)
    return s

def extract_html_text(raw_html: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw_html or "")
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    return normalize_space(text)

@st.cache_data(show_spinner=False)
def verify_professor_from_official_pages(author_name: str) -> Optional[Dict]:
    name = normalize_space(author_name)
    if not name:
        return None

    norm_name = normalize_name_for_match(name)

    candidate_urls = [
        f"https://scholar.pusan.ac.kr/search?q={name}",
        f"https://scholar.pusan.ac.kr/search?q={name.replace('-', ' ')}",
    ]

    # 영문 이름이면 공백 제거/하이픈 제거 버전도 같이 시도
    compact_name = re.sub(r"[-\s]+", "", name)
    if compact_name and compact_name != name:
        candidate_urls.append(f"https://scholar.pusan.ac.kr/search?q={compact_name}")

    for url in candidate_urls:
        try:
            r = SESSION.get(url, timeout=8)
            if r.status_code != 200:
                continue

            page_text = extract_html_text(r.text)[:30000]
            norm_page = normalize_name_for_match(page_text)

            if norm_name not in norm_page:
                continue

            # scholar.pusan.ac.kr 내부에서 연구자/교수 흔적 점수화
            score = 0
            low_text = page_text.lower()
            if "scholar" in url:
                score += 2
            if any(k in low_text for k in ["professor", "faculty", "교수", "조교수", "부교수", "정교수", "researcher", "연구자"]):
                score += 2
            if any(k in low_text for k in ["department", "school", "college", "학과", "학부", "대학", "전공", "연구실", "laboratory", "lab"]):
                score += 1

            if score < 2:
                continue

            evidence = extract_evidence_snippet(page_text, name)
            dept = extract_department_from_page(page_text)
            field = extract_field_from_page(page_text)

            return {
                "official_name": name,
                "department": dept or "PNU Scholar 기반 확인",
                "field": field or "Scholar 페이지 기반 확인",
                "link": url,
                "evidence": evidence or "scholar.pusan.ac.kr 검색 결과에서 이름 확인",
            }
        except Exception:
            continue

    return None

    candidate_names = build_name_variants(name)
    queries = []
    for variant in candidate_names[:5]:
        queries.extend([
            f'site:pusan.ac.kr "{variant}"',
            f'site:scholar.pusan.ac.kr "{variant}"',
            f'site:pusan.ac.kr "{variant}" professor',
            f'site:pusan.ac.kr "{variant}" 교수',
            f'site:pusan.ac.kr "{variant}" 연구실',
            f'site:pusan.ac.kr "{variant}" 학과',
        ])
    queries = unique_keep_order(queries)

    best = None
    best_score = 0
    best_name = name
    checked_urls = set()

    for query in queries[:24]:
        results = duckduckgo_search(query, max_results=8)
        for item in results:
            url = item.get("url", "")
            if not is_official_domain(url) or url in checked_urls:
                continue
            checked_urls.add(url)
            page_text = fetch_page_text(url)
            if not page_text:
                continue

            score, matched_name = compute_verification_score(page_text, candidate_names, url=url)
            if score < 5:
                continue

            dept = extract_department_from_page(page_text)
            field = extract_field_from_page(page_text)
            evidence = extract_evidence_snippet(page_text, matched_name or candidate_names[0])
            if dept:
                score += 1
            if field:
                score += 1

            current = {
                "official_name": matched_name or name,
                "department": dept or "확인됨(세부 학과 추출 실패)",
                "field": field or "공식 페이지에서 상세 연구분야 자동 추출 실패",
                "link": url,
                "evidence": evidence or "공식 페이지에서 이름과 교수/연구자 표현을 확인함",
            }
            if score > best_score:
                best = current
                best_score = score
                best_name = matched_name or name

    if best:
        best["official_name"] = best_name
    return best


def extract_department_from_page(text: str) -> str:
    patterns = [
        r"(Department of [A-Za-z0-9\-&, ]{3,80})",
        r"(School of [A-Za-z0-9\-&, ]{3,80})",
        r"([가-힣A-Za-z0-9·\- ]{2,40}학과)",
        r"([가-힣A-Za-z0-9·\- ]{2,40}학부)",
        r"([가-힣A-Za-z0-9·\- ]{2,40}대학)",
        r"([가-힣A-Za-z0-9·\- ]{2,40}전공)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return normalize_space(m.group(1))[:120]
    return ""


def extract_field_from_page(text: str) -> str:
    patterns = [
        r"(Research Interests?[:\s][A-Za-z0-9,;\-()/. ]{10,200})",
        r"(Research Areas?[:\s][A-Za-z0-9,;\-()/. ]{10,200})",
        r"(연구분야[:\s][가-힣A-Za-z0-9,;·\-()/. ]{5,200})",
        r"(주요 연구분야[:\s][가-힣A-Za-z0-9,;·\-()/. ]{5,200})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return normalize_space(m.group(1))[:200]
    return ""


def extract_evidence_snippet(text: str, author_name: str) -> str:
    idx = text.lower().find(author_name.lower())
    if idx == -1:
        norm_author = normalize_name_for_match(author_name)
        norm_text = normalize_name_for_match(text)
        idx = norm_text.find(norm_author)
        if idx == -1:
            return ""
        return normalize_space(text[:300])
    start = max(0, idx - 120)
    end = min(len(text), idx + 220)
    return normalize_space(text[start:end])


# -----------------------------
# Paper/Patent summary
# -----------------------------
def summarize_papers(valid_papers: List[Dict]) -> Dict[str, Dict[str, str]]:
    if not valid_papers:
        return {}
    blocks = []
    for i, p in enumerate(valid_papers, start=1):
        abs_text = reconstruct_abstract(p.get("abstract_inverted_index"))
        blocks.append(f"[{i}] Title: {p.get('title')}\nAbstract: {abs_text[:700]}\n")

    prompt = f"""
아래 논문들에 대해 각 번호별로
1) 한국어 번역 제목
2) 기술 핵심 요약 한 줄
을 작성하세요.

출력 형식:
[번호] 번역제목 | 요약내용

{chr(10).join(blocks)}
"""
    if client is None:
        return {}
    try:
        res = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        parsed = {}
        for line in getattr(res, "text", "").split("\n"):
            if "|" in line and line.strip().startswith("[") and "]" in line:
                parts = line.split("]", 1)
                idx = parts[0].replace("[", "").strip()
                title_part, sum_part = parts[1].split("|", 1)
                parsed[idx] = {
                    "title": title_part.strip(),
                    "sum": sum_part.strip(),
                }
        return parsed
    except Exception:
        return {}


# -----------------------------
# Result assembly
# -----------------------------
def build_verified_researcher_map(
    valid_papers: List[Dict],
    valid_patents: List[Dict],
    verified_authors: Dict[str, Dict],
    parsed_papers: Dict,
    parsed_patents: Dict,
) -> Dict:
    researcher_map = {}

    for i, p in enumerate(valid_papers, start=1):
        info = parsed_papers.get(str(i), {})
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
            if not is_pnu or name not in verified_authors:
                continue
            db = verified_authors[name]
            if name not in researcher_map:
                researcher_map[name] = {
                    "dept": db.get("department", "공식 확인됨"),
                    "field": db.get("field", "공식 페이지 기반 확인"),
                    "link": db.get("link", "#"),
                    "evidence": db.get("evidence", "공식 페이지 확인"),
                    "papers": [],
                    "patents": [],
                }
            researcher_map[name]["papers"].append(paper_obj)

    for i, p in enumerate(valid_patents, start=1):
        info = parsed_patents.get(str(i), {})
        patent_obj = {
            "title": p.get("title", "발명의 명칭 미상"),
            "k_title": info.get("title", p.get("title", "발명의 명칭 미상")),
            "summary": info.get("sum", "요약 없음"),
            "application_number": p.get("application_number", "출원번호 미상"),
            "application_date": p.get("application_date", "출원일자 미상"),
            "register_number": p.get("register_number", ""),
            "register_date": p.get("register_date", ""),
            "register_status": p.get("register_status", ""),
            "patent_relevance": p.get("patent_relevance", "Unknown"),
            "patent_score": p.get("patent_score", 0),
            "patent_reason": p.get("patent_reason", ""),
            "applicant_names": p.get("applicant_names", []),
        }
        for name in p.get("inventor_names", []):
            if name not in verified_authors:
                continue
            db = verified_authors[name]
            if name not in researcher_map:
                researcher_map[name] = {
                    "dept": db.get("department", "공식 확인됨"),
                    "field": db.get("field", "공식 페이지 기반 확인"),
                    "link": db.get("link", "#"),
                    "evidence": db.get("evidence", "공식 페이지 확인"),
                    "papers": [],
                    "patents": [],
                }
            researcher_map[name]["patents"].append(patent_obj)

    for _, data in researcher_map.items():
        unique_papers, seen_papers = [], set()
        for p in sorted(data["papers"], key=lambda x: (-int(x.get("paper_score", 0)), x.get("title", ""))):
            key = (p.get("title"), p.get("date"))
            if key in seen_papers:
                continue
            seen_papers.add(key)
            unique_papers.append(p)
        data["papers"] = unique_papers

        unique_patents, seen_patents = [], set()
        for p in sorted(data["patents"], key=lambda x: (-int(x.get("patent_score", 0)), x.get("application_number", ""), x.get("title", ""))):
            key = (p.get("application_number"), p.get("title"))
            if key in seen_patents:
                continue
            seen_patents.add(key)
            unique_patents.append(p)
        data["patents"] = unique_patents
    return researcher_map


# -----------------------------
# Main analysis
# -----------------------------
def unified_analyze(uploaded_file, manual_text: str, progress_callback=None) -> str:
    def report(step: int, total: int, label: str, detail: str = ""):
        if progress_callback:
            progress_callback(step, total, label, detail)

    total_steps = 11 if kipris_enabled() else 8
    report(0, total_steps, "입력 확인", "파일 또는 직접 입력 내용을 점검하는 중입니다.")
    query_text = (file_text(uploaded_file) if uploaded_file else "").strip() or (manual_text or "").strip()
    if len(query_text) < 5:
        return "분석할 내용이 없습니다. 파일을 업로드하거나 내용을 입력해주세요."

    report(1, total_steps, "기본 정보 추출", "기업명과 수요기술 요약을 정리하는 중입니다.")
    request_meta = extract_request_metadata(query_text)

    report(2, total_steps, "기술 프로파일 생성", "논문/특허 검색용 키워드를 만드는 중입니다.")
    profile = extract_search_profile(query_text)
    if (not request_meta.get("tech_summary")) and profile.get("korean_summary"):
        request_meta["tech_summary"] = profile.get("korean_summary")
    keywords_text = format_keyword_text(profile)
    patent_keywords_text = format_patent_keyword_text(profile)

    report(3, total_steps, "논문 검색", "OpenAlex에서 부산대 관련 논문을 수집하는 중입니다.")
    raw_papers = search_openalex(
        tuple(profile.get("search_keywords", [])),
        tuple(profile.get("applications", [])),
        tuple(profile.get("core_tech", [])),
    )

    report(4, total_steps, "부산대 논문 필터링", f"수집 논문 {len(raw_papers)}건에서 부산대 저자를 식별하는 중입니다.")
    pnu_papers, _ = filter_pnu_papers(raw_papers)

    report(5, total_steps, "논문 적합성 검토", f"부산대 논문 {len(pnu_papers)}건의 적합도를 평가하는 중입니다.")
    relevance_map = score_paper_relevance(pnu_papers, profile, request_meta.get("tech_summary", ""))
    valid_papers = select_relevant_papers(pnu_papers, relevance_map)
    if not valid_papers:
        valid_papers = pnu_papers[:MIN_RELEVANT_PAPERS]

    filtered_paper_authors = []
    seen = set()
    for p in valid_papers:
        for name, is_pnu in p.get("raw_authors_info", []):
            if is_pnu and name not in seen:
                seen.add(name)
                filtered_paper_authors.append(name)

    valid_patents = []
    patent_inventors = []
    patent_relevance_map = {}
    if kipris_enabled():
        report(6, total_steps, "특허 검색", "KIPRIS에서 수요기술 연관 특허를 검색하는 중입니다.")
        raw_patents = search_kipris_patents(tuple(profile.get("korean_patent_keywords", [])))

        report(7, total_steps, "산학협력단 단독출원 특허 필터링", f"수집 특허 {len(raw_patents)}건에서 부산대학교 산학협력단 단독출원 여부를 확인하는 중입니다.")
        pnu_iucf_patents, patent_inventors = enrich_and_filter_pnu_iucf_patents(raw_patents)

        report(8, total_steps, "특허 적합성 검토", f"단독출원 특허 {len(pnu_iucf_patents)}건의 적합도를 평가하는 중입니다.")
        patent_relevance_map = score_patent_relevance(pnu_iucf_patents, profile, request_meta.get("tech_summary", ""))
        valid_patents = select_relevant_patents(pnu_iucf_patents, patent_relevance_map)
        if not valid_patents:
            valid_patents = pnu_iucf_patents[:MIN_RELEVANT_PATENTS]

        verify_step = 9
        summarize_step = 10
    else:
        verify_step = 6
        summarize_step = 7

    all_people = unique_keep_order(filtered_paper_authors + patent_inventors)[:MAX_AUTHORS]
    report(verify_step, total_steps, "공식 교수 페이지 검증", f"논문 저자/특허 발명자 {len(all_people)}명의 공식 페이지를 확인하는 중입니다.")
    verified_authors = {}
    unverified_authors = []
    for name in all_people:
        verified = verify_professor_from_official_pages(name)
        if verified:
            verified_authors[name] = verified
        else:
            unverified_authors.append(name)

    report(summarize_step, total_steps, "요약 및 결과 정리", f"공식 검증 통과 인원 {len(verified_authors)}명을 정리하는 중입니다.")
    parsed_papers = summarize_papers(valid_papers)
    parsed_patents = summarize_patents(valid_patents) if kipris_enabled() else {}
    researcher_map = build_verified_researcher_map(valid_papers, valid_patents, verified_authors, parsed_papers, parsed_patents)

    high_count = sum(1 for p in valid_papers if p.get("paper_relevance") == "High")
    medium_count = sum(1 for p in valid_papers if p.get("paper_relevance") == "Medium")
    patent_high_count = sum(1 for p in valid_patents if p.get("patent_relevance") == "High")
    patent_medium_count = sum(1 for p in valid_patents if p.get("patent_relevance") == "Medium")

    lines = []
    lines.append(f"### 🏢 기업명: **{request_meta.get('company_name', '미확인')}**")
    lines.append("")
    lines.append("### 📝 수요기술 요약")
    lines.append(request_meta.get("tech_summary", "입력된 수요기술 설명을 바탕으로 연구자 매칭을 수행했습니다."))
    lines.append("")
    lines.append(f"### 🔍 논문 분석 키워드: **{keywords_text}**")
    if kipris_enabled():
        lines.append(f"### 🔎 특허 분석 키워드: **{patent_keywords_text}**")
    lines.append("")
    lines.append(f"- 검토 논문 수: **{len(pnu_papers)}건**")
    lines.append(f"- 적합성 통과 논문 수: **{len(valid_papers)}건** (High {high_count}건 / Medium {medium_count}건)")
    if kipris_enabled():
        lines.append(f"- 적합성 통과 특허 수: **{len(valid_patents)}건** (High {patent_high_count}건 / Medium {patent_medium_count}건)")
        lines.append(f"- 특허 필터 기준: **출원인에 부산대학교 산학협력단 포함**")
    lines.append(f"- 검토 연구자 수(논문 저자+특허 발명자): **{len(all_people)}명**")
    lines.append(f"- 공식 페이지 검증 통과 교수 수: **{len(researcher_map)}명**")
    if unverified_authors:
        lines.append(f"- 공식 페이지 미검증 인원 수: **{len(unverified_authors)}명**")
    if not kipris_enabled():
        lines.append("- KIPRIS_API_KEY가 없어 특허 검색은 건너뜀")
    lines.append("")
    lines.append("---")

    if not researcher_map:
        lines.append("## ⚠️ 증거형 추천 결과")
        lines.append("")
        lines.append("- 논문 저자 또는 특허 발명자는 확인되었지만, 공식 부산대 교수 페이지까지 확인된 후보가 없어 교수명을 출력하지 않았습니다.")
        lines.append("- 증거형 추천 원칙상 공식 페이지 확인 없는 이름은 제외했습니다.")
        if unverified_authors:
            lines.append(f"- 미검증 인원: {', '.join(unverified_authors[:15])}")
        return "\n".join(lines)

    sorted_researchers = sorted(
        researcher_map.items(),
        key=lambda x: (
            -(sum(int(p.get("paper_score", 0)) for p in x[1].get("papers", [])) + sum(int(p.get("patent_score", 0)) for p in x[1].get("patents", []))),
            -(len(x[1].get("papers", [])) + len(x[1].get("patents", []))),
            x[0],
        )
    )

    for name, data in sorted_researchers:
        lines.append(f"## 🏫 {data['dept']} | {name}")
        lines.append(f"- **검증 방식:** 공식 부산대 페이지 검색 확인")
        lines.append(f"- **공식 페이지 근거:** {data['evidence']}")
        lines.append(f"- **주요 연구분야(페이지 추출):** {data['field']}")
        lines.append(f"- **공식 링크:** [링크 바로가기]({data['link']})")
        lines.append("")

        if data["papers"]:
            lines.append("#### 📄 관련 논문")
            for idx, paper in enumerate(data["papers"], start=1):
                lines.append(f"{idx}. **{paper['k_title']}**")
                lines.append(f"   - 원제: {paper['title']}")
                lines.append(f"   - 논문 적합도: {paper.get('paper_relevance', 'Unknown')} ({paper.get('paper_score', 0)}점)")
                if paper.get("paper_reason"):
                    lines.append(f"   - 적합성 근거: {paper['paper_reason']}")
                lines.append(f"   - 요약: {paper['summary']} ({paper['date']}, {paper['venue']})")
            lines.append("")

        if data["patents"]:
            lines.append("#### 🧾 관련 특허")
            for idx, patent in enumerate(data["patents"], start=1):
                lines.append(f"{idx}. **{patent['k_title']}**")
                lines.append(f"   - 발명의 명칭: {patent['title']}")
                lines.append(f"   - 특허 적합도: {patent.get('patent_relevance', 'Unknown')} ({patent.get('patent_score', 0)}점)")
                if patent.get("patent_reason"):
                    lines.append(f"   - 적합성 근거: {patent['patent_reason']}")
                lines.append(f"   - 출원번호/일자: {patent['application_number']} / {patent['application_date']}")
                if patent.get("register_number") or patent.get("register_date"):
                    lines.append(f"   - 등록정보: {patent.get('register_number', '-') or '-'} / {patent.get('register_date', '-') or '-'} / {patent.get('register_status', '-') or '-'}")
                lines.append(f"   - 출원인: {', '.join(patent.get('applicant_names', [])) or PNU_IUCF_APPLICANT_KR}")
                lines.append(f"   - 요약: {patent['summary']}")
            lines.append("")

        lines.append("---")

    return "\n".join(lines)


# -----------------------------
# UI
# -----------------------------
st.title("🎓 PNU 수요기술-연구자 증거형 매칭 시스템")
st.caption("수요기술에 맞는 부산대 논문 저자와 출원인에 산학협력단이 포함된 특허 발명자를 찾고, 공식 페이지까지 확인된 교수만 출력합니다")

with st.sidebar:
    st.header("설정 안내")
    st.markdown(
        f"""
- PDF, DOCX, TXT, MD 업로드 가능
        """
    )

uploaded_file = st.file_uploader("1. 수요기술조사서 업로드", type=["pdf", "docx", "txt", "md"])
manual_text = st.text_area(
    "2. 또는 기술 내용 직접 입력 (선택)",
    placeholder="기술 설명, 적용 분야, 핵심 성능, 장치/공정/소재 정보를 넣으면 정확도가 올라갑니다.",
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
