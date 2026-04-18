import os
import re
import json
import time
from typing import Dict, List, Tuple, Optional
from urllib.parse import quote_plus, urlparse

import requests
import pdfplumber
import streamlit as st
from docx import Document
from google import genai

st.set_page_config(page_title="부산대 수요기술-연구자 증거형 매칭 시스템", layout="wide")

OPENALEX_URL = "https://api.openalex.org/works"
MAX_PAPERS = 20
MAX_AUTHORS = 30
MIN_RELEVANT_PAPERS = 3
USER_AGENT = "Mozilla/5.0 (EvidenceOnlyPNUMatcher/1.0)"
REQUEST_TIMEOUT = 20
OFFICIAL_DOMAINS = [
    "pusan.ac.kr",
    "pnu.edu",
]


# -----------------------------
# Environment / Client
# -----------------------------
def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


GEMINI_API_KEY = get_env("GEMINI_API_KEY")
OPENALEX_API_KEY = get_env("OPENALEX_API_KEY")


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
    last_error = None
    for model_name in ["gemini-2.5-flash", "gemini-2.5-flash-lite"]:
        for attempt in range(retries):
            try:
                res = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return extract_json_object(getattr(res, "text", ""))
            except Exception as e:
                last_error = e
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
당신은 OpenAlex 논문 검색 프로파일 설계기입니다.
아래 수요기술 설명을 바탕으로 논문 검색용 키워드 JSON을 작성하세요.

반드시 포함할 항목:
- core_tech: 핵심 기술 2~4개 (영어)
- materials_or_methods: 재료/방법 2~4개 (영어)
- properties: 요구 특성 1~4개 (영어)
- applications: 적용처 1~3개 (영어)
- search_keywords: 검색용 핵심 키워드 4~6개 (영어 짧은 구)
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
        ]:
            data[key] = [str(x).strip() for x in data.get(key, []) if str(x).strip()]
        data["korean_summary"] = str(data.get("korean_summary", "")).strip()
        return data

    fallback = [t for t in fallback_tokens if len(t) >= 4][:5]
    return {
        "core_tech": fallback[:2],
        "materials_or_methods": fallback[2:4],
        "properties": [],
        "applications": [],
        "search_keywords": fallback or ["Pusan National University"],
        "exclude_keywords": [],
        "korean_summary": "입력된 수요기술 설명을 바탕으로 검색 키워드를 구성했습니다.",
    }


def format_keyword_text(profile: Dict) -> str:
    return ", ".join(profile.get("search_keywords", []))


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
            inst_names = [str((inst or {}).get("display_name") or "").lower() for inst in institutions]
            raw_aff = str(authorship.get("raw_affiliation_string") or "").lower()
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
# Evidence-only professor verification
# -----------------------------
def is_official_domain(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return False
    official_domains = set(OFFICIAL_DOMAINS + [
        "pusan.ac.kr",
        "www.pusan.ac.kr",
        "scholar.pusan.ac.kr",
        "sites.pusan.ac.kr",
        "home.pusan.ac.kr",
    ])
    return any(netloc == domain or netloc.endswith("." + domain) for domain in official_domains)


def normalize_name_for_match(name: str) -> str:
    s = normalize_space(name or "").lower()
    s = s.replace("–", "-").replace("—", "-").replace("‑", "-")
    s = re.sub(r"[^a-z0-9가-힣]", "", s)
    return s


def build_name_variants(author_name: str) -> List[str]:
    base = normalize_space(author_name)
    variants = {base}
    if not base:
        return []

    variants.add(base.replace("-", " "))
    variants.add(base.replace("–", " "))
    variants.add(base.replace("—", " "))
    variants.add(base.replace("·", " "))
    variants.add(base.replace(".", " "))

    tokens = [t for t in re.split(r"[\s\-–—·.]+", base) if t]
    if len(tokens) >= 2:
        variants.add(" ".join(tokens))
        variants.add("-".join(tokens))
        variants.add("".join(tokens))
        variants.add(f"{tokens[-1]} {' '.join(tokens[:-1])}".strip())

    return [v for v in sorted(variants) if v.strip()]


@st.cache_data(show_spinner=False)
def duckduckgo_search(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    url = "https://html.duckduckgo.com/html/"
    results = []
    seen = set()
    try:
        r = SESSION.post(url, data={"q": query}, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []
        html = r.text
        for m in re.finditer(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.I | re.S):
            href = m.group(1).strip()
            title = normalize_space(re.sub(r"<[^>]+>", " ", m.group(2)))
            if href and title and href not in seen:
                seen.add(href)
                results.append({"title": title, "url": href})
                if len(results) >= max_results:
                    break
    except Exception:
        return []
    return results


@st.cache_data(show_spinner=False)
def fetch_page_text(url: str) -> str:
    try:
        r = SESSION.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return ""
        html = r.text
        html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
        html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", html)
        return normalize_space(text)[:20000]
    except Exception:
        return ""


def name_present_in_text(author_name: str, page_text: str) -> bool:
    norm_page = normalize_name_for_match(page_text)
    for variant in build_name_variants(author_name):
        norm_variant = normalize_name_for_match(variant)
        if norm_variant and norm_variant in norm_page:
            return True
    return False


def score_official_profile_page(author_name: str, url: str, page_text: str) -> int:
    score = 0
    low_text = (page_text or "").lower()
    low_url = (url or "").lower()

    if not is_official_domain(url):
        return -1
    score += 2

    if name_present_in_text(author_name, page_text):
        score += 4

    professor_keywords = [
        "professor", "faculty", "assistant professor", "associate professor", "full professor",
        "교수", "조교수", "부교수", "정교수", "교원", "faculty member", "faculty profile"
    ]
    department_keywords = [
        "department", "school", "college", "학과", "학부", "대학", "대학원", "전공", "lab", "laboratory", "연구실"
    ]
    researcher_keywords = ["researcher", "research", "scholar", "연구자", "연구실적", "research interests", "research area", "연구분야"]

    if any(k in low_text for k in professor_keywords):
        score += 3
    if any(k in low_text for k in department_keywords):
        score += 2
    if any(k in low_text for k in researcher_keywords):
        score += 1

    if any(k in low_url for k in ["/subview.do", "/people/", "/faculty", "/professor", "/researchers/"]):
        score += 1
    if "scholar.pusan.ac.kr" in low_url:
        score += 2

    return score


@st.cache_data(show_spinner=False)
def verify_professor_from_official_pages(author_name: str) -> Optional[Dict]:
    variants = build_name_variants(author_name)
    queries = []
    for variant in variants[:6]:
        queries.extend([
            f'site:pusan.ac.kr "{variant}"',
            f'site:pusan.ac.kr "{variant}" 교수',
            f'site:pusan.ac.kr "{variant}" professor',
            f'site:scholar.pusan.ac.kr "{variant}"',
        ])

    best = None
    best_score = -1
    seen_urls = set()

    for query in queries[:18]:
        results = duckduckgo_search(query, max_results=8)
        for item in results:
            url = item.get("url", "")
            if not url or url in seen_urls or not is_official_domain(url):
                continue
            seen_urls.add(url)
            page_text = fetch_page_text(url)
            if not page_text or not name_present_in_text(author_name, page_text):
                continue

            score = score_official_profile_page(author_name, url, page_text)
            if score < 6:
                continue

            dept = extract_department_from_page(page_text)
            field = extract_field_from_page(page_text)
            evidence = extract_evidence_snippet(page_text, author_name)
            candidate = {
                "official_name": author_name,
                "department": dept or "부산대 공식 페이지 확인",
                "field": field or "공식 페이지에서 상세 연구분야 자동 추출 실패",
                "link": url,
                "evidence": evidence or "공식 페이지에서 이름 및 교수/연구자 정보를 확인함",
                "verify_score": score,
            }
            if score > best_score:
                best_score = score
                best = candidate

    return best


def extract_department_from_page(text: str) -> str:
    patterns = [
        r"(Department of [A-Za-z0-9\-&, ]{3,120})",
        r"(School of [A-Za-z0-9\-&, ]{3,120})",
        r"(College of [A-Za-z0-9\-&, ]{3,120})",
        r"([가-힣A-Za-z0-9·\- ]{2,60}학과)",
        r"([가-힣A-Za-z0-9·\- ]{2,60}학부)",
        r"([가-힣A-Za-z0-9·\- ]{2,60}대학)",
        r"([가-힣A-Za-z0-9·\- ]{2,60}대학원)",
        r"([가-힣A-Za-z0-9·\- ]{2,60}전공)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return normalize_space(m.group(1))[:120]
    return ""


def extract_field_from_page(text: str) -> str:
    patterns = [
        r"(Research Interests?[:\s][A-Za-z0-9,;\-()/. ]{10,250})",
        r"(Research Areas?[:\s][A-Za-z0-9,;\-()/. ]{10,250})",
        r"(Keyword[s]?[:\s][A-Za-z0-9,;\-()/. ]{10,250})",
        r"(연구분야[:\s][가-힣A-Za-z0-9,;·\-()/. ]{5,250})",
        r"(주요 연구분야[:\s][가-힣A-Za-z0-9,;·\-()/. ]{5,250})",
        r"(연구키워드[:\s][가-힣A-Za-z0-9,;·\-()/. ]{5,250})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return normalize_space(m.group(1))[:220]
    return ""


def extract_evidence_snippet(text: str, author_name: str) -> str:
    variants = build_name_variants(author_name)
    low_text = text.lower()
    idx = -1
    hit = author_name
    for variant in variants:
        idx = low_text.find(variant.lower())
        if idx != -1:
            hit = variant
            break
    if idx == -1:
        return ""
    start = max(0, idx - 120)
    end = min(len(text), idx + max(len(hit), 20) + 220)
    return normalize_space(text[start:end])


# -----------------------------
# Paper summary
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
def build_verified_professor_map(valid_papers: List[Dict], verified_authors: Dict[str, Dict], parsed_results: Dict) -> Dict:
    professor_map = {}
    for i, p in enumerate(valid_papers, start=1):
        info = parsed_results.get(str(i), {})
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
            if name not in professor_map:
                professor_map[name] = {
                    "dept": db.get("department", "공식 확인됨"),
                    "field": db.get("field", "공식 페이지 기반 확인"),
                    "link": db.get("link", "#"),
                    "evidence": db.get("evidence", "공식 페이지 확인"),
                    "papers": [],
                }
            professor_map[name]["papers"].append(paper_obj)

    for _, data in professor_map.items():
        unique = []
        seen = set()
        for p in sorted(data["papers"], key=lambda x: (-int(x.get("paper_score", 0)), x.get("title", ""))):
            key = (p.get("title"), p.get("date"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(p)
        data["papers"] = unique
    return professor_map


# -----------------------------
# Main analysis
# -----------------------------
def unified_analyze(uploaded_file, manual_text: str, progress_callback=None) -> str:
    def report(step: int, total: int, label: str, detail: str = ""):
        if progress_callback:
            progress_callback(step, total, label, detail)

    total_steps = 8
    report(0, total_steps, "입력 확인", "파일 또는 직접 입력 내용을 점검하는 중입니다.")
    query_text = (file_text(uploaded_file) if uploaded_file else "").strip() or (manual_text or "").strip()
    if len(query_text) < 5:
        return "분석할 내용이 없습니다. 파일을 업로드하거나 내용을 입력해주세요."

    report(1, total_steps, "기본 정보 추출", "기업명과 수요기술 요약을 정리하는 중입니다.")
    request_meta = extract_request_metadata(query_text)

    report(2, total_steps, "기술 프로파일 생성", "OpenAlex 검색용 키워드를 만드는 중입니다.")
    profile = extract_search_profile(query_text)
    if (not request_meta.get("tech_summary")) and profile.get("korean_summary"):
        request_meta["tech_summary"] = profile.get("korean_summary")
    keywords_text = format_keyword_text(profile)

    report(3, total_steps, "논문 검색", "OpenAlex에서 부산대 관련 논문을 수집하는 중입니다.")
    raw_papers = search_openalex(
        tuple(profile.get("search_keywords", [])),
        tuple(profile.get("applications", [])),
        tuple(profile.get("core_tech", [])),
    )

    report(4, total_steps, "부산대 논문 필터링", f"수집 논문 {len(raw_papers)}건에서 부산대 저자를 식별하는 중입니다.")
    pnu_papers, pnu_authors = filter_pnu_papers(raw_papers)
    if not pnu_papers:
        return "\n".join([
            f"### 🏢 기업명: **{request_meta.get('company_name', '미확인')}**",
            "",
            "### 📝 수요기술 요약",
            request_meta.get("tech_summary", "입력된 수요기술 설명을 바탕으로 연구자 매칭을 수행했습니다."),
            "",
            f"### 🔍 분석 키워드: **{keywords_text}**",
            "",
            "- OpenAlex에서 부산대 소속 논문을 찾지 못했습니다.",
            "- 증거형 추천 원칙에 따라 교수 이름을 추정 생성하지 않았습니다.",
            "- 수요기술 설명을 더 구체화하거나 검색 키워드를 조정한 뒤 다시 시도해주세요.",
        ])

    report(5, total_steps, "논문 적합성 검토", f"부산대 논문 {len(pnu_papers)}건의 적합도를 평가하는 중입니다.")
    relevance_map = score_paper_relevance(pnu_papers, profile, request_meta.get("tech_summary", ""))
    valid_papers = select_relevant_papers(pnu_papers, relevance_map)
    if not valid_papers:
        valid_papers = pnu_papers[:MIN_RELEVANT_PAPERS]

    filtered_authors = []
    seen = set()
    for p in valid_papers:
        for name, is_pnu in p.get("raw_authors_info", []):
            if is_pnu and name not in seen:
                seen.add(name)
                filtered_authors.append(name)

    report(6, total_steps, "공식 교수 페이지 검증", f"논문 저자 {len(filtered_authors[:MAX_AUTHORS])}명의 공식 페이지를 확인하는 중입니다.")
    verified_authors = {}
    unverified_authors = []
    for name in filtered_authors[:MAX_AUTHORS]:
        verified = verify_professor_from_official_pages(name)
        if verified:
            verified_authors[name] = verified
        else:
            unverified_authors.append(name)

    report(7, total_steps, "논문 요약 및 결과 정리", f"공식 검증 통과 저수 {len(verified_authors)}명을 정리하는 중입니다.")
    parsed_results = summarize_papers(valid_papers)
    professor_map = build_verified_professor_map(valid_papers, verified_authors, parsed_results)

    high_count = sum(1 for p in valid_papers if p.get("paper_relevance") == "High")
    medium_count = sum(1 for p in valid_papers if p.get("paper_relevance") == "Medium")

    lines = []
    lines.append(f"### 🏢 기업명: **{request_meta.get('company_name', '미확인')}**")
    lines.append("")
    lines.append("### 📝 수요기술 요약")
    lines.append(request_meta.get("tech_summary", "입력된 수요기술 설명을 바탕으로 연구자 매칭을 수행했습니다."))
    lines.append("")
    lines.append(f"### 🔍 분석 키워드: **{keywords_text}**")
    lines.append("")
    lines.append(f"- 검토 논문 수: **{len(pnu_papers)}건**")
    lines.append(f"- 적합성 통과 논문 수: **{len(valid_papers)}건** (High {high_count}건 / Medium {medium_count}건)")
    lines.append(f"- 검토 부산대 저자 수: **{len(filtered_authors)}명**")
    lines.append(f"- 공식 페이지 검증 통과 교수 수: **{len(professor_map)}명**")
    if unverified_authors:
        lines.append(f"- 공식 페이지 미검증 저자 수: **{len(unverified_authors)}명**")
    lines.append("")
    lines.append("---")

    if not professor_map:
        lines.append("## ⚠️ 증거형 추천 결과")
        lines.append("")
        lines.append("- 논문 저자는 확인되었지만, 공식 부산대 교수 페이지까지 확인된 후보가 없어 교수명을 출력하지 않았습니다.")
        lines.append("- 증거형 추천 원칙상 공식 페이지 확인 없는 이름은 제외했습니다.")
        if unverified_authors:
            lines.append(f"- 미검증 논문 저자: {', '.join(unverified_authors[:15])}")
        return "\n".join(lines)

    sorted_professors = sorted(
        professor_map.items(),
        key=lambda x: (-sum(int(p.get("paper_score", 0)) for p in x[1].get("papers", [])), -len(x[1].get("papers", [])), x[0])
    )

    for eng_name, data in sorted_professors:
        lines.append(f"## 🏫 {data['dept']} | {eng_name}")
        lines.append(f"- **검증 방식:** 공식 부산대 페이지 검색 확인")
        lines.append(f"- **공식 페이지 근거:** {data['evidence']}")
        lines.append(f"- **주요 연구분야(페이지 추출):** {data['field']}")
        lines.append(f"- **공식 링크:** [링크 바로가기]({data['link']})")
        lines.append("")
        lines.append("#### 📄 관련 논문")
        for idx, paper in enumerate(data["papers"], start=1):
            lines.append(f"{idx}. **{paper['k_title']}**")
            lines.append(f"   - 원제: {paper['title']}")
            lines.append(f"   - 논문 적합도: {paper.get('paper_relevance', 'Unknown')} ({paper.get('paper_score', 0)}점)")
            if paper.get("paper_reason"):
                lines.append(f"   - 적합성 근거: {paper['paper_reason']}")
            lines.append(f"   - 요약: {paper['summary']} ({paper['date']}, {paper['venue']})")
        lines.append("")
        lines.append("---")

    return "\n".join(lines)


# -----------------------------
# UI
# -----------------------------
st.title("🎓 PNU 수요기술-연구자 증거형 매칭 시스템")
st.caption("수요기술에 맞는 부산대 논문 저자를 찾고, 공식 페이지까지 확인된 교수만 출력합니다")

with st.sidebar:
    st.header("설정 안내")
    st.markdown(
        """
- PDF, DOCX, TXT, MD 업로드 가능
- OpenAlex 논문 검색 + 부산대 저자 필터링
- **공식 부산대 페이지가 확인된 교수만 출력**
- 공식 페이지가 확인되지 않으면 이름을 추정 생성하지 않음
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
