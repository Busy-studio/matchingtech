import os
import re
import json
import time
import html
import difflib
from typing import Dict, List, Tuple, Optional, Any
import xml.etree.ElementTree as ET

import requests
import streamlit as st
from docx import Document
from google import genai

try:
    import pdfplumber
except Exception:
    pdfplumber = None


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="부산대 수요기술-연구자 증거형 매칭 시스템",
    layout="wide",
)

OPENALEX_URL = "https://api.openalex.org/works"
PNU_SCHOLAR_API_URL = "https://scholar.pusan.ac.kr/wp-json/rm/v1/scholars"
KIPRIS_BASE_URL = "https://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice"

MAX_PAPERS = 20
MAX_PATENTS = 20
MAX_RESEARCHERS = 40
MAX_SCHOLAR_PAGES = 200
SCHOLAR_RECORD_PER_PAGE = 24

MIN_RELEVANT_PAPERS = 3
MIN_RELEVANT_PATENTS = 3

REQUEST_TIMEOUT = 20
USER_AGENT = "Mozilla/5.0 (EvidenceOnlyPNUMatcher/2.0)"

PNU_IUCF_APPLICANT_KR = "부산대학교 산학협력단"
PNU_AFFILIATION_HINTS = [
    "pusan national",
    "busan national",
    "부산대",
    "부산대학교",
]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


# =========================================================
# 환경변수 / API 클라이언트
# =========================================================
def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


GEMINI_API_KEY = get_env("GEMINI_API_KEY")
OPENALEX_API_KEY = get_env("OPENALEX_API_KEY")
KIPRIS_API_KEY = get_env("KIPRIS_API_KEY")


@st.cache_resource(show_spinner=False)
def init_gemini_client():
    if not GEMINI_API_KEY:
        return None
    try:
        return genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        return None


client = init_gemini_client()


# =========================================================
# 공통 유틸
# =========================================================
def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def compact_text(text: str, limit: int = 5000) -> str:
    return normalize_space(text)[:limit]


def strip_tags(raw_html: str) -> str:
    if not raw_html:
        return ""
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw_html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    return normalize_space(text)


def safe_lower(x: Any) -> str:
    return str(x or "").lower()


def unique_keep_order(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for item in items:
        val = normalize_space(item)
        if not val:
            continue
        key = val.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(val)
    return out


def clamp_score(value: Any, default: int = 60) -> int:
    try:
        score = int(value)
    except Exception:
        return default
    return max(0, min(100, score))


def normalize_name_for_match(name: str) -> str:
    s = normalize_space(name).lower()
    s = s.replace("–", "-").replace("—", "-").replace("-", "-")
    s = re.sub(r"[^a-z0-9가-힣]", "", s)
    return s


def has_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", str(text or "")))


def has_english(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", str(text or "")))


def build_name_variants(name: str) -> List[str]:
    """
    영문명/한글명 매칭 실패를 줄이기 위한 이름 변형 생성.
    예:
    - Jae-Hong Kim
    - Jae Hong Kim
    - JaeHongKim
    - Kim Jae Hong
    - Kim Jae-Hong
    - Kim, Jae-Hong
    """
    base = normalize_space(name)
    if not base:
        return []

    variants = [base]

    cleaned = base.replace("–", "-").replace("—", "-").replace("-", "-")
    variants.append(cleaned)
    variants.append(cleaned.replace("-", " "))
    variants.append(cleaned.replace("-", ""))
    variants.append(cleaned.replace(" ", ""))
    variants.append(cleaned.replace(",", ""))

    tokens = re.split(r"[\s,\-]+", cleaned)
    tokens = [t for t in tokens if t]

    if len(tokens) >= 2 and all(re.search(r"[A-Za-z]", t) for t in tokens):
        first_parts = tokens[:-1]
        last = tokens[-1]
        reversed_tokens = [last] + first_parts

        variants.append(" ".join(reversed_tokens))
        variants.append("-".join(reversed_tokens))
        variants.append(",".join([last, " ".join(first_parts)]))
        variants.append(last + "".join(first_parts))
        variants.append("".join(reversed_tokens))
        variants.append(" ".join(first_parts) + " " + last)
        variants.append("".join(first_parts) + last)

    return unique_keep_order(variants)


def name_similarity(a: str, b: str) -> float:
    na = normalize_name_for_match(a)
    nb = normalize_name_for_match(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def best_name_match_score(query_name: str, candidate_names: List[str]) -> Tuple[float, str]:
    q_variants = build_name_variants(query_name)
    c_variants = []
    for cn in candidate_names:
        c_variants.extend(build_name_variants(cn))
    c_variants = unique_keep_order(c_variants)

    best_score = 0.0
    best_candidate = ""

    for q in q_variants:
        for c in c_variants:
            score = name_similarity(q, c)
            if score > best_score:
                best_score = score
                best_candidate = c

    return best_score, best_candidate


def is_probable_name(text: str) -> bool:
    t = normalize_space(strip_tags(text))
    if not t:
        return False
    if len(t) > 80:
        return False
    if re.search(r"(research|publication|department|college|university|논문|연구|학과|대학|특허)", t, re.I):
        if len(t) > 25:
            return False
    return bool(re.search(r"[A-Za-z가-힣]", t))


# =========================================================
# 파일 읽기
# =========================================================
@st.cache_data(show_spinner=False)
def file_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    ext = os.path.splitext(uploaded_file.name)[1].lower()

    try:
        if ext == ".pdf":
            if pdfplumber is None:
                return ""
            text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
            return text

        if ext == ".docx":
            doc = Document(uploaded_file)
            return "\n".join(p.text for p in doc.paragraphs)

        if ext in {".txt", ".md"}:
            return uploaded_file.read().decode("utf-8", errors="ignore")

    except Exception:
        return ""

    return ""


# =========================================================
# Gemini JSON 처리
# =========================================================
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

    model_names = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]

    for model_name in model_names:
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


def safe_gemini_text(prompt: str, retries: int = 2) -> str:
    if client is None:
        return ""

    model_names = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]

    for model_name in model_names:
        for attempt in range(retries):
            try:
                res = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return getattr(res, "text", "") or ""
            except Exception as e:
                if any(code in str(e) for code in ["429", "503"]) and attempt < retries - 1:
                    time.sleep(2)
                    continue
                break

    return ""


# =========================================================
# 입력 텍스트 → 수요기술 프로파일
# =========================================================
def extract_request_metadata(query_text: str) -> Dict[str, str]:
    prompt = f"""
당신은 대학 산학협력 실무용 입력정보 정리기입니다.
아래 텍스트에서 기업명과 수요기술 요약만 JSON으로 추출하세요.

규칙:
- 기업명이 명확하지 않으면 "미확인"
- 수요기술 요약은 한국어 1~2문장
- 과장 없이 핵심 기술/성능/적용처 중심
- 출력은 JSON만

형식:
{{
  "company_name": "기업명 또는 미확인",
  "tech_summary": "수요기술 요약"
}}

입력:
{compact_text(query_text)}
"""
    data = safe_gemini_json(prompt)

    return {
        "company_name": str(data.get("company_name") or "미확인").strip() or "미확인",
        "tech_summary": str(
            data.get("tech_summary")
            or "입력된 수요기술 설명을 바탕으로 연구자 매칭을 수행했습니다."
        ).strip(),
    }


def extract_search_profile(query_text: str) -> Dict:
    fallback_tokens = []
    for token in [x.strip(",.()[]{}") for x in query_text.replace("\n", " ").split() if x.strip()]:
        if len(token) >= 3:
            fallback_tokens.append(token)
        if len(fallback_tokens) >= 8:
            break

    prompt = f"""
당신은 대학 산학협력용 검색 프로파일 설계기입니다.
아래 수요기술 설명을 바탕으로 논문/특허 검색용 키워드 JSON을 작성하세요.

반드시 포함할 항목:
- core_tech: 핵심 기술 2~4개, 영어
- materials_or_methods: 재료/방법 2~4개, 영어
- properties: 요구 특성 1~4개, 영어
- applications: 적용처 1~3개, 영어
- search_keywords: 논문 검색용 핵심 키워드 4~6개, 영어 짧은 구
- korean_patent_keywords: 특허 검색용 한국어 핵심 키워드 4~8개
- exclude_keywords: 배제 키워드 0~4개, 영어
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
            "korean_patent_keywords",
            "exclude_keywords",
        ]:
            data[key] = [str(x).strip() for x in data.get(key, []) if str(x).strip()]
        data["korean_summary"] = str(data.get("korean_summary", "")).strip()
        return data

    fallback = fallback_tokens[:6] or ["Pusan National University"]

    return {
        "core_tech": fallback[:2],
        "materials_or_methods": fallback[2:4],
        "properties": [],
        "applications": [],
        "search_keywords": fallback,
        "korean_patent_keywords": fallback,
        "exclude_keywords": [],
        "korean_summary": "입력된 수요기술 설명을 바탕으로 검색 키워드를 구성했습니다.",
    }


# =========================================================
# PNU Scholar API 기반 연구자 DB
# =========================================================
def flatten_json_strings(obj: Any, prefix: str = "") -> Dict[str, str]:
    out = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_json_strings(v, key))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}.{i}" if prefix else str(i)
            out.update(flatten_json_strings(v, key))

    else:
        if obj is not None:
            val = normalize_space(strip_tags(str(obj)))
            if val:
                out[prefix.lower()] = val

    return out


def find_list_candidates(data: Any) -> List[List[Dict]]:
    lists = []

    def walk(x):
        if isinstance(x, list):
            if x and all(isinstance(i, dict) for i in x):
                lists.append(x)
            for i in x:
                walk(i)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)

    walk(data)
    return lists


def extract_scholar_records_from_response(data: Any) -> List[Dict]:
    """
    PNU Scholar API 응답 구조가 바뀌어도 최대한 연구자 record list를 찾기 위한 함수.
    우선순위:
    1) data, results, items, scholars, posts 등 명시 키
    2) dict list 중 name/title/link 계열이 많은 리스트
    """
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if not isinstance(data, dict):
        return []

    preferred_keys = [
        "data",
        "results",
        "items",
        "scholars",
        "researchers",
        "posts",
        "records",
    ]

    for key in preferred_keys:
        val = data.get(key)
        if isinstance(val, list) and val and all(isinstance(x, dict) for x in val):
            return val
        if isinstance(val, dict):
            nested = extract_scholar_records_from_response(val)
            if nested:
                return nested

    candidates = find_list_candidates(data)
    if not candidates:
        return []

    def score_list(lst: List[Dict]) -> int:
        score = 0
        for rec in lst[:5]:
            flat = flatten_json_strings(rec)
            keys = " ".join(flat.keys()).lower()
            vals = " ".join(flat.values()).lower()
            if any(k in keys for k in ["name", "title", "author", "scholar", "researcher"]):
                score += 3
            if any(k in keys for k in ["department", "dept", "college", "affiliation", "major"]):
                score += 2
            if any(k in keys for k in ["link", "url", "permalink"]):
                score += 1
            if any(k in vals for k in ["professor", "교수", "부산", "pusan"]):
                score += 1
        score += min(len(lst), 10)
        return score

    candidates = sorted(candidates, key=score_list, reverse=True)
    return candidates[0]


def pick_first_by_key_hints(flat: Dict[str, str], hints: List[str], max_len: int = 200) -> str:
    for key, val in flat.items():
        if any(h in key.lower() for h in hints):
            v = normalize_space(val)
            if v and len(v) <= max_len:
                return v
    return ""


def pick_all_by_key_hints(flat: Dict[str, str], hints: List[str], max_len: int = 120) -> List[str]:
    out = []
    for key, val in flat.items():
        if any(h in key.lower() for h in hints):
            v = normalize_space(val)
            if v and len(v) <= max_len:
                out.append(v)
    return unique_keep_order(out)


def normalize_pnu_scholar_record(record: Dict) -> Dict:
    flat = flatten_json_strings(record)
    lower_keys = {k.lower(): v for k, v in flat.items()}

    name_hints = [
        "kor_name",
        "korean_name",
        "kr_name",
        "name_ko",
        "name.kr",
        "eng_name",
        "english_name",
        "en_name",
        "name_en",
        "display_name",
        "full_name",
        "scholar_name",
        "researcher_name",
        "author_name",
        "post_title",
        "title.rendered",
        "title",
        "name",
    ]

    raw_names = pick_all_by_key_hints(lower_keys, name_hints, max_len=100)

    cleaned_names = []
    for n in raw_names:
        n = strip_tags(n)
        n = re.sub(r"\s*[|｜].*$", "", n).strip()
        n = re.sub(r"\s*-\s*(Professor|교수|Researcher|연구자).*$", "", n, flags=re.I).strip()
        if is_probable_name(n):
            cleaned_names.append(n)

    cleaned_names = unique_keep_order(cleaned_names)

    korean_names = [n for n in cleaned_names if has_korean(n)]
    english_names = [n for n in cleaned_names if has_english(n)]

    primary_name = ""
    if korean_names:
        primary_name = korean_names[0]
    elif english_names:
        primary_name = english_names[0]
    elif cleaned_names:
        primary_name = cleaned_names[0]

    department = pick_first_by_key_hints(
        lower_keys,
        [
            "department",
            "dept",
            "affiliation",
            "college",
            "school",
            "major",
            "organization",
            "org",
            "소속",
            "학과",
            "학부",
            "대학",
            "전공",
        ],
        max_len=200,
    )

    field = pick_first_by_key_hints(
        lower_keys,
        [
            "research",
            "field",
            "interest",
            "keyword",
            "area",
            "specialty",
            "전공분야",
            "연구분야",
            "관심분야",
            "키워드",
        ],
        max_len=300,
    )

    link = pick_first_by_key_hints(
        lower_keys,
        [
            "permalink",
            "profile_url",
            "profile",
            "link",
            "url",
            "href",
        ],
        max_len=500,
    )

    if link and link.startswith("/"):
        link = "https://scholar.pusan.ac.kr" + link

    if link and not link.startswith("http"):
        link = ""

    scholar_id = pick_first_by_key_hints(
        lower_keys,
        [
            "id",
            "researcher_id",
            "scholar_id",
            "post_id",
        ],
        max_len=100,
    )

    evidence_parts = []
    if primary_name:
        evidence_parts.append(primary_name)
    if department:
        evidence_parts.append(department)
    if field:
        evidence_parts.append(field)

    return {
        "scholar_id": scholar_id,
        "official_name": primary_name,
        "all_names": cleaned_names,
        "korean_names": korean_names,
        "english_names": english_names,
        "department": department or "PNU Scholar API 기반 확인",
        "field": field or "PNU Scholar API에서 상세 연구분야 자동 추출 실패",
        "link": link or "https://scholar.pusan.ac.kr/researchers",
        "evidence": " / ".join(evidence_parts) if evidence_parts else "PNU Scholar API record 확인",
        "raw": record,
    }


@st.cache_data(show_spinner=False)
def fetch_pnu_scholars_page(page: int = 1) -> List[Dict]:
    params = {
        "page": page,
        "current_page": page,
        "record_per_page": SCHOLAR_RECORD_PER_PAGE,
        "order_by": "date",
        "order": "desc",
    }

    try:
        r = SESSION.get(PNU_SCHOLAR_API_URL, params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []
        data = r.json()
        records = extract_scholar_records_from_response(data)
        return records
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def fetch_all_pnu_scholars(max_pages: int = MAX_SCHOLAR_PAGES) -> List[Dict]:
    scholars = []
    seen_keys = set()
    empty_count = 0

    for page in range(1, max_pages + 1):
        records = fetch_pnu_scholars_page(page)

        if not records:
            empty_count += 1
            if empty_count >= 2:
                break
            continue

        empty_count = 0

        for rec in records:
            norm = normalize_pnu_scholar_record(rec)
            if not norm.get("official_name") and not norm.get("all_names"):
                continue

            key_source = norm.get("scholar_id") or "|".join(norm.get("all_names", [])) or json.dumps(rec, ensure_ascii=False)[:300]
            key = normalize_name_for_match(key_source)

            if key in seen_keys:
                continue

            seen_keys.add(key)
            scholars.append(norm)

        if len(records) < SCHOLAR_RECORD_PER_PAGE:
            break

    return scholars


def match_author_to_pnu_scholar(author_name: str, scholar_db: List[Dict]) -> Optional[Dict]:
    """
    논문 저자명 또는 특허 발명자명을 PNU Scholar API DB와 매칭.
    - 한글명은 거의 exact 위주
    - 영문명은 하이픈/공백/성-이름 순서 변형 허용
    """
    name = normalize_space(author_name)
    if not name:
        return None

    best = None
    best_score = 0.0

    query_is_korean = has_korean(name)

    for scholar in scholar_db:
        candidate_names = scholar.get("all_names", []) or []
        if not candidate_names:
            continue

        score, matched_variant = best_name_match_score(name, candidate_names)

        # 한글명은 짧아서 유사도 오탐이 많으므로 기준 강화
        if query_is_korean:
            query_norm = normalize_name_for_match(name)
            cand_norms = [normalize_name_for_match(c) for c in candidate_names]

            if query_norm in cand_norms:
                score = 1.0
            elif len(query_norm) <= 4:
                # 김철수 같은 3글자명은 유사도만으로 매칭하지 않음
                score = 0.0

        if score > best_score:
            best_score = score
            best = {
                **scholar,
                "match_score": round(score, 3),
                "matched_variant": matched_variant,
                "query_name": name,
            }

    if not best:
        return None

    # 기준값
    # - 한글 exact: 1.0
    # - 영문명 변형: 0.88 이상
    # - 긴 영문명은 0.84 이상도 허용
    if query_is_korean:
        return best if best_score >= 0.99 else None

    if len(normalize_name_for_match(name)) >= 10:
        return best if best_score >= 0.84 else None

    return best if best_score >= 0.88 else None


# =========================================================
# OpenAlex 논문 검색
# =========================================================
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
def search_openalex(
    search_keywords: Tuple[str, ...],
    applications: Tuple[str, ...],
    core_tech: Tuple[str, ...],
) -> List[Dict]:
    keywords = list(search_keywords)
    apps = list(applications)
    techs = list(core_tech)

    queries = []
    base = keywords[:4] if keywords else techs[:3]

    if base:
        queries.append(" ".join(base[:3]) + " Pusan National University")
        queries.append(" ".join(base[:2]) + " Busan National University")
        queries.append(" ".join(base[:3]))

    if techs and apps:
        queries.append(f"{' '.join(techs[:2])} {' '.join(apps[:2])} Pusan National University")

    if techs:
        queries.append(" ".join(techs[:3]))

    queries = unique_keep_order(queries)

    seen = set()
    collected = []

    for q in queries:
        params = {
            "search": q,
            "sort": "publication_date:desc",
            "per_page": 50,
            "select": "id,title,authorships,publication_date,abstract_inverted_index,primary_location,doi",
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
            item_id = item.get("id") or item.get("doi") or item.get("title")
            if not item_id or item_id in seen:
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
            author = authorship.get("author") or {}
            name = str(author.get("display_name") or "Unknown")

            institutions = authorship.get("institutions") or []
            inst_names = [safe_lower((inst or {}).get("display_name")) for inst in institutions]
            raw_aff = safe_lower(authorship.get("raw_affiliation_string"))

            combined = " ".join(inst_names) + " " + raw_aff
            is_pnu = any(k in combined for k in PNU_AFFILIATION_HINTS)

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

    return valid_papers, unique_pnu_authors


def score_paper_relevance(valid_papers: List[Dict], profile: Dict, tech_summary: str) -> Dict[str, Dict]:
    if not valid_papers:
        return {}

    if client is None:
        return {
            str(i): {
                "relevance": "Medium",
                "score": 60,
                "reason": "Gemini API 미설정으로 기본 적합도 적용",
            }
            for i, _ in enumerate(valid_papers, start=1)
        }

    blocks = []

    for i, p in enumerate(valid_papers, start=1):
        abs_text = reconstruct_abstract(p.get("abstract_inverted_index"))
        blocks.append(
            f"[{i}] Title: {p.get('title', '')}\n"
            f"Abstract: {abs_text[:900]}\n"
        )

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

    return {
        str(i): {
            "relevance": "Medium",
            "score": 60,
            "reason": "적합성 평가 실패로 기본값 적용",
        }
        for i, _ in enumerate(valid_papers, start=1)
    }


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
        selected.extend(
            sorted(low, key=lambda x: x.get("paper_score", 0), reverse=True)[
                : MIN_RELEVANT_PAPERS - len(selected)
            ]
        )

    return selected[:MAX_PAPERS]


# =========================================================
# KIPRIS 특허 검색
# =========================================================
def kipris_enabled() -> bool:
    return bool(KIPRIS_API_KEY)


def extract_texts_by_tag(root: ET.Element, tag_name: str) -> List[str]:
    values = []

    for elem in root.iter():
        if elem.tag.split("}")[-1] == tag_name:
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

    words = [x for x in korean_keywords if x]
    queries = []

    if words:
        queries.append(" ".join(words[:3]))
        queries.append(" ".join(words[:2]))
        for kw in words[:6]:
            queries.append(kw)

    queries = unique_keep_order(queries)
    applicant_queries = [PNU_IUCF_APPLICANT_KR, "부산대학교"]

    collected = []
    seen = set()

    for q in queries[:8]:
        for applicant in applicant_queries:
            root = kipris_call(
                "getAdvancedSearch",
                tuple(
                    sorted(
                        {
                            "word": q,
                            "applicant": applicant,
                            "patent": "true",
                            "utility": "false",
                            "lastvalue": "R",
                            "sortSpec": "AD",
                            "descSort": "true",
                            "pageNo": "1",
                            "numOfRows": "40",
                        }.items()
                    )
                ),
            )

            if root is None:
                continue

            items_nodes = [elem for elem in root.iter() if elem.tag.split("}")[-1] == "items"]
            if not items_nodes:
                continue

            for item in items_nodes[0]:
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

                collected.append(
                    {
                        "application_number": app_no,
                        "title": title or "발명의 명칭 미상",
                        "abstract": abstract,
                        "application_date": application_date,
                        "register_number": register_number,
                        "register_date": register_date,
                        "register_status": register_status,
                        "applicant_name_summary": applicant_name,
                        "search_query": q,
                    }
                )

                if len(collected) >= 60:
                    return collected

    return collected


@st.cache_data(show_spinner=False)
def get_kipris_bibliography_detail(application_number: str) -> Optional[Dict]:
    if not application_number:
        return None

    root = kipris_call(
        "getBibliographyDetailInfoSearch",
        tuple(sorted({"applicationNumber": application_number}.items())),
    )

    if root is None:
        return None

    def extract_nested_names(array_tag: str) -> List[str]:
        vals = []
        for elem in root.iter():
            if elem.tag.split("}")[-1] == array_tag:
                for sub in elem.iter():
                    if sub.tag.split("}")[-1] == "name":
                        val = normalize_space(sub.text or "")
                        if val:
                            vals.append(val)
        return unique_keep_order(vals)

    applicant_names = extract_nested_names("applicantInfoArray")
    inventor_names = extract_nested_names("inventorInfoArray")

    if not applicant_names:
        applicant_names = unique_keep_order(extract_texts_by_tag(root, "name"))[:5]

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
        "inventor_names": inventor_names[:20],
        "ipc_numbers": unique_keep_order(extract_texts_by_tag(root, "ipcNumber"))[:8],
    }

    return detail


def is_pnu_iucf_included(detail: Dict) -> bool:
    applicants = [normalize_space(x) for x in detail.get("applicant_names", []) if normalize_space(x)]

    if not applicants:
        return False

    for name in applicants:
        if PNU_IUCF_APPLICANT_KR in name:
            return True
        if "부산대학교" in name:
            return True
        low = name.lower()
        if "pusan national university" in low or "busan national university" in low:
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

    return valid, inventors


def score_patent_relevance(valid_patents: List[Dict], profile: Dict, tech_summary: str) -> Dict[str, Dict]:
    if not valid_patents:
        return {}

    if client is None:
        return {
            str(i): {
                "relevance": "Medium",
                "score": 60,
                "reason": "Gemini API 미설정으로 기본 적합도 적용",
            }
            for i, _ in enumerate(valid_patents, start=1)
        }

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

    return {
        str(i): {
            "relevance": "Medium",
            "score": 60,
            "reason": "적합성 평가 실패로 기본값 적용",
        }
        for i, _ in enumerate(valid_patents, start=1)
    }


def select_relevant_patents(valid_patents: List[Dict], relevance_map: Dict[str, Dict]) -> List[Dict]:
    selected = []
    medium = []
    low = []

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
        selected.extend(
            sorted(low, key=lambda x: x.get("patent_score", 0), reverse=True)[
                : MIN_RELEVANT_PATENTS - len(selected)
            ]
        )

    return selected[:MAX_PATENTS]


# =========================================================
# 논문/특허 요약
# =========================================================
def summarize_papers(valid_papers: List[Dict]) -> Dict[str, Dict[str, str]]:
    if not valid_papers:
        return {}

    if client is None:
        return {}

    blocks = []

    for i, p in enumerate(valid_papers, start=1):
        abs_text = reconstruct_abstract(p.get("abstract_inverted_index"))
        blocks.append(
            f"[{i}] Title: {p.get('title')}\n"
            f"Abstract: {abs_text[:700]}\n"
        )

    prompt = f"""
아래 논문들에 대해 각 번호별로
1) 한국어 번역 제목
2) 기술 핵심 요약 한 줄
을 작성하세요.

출력 형식:
[번호] 번역제목 | 요약내용

{chr(10).join(blocks)}
"""

    text = safe_gemini_text(prompt)
    parsed = {}

    for line in text.split("\n"):
        if "|" in line and line.strip().startswith("[") and "]" in line:
            try:
                parts = line.split("]", 1)
                idx = parts[0].replace("[", "").strip()
                title_part, sum_part = parts[1].split("|", 1)
                parsed[idx] = {
                    "title": title_part.strip(),
                    "sum": sum_part.strip(),
                }
            except Exception:
                continue

    return parsed


def summarize_patents(valid_patents: List[Dict]) -> Dict[str, Dict[str, str]]:
    if not valid_patents:
        return {}

    if client is None:
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

    text = safe_gemini_text(prompt)
    parsed = {}

    for line in text.split("\n"):
        if "|" in line and line.strip().startswith("[") and "]" in line:
            try:
                parts = line.split("]", 1)
                idx = parts[0].replace("[", "").strip()
                title_part, sum_part = parts[1].split("|", 1)
                parsed[idx] = {
                    "title": title_part.strip(),
                    "sum": sum_part.strip(),
                }
            except Exception:
                continue

    return parsed


# =========================================================
# 연구자 맵 구성
# =========================================================
def make_unverified_researcher(
    name: str,
    source_type: str,
) -> Dict:
    if source_type == "paper":
        evidence = "OpenAlex 논문 저자 정보에서 부산대 소속으로 확인되었으나, PNU Scholar API 자동 매칭은 실패함"
        field = "논문 저자 및 부산대 소속 정보 기준 후보"
    else:
        evidence = "KIPRIS 특허 발명자 정보에서 확인되었으나, PNU Scholar API 자동 매칭은 실패함"
        field = "특허 발명자 정보 기준 후보"

    return {
        "official_name": name,
        "department": "PNU Scholar 자동 매칭 미확인",
        "field": field,
        "link": "#",
        "evidence": evidence,
        "verified": False,
        "match_score": 0,
        "matched_variant": "",
    }


def build_researcher_map(
    valid_papers: List[Dict],
    valid_patents: List[Dict],
    scholar_matches: Dict[str, Dict],
    parsed_papers: Dict,
    parsed_patents: Dict,
) -> Dict:
    researcher_map = {}

    # 논문 기반
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
            if not is_pnu:
                continue

            db = scholar_matches.get(name) or make_unverified_researcher(name, "paper")
            key = db.get("official_name") or name

            if key not in researcher_map:
                researcher_map[key] = {
                    "name": key,
                    "query_names": [],
                    "dept": db.get("department", "확인 실패"),
                    "field": db.get("field", "상세 분야 확인 실패"),
                    "link": db.get("link", "#"),
                    "evidence": db.get("evidence", ""),
                    "verified": bool(db.get("verified", False)),
                    "match_score": db.get("match_score", 0),
                    "matched_variant": db.get("matched_variant", ""),
                    "papers": [],
                    "patents": [],
                }

            researcher_map[key]["query_names"].append(name)
            researcher_map[key]["papers"].append(paper_obj)

    # 특허 기반
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
            db = scholar_matches.get(name) or make_unverified_researcher(name, "patent")
            key = db.get("official_name") or name

            if key not in researcher_map:
                researcher_map[key] = {
                    "name": key,
                    "query_names": [],
                    "dept": db.get("department", "확인 실패"),
                    "field": db.get("field", "상세 분야 확인 실패"),
                    "link": db.get("link", "#"),
                    "evidence": db.get("evidence", ""),
                    "verified": bool(db.get("verified", False)),
                    "match_score": db.get("match_score", 0),
                    "matched_variant": db.get("matched_variant", ""),
                    "papers": [],
                    "patents": [],
                }

            researcher_map[key]["query_names"].append(name)
            researcher_map[key]["patents"].append(patent_obj)

    # 중복 제거 및 정렬
    for _, data in researcher_map.items():
        data["query_names"] = unique_keep_order(data.get("query_names", []))

        unique_papers = []
        seen_papers = set()

        for paper in sorted(
            data["papers"],
            key=lambda x: (-int(x.get("paper_score", 0)), x.get("title", "")),
        ):
            key = (paper.get("title"), paper.get("date"))
            if key in seen_papers:
                continue
            seen_papers.add(key)
            unique_papers.append(paper)

        data["papers"] = unique_papers

        unique_patents = []
        seen_patents = set()

        for patent in sorted(
            data["patents"],
            key=lambda x: (
                -int(x.get("patent_score", 0)),
                x.get("application_number", ""),
                x.get("title", ""),
            ),
        ):
            key = (patent.get("application_number"), patent.get("title"))
            if key in seen_patents:
                continue
            seen_patents.add(key)
            unique_patents.append(patent)

        data["patents"] = unique_patents

    return researcher_map


# =========================================================
# 전체 분석 파이프라인
# =========================================================
def unified_analyze(uploaded_file, manual_text: str, progress_callback=None) -> str:
    def report(step: int, total: int, label: str, detail: str = ""):
        if progress_callback:
            progress_callback(step, total, label, detail)

    total_steps = 12 if kipris_enabled() else 9

    report(0, total_steps, "입력 확인", "파일 또는 직접 입력 내용을 점검하는 중입니다.")
    query_text = (file_text(uploaded_file) if uploaded_file else "").strip() or (manual_text or "").strip()

    if len(query_text) < 5:
        return "분석할 내용이 없습니다. 파일을 업로드하거나 내용을 입력해주세요."

    report(1, total_steps, "PNU Scholar 연구자 DB 수집", "PNU Scholar API에서 부산대 연구자 목록을 불러오는 중입니다.")
    scholar_db = fetch_all_pnu_scholars()

    report(2, total_steps, "기본 정보 추출", "기업명과 수요기술 요약을 정리하는 중입니다.")
    request_meta = extract_request_metadata(query_text)

    report(3, total_steps, "기술 프로파일 생성", "논문/특허 검색용 키워드를 만드는 중입니다.")
    profile = extract_search_profile(query_text)

    if not request_meta.get("tech_summary") and profile.get("korean_summary"):
        request_meta["tech_summary"] = profile.get("korean_summary")

    keywords_text = ", ".join(profile.get("search_keywords", []))
    patent_keywords_text = ", ".join(profile.get("korean_patent_keywords", []))

    report(4, total_steps, "논문 검색", "OpenAlex에서 부산대 관련 논문을 수집하는 중입니다.")
    raw_papers = search_openalex(
        tuple(profile.get("search_keywords", [])),
        tuple(profile.get("applications", [])),
        tuple(profile.get("core_tech", [])),
    )

    report(5, total_steps, "부산대 논문 필터링", f"수집 논문 {len(raw_papers)}건에서 부산대 저자를 식별하는 중입니다.")
    pnu_papers, paper_authors = filter_pnu_papers(raw_papers)

    report(6, total_steps, "논문 적합성 검토", f"부산대 논문 {len(pnu_papers)}건의 적합도를 평가하는 중입니다.")
    paper_relevance_map = score_paper_relevance(
        pnu_papers,
        profile,
        request_meta.get("tech_summary", ""),
    )
    valid_papers = select_relevant_papers(pnu_papers, paper_relevance_map)

    if not valid_papers:
        valid_papers = pnu_papers[:MIN_RELEVANT_PAPERS]

    filtered_paper_authors = []
    seen_paper_author = set()

    for p in valid_papers:
        for name, is_pnu in p.get("raw_authors_info", []):
            if is_pnu and name not in seen_paper_author:
                seen_paper_author.add(name)
                filtered_paper_authors.append(name)

    valid_patents = []
    patent_inventors = []

    if kipris_enabled():
        report(7, total_steps, "특허 검색", "KIPRIS에서 수요기술 연관 특허를 검색하는 중입니다.")
        raw_patents = search_kipris_patents(tuple(profile.get("korean_patent_keywords", [])))

        report(8, total_steps, "부산대 특허 필터링", f"수집 특허 {len(raw_patents)}건에서 부산대 출원 여부를 확인하는 중입니다.")
        pnu_patents, patent_inventors = enrich_and_filter_pnu_iucf_patents(raw_patents)

        report(9, total_steps, "특허 적합성 검토", f"부산대 특허 {len(pnu_patents)}건의 적합도를 평가하는 중입니다.")
        patent_relevance_map = score_patent_relevance(
            pnu_patents,
            profile,
            request_meta.get("tech_summary", ""),
        )
        valid_patents = select_relevant_patents(pnu_patents, patent_relevance_map)

        if not valid_patents:
            valid_patents = pnu_patents[:MIN_RELEVANT_PATENTS]

        match_step = 10
        summarize_step = 11
    else:
        match_step = 7
        summarize_step = 8

    all_people = unique_keep_order(filtered_paper_authors + patent_inventors)[:MAX_RESEARCHERS]

    report(
        match_step,
        total_steps,
        "PNU Scholar 연구자명 매칭",
        f"논문 저자/특허 발명자 {len(all_people)}명을 PNU Scholar API 연구자 DB와 비교하는 중입니다.",
    )

    scholar_matches = {}
    unmatched_people = []

    for name in all_people:
        matched = match_author_to_pnu_scholar(name, scholar_db)
        if matched:
            matched["verified"] = True
            scholar_matches[name] = matched
        else:
            unmatched_people.append(name)

    report(
        summarize_step,
        total_steps,
        "요약 및 결과 정리",
        f"PNU Scholar 매칭 성공 {len(scholar_matches)}명, 미매칭 {len(unmatched_people)}명을 정리하는 중입니다.",
    )

    parsed_papers = summarize_papers(valid_papers)
    parsed_patents = summarize_patents(valid_patents) if kipris_enabled() else {}

    researcher_map = build_researcher_map(
        valid_papers,
        valid_patents,
        scholar_matches,
        parsed_papers,
        parsed_patents,
    )

    # 통계
    high_count = sum(1 for p in valid_papers if p.get("paper_relevance") == "High")
    medium_count = sum(1 for p in valid_papers if p.get("paper_relevance") == "Medium")

    patent_high_count = sum(1 for p in valid_patents if p.get("patent_relevance") == "High")
    patent_medium_count = sum(1 for p in valid_patents if p.get("patent_relevance") == "Medium")

    verified_count = sum(1 for _, data in researcher_map.items() if data.get("verified"))
    unverified_count = len(researcher_map) - verified_count

    # 결과 생성
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
    lines.append("### 📊 분석 요약")
    lines.append(f"- PNU Scholar API 수집 연구자 수: **{len(scholar_db)}명**")
    lines.append(f"- 검토 논문 수: **{len(pnu_papers)}건**")
    lines.append(f"- 적합성 통과 논문 수: **{len(valid_papers)}건** (High {high_count}건 / Medium {medium_count}건)")

    if kipris_enabled():
        lines.append(f"- 적합성 통과 특허 수: **{len(valid_patents)}건** (High {patent_high_count}건 / Medium {patent_medium_count}건)")
        lines.append("- 특허 필터 기준: **출원인에 부산대학교 또는 부산대학교 산학협력단 포함**")
    else:
        lines.append("- KIPRIS_API_KEY가 없어 특허 검색은 건너뜀")

    lines.append(f"- 검토 연구자 수: **{len(all_people)}명**")
    lines.append(f"- 최종 추천 후보 수: **{len(researcher_map)}명**")
    lines.append(f"- PNU Scholar 매칭 완료: **{verified_count}명**")
    lines.append(f"- 논문/특허 기반 미매칭 후보: **{unverified_count}명**")

    if unmatched_people:
        lines.append(f"- 미매칭 후보 일부: {', '.join(unmatched_people[:15])}")

    lines.append("")
    lines.append("---")

    if not researcher_map:
        lines.append("## ⚠️ 추천 결과 없음")
        lines.append("")
        lines.append("- 논문 또는 특허 후보는 일부 확인되었으나, 최종 연구자 맵 구성에 실패했습니다.")
        lines.append("- 입력 기술 키워드가 너무 포괄적이거나, OpenAlex/KIPRIS 검색 결과가 부족할 수 있습니다.")
        return "\n".join(lines)

    sorted_researchers = sorted(
        researcher_map.items(),
        key=lambda x: (
            not x[1].get("verified", False),
            -(
                sum(int(p.get("paper_score", 0)) for p in x[1].get("papers", []))
                + sum(int(p.get("patent_score", 0)) for p in x[1].get("patents", []))
            ),
            -(len(x[1].get("papers", [])) + len(x[1].get("patents", []))),
            x[0],
        ),
    )

    for name, data in sorted_researchers:
        verify_label = "PNU Scholar 매칭 완료" if data.get("verified") else "논문/특허 기반 후보"

        if data.get("verified"):
            header_icon = "🏫"
        else:
            header_icon = "🟡"

        lines.append(f"## {header_icon} {data['dept']} | {name}")
        lines.append(f"- **검증 상태:** {verify_label}")

        if data.get("query_names"):
            qnames = ", ".join(data.get("query_names", []))
            if qnames and qnames != name:
                lines.append(f"- **원천 데이터상 이름:** {qnames}")

        if data.get("verified"):
            lines.append(f"- **매칭 점수:** {data.get('match_score', 0)}")
            if data.get("matched_variant"):
                lines.append(f"- **매칭 이름 변형:** {data.get('matched_variant')}")

        lines.append(f"- **근거:** {data['evidence']}")
        lines.append(f"- **주요 연구분야:** {data['field']}")

        if data.get("link") and data.get("link") != "#":
            lines.append(f"- **공식 링크:** [PNU Scholar 바로가기]({data['link']})")
        else:
            lines.append("- **공식 링크:** 자동 확인 실패")

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
                    lines.append(
                        f"   - 등록정보: "
                        f"{patent.get('register_number', '-') or '-'} / "
                        f"{patent.get('register_date', '-') or '-'} / "
                        f"{patent.get('register_status', '-') or '-'}"
                    )

                lines.append(f"   - 출원인: {', '.join(patent.get('applicant_names', [])) or PNU_IUCF_APPLICANT_KR}")
                lines.append(f"   - 요약: {patent['summary']}")

            lines.append("")

        lines.append("---")

    return "\n".join(lines)


# =========================================================
# Streamlit UI
# =========================================================
st.title("🎓 PNU 수요기술-연구자 증거형 매칭 시스템")
st.caption(
    "수요기술에 맞는 부산대 논문 저자와 특허 발명자를 찾고, "
    "PNU Scholar API 기반 연구자 DB와 이름을 매칭합니다."
)

with st.sidebar:
    st.header("설정 안내")
    st.markdown(
        """
### 필수/선택 API

- `GEMINI_API_KEY`: 기술 키워드 추출, 적합성 평가, 요약 생성에 사용
- `OPENALEX_API_KEY`: 선택사항. 없어도 OpenAlex 검색 가능
- `KIPRIS_API_KEY`: 있으면 특허 검색 포함, 없으면 논문만 분석

### 주요 변경점

- 기존 검색페이지 크롤링 방식 제거
- PNU Scholar 실제 API 직접 호출
- 영문명/한글명/하이픈 이름 변형 매칭
- 공식 매칭 실패자도 후보로 표시
- 검증 완료 후보와 미검증 후보 구분 출력
        """
    )

uploaded_file = st.file_uploader(
    "1. 수요기술조사서 업로드",
    type=["pdf", "docx", "txt", "md"],
)

manual_text = st.text_area(
    "2. 또는 기술 내용 직접 입력",
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
        result = unified_analyze(
            uploaded_file,
            manual_text,
            progress_callback=update_progress,
        )

        progress_bar.progress(1.0)
        status_box.update(label="분석 완료", state="complete", expanded=False)
        step_placeholder.success("분석이 완료되었습니다. 아래 결과를 확인하세요.")
        st.markdown(result)

    except Exception as e:
        status_box.update(label="분석 중 오류 발생", state="error", expanded=True)
        progress_bar.progress(0)
        step_placeholder.error(f"오류가 발생했습니다: {e}")
