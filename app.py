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
    page_title="부산대 수요기술-연구자 매칭 시스템",
    layout="wide",
)

OPENALEX_URL = "https://api.openalex.org/works"

PNU_SCHOLAR_API_URL = "https://scholar.pusan.ac.kr/wp-json/rm/v1/scholars"
PNU_SCHOLAR_SEARCH_PAGE = "https://scholar.pusan.ac.kr/researchers/"
PNU_SCHOLAR_KEYWORD_CLOUD_API_URL = "https://scholar.pusan.ac.kr/wp-json/rm/v1/stats/keyword_cloud"

KIPRIS_BASE_URL = "https://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice"

MAX_PAPERS = 20
MAX_PATENTS = 20
MAX_RESEARCHERS = 40

MIN_RELEVANT_PAPERS = 3
MIN_RELEVANT_PATENTS = 3

REQUEST_TIMEOUT = 20
USER_AGENT = "Mozilla/5.0 (EvidenceOnlyPNUMatcher/3.1; PNU-Scholar-Search)"

PNU_IUCF_APPLICANT_KR = "부산대학교 산학협력단"
PNU_AFFILIATION_HINTS = [
    "pusan national",
    "busan national",
    "부산대",
    "부산대학교",
]

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json,*/*;q=0.8",
        "Accept-Language": "ko,en-US;q=0.9,en;q=0.8",
    }
)


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
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", str(raw_html))
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


def name_similarity(a: str, b: str) -> float:
    na = normalize_name_for_match(a)
    nb = normalize_name_for_match(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def build_name_variants(name: str) -> List[str]:
    base = normalize_space(name)
    if not base:
        return []

    cleaned = base.replace("–", "-").replace("—", "-").replace("-", "-")
    variants = [
        base,
        cleaned,
        cleaned.replace("-", " "),
        cleaned.replace("-", ""),
        cleaned.replace(" ", ""),
        cleaned.replace(",", ""),
    ]

    tokens = re.split(r"[\s,\-]+", cleaned)
    tokens = [t for t in tokens if t]

    if len(tokens) >= 2 and all(re.search(r"[A-Za-z]", t) for t in tokens):
        first_parts = tokens[:-1]
        last = tokens[-1]

        variants.extend(
            [
                " ".join([last] + first_parts),
                "-".join([last] + first_parts),
                f"{last}, {' '.join(first_parts)}",
                f"{last}, {'-'.join(first_parts)}",
                last + "".join(first_parts),
                "".join([last] + first_parts),
                " ".join(first_parts + [last]),
                "".join(first_parts + [last]),
            ]
        )

    return unique_keep_order(variants)


def split_display_name(display_name: str) -> Tuple[str, str]:
    text = strip_tags(display_name)
    text = normalize_space(text)

    m = re.match(r"^(.*?)\((.*?)\)\s*$", text)
    if m:
        eng = normalize_space(m.group(1))
        kor = normalize_space(m.group(2))
        return eng, kor

    if has_korean(text) and not has_english(text):
        return "", text

    if has_english(text) and not has_korean(text):
        return text, ""

    return text, ""


def parse_affiliation_text(text: str) -> Tuple[str, str]:
    text = strip_tags(text)
    text = normalize_space(text)
    if not text:
        return "", ""

    text = text.replace("ㆍ", "·").replace("|", "·").replace("/", "·")
    parts = [normalize_space(p) for p in text.split("·") if normalize_space(p)]

    if len(parts) >= 2:
        return parts[0], parts[1]

    if re.search(r"(학과|학부|전공|부|과)$", text):
        return text, ""

    if re.search(r"(대학|대학원|전문대학원)$", text):
        return "", text

    return text, ""


def format_department(affiliation: str, department: str = "", college: str = "") -> str:
    affiliation = normalize_space(affiliation)
    department = normalize_space(department)
    college = normalize_space(college)

    if department and college:
        return f"{department} · {college}"
    if affiliation:
        return affiliation
    if department:
        return department
    if college:
        return college
    return "검색 결과 확인"


def is_name_like(text: str) -> bool:
    text = normalize_space(strip_tags(text))
    if not text:
        return False
    if len(text) > 120:
        return False

    if has_english(text) and re.search(r"\([가-힣A-Za-z\s]{2,50}\)", text):
        return True

    if re.fullmatch(r"[가-힣\s]{2,30}", text):
        return True

    if re.fullmatch(r"[A-Za-z,\-\.\s]{2,100}", text):
        bad = ["research", "department", "college", "university", "school", "profile", "journal"]
        return not any(b in text.lower() for b in bad)

    return False


def is_affiliation_like(text: str) -> bool:
    text = normalize_space(strip_tags(text))
    if not text:
        return False
    if len(text) > 150:
        return False

    if "·" in text and re.search(r"(학과|학부|전공|대학|대학원|School|College|Department)", text, re.I):
        return True

    if re.search(r"(학과|학부|전공|대학|대학원|전문대학원)", text):
        return True

    if re.search(r"(Department|College|School|Faculty|Division|Major)", text, re.I):
        return True

    return False


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
# Gemini 처리
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
        snippet = cleaned[start : end + 1]
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
        if len(fallback_tokens) >= 10:
            break

    prompt = f"""
당신은 대학 산학협력 기술수요를 논문·특허 검색에 최적화하는 전문 검색전략 설계기입니다.

아래 입력문은 기업 또는 수요자가 자유롭게 작성한 기술 설명입니다.
입력문을 그대로 검색하지 말고, 논문·특허 검색에 적합하도록 기술 개념을 정규화하고 확장하세요.

목표:
- OpenAlex 논문 검색에서 부산대학교 연구자 논문이 잘 검색되도록 영어 검색어를 구성
- KIPRIS 특허 검색에서 부산대학교/부산대학교 산학협력단 특허가 잘 검색되도록 한국어 검색어를 구성
- 너무 넓은 일반어는 줄이고, 장치·소재·공정·알고리즘·성능·적용분야 중심으로 구체화
- 기업명, 사업명, 지역명, 불필요한 행정 문구는 검색 키워드에서 제외
- 기술이 너무 포괄적이면 세부기술 후보를 2~4개로 나누어 검색 가능하게 구성

반드시 JSON만 출력하세요.

출력 형식:
{{
  "optimized_query_ko": "입력 기술을 검색 친화적으로 정리한 한국어 설명 2~3문장",
  "optimized_query_en": "Search-optimized English technical description in 2-3 sentences",
  "core_tech": ["핵심 기술 영어 2~5개"],
  "materials_or_methods": ["소재/방법/공정/알고리즘 영어 2~6개"],
  "properties": ["성능/특성 영어 2~5개"],
  "applications": ["적용처 영어 1~4개"],
  "search_keywords": ["OpenAlex용 짧은 영어 키워드 6~10개"],
  "openalex_queries": [
    "핵심기술 + 적용처 + Pusan National University",
    "소재/방법 + 성능 + Pusan National University",
    "핵심기술 동의어 + Busan National University"
  ],
  "korean_patent_keywords": ["KIPRIS 특허 검색용 한국어 키워드 6~12개"],
  "exclude_keywords": ["배제 키워드 영어 0~5개"],
  "korean_summary": "수요기술 요약 1~2문장"
}}

검색어 작성 규칙:
- openalex_queries는 실제 검색창에 넣을 수 있는 짧은 영어 구문으로 작성
- 각 openalex_queries에는 가능하면 Pusan National University 또는 Busan National University를 포함
- search_keywords는 1~4단어 이내의 짧은 기술명 중심
- korean_patent_keywords는 명사형 중심으로 작성
- 약어가 있으면 풀네임과 약어를 모두 고려
- 의료·바이오·로봇·AI·소재·반도체 등 분야별 전문용어를 적극 반영
- 단, 입력문에 없는 기술을 과도하게 창작하지 말 것

입력문:
{compact_text(query_text, 7000)}
"""

    data = safe_gemini_json(prompt)

    if isinstance(data, dict) and (data.get("search_keywords") or data.get("openalex_queries")):
        list_keys = [
            "core_tech",
            "materials_or_methods",
            "properties",
            "applications",
            "search_keywords",
            "openalex_queries",
            "korean_patent_keywords",
            "exclude_keywords",
        ]

        for key in list_keys:
            data[key] = [str(x).strip() for x in data.get(key, []) if str(x).strip()]

        data["optimized_query_ko"] = str(data.get("optimized_query_ko", "")).strip()
        data["optimized_query_en"] = str(data.get("optimized_query_en", "")).strip()
        data["korean_summary"] = str(data.get("korean_summary", "")).strip()

        if not data.get("openalex_queries"):
            base = data.get("search_keywords", [])[:4]
            data["openalex_queries"] = [
                " ".join(base[:3]) + " Pusan National University",
                " ".join(base[:3]) + " Busan National University",
            ]

        return data

    fallback = fallback_tokens[:8] or ["Pusan National University"]

    return {
        "optimized_query_ko": "입력된 기술 설명을 바탕으로 논문·특허 검색용 키워드를 구성했습니다.",
        "optimized_query_en": "Search profile generated from the provided technology description.",
        "core_tech": fallback[:3],
        "materials_or_methods": fallback[3:6],
        "properties": [],
        "applications": [],
        "search_keywords": fallback,
        "openalex_queries": [
            " ".join(fallback[:3]) + " Pusan National University",
            " ".join(fallback[:3]) + " Busan National University",
        ],
        "korean_patent_keywords": fallback,
        "exclude_keywords": [],
        "korean_summary": "입력된 수요기술 설명을 바탕으로 검색 키워드를 구성했습니다.",
    }


# =========================================================
# PNU Scholar 검색 기반 매칭
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
            if any(k in vals for k in ["교수", "부산", "pusan", "department", "college"]):
                score += 1
        score += min(len(lst), 10)
        return score

    candidates = sorted(candidates, key=score_list, reverse=True)
    return candidates[0]


def pick_values_by_key(flat: Dict[str, str], hints: List[str], max_len: int = 500) -> List[str]:
    values = []
    seen = set()

    for key, val in flat.items():
        k = key.lower()
        if any(h in k for h in hints):
            v = normalize_space(val)
            if v and len(v) <= max_len and v not in seen:
                seen.add(v)
                values.append(v)

    return values


def find_display_name_from_flat(flat: Dict[str, str]) -> str:
    name_hints = [
        "title.rendered",
        "post_title",
        "display_name",
        "full_name",
        "scholar_name",
        "researcher_name",
        "author_name",
        "kor_name",
        "korean_name",
        "kr_name",
        "eng_name",
        "english_name",
        "en_name",
        "name_en",
        "name_ko",
        "name",
        "title",
    ]

    for hint in name_hints:
        vals = pick_values_by_key(flat, [hint], max_len=150)
        for v in vals:
            cleaned = strip_tags(v)
            if is_name_like(cleaned):
                return cleaned

    for v in flat.values():
        cleaned = strip_tags(v)
        if has_english(cleaned) and re.search(r"\([가-힣A-Za-z\s]{2,50}\)", cleaned):
            if is_name_like(cleaned):
                return cleaned

    return ""


def find_affiliation_from_flat(flat: Dict[str, str]) -> str:
    affiliation_hints = [
        "department",
        "dept",
        "affiliation",
        "college",
        "school",
        "major",
        "organization",
        "org",
        "division",
        "faculty",
        "belong",
        "소속",
        "학과",
        "학부",
        "대학",
        "전공",
    ]

    for hint in affiliation_hints:
        vals = pick_values_by_key(flat, [hint], max_len=300)
        for v in vals:
            cleaned = strip_tags(v)
            if is_affiliation_like(cleaned):
                return cleaned

    for v in flat.values():
        cleaned = strip_tags(v)
        if is_affiliation_like(cleaned):
            return cleaned

    return ""


def find_profile_link_from_flat(flat: Dict[str, str]) -> str:
    link_hints = [
        "permalink",
        "profile_url",
        "profile",
        "link",
        "url",
        "href",
        "guid.rendered",
    ]

    for hint in link_hints:
        vals = pick_values_by_key(flat, [hint], max_len=800)
        for v in vals:
            link = normalize_space(v)
            if link.startswith("/"):
                link = "https://scholar.pusan.ac.kr" + link
            if link.startswith("http"):
                return link

    return ""


# =========================================================
# PNU Scholar 상세페이지 / 한글명 / Keyword Cloud 보강
# =========================================================
def extract_korean_name_from_anywhere(data: Dict, fallback_name: str = "") -> str:
    candidates = []

    candidates.extend(
        [
            data.get("korean_name", ""),
            data.get("official_name", ""),
            data.get("display_name", ""),
            data.get("matched_variant", ""),
            data.get("search_keyword", ""),
            data.get("name", ""),
            fallback_name,
        ]
    )

    for q in data.get("query_names", []) or []:
        candidates.append(q)

    # English Name(한글명) 형태에서 괄호 안 한글명 우선 추출
    for value in candidates:
        text = normalize_space(strip_tags(value))
        if not text:
            continue
        m = re.search(r"\(([가-힣\s]{2,30})\)", text)
        if m:
            return normalize_space(m.group(1))

    # 순수 한글명 우선
    for value in candidates:
        text = normalize_space(strip_tags(value))
        if re.fullmatch(r"[가-힣\s]{2,30}", text):
            return text

    # 한글이 섞여 있으면 한글 부분만 추출
    for value in candidates:
        text = normalize_space(strip_tags(value))
        m = re.search(r"([가-힣]{2,10})", text)
        if m:
            return m.group(1)

    return normalize_space(fallback_name or data.get("name") or data.get("official_name") or data.get("display_name") or "")


def find_researcher_id_from_flat(flat: Dict[str, str]) -> str:
    # 1) 링크 값 안의 /researchers/숫자 우선 추출
    for val in flat.values():
        text = normalize_space(val)
        m = re.search(r"/researchers/(\d+)/?", text)
        if m:
            return m.group(1)

    # 2) API 원본의 id/post_id/researcher_id/scholar_id 계열 후보 추출
    id_hints = ["researcher_id", "researcher", "scholar_id", "author_id", "user_id", "post_id", "rid", "id"]
    for key, val in flat.items():
        k = key.lower()
        v = normalize_space(val)
        if not re.fullmatch(r"\d{3,10}", v):
            continue
        if any(h in k for h in id_hints):
            return v

    return ""


def normalize_pnu_profile_link(link: str = "", researcher_id: str = "") -> str:
    link = normalize_space(link)

    if link.startswith("/"):
        link = "https://scholar.pusan.ac.kr" + link

    if link.startswith("http") and "/researchers/" in link:
        m = re.search(r"/researchers/(\d+)/?", link)
        if m:
            return f"https://scholar.pusan.ac.kr/researchers/{m.group(1)}/"
        return link.rstrip("/") + "/"

    if researcher_id:
        return f"https://scholar.pusan.ac.kr/researchers/{researcher_id}/"

    return "https://scholar.pusan.ac.kr/researchers/"


@st.cache_data(show_spinner=False)
def fetch_pnu_profile_html(profile_link: str) -> str:
    profile_link = normalize_space(profile_link)
    if not profile_link or profile_link == "#" or profile_link.rstrip("/").endswith("/researchers"):
        return ""

    try:
        r = SESSION.get(profile_link, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            return r.text or ""
    except Exception:
        return ""

    return ""


def extract_keyword_cloud_terms_from_html(html_text: str, max_terms: int = 5) -> List[str]:
    if not html_text:
        return []

    plain = strip_tags(html_text)
    plain = normalize_space(plain)

    if "Keyword Cloud" not in plain:
        return []

    after = plain.split("Keyword Cloud", 1)[1]

    stop_words = [
        "Publication",
        "Publications",
        "Research Area",
        "Co-author",
        "Coauthors",
        "Similar Researchers",
        "Researcher",
        "Journals",
        "Departments",
        "Highlights",
        "Notice",
        "Q&A",
        "논문",
        "저널",
        "학과",
    ]

    cut_positions = []
    for sw in stop_words:
        pos = after.find(sw)
        if pos > 0:
            cut_positions.append(pos)

    if cut_positions:
        after = after[: min(cut_positions)]

    after = after[:4000]

    candidates = re.findall(r"[A-Za-z][A-Za-z0-9\-\/\s]{2,80}", after)

    blocked = {
        "keyword cloud",
        "researcher",
        "researchers",
        "publication",
        "publications",
        "notice",
        "login",
        "search",
        "department",
        "departments",
        "journal",
        "journals",
        "highlight",
        "highlights",
        "privacy policy",
        "need help",
        "pnu scholar",
        "pusan national university",
        "busan national university",
        "all rights reserved",
        "close",
        "image",
    }

    cleaned = []
    for c in candidates:
        term = normalize_space(c).strip(" -_.,;:()[]{}")
        term_low = term.lower()

        if not term:
            continue
        if len(term) < 3 or len(term) > 80:
            continue
        if term_low in blocked:
            continue
        if any(b in term_low for b in ["copyright", "privacy", "login", "notice", "research guide", "all rights"]):
            continue
        if re.fullmatch(r"\d+", term):
            continue

        cleaned.append(term)

    return unique_keep_order(cleaned)[:max_terms]


def extract_keyword_terms_from_api_result(data: Any, max_terms: int = 12) -> List[str]:
    """
    PNU Scholar Keyword Cloud REST API 응답에서 키워드 문자열만 최대한 보수적으로 추출합니다.
    API 응답 구조가 바뀌어도 result/data/items/list 내부를 재귀적으로 탐색합니다.
    """
    terms = []

    def add_term(value: Any):
        if value is None:
            return
        if isinstance(value, (int, float, bool)):
            return

        text = normalize_space(strip_tags(str(value)))
        if not text:
            return

        # 쉼표/세미콜론으로 묶여 오는 경우까지 분리
        pieces = re.split(r"[,;|]", text) if len(text) > 80 else [text]
        for piece in pieces:
            piece = normalize_space(piece).strip(" -_.,;:()[]{}\"'")
            if piece:
                terms.append(piece)

    def walk(obj: Any):
        if isinstance(obj, dict):
            keyword_keys = {
                "keyword",
                "keywords",
                "name",
                "text",
                "label",
                "word",
                "term",
                "title",
                "key",
            }

            for k, v in obj.items():
                kl = str(k).lower()

                # count, value, weight, size 같은 수치 필드는 키워드로 쓰지 않음
                if kl in {"count", "cnt", "value", "weight", "size", "score", "rank"}:
                    continue

                if any(h == kl or h in kl for h in keyword_keys):
                    if isinstance(v, (str, int, float)):
                        add_term(v)
                    else:
                        walk(v)
                else:
                    walk(v)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

        elif isinstance(obj, str):
            add_term(obj)

    walk(data)

    blocked = {
        "keyword cloud",
        "researcher",
        "researchers",
        "publication",
        "publications",
        "pnu scholar",
        "pusan national university",
        "busan national university",
        "loading",
        "null",
        "none",
        "true",
        "false",
        "status",
        "result",
    }

    cleaned = []
    for term in terms:
        t = normalize_space(term).strip(" -_.,;:()[]{}\"'")
        tl = t.lower()

        if not t:
            continue
        if len(t) < 2 or len(t) > 80:
            continue
        if tl in blocked:
            continue
        if re.fullmatch(r"\d+", t):
            continue
        if re.search(r"^(http|www\.)", tl):
            continue
        if any(b in tl for b in ["copyright", "privacy", "login", "notice", "all rights"]):
            continue

        cleaned.append(t)

    return unique_keep_order(cleaned)[:max_terms]


@st.cache_data(show_spinner=False)
def fetch_keyword_cloud_by_researcher_id(researcher_id: str) -> List[str]:
    """
    F12 Network에서 확인되는 /wp-json/rm/v1/stats/keyword_cloud API를 직접 호출합니다.
    파라미터명이 환경마다 다를 수 있어 GET/POST + 여러 후보명을 순차적으로 시도합니다.
    """
    researcher_id = normalize_space(researcher_id)
    if not researcher_id:
        return []

    param_candidates = [
        {"id": researcher_id},
        {"researcher_id": researcher_id},
        {"researcher": researcher_id},
        {"author_id": researcher_id},
        {"scholar_id": researcher_id},
        {"user_id": researcher_id},
        {"post_id": researcher_id},
        {"rid": researcher_id},
    ]

    # 1) GET query string 방식
    for params in param_candidates:
        try:
            r = SESSION.get(
                PNU_SCHOLAR_KEYWORD_CLOUD_API_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code != 200:
                continue

            data = r.json()
            terms = extract_keyword_terms_from_api_result(data)
            if terms:
                return terms
        except Exception:
            continue

    # 2) POST form 방식
    for params in param_candidates:
        try:
            r = SESSION.post(
                PNU_SCHOLAR_KEYWORD_CLOUD_API_URL,
                data=params,
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code != 200:
                continue

            data = r.json()
            terms = extract_keyword_terms_from_api_result(data)
            if terms:
                return terms
        except Exception:
            continue

    # 3) POST JSON 방식
    for params in param_candidates:
        try:
            r = SESSION.post(
                PNU_SCHOLAR_KEYWORD_CLOUD_API_URL,
                json=params,
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code != 200:
                continue

            data = r.json()
            terms = extract_keyword_terms_from_api_result(data)
            if terms:
                return terms
        except Exception:
            continue

    return []


@st.cache_data(show_spinner=False)
def translate_keyword_cloud_to_korean_cached(keywords_tuple: Tuple[str, ...]) -> List[str]:
    keywords = [normalize_space(x) for x in keywords_tuple if normalize_space(x)]
    if not keywords:
        return []

    if client is None:
        return keywords

    prompt = f"""
아래 PNU Scholar Keyword Cloud 영어 키워드를 한국어 연구분야 키워드로 번역하세요.

규칙:
- 각 키워드는 짧은 한국어 명사구로 번역
- 원문 순서를 유지
- 원문 개수와 동일하게 JSON 배열만 출력
- 설명 문장 금지
- 고유한 학술 용어는 자연스러운 한국어 연구 키워드로 번역

입력 키워드:
{json.dumps(keywords, ensure_ascii=False)}

출력 예:
["기계학습", "탄소나노튜브", "단백질 발현"]
"""

    raw = safe_gemini_text(prompt)
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            translated = [normalize_space(str(x)) for x in data if normalize_space(str(x))]
            if translated:
                return translated[: len(keywords)]
    except Exception:
        pass

    return keywords


def get_korean_keyword_cloud_from_profile(profile_link: str, researcher_id: str = "") -> str:
    """
    연구자 상세페이지 번호가 있으면 Keyword Cloud API를 먼저 호출하고,
    실패할 경우 기존 HTML 파싱 방식으로 보조 추출합니다.
    단, 번역 비용과 속도를 줄이기 위해 상위 5개 키워드만 번역합니다.
    """
    profile_link = normalize_space(profile_link)
    researcher_id = normalize_space(researcher_id)

    if not researcher_id and profile_link:
        m = re.search(r"/researchers/(\d+)/?", profile_link)
        if m:
            researcher_id = m.group(1)

    # 1순위: F12에서 확인된 REST API 직접 호출
    keywords = fetch_keyword_cloud_by_researcher_id(researcher_id)

    # 2순위: 상세페이지 HTML 내 Keyword Cloud 텍스트 파싱
    if not keywords:
        html_text = fetch_pnu_profile_html(profile_link)
        keywords = extract_keyword_cloud_terms_from_html(html_text)

    if not keywords:
        return "Keyword Cloud 자동 추출 실패"

    # 핵심 수정: 상위 5개만 번역 요청
    keywords = unique_keep_order(keywords)[:5]

    translated = translate_keyword_cloud_to_korean_cached(tuple(keywords))
    translated = unique_keep_order(translated)

    if not translated:
        return "Keyword Cloud 자동 추출 실패"

    return ", ".join(translated[:5])


def normalize_scholar_api_record(record: Dict, search_keyword: str = "") -> Optional[Dict]:
    flat = flatten_json_strings(record)

    display_name = find_display_name_from_flat(flat)
    if not display_name:
        return None

    eng_name, kor_name = split_display_name(display_name)

    affiliation = find_affiliation_from_flat(flat)
    dept_major, college = parse_affiliation_text(affiliation)

    raw_profile_link = find_profile_link_from_flat(flat)
    researcher_id = find_researcher_id_from_flat(flat)
    profile_link = normalize_pnu_profile_link(raw_profile_link, researcher_id)

    keyword_cloud_ko = get_korean_keyword_cloud_from_profile(profile_link, researcher_id)

    all_names = unique_keep_order(
        [
            display_name,
            eng_name,
            kor_name,
            search_keyword if has_korean(search_keyword) else "",
            eng_name.replace(",", "") if eng_name else "",
            eng_name.replace("-", " ") if eng_name else "",
            eng_name.replace("-", "") if eng_name else "",
            eng_name.replace(" ", "") if eng_name else "",
        ]
    )

    if kor_name:
        official_name = kor_name
    elif has_korean(search_keyword):
        official_name = search_keyword
    else:
        official_name = eng_name or display_name

    department = format_department(affiliation, dept_major, college)

    return {
        "official_name": official_name,
        "display_name": display_name,
        "english_name": eng_name,
        "korean_name": kor_name or (search_keyword if has_korean(search_keyword) else ""),
        "all_names": all_names,
        "department": department,
        "field": keyword_cloud_ko,
        "link": profile_link,
        "researcher_id": researcher_id,
        "evidence": f"PNU Scholar 검색어 '{search_keyword}'로 검색 결과 확인: {display_name}",
        "source": "pnu_scholar_api",
        "search_keyword": search_keyword,
        "raw": record,
    }


def parse_scholar_html_results(html_text: str, search_keyword: str = "") -> List[Dict]:
    results = []

    # 1) 검색 결과 HTML의 연구자 상세 링크 우선 추출
    anchor_pattern = r'''<a[^>]+href=["\']([^"\']*?/researchers/(\d+)/?[^"\']*)["\'][^>]*>(.*?)</a>'''
    for href, researcher_id, inner_html in re.findall(anchor_pattern, html_text, flags=re.I | re.S):
        display_name = strip_tags(inner_html)
        display_name = normalize_space(display_name)

        if not is_name_like(display_name):
            continue

        eng_name, kor_name = split_display_name(display_name)
        if not display_name or not (eng_name or kor_name):
            continue

        profile_link = normalize_pnu_profile_link(href, researcher_id)
        keyword_cloud_ko = get_korean_keyword_cloud_from_profile(profile_link, researcher_id)

        all_names = unique_keep_order(
            [
                display_name,
                eng_name,
                kor_name,
                search_keyword if has_korean(search_keyword) else "",
                eng_name.replace(",", "") if eng_name else "",
                eng_name.replace("-", " ") if eng_name else "",
                eng_name.replace("-", "") if eng_name else "",
                eng_name.replace(" ", "") if eng_name else "",
            ]
        )

        official_name = kor_name or (search_keyword if has_korean(search_keyword) else "") or eng_name or display_name

        results.append(
            {
                "official_name": official_name,
                "display_name": display_name,
                "english_name": eng_name,
                "korean_name": kor_name or (search_keyword if has_korean(search_keyword) else ""),
                "all_names": all_names,
                "department": "PNU Scholar 검색 결과 기반 확인",
                "field": keyword_cloud_ko,
                "link": profile_link,
                "researcher_id": researcher_id,
                "evidence": f"PNU Scholar 검색어 '{search_keyword}'로 HTML 검색 결과 확인: {display_name}",
                "source": "pnu_scholar_html",
                "search_keyword": search_keyword,
            }
        )

    if results:
        return dedupe_scholar_results(results)

    # 2) 링크 파싱 실패 시 기존 텍스트 패턴으로 보조 확인
    text = strip_tags(html_text)
    pattern = r"([A-Z][A-Za-z,\-\.\s]{1,90}\([가-힣A-Za-z\s]{2,50}\))"

    for display_name in re.findall(pattern, text):
        display_name = normalize_space(display_name)
        eng_name, kor_name = split_display_name(display_name)

        if not display_name or not (eng_name or kor_name):
            continue

        all_names = unique_keep_order(
            [
                display_name,
                eng_name,
                kor_name,
                search_keyword if has_korean(search_keyword) else "",
                eng_name.replace(",", "") if eng_name else "",
                eng_name.replace("-", " ") if eng_name else "",
                eng_name.replace("-", "") if eng_name else "",
                eng_name.replace(" ", "") if eng_name else "",
            ]
        )

        official_name = kor_name or (search_keyword if has_korean(search_keyword) else "") or eng_name or display_name

        results.append(
            {
                "official_name": official_name,
                "display_name": display_name,
                "english_name": eng_name,
                "korean_name": kor_name or (search_keyword if has_korean(search_keyword) else ""),
                "all_names": all_names,
                "department": "PNU Scholar 검색 결과 기반 확인",
                "field": "Keyword Cloud 자동 추출 실패",
                "link": "https://scholar.pusan.ac.kr/researchers/",
                "researcher_id": "",
                "evidence": f"PNU Scholar 검색어 '{search_keyword}'로 HTML 검색 결과 확인: {display_name}",
                "source": "pnu_scholar_html",
                "search_keyword": search_keyword,
            }
        )

    return dedupe_scholar_results(results)


def dedupe_scholar_results(results: List[Dict]) -> List[Dict]:
    unique = []
    seen = set()

    for r in results:
        key = normalize_name_for_match(
            f"{r.get('display_name')}|{r.get('english_name')}|{r.get('korean_name')}|{r.get('link')}"
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    return unique


@st.cache_data(show_spinner=False)
def search_pnu_scholar_by_keyword(keyword: str) -> List[Dict]:
    keyword = normalize_space(keyword)
    if not keyword:
        return []

    results = []

    try:
        params = {
            "sub_ks": keyword,
            "order_by": "score",
            "order": "desc",
            "record_per_page": 12,
            "page": 1,
            "current_page": 1,
        }

        r = SESSION.get(PNU_SCHOLAR_API_URL, params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            records = extract_scholar_records_from_response(data)
            for rec in records:
                norm = normalize_scholar_api_record(rec, search_keyword=keyword)
                if norm:
                    results.append(norm)
    except Exception:
        pass

    if results:
        return dedupe_scholar_results(results)

    try:
        params = {
            "sub_ks": keyword,
            "order_by": "score",
        }

        r = SESSION.get(PNU_SCHOLAR_SEARCH_PAGE, params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            results = parse_scholar_html_results(r.text, search_keyword=keyword)
            if results:
                return dedupe_scholar_results(results)
    except Exception:
        pass

    return []


def score_scholar_result_against_author(author_name: str, result: Dict) -> Tuple[float, str]:
    candidate_names = result.get("all_names", []) or []
    candidate_names.extend(
        [
            result.get("display_name", ""),
            result.get("english_name", ""),
            result.get("korean_name", ""),
            result.get("official_name", ""),
        ]
    )
    candidate_names = unique_keep_order(candidate_names)

    variants = build_name_variants(author_name)
    best_score = 0.0
    best_variant = ""

    for q in variants:
        for c in candidate_names:
            score = name_similarity(q, c)
            if score > best_score:
                best_score = score
                best_variant = c

    if has_korean(author_name):
        q_norm = normalize_name_for_match(author_name)
        for c in candidate_names:
            if q_norm and q_norm == normalize_name_for_match(c):
                return 1.0, c

    return best_score, best_variant


def match_author_to_pnu_scholar(author_name: str) -> Optional[Dict]:
    name = normalize_space(author_name)
    if not name:
        return None

    search_queries = build_name_variants(name)
    all_results = []

    for q in search_queries[:10]:
        results = search_pnu_scholar_by_keyword(q)

        if results:
            for r in results:
                r["used_query"] = q
            all_results.extend(results)

        time.sleep(0.12)

        if len(all_results) >= 5:
            break

    all_results = dedupe_scholar_results(all_results)

    if not all_results:
        return None

    best = None
    best_score = 0.0
    best_variant = ""

    for r in all_results:
        score, matched_variant = score_scholar_result_against_author(name, r)
        if score > best_score:
            best_score = score
            best_variant = matched_variant
            best = r

    if not best:
        return None

    if has_korean(name):
        threshold = 0.99
    else:
        norm_len = len(normalize_name_for_match(name))
        threshold = 0.78 if norm_len >= 10 else 0.82

    query_exact_hit = False
    for q in search_queries:
        qn = normalize_name_for_match(q)
        display_blob = normalize_name_for_match(
            " ".join(
                [
                    best.get("display_name", ""),
                    best.get("english_name", ""),
                    best.get("korean_name", ""),
                    best.get("official_name", ""),
                ]
            )
        )
        if qn and qn in display_blob:
            query_exact_hit = True
            break

    if best_score < threshold and not query_exact_hit:
        return None

    used_query = best.get("used_query") or best.get("search_keyword") or search_queries[0]

    best["verified"] = True
    best["match_score"] = round(max(best_score, 0.8 if query_exact_hit else best_score), 3)
    best["matched_variant"] = best_variant
    best["query_name"] = name
    best["search_keyword"] = used_query
    best["evidence"] = (
        f"PNU Scholar 연구자 검색에서 '{used_query}' 검색 결과 확인: "
        f"{best.get('display_name') or best.get('official_name')}"
    )

    return best


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
    search_terms: Tuple[str, ...],
    applications: Tuple[str, ...],
    core_tech: Tuple[str, ...],
) -> List[Dict]:
    terms = list(search_terms)
    apps = list(applications)
    techs = list(core_tech)

    queries = []

    for term in terms[:8]:
        term = normalize_space(term)
        if not term:
            continue

        if "pusan national university" in term.lower() or "busan national university" in term.lower():
            queries.append(term)
        else:
            queries.append(term + " Pusan National University")
            queries.append(term + " Busan National University")

    if techs and apps:
        queries.append(f"{' '.join(techs[:2])} {' '.join(apps[:2])} Pusan National University")

    if techs:
        queries.append(" ".join(techs[:3]) + " Pusan National University")

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

검색 최적화 프로파일:
- optimized_query_ko: {profile.get('optimized_query_ko', '')}
- optimized_query_en: {profile.get('optimized_query_en', '')}
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

검색 최적화 프로파일:
- optimized_query_ko: {profile.get('optimized_query_ko', '')}
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
def make_unverified_researcher(name: str, source_type: str) -> Dict:
    if source_type == "paper":
        evidence = "OpenAlex 논문 저자 정보에서 부산대 소속으로 확인되었으나, PNU Scholar 연구자 검색에서는 자동 확인되지 않음"
        field = "논문 저자 및 부산대 소속 정보 기준 후보"
    else:
        evidence = "KIPRIS 특허 발명자 정보에서 확인되었으나, PNU Scholar 연구자 검색에서는 자동 확인되지 않음"
        field = "특허 발명자 정보 기준 후보"

    return {
        "official_name": name,
        "department": "PNU Scholar 검색 미확인",
        "field": field,
        "link": "#",
        "evidence": evidence,
        "verified": False,
        "match_score": 0,
        "matched_variant": "",
        "display_name": name,
        "english_name": name if has_english(name) else "",
        "korean_name": name if has_korean(name) else "",
        "search_keyword": "",
    }


def build_researcher_map(
    valid_papers: List[Dict],
    valid_patents: List[Dict],
    scholar_matches: Dict[str, Dict],
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
                    "display_name": db.get("display_name", ""),
                    "english_name": db.get("english_name", ""),
                    "korean_name": db.get("korean_name", ""),
                    "search_keyword": db.get("search_keyword", "") or db.get("used_query", ""),
                    "papers": [],
                    "patents": [],
                }

            researcher_map[key]["query_names"].append(name)
            researcher_map[key]["papers"].append(paper_obj)

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
                    "display_name": db.get("display_name", ""),
                    "english_name": db.get("english_name", ""),
                    "korean_name": db.get("korean_name", ""),
                    "search_keyword": db.get("search_keyword", "") or db.get("used_query", ""),
                    "papers": [],
                    "patents": [],
                }

            researcher_map[key]["query_names"].append(name)
            researcher_map[key]["patents"].append(patent_obj)

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
# 결과 출력용 마크다운 생성
# =========================================================
def append_researcher_block(target_lines: List[str], name: str, data: Dict, is_verified: bool):
    researcher_name = extract_korean_name_from_anywhere(data, fallback_name=name)

    department = (
        data.get("dept")
        or data.get("department")
        or "PNU Scholar 검색 결과 기반 확인"
    )

    research_field = (
        data.get("field")
        or "Keyword Cloud 자동 추출 실패"
    )

    official_link = data.get("link") or "#"

    # =====================================================
    # 1. 연구자 기본 정보 출력: 요청 형식 4개만 유지
    # =====================================================
    if is_verified:
        target_lines.append(f"## 🏫 PNU Scholar 검색 결과 기반 확인 | {researcher_name}")
        target_lines.append(f"- **연구자명:** {researcher_name}")
        target_lines.append(f"- **소속(학과):** {department}")
        target_lines.append(f"- **연구분야:** {research_field}")

        if official_link and official_link != "#":
            target_lines.append(f"- **공식링크:** [PNU Scholar 바로가기]({official_link})")
        else:
            target_lines.append("- **공식링크:** 자동 확인 실패")

    else:
        target_lines.append(f"## 🟡 PNU Scholar 미확인 후보 | {researcher_name}")
        target_lines.append(f"- **연구자명:** {researcher_name}")
        target_lines.append("- **소속(학과):** PNU Scholar 검색 미확인")
        target_lines.append("- **연구분야:** 논문·특허 기반 후보")
        target_lines.append("- **공식링크:** 자동 확인 실패")

    target_lines.append("")

    # =====================================================
    # 2. 관련 논문 출력
    # =====================================================
    if data["papers"]:
        target_lines.append("#### 📄 관련 논문")

        for idx, paper in enumerate(data["papers"], start=1):
            target_lines.append(f"{idx}. **{paper['k_title']}**")
            target_lines.append(f"   - 원제: {paper['title']}")
            target_lines.append(f"   - 논문 적합도: {paper.get('paper_relevance', 'Unknown')} ({paper.get('paper_score', 0)}점)")

            if paper.get("paper_reason"):
                target_lines.append(f"   - 적합성 근거: {paper['paper_reason']}")

            target_lines.append(f"   - 요약: {paper['summary']} ({paper['date']}, {paper['venue']})")

        target_lines.append("")

    # =====================================================
    # 3. 관련 특허 출력
    # =====================================================
    if data["patents"]:
        target_lines.append("#### 🧾 관련 특허")

        for idx, patent in enumerate(data["patents"], start=1):
            target_lines.append(f"{idx}. **{patent['k_title']}**")
            target_lines.append(f"   - 발명의 명칭: {patent['title']}")
            target_lines.append(f"   - 특허 적합도: {patent.get('patent_relevance', 'Unknown')} ({patent.get('patent_score', 0)}점)")

            if patent.get("patent_reason"):
                target_lines.append(f"   - 적합성 근거: {patent['patent_reason']}")

            target_lines.append(f"   - 출원번호/일자: {patent['application_number']} / {patent['application_date']}")

            if patent.get("register_number") or patent.get("register_date"):
                target_lines.append(
                    f"   - 등록정보: "
                    f"{patent.get('register_number', '-') or '-'} / "
                    f"{patent.get('register_date', '-') or '-'} / "
                    f"{patent.get('register_status', '-') or '-'}"
                )

            target_lines.append(f"   - 출원인: {', '.join(patent.get('applicant_names', [])) or PNU_IUCF_APPLICANT_KR}")
            target_lines.append(f"   - 요약: {patent['summary']}")

        target_lines.append("")

    target_lines.append("---")


# =========================================================
# 전체 분석 파이프라인
# =========================================================
def unified_analyze(uploaded_file, manual_text: str, progress_callback=None) -> Dict:
    def report(step: int, total: int, label: str, detail: str = ""):
        if progress_callback:
            progress_callback(step, total, label, detail)

    total_steps = 11 if kipris_enabled() else 8

    report(0, total_steps, "입력 확인", "파일 또는 직접 입력 내용을 점검하는 중입니다.")
    query_text = (file_text(uploaded_file) if uploaded_file else "").strip() or (manual_text or "").strip()

    if len(query_text) < 5:
        return {
            "main_markdown": "분석할 내용이 없습니다. 파일을 업로드하거나 내용을 입력해주세요.",
            "unconfirmed_markdown": "",
            "unconfirmed_count": 0,
        }

    report(1, total_steps, "기본 정보 추출", "기업명과 수요기술 요약을 정리하는 중입니다.")
    request_meta = extract_request_metadata(query_text)

    report(2, total_steps, "검색 최적화 프로파일 생성", "기술 내용을 논문·특허 검색용 키워드로 확장/정리하는 중입니다.")
    profile = extract_search_profile(query_text)

    if not request_meta.get("tech_summary") and profile.get("korean_summary"):
        request_meta["tech_summary"] = profile.get("korean_summary")

    report(3, total_steps, "논문 검색", "OpenAlex에서 부산대 관련 논문을 수집하는 중입니다.")
    openalex_search_terms = profile.get("openalex_queries") or profile.get("search_keywords", [])

    raw_papers = search_openalex(
        tuple(openalex_search_terms),
        tuple(profile.get("applications", [])),
        tuple(profile.get("core_tech", [])),
    )

    report(4, total_steps, "부산대 논문 필터링", f"수집 논문 {len(raw_papers)}건에서 부산대 저자를 식별하는 중입니다.")
    pnu_papers, _ = filter_pnu_papers(raw_papers)

    report(5, total_steps, "논문 적합성 검토", f"부산대 논문 {len(pnu_papers)}건의 적합도를 평가하는 중입니다.")
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
        report(6, total_steps, "특허 검색", "KIPRIS에서 수요기술 연관 특허를 검색하는 중입니다.")
        raw_patents = search_kipris_patents(tuple(profile.get("korean_patent_keywords", [])))

        report(7, total_steps, "부산대 특허 필터링", f"수집 특허 {len(raw_patents)}건에서 부산대 출원 여부를 확인하는 중입니다.")
        pnu_patents, patent_inventors = enrich_and_filter_pnu_iucf_patents(raw_patents)

        report(8, total_steps, "특허 적합성 검토", f"부산대 특허 {len(pnu_patents)}건의 적합도를 평가하는 중입니다.")
        patent_relevance_map = score_patent_relevance(
            pnu_patents,
            profile,
            request_meta.get("tech_summary", ""),
        )
        valid_patents = select_relevant_patents(pnu_patents, patent_relevance_map)

        if not valid_patents:
            valid_patents = pnu_patents[:MIN_RELEVANT_PATENTS]

        match_step = 9
        summarize_step = 10
    else:
        match_step = 6
        summarize_step = 7

    all_people = unique_keep_order(filtered_paper_authors + patent_inventors)[:MAX_RESEARCHERS]

    report(
        match_step,
        total_steps,
        "PNU Scholar 연구자 검색",
        f"논문 저자/특허 발명자 {len(all_people)}명을 PNU Scholar 검색창 방식으로 확인하는 중입니다.",
    )

    scholar_matches = {}
    unmatched_people = []

    for name in all_people:
        matched = match_author_to_pnu_scholar(name)
        if matched:
            matched["verified"] = True
            scholar_matches[name] = matched
        else:
            unmatched_people.append(name)

    report(
        summarize_step,
        total_steps,
        "요약 및 결과 정리",
        f"PNU Scholar 검색 확인 {len(scholar_matches)}명, 미확인 {len(unmatched_people)}명을 정리하는 중입니다.",
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

    high_count = sum(1 for p in valid_papers if p.get("paper_relevance") == "High")
    medium_count = sum(1 for p in valid_papers if p.get("paper_relevance") == "Medium")

    patent_high_count = sum(1 for p in valid_patents if p.get("patent_relevance") == "High")
    patent_medium_count = sum(1 for p in valid_patents if p.get("patent_relevance") == "Medium")

    verified_count = sum(1 for _, data in researcher_map.items() if data.get("verified"))
    unverified_count = len(researcher_map) - verified_count

    lines = []
    unconfirmed_lines = []

    lines.append(f"### 🏢 기업명: **{request_meta.get('company_name', '미확인')}**")
    lines.append("")
    lines.append("### 📝 수요기술 요약")
    lines.append(request_meta.get("tech_summary", "입력된 수요기술 설명을 바탕으로 연구자 매칭을 수행했습니다."))
    lines.append("")

    if profile.get("optimized_query_ko") or profile.get("optimized_query_en"):
        lines.append("### 🧭 검색 최적화 기술 프로파일")
        if profile.get("optimized_query_ko"):
            lines.append(f"- **검색용 정리문:** {profile.get('optimized_query_ko')}")
        if profile.get("optimized_query_en"):
            lines.append(f"- **English Search Profile:** {profile.get('optimized_query_en')}")
        lines.append("")

    keywords_text = ", ".join(profile.get("search_keywords", []))
    openalex_query_text = " / ".join(profile.get("openalex_queries", []))
    patent_keywords_text = ", ".join(profile.get("korean_patent_keywords", []))

    lines.append("### 🔍 논문·특허 분석 키워드")
    lines.append(f"- **핵심 키워드:** {keywords_text}")
    if openalex_query_text:
        lines.append(f"- **OpenAlex 검색식:** {openalex_query_text}")
    if kipris_enabled():
        lines.append(f"- **특허 분석 키워드:** {patent_keywords_text}")

    lines.append("")
    lines.append("### 📊 분석 요약")
    lines.append("- PNU Scholar 확인 방식: **연구자 검색창 sub_ks 검색 기반 확인**")
    lines.append(f"- 검토 논문 수: **{len(pnu_papers)}건**")
    lines.append(f"- 적합성 통과 논문 수: **{len(valid_papers)}건** (High {high_count}건 / Medium {medium_count}건)")

    if kipris_enabled():
        lines.append(f"- 적합성 통과 특허 수: **{len(valid_patents)}건** (High {patent_high_count}건 / Medium {patent_medium_count}건)")
        lines.append("- 특허 필터 기준: **출원인에 부산대학교 또는 부산대학교 산학협력단 포함**")
    else:
        lines.append("- KIPRIS_API_KEY가 없어 특허 검색은 건너뜀")

    lines.append(f"- 검토 연구자 수: **{len(all_people)}명**")
    lines.append(f"- 최종 추천 후보 수: **{len(researcher_map)}명**")
    lines.append(f"- PNU Scholar 검색 확인: **{verified_count}명**")
    lines.append(f"- PNU Scholar 미확인 후보: **{unverified_count}명**")
    lines.append("")
    lines.append("---")

    if not researcher_map:
        lines.append("## ⚠️ 추천 결과 없음")
        lines.append("")
        lines.append("- 논문 또는 특허 후보는 일부 확인되었으나, 최종 연구자 맵 구성에 실패했습니다.")
        lines.append("- 입력 기술 키워드가 너무 포괄적이거나, OpenAlex/KIPRIS 검색 결과가 부족할 수 있습니다.")

        return {
            "main_markdown": "\n".join(lines),
            "unconfirmed_markdown": "",
            "unconfirmed_count": 0,
        }

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

    lines.append("## ✅ PNU Scholar 검색 확인 연구자")
    lines.append("")

    confirmed_output_count = 0
    unconfirmed_output_count = 0

    for name, data in sorted_researchers:
        is_verified = bool(data.get("verified"))

        if is_verified:
            confirmed_output_count += 1
            append_researcher_block(lines, name, data, True)
        else:
            unconfirmed_output_count += 1
            append_researcher_block(unconfirmed_lines, name, data, False)

    if confirmed_output_count == 0:
        lines.append("- PNU Scholar에서 자동 확인된 연구자가 없습니다.")
        lines.append("- 아래 **연구자 미확인 건 보기**에서 논문·특허 기반 후보를 확인하세요.")
        lines.append("")

    if unconfirmed_output_count > 0:
        unconfirmed_lines.insert(0, f"### 🟡 PNU Scholar 미확인 연구자 후보 {unconfirmed_output_count}명")
        unconfirmed_lines.insert(1, "")
        unconfirmed_lines.insert(2, "- 아래 후보는 논문 저자 또는 특허 발명자 정보에서는 확인되었으나, PNU Scholar 검색창 방식으로 자동 확인되지 않은 인원입니다.")
        unconfirmed_lines.insert(3, "- 동명이인, 영문명 표기 차이, 한글명만 등록된 경우, Scholar 미등록 연구자일 가능성이 있습니다.")
        unconfirmed_lines.insert(4, "")

    return {
        "main_markdown": "\n".join(lines),
        "unconfirmed_markdown": "\n".join(unconfirmed_lines),
        "unconfirmed_count": unconfirmed_output_count,
    }


# =========================================================
# Streamlit UI
# =========================================================
st.title("🎓 PNU 수요기술-연구자 매칭 시스템")
st.caption(
    "수요기술에 맞는 부산대 논문 저자와 특허 발명자를 찾고, "
    "적합한 PNU 연구자와 수요기술을 매칭합니다."
)

debug_name = st.sidebar.text_input(
    "디버그: 이름 직접 검색",
    placeholder="예: Gil-Dong Hong 또는 홍길동",
)

if debug_name:
    st.sidebar.markdown("#### 검색 결과")
    debug_match = match_author_to_pnu_scholar(debug_name)
    if debug_match:
        st.sidebar.success("검색 확인됨")
        st.sidebar.json(
            {
                "입력이름": debug_name,
                "공식표시명": debug_match.get("display_name"),
                "한글이름": debug_match.get("korean_name"),
                "영문이름": debug_match.get("english_name"),
                "소속": debug_match.get("department"),
                "검색어": debug_match.get("search_keyword") or debug_match.get("used_query"),
                "매칭점수": debug_match.get("match_score"),
                "연구분야": debug_match.get("field"),
                "연구자ID": debug_match.get("researcher_id"),
                "링크": debug_match.get("link"),
            }
        )
    else:
        st.sidebar.warning("검색 결과 자동 확인 실패")

uploaded_file = st.file_uploader(
    "1. 수요기술조사서 업로드",
    type=["pdf", "docx", "txt", "md"],
)

manual_text = st.text_area(
    "2. 또는 기술 내용 직접 입력",
    placeholder="기술 설명, 적용 분야, 핵심 성능, 장치/공정/소재 정보를 넣으면 검색용 키워드로 확장·정리됩니다.",
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

        st.markdown(result.get("main_markdown", ""))

        unconfirmed_count = int(result.get("unconfirmed_count", 0) or 0)
        unconfirmed_markdown = result.get("unconfirmed_markdown", "")

        if unconfirmed_count > 0 and unconfirmed_markdown:
            with st.expander(f"연구자 미확인 건 보기 ({unconfirmed_count}명)", expanded=False):
                st.markdown(unconfirmed_markdown)

    except Exception as e:
        status_box.update(label="분석 중 오류 발생", state="error", expanded=True)
        progress_bar.progress(0)
        step_placeholder.error(f"오류가 발생했습니다: {e}")
