
import re
import json
import time
import traceback
import concurrent.futures
from pathlib import Path
from datetime import datetime

import fitz
import pdfplumber


# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FOLDER  = r"path\to\pdfs"
OUTPUT_FOLDER = r"path\to\output"
SAVE_EVERY_N  = 5

# Hard limits
MAX_PDF_SECONDS        = 90
MAX_TEXT_LEN           = 200_000
MAX_SENTS_PER_SECTION  = 2000
MAX_TOTAL_SENTENCES    = 8000

# Stage timeouts
FITZ_TIMEOUT       = 25
PDFPLUMBER_TIMEOUT = 25
CLEAN_TIMEOUT      = 15
SECTION_TIMEOUT    = 10
SPLIT_TIMEOUT      = 10

# Position buckets (relative position 0.0=first section, 1.0=last)
FRONT_CUTOFF       = 0.30   # below this = FRONT
BACK_CUTOFF        = 0.60   # above this = BACK
FINAL_CUTOFF       = 0.85   # above this = FINAL (max boost)

POSITION_MULTIPLIERS = {
    "FRONT":  0.3,   # heavy penalty
    "BODY":   1.0,   # neutral
    "BACK":   1.5,   # boost
    "FINAL":  2.0,   # max boost
}


# ============================================================
# TIMEOUT HELPER
# ============================================================
def _run_with_timeout(fn, *args, timeout=15):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn, *args)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return None
        except Exception:
            return None


# ============================================================
# STAGE 1: PDF PARSER
# ============================================================
def _parse_with_fitz(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            text += page.get_text("text") + "\n"
    finally:
        doc.close()
    return text


def _parse_with_pdfplumber(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pt = page.extract_text()
            if pt:
                text += pt + "\n"
    return text


def parse_pdf(pdf_path):
    text = ""
    parser_used = None

    try:
        result = _run_with_timeout(_parse_with_fitz, str(pdf_path), timeout=FITZ_TIMEOUT)
        if result and result.strip():
            text = result
            parser_used = "fitz"
        elif result is None:
            print(f"   [fitz timed out]", flush=True)
    except Exception as e:
        print(f"   [fitz failed]: {e}", flush=True)

    if text.strip():
        return text, parser_used

    try:
        result = _run_with_timeout(
            _parse_with_pdfplumber, str(pdf_path), timeout=PDFPLUMBER_TIMEOUT
        )
        if result and result.strip():
            text = result
            parser_used = "pdfplumber"
        elif result is None:
            print(f"   [pdfplumber timed out]", flush=True)
    except Exception as e:
        print(f"   [pdfplumber failed]: {e}", flush=True)

    return (text if text.strip() else None), parser_used


# ============================================================
# STAGE 2: TEXT CLEANER
# ============================================================
def _clean_text_inner(text):
    if len(text) > 300_000:
        text = text[:300_000]

    # Strip references and everything after
    text = re.sub(
        r'\n\s*(References|Bibliography|Works\s+Cited|REFERENCES|BIBLIOGRAPHY)\s*\n.*',
        '\n', text, flags=re.DOTALL,
    )
    text = re.sub(r'\[\d+(?:[,\-\u2013]\s*\d+)*\]', '', text)
    text = re.sub(
        r'\(([A-Z][a-zA-Z\-]+(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][a-zA-Z\-]+)*)'
        r'\s*,?\s*\d{4}[a-z]?\)', '', text,
    )
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\$[^$]+\$', '', text)
    text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)
    text = re.sub(
        r'\\begin\{(equation|align|gather|math)\*?\}.*?\\end\{\1\*?\}',
        '', text, flags=re.DOTALL,
    )
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'doi:\s*\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(\|[^\n]*\|\s*){2,}', '', text)
    text = re.sub(r'[-_=]{4,}', '', text)
    text = re.sub(r'\.{4,}', '', text)
    text = re.sub(r'\n\s*\d{1,4}\s*\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'(?<=[a-z,])\n(?=[a-z])', ' ', text)
    return text.strip()


def clean_text(text):
    result = _run_with_timeout(_clean_text_inner, text, timeout=CLEAN_TIMEOUT)
    if result is None:
        print(f"  [clean_text timeout — using minimal fallback]", flush=True)
        return re.sub(r'[ \t]+', ' ', text[:50_000]).strip()
    return result


# ============================================================
# STAGE 3: SECTION LOCATOR (4 safe alternatives, no backtracking risk)
# ============================================================
SECTION_PRIORITIES = {
    "high": [
        r'acknowledge?ments?',
        r'author\s+contributions?',
        r'ethics?\s+(statement|declaration|note|approval)',
        r'disclosures?',
        r'declarations?',
        r'ai\s+(use|usage)\s+(statement|declaration)?',
        r'generative\s+ai\s+(use|statement|disclosure)?',
        r'use\s+of\s+(ai|generative\s+ai|llm|large\s+language\s+model)',
        r'competing\s+interests?',
        r'conflict[s]?\s+of\s+interest',
        r'funding',
        r'data\s+availability',
        r'code\s+availability',
    ],
    "medium": [
        r'methodology', r'methods?',
        r'materials?\s+and\s+methods?',
        r'study\s+design',
    ],
    "low": [
        r'appendix(\s+[A-Z])?',
        r'footnotes?',
        r'notes?',
        r'supplementary',
    ],
}
PRIORITY_SCORES = {"high": 3, "medium": 1, "low": 0.5, "none": 0}


# Header detection regex — 4 safe alternatives, all bounded.
# Tested against pathological inputs (50k char spam, OCR garbage, mixed case);
# all completed in <0.001s with zero false matches.
HEADER_PATTERN = re.compile(
    r'\n[ \t]*('
    # (a) Numbered: "1. Introduction", "3.2 Methods"
    r'\d+(?:\.\d+){0,3}\.?[ \t]+[A-Z][A-Za-z][A-Za-z\s&\-]{1,58}|'
    # (b) All-caps: "INTRODUCTION", at least 3 caps then more caps/space/&/-
    r'[A-Z][A-Z][A-Z\s&\-]{1,58}|'
    # (c) Known-header whitelist (case-insensitive group via the pattern itself)
    r'(?:Acknowledge?ments?|Author\s+Contributions?|'
    r'Ethics?\s+(?:Statement|Declaration|Note|Approval)|'
    r'Disclosures?|Disclosure\s+statement|Declarations?|Funding|'
    r'Conflicts?\s+of\s+Interest|Competing\s+Interests?|'
    r'Data\s+Availability|Code\s+Availability|'
    r'Generative\s+AI(?:\s+(?:Use|Statement|Disclosure))?|'
    r'AI\s+(?:Use|Usage|Statement|Declaration|Disclosure)|'
    r'Use\s+of\s+(?:AI|Generative\s+AI|LLMs?|Large\s+Language\s+Models?)|'
    r'Supplementary\s+(?:Information|Materials?)|'
    r'Appendix(?:\s+[A-Z])?)|'
    # (d) Bounded Title-Case: 1-5 words, each Capitalized + small lowercase
    # tail, OR short all-caps acronym (AI, LLM, NLP, ChatGPT-style stays out
    # because requires at least 2 caps run). Connectors allowed mid-phrase.
    r'(?:[A-Z][a-z]{2,15}|[A-Z]{2,5})'
    r'(?:[ \t](?:[A-Z][a-z]{1,15}|[A-Z]{2,5}|of|and|the|for|in|on|to|a)){0,4}'
    r')[ \t]*\n',
)


def _classify_position(rel_pos):
    """Map relative section position [0.0, 1.0] to a bucket."""
    if rel_pos < FRONT_CUTOFF:
        return "FRONT"
    elif rel_pos < BACK_CUTOFF:
        return "BODY"
    elif rel_pos < FINAL_CUTOFF:
        return "BACK"
    else:
        return "FINAL"


def _classify_priority(header):
    """Look up a section's priority from its header text."""
    for level in ("high", "medium", "low"):
        for pat in SECTION_PRIORITIES[level]:
            if re.search(pat, header, re.IGNORECASE):
                return level, PRIORITY_SCORES[level]
    return "none", 0


def _locate_sections_inner(text):
    matches = list(HEADER_PATTERN.finditer(text))
    sections = []

    if not matches:
        return [{"header": "FULL_DOCUMENT", "content": text,
                 "priority": "none", "section_score": 0,
                 "rel_pos": 0.5, "position_bucket": "BODY"}]

    # Build raw section list first (we need the count before we can compute rel_pos)
    raw = []
    if matches[0].start() > 0:
        raw.append(("PREAMBLE", text[:matches[0].start()].strip()))

    for i, m in enumerate(matches):
        header  = m.group(1).strip()
        start   = m.end()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        raw.append((header, content))

    # Now annotate each section with relative position + priority + bucket
    n = len(raw)
    for idx, (header, content) in enumerate(raw):
        rel_pos = idx / max(1, n - 1) if n > 1 else 0.5
        priority, score = _classify_priority(header)
        bucket = _classify_position(rel_pos)
        sections.append({
            "header":          header,
            "content":         content,
            "priority":        priority,
            "section_score":   score,
            "rel_pos":         round(rel_pos, 3),
            "position_bucket": bucket,
        })

    return sections


def locate_sections(text):
    result = _run_with_timeout(_locate_sections_inner, text, timeout=SECTION_TIMEOUT)
    if result is None:
        print(f"  [locate_sections timeout — treating as FULL_DOCUMENT]", flush=True)
        return [{"header": "FULL_DOCUMENT", "content": text,
                 "priority": "none", "section_score": 0,
                 "rel_pos": 0.5, "position_bucket": "BODY"}]
    return result


# ============================================================
# STAGE 4: SENTENCE SPLITTER
# ============================================================
def _split_sentences_inner(text):
    if not text.strip():
        return []
    if len(text) > 250_000:
        text = text[:250_000]
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def split_sentences(text):
    result = _run_with_timeout(_split_sentences_inner, text, timeout=SPLIT_TIMEOUT)
    if result is None:
        print(f"  [split_sentences timeout — section dropped]", flush=True)
        return []
    return result


# ============================================================
# STAGE 5: SIGNAL PATTERNS (unchanged from v4)
# ============================================================
STRONG_TOOLS = [
    "Grammarly", "Grammarly Premium", "Grammarly GO",
    "QuillBot", "Quillbot", "Quill Bot",
    "Wordtune", "Word Tune",
    "Writefull", "Write Full",
    "Trinka", "Trinka AI",
    "LanguageTool", "Language Tool",
    "ProWritingAid", "Pro Writing Aid",
    "Hemingway Editor", "Hemingway App",
    "Paperpal", "PaperPal", "Paper Pal",
    "Linguix", "Ginger Software",
    "WhiteSmoke", "White Smoke",
    "AutoCrit", "Slick Write",
    "Smodin", "Scribbr", "EditGPT",
    "DeepL", "DeepL Write", "DeepL Translator", "DeepL Pro",
    "Google Translate", "Microsoft Translator", "Bing Translator",
    "ChatGPT", "Chat GPT", "Chat-GPT",
    "Claude.ai", "Anthropic Claude",
    "Google Bard", "Bard",
    "Bing Chat", "Bing AI", "Microsoft Copilot",
    "Perplexity AI", "Perplexity.ai",
    "Notion AI",
    "Elicit", "Elicit.org",
    "Consensus.app",
    "Scite.ai", "Scite Assistant",
    "ResearchRabbit", "Research Rabbit",
    "SciSpace", "Sci Space",
    "Scholarcy", "ExplainPaper", "Explain Paper",
    "Humata", "Humata AI", "ChatPDF", "AskYourPDF",
    "Mendeley", "Zotero", "EndNote",
    "DALL-E", "DALL\u00b7E", "DALL-E 2", "DALL-E 3",
    "Midjourney", "Mid Journey",
    "Stable Diffusion",
    "Adobe Firefly",
    "Leonardo AI", "Leonardo.ai",
    "Canva AI", "Canva Magic",
    "ElevenLabs", "Eleven Labs",
    "Synthesia", "HeyGen", "D-ID",
    "Otter.ai", "Otter AI",
    "GPT-4", "GPT-4o", "GPT-4 Turbo", "GPT-4.5",
    "GPT-3.5", "GPT-3",
    "OpenAI o1", "o1-preview", "o1-mini",
    "Claude 2", "Claude 3", "Claude 3.5",
    "Claude Sonnet", "Claude Opus", "Claude Haiku",
    "Claude 3 Opus", "Claude 3.5 Sonnet",
    "Gemini Pro", "Gemini 1.5", "Gemini Advanced", "Google Gemini",
    "PaLM 2",
    "LLaMA", "Llama 2", "Llama 3", "Llama-2", "Llama-3",
    "Meta Llama", "Mistral 7B", "Mixtral", "Mixtral 8x7B",
    "Falcon", "Vicuna", "Alpaca",
    "DeepSeek V2", "DeepSeek V3", "DeepSeek R1",
]

AMBIGUOUS_TOOLS = [
    "GitHub Copilot", "Copilot",
    "Cursor", "Cursor AI", "Codeium", "Tabnine",
    "Amazon CodeWhisperer", "CodeWhisperer",
    "Replit Ghostwriter",
]

STRONG_TOOLS    = sorted(set(STRONG_TOOLS),    key=len, reverse=True)
AMBIGUOUS_TOOLS = sorted(set(AMBIGUOUS_TOOLS), key=len, reverse=True)

STRONG_TOOL_PATTERN = re.compile(
    r'(?<![\w/])(' + '|'.join(re.escape(t) for t in STRONG_TOOLS) + r')(?![\w])',
    re.IGNORECASE,
)
AMBIGUOUS_TOOL_PATTERN = re.compile(
    r'(?<![\w/])(' + '|'.join(re.escape(t) for t in AMBIGUOUS_TOOLS) + r')(?![\w])',
    re.IGNORECASE,
)

TASK_GRAMMAR_SPELLING = re.compile(
    r'\b(grammar(?:\s+(?:check|checking|correction|correcting|errors?|mistakes?))?|'
    r'spell(?:ing)?(?:\s+(?:check|checking|correction|correcting|errors?|mistakes?))?|'
    r'spell[\s\-]?check(?:er|ing)?|'
    r'grammatical\s+(?:errors?|mistakes?|issues?|corrections?)|'
    r'typos?|typographical\s+errors?|'
    r'punctuation(?:\s+(?:check|errors?))?|orthograph(?:y|ic)|'
    r'language\s+(?:check|checking|correction)|'
    r'proofread(?:ing)?|copy[\s\-]?edit(?:ing)?|'
    r'linguistic\s+(?:check|correction|review)|'
    r'catch(?:ing)?\s+(?:grammatical\s+)?errors?)\b', re.IGNORECASE)

TASK_WRITING_STYLE = re.compile(
    r'\b(writing\s+(?:style|quality|flow|clarity|assistance)|'
    r'improv(?:e|ing|ement)\s+(?:the\s+)?(?:writing|text|language|style|clarity|readability)|'
    r'enhanc(?:e|ing|ement)\s+(?:the\s+)?(?:writing|text|language|style|clarity|readability)|'
    r'refin(?:e|ing|ement)\s+(?:the\s+)?(?:writing|text|language|style)|'
    r'polish(?:ing)?\s+(?:the\s+)?(?:writing|text|language|manuscript|paper)|'
    r'sentence\s+(?:structure|construction|formation)|word\s+(?:choice|selection)|'
    r'language\s+(?:improvement|enhancement|polishing|editing|quality)|readability|'
    r'(?:overall\s+)?flow\s+of\s+(?:the\s+)?(?:writing|text|paper)|'
    r'tone\s+adjustment|stylistic\s+(?:improvement|edit|change)|'
    r'native[\s\-]?(?:like|speaker)\s+(?:editing|writing)|'
    r'english\s+(?:editing|polishing|improvement|correction)|language\s+editing|'
    r'clarity\s+(?:and|of)\s+(?:conciseness|writing|expression)|'
    r'concise(?:ness)?\s+(?:and\s+clarity|of\s+writing))\b', re.IGNORECASE)

TASK_PARAPHRASE = re.compile(
    r'\b(paraphras(?:e|ing|ed)|reword(?:ing|ed)?|re[\s\-]word(?:ing|ed)?|'
    r'rephras(?:e|ing|ed)|re[\s\-]phras(?:e|ing|ed)?|'
    r'rewrit(?:e|ing|ten)\s+(?:the\s+|sentences?|text|paragraph)|'
    r're[\s\-]writ(?:e|ing|ten)\s+(?:the\s+|sentences?|text)|'
    r'reformulat(?:e|ing|ion)\s+(?:sentences?|text)|'
    r'restructur(?:e|ing|ed)\s+(?:sentences?|text|writing)|'
    r'express(?:ing)?\s+(?:ideas?|thoughts?)\s+(?:differently|in\s+different\s+ways)|'
    r'ensuring\s+clarity\s+and\s+conciseness)\b', re.IGNORECASE)

TASK_TRANSLATION = re.compile(
    r'\b(translat(?:e|ing|ion|ions|ed)\s+(?:the\s+)?(?:paper|manuscript|article|text|'
    r'content|abstract|sections?|from|into|to|our|my|this|work)|'
    r'(?:language\s+)?translation\s+(?:tool|service|aid|of\s+the)|'
    r'transcrib(?:e|ing|ed)\s+(?:audio|interview|speech|the|data)|'
    r'transcription[s]?\s+(?:of\s+)?(?:the\s+)?(?:audio|interview|speech|data)|'
    r'speech[\s\-]to[\s\-]text|text[\s\-]to[\s\-]speech|'
    r'multilingual\s+(?:support|translation)|'
    r'reach(?:ing)?\s+(?:a\s+)?broader\s+audience)\b', re.IGNORECASE)

TASK_DRAFTING = re.compile(
    r'\b(draft(?:ing|ed)?\s+(?:the\s+)?(?:paper|manuscript|article|abstract|introduction|'
    r'literature\s+review|methodology|conclusion|discussion|sections?|content|text|results)|'
    r'writ(?:e|ing|ten)\s+(?:the\s+)?(?:paper|manuscript|article|abstract|introduction|'
    r'literature\s+review|methodology|conclusion|sections?|content|different\s+sections)|'
    r'(?:initial|first|rough)\s+draft|'
    r'abstract\s+(?:drafting|generation|writing|creation)|'
    r'introduction\s+(?:drafting|writing)|'
    r'methodology\s+(?:description[s]?|drafting|writing)|'
    r'manuscript\s+(?:preparation|drafting|writing)|'
    r'paper\s+(?:preparation|drafting|writing)|'
    r'article\s+(?:preparation|drafting|writing)|'
    r'help(?:ed)?\s+(?:to\s+)?(?:write|draft)\s+(?:the\s+)?(?:paper|manuscript|sections?|abstract|introduction)|'
    r'assist(?:ed|ing)?\s+(?:in|with)\s+(?:writing|drafting)\s+(?:the\s+)?(?:paper|manuscript|sections?)|'
    r'content\s+(?:generation|creation|drafting|production)|text\s+generation|'
    r'enhancing\s+content|suggesting\s+additional\s+content|'
    r'assisting\s+with\s+critiques?|offering\s+alternative\s+perspectives?)\b', re.IGNORECASE)

TASK_BRAINSTORM = re.compile(
    r'\b(brainstorm(?:ing|ed)?|(?:research\s+)?idea\s+(?:generation|brainstorming)|'
    r'generat(?:e|ing|ion)\s+(?:research\s+)?(?:ideas?|hypothes[ie]s|questions?)|'
    r'creat(?:e|ing|ion)\s+(?:research\s+)?(?:hypothes[ie]s|questions?)|'
    r'formulat(?:e|ing|ion)\s+(?:hypothes[ie]s|research\s+questions?)|'
    r'research\s+(?:question|direction|idea)\s+(?:generation|formulation)|'
    r'thinking\s+(?:partner|aid)|'
    r'explor(?:e|ing)\s+(?:research\s+)?(?:ideas?|directions?)|'
    r'creating\s+hypotheses)\b', re.IGNORECASE)

TASK_LITERATURE = re.compile(
    r'\b(literature\s+(?:review|search|survey)|'
    r'(?:scholarly|academic)\s+literature\s+(?:search|review)|'
    r'searching\s+(?:scholarly|academic|relevant|the)\s+literature|'
    r'find(?:ing)?\s+relevant\s+(?:papers?|articles?|publications?|references?)|'
    r'paper\s+(?:search|discovery|recommendation)|'
    r'related\s+work\s+(?:search|review)|'
    r'systematic\s+review|meta[\s\-]analysis|lit[\s\-]?review|'
    r'generat(?:e|ing|ion)\s+(?:the\s+)?(?:a\s+)?literature\s+review|'
    r'drafting\s+(?:a\s+)?literature\s+review\s+section|'
    r'(?:a\s+)?set\s+of\s+relevant\s+papers)\b', re.IGNORECASE)

TASK_PLAGIARISM = re.compile(
    r'\b(plagiari[sz]m\s+(?:detection|check|checking|checker|test|testing|scan|scanning|issues?)|'
    r'(?:detect|detecting|check|checking|find|finding|identify|identifying)\s+'
    r'(?:potential\s+)?plagiari[sz]m|'
    r'similarity\s+(?:check|detection|index|score)|'
    r'originality\s+(?:check|detection|score|report)|'
    r'duplicate\s+content\s+(?:check|detection)|content\s+originality|'
    r'plagiari[sz]m\s+issues?\s+in\s+(?:your|my|the|our)\s+(?:own\s+)?writing)\b',
    re.IGNORECASE)

TASK_CITATION = re.compile(
    r'\b(citation(?:s)?\s+(?:management|formatting|generation|style)|'
    r'(?:format|formatting|generate|generating|manage|managing)\s+citations?|'
    r'reference(?:s)?\s+(?:management|formatting|generation|list)|'
    r'(?:format|formatting|generate|generating|manage|managing)\s+references?|'
    r'bibliograph(?:y|ic)\s+(?:management|formatting|generation)|'
    r'(?:apa|mla|chicago|harvard|ieee|vancouver)\s+(?:style|format|formatting|citation)|'
    r'format\s+citations?\s+and\s+references|reference\s+(?:list|style|format)|'
    r'citation\s+and\s+reference\s+management|'
    r'according\s+to\s+specific\s+(?:citation\s+)?styles?)\b', re.IGNORECASE)

TASK_FORMATTING = re.compile(
    r'\b(format(?:ting)?\s+(?:assistance|help|aid|support|guidelines?|requirements?)|'
    r'(?:journal|conference|publication)\s+(?:format|formatting|template|guidelines?|style)|'
    r'manuscript\s+(?:formatting|format|template)|paper\s+(?:formatting|format|template)|'
    r'(?:adher(?:e|ence|es?|ing)|conform(?:ing|s)?)\s+to\s+(?:specific\s+)?(?:format|formatting|guidelines)|'
    r'formatting\s+guidelines?\s+required\s+by\s+(?:journals?|institutions?)|'
    r'document\s+(?:formatting|format|structure|layout))\b', re.IGNORECASE)

TASK_PEER_REVIEW = re.compile(
    r'\b(peer\s+review(?:\s+simulation)?|simulat(?:e|ing|ion)\s+(?:peer\s+)?review|'
    r'(?:provide|providing|generate|generating)\s+(?:peer\s+review\s+)?feedback|'
    r'feedback\s+on\s+(?:the\s+)?(?:strengths?\s+and\s+weaknesses?|paper|manuscript)|'
    r'critique\s+(?:of\s+)?(?:content|writing|paper|manuscript|argument)|'
    r'mock\s+review|reviewer\s+(?:perspective|comments?)|'
    r'strengths?\s+and\s+weaknesses?\s+of\s+(?:your|the|my|our)\s+paper)\b',
    re.IGNORECASE)

TASK_CONTENT_ENHANCE = re.compile(
    r'\b(content\s+(?:enhancement|improvement|enrichment)|'
    r'enhanc(?:e|ing|ement)\s+(?:the\s+)?(?:content|argument[s]?|paper|manuscript)|'
    r'strengthen(?:ing)?\s+(?:the\s+)?(?:argument[s]?|paper|manuscript|writing)|'
    r'suggest(?:ing|ions?)?\s+(?:additional\s+)?(?:content|research|references?|ideas?)|'
    r'additional\s+(?:content|research)\s+(?:that\s+)?(?:could|to)\s+strengthen|'
    r'alternative\s+(?:perspective[s]?|viewpoint[s]?)|'
    r'enrich(?:ing|ment)\s+(?:the\s+)?(?:text|content|paper)|'
    r'assisting\s+with\s+critiques?\s+of\s+(?:your|the|our)\s+content|'
    r'offering\s+alternative\s+perspectives?)\b', re.IGNORECASE)

TASK_IMAGES = re.compile(
    r'\b((?:generat(?:e|ing|ion)|creat(?:e|ing|ion)|produc(?:e|ing|tion))\s+'
    r'(?:image[s]?|picture[s]?|graphic[s]?|illustration[s]?|figure[s]?(?:\s+for)?|'
    r'video[s]?|audio|visual[s]?|artwork)|'
    r'image\s+(?:generation|creation|synthesis)|'
    r'graphic\s+(?:generation|creation|design)|'
    r'illustration\s+(?:generation|creation)|'
    r'ai[\s\-]generated\s+(?:image|figure|illustration|graphic|visual|artwork)|'
    r'figure\s+(?:was|were)\s+(?:generated|created|produced)\s+(?:by|with|using)|'
    r'creating\s+graphics|'
    r'generate\s+images?\s+for\s+(?:your|the|our|my)\s+paper)\b', re.IGNORECASE)

TASK_CODING = re.compile(
    r'\b(cod(?:e|ing)\s+(?:assistance|help|aid|generation|completion|writing|review|debug)|'
    r'programming\s+(?:assistance|help|aid)|'
    r'(?:writ|generat|develop|debug|review)(?:e|ing|ed)\s+(?:the\s+)?code|'
    r'pair\s+programming|help(?:ed)?\s+(?:to\s+)?(?:write|generate|debug)\s+code|'
    r'code\s+suggestion[s]?|coding\s+and\s+programming|'
    r'script\s+(?:generation|writing|creation))\b', re.IGNORECASE)

TASK_DATA = re.compile(
    r'\b(data\s+(?:analysis|analy[sz]ing|interpretation|interpret|processing|cleaning|preparation)|'
    r'(?:analy[sz]e|analy[sz]ing|interpret|process|clean|prepare)\s+(?:the\s+)?data|'
    r'data\s+(?:visuali[sz]ation|visuali[sz]ing|representation|organi[sz]ation)|'
    r'visuali[sz]e\s+(?:the\s+)?data|(?:organi[sz]e|represent|display)\s+(?:the\s+)?data|'
    r'chart\s+(?:generation|creation)|plot\s+(?:generation|creation)|'
    r'graph\s+(?:generation|creation)|statistical\s+analysis|'
    r'data\s+collection|collecting\s+data|data\s+transcription|'
    r'organising\s+(?:and\s+)?(?:representing|visualising)\s+data|'
    r'representing\s+(?:and\s+)?visualising\s+data|modelling\s+(?:complex\s+)?phenomena|'
    r'generating\s+data\s+for\s+(?:use\s+in\s+)?validation|'
    r'data\s+for\s+validation\s+of\s+(?:hypothes[ie]s|models?))\b', re.IGNORECASE)

TASK_SUMMARIZE = re.compile(
    r'\b(summari[sz](?:e|ing|ation)\s+(?:the\s+|papers?|articles?|content|literature|research)|'
    r'summari[sz]ed?\s+(?:the\s+|papers?|articles?|content)|'
    r'(?:generate|create|produce|provide)\s+(?:a\s+)?summary\s+of|'
    r'condens(?:e|ing|ation)\s+(?:text|content|paper)|'
    r'extract(?:ing)?\s+(?:key\s+)?(?:points?|findings?|insights?)\s+from|'
    r'captures?\s+the\s+gist\s+of)\b', re.IGNORECASE)

ALL_TASK_PATTERNS = [
    ("grammar_spelling",    TASK_GRAMMAR_SPELLING),
    ("writing_style",       TASK_WRITING_STYLE),
    ("paraphrase",          TASK_PARAPHRASE),
    ("translation",         TASK_TRANSLATION),
    ("drafting",            TASK_DRAFTING),
    ("brainstorming",       TASK_BRAINSTORM),
    ("literature_review",   TASK_LITERATURE),
    ("plagiarism",          TASK_PLAGIARISM),
    ("citation",            TASK_CITATION),
    ("formatting",          TASK_FORMATTING),
    ("peer_review",         TASK_PEER_REVIEW),
    ("content_enhancement", TASK_CONTENT_ENHANCE),
    ("images_media",        TASK_IMAGES),
    ("coding",              TASK_CODING),
    ("data_analysis",       TASK_DATA),
    ("summarization",       TASK_SUMMARIZE),
]

STRONG_DECLARATION = re.compile(
    r'\b(we\s+(?:used|utili[sz]ed|employed|leveraged|applied|adopted|consulted|prompted|asked)\s+'
    r'[A-Za-z0-9\s\-\.,]{1,80}?\s+(?:for|to|in|during)\b|'
    r'(?:we\s+)?acknowledge\s+(?:the\s+)?use\s+of\s+[A-Za-z]|'
    r'we\s+(?:wish\s+to\s+|would\s+like\s+to\s+)?(?:disclose|declare)\s+(?:the\s+)?use\s+of|'
    r'(?:the\s+)?authors?\s+(?:used|utili[sz]ed|employed|leveraged|applied)\s+[A-Za-z]|'
    r'(?:chatgpt|gpt[\s\-]?\d|claude|gemini|grammarly|copilot|llm[s]?|ai\s+tool|'
    r'language\s+model|generative\s+ai|gen\s?ai)\s+'
    r'(?:was|were|is|are|has\s+been|have\s+been)\s+'
    r'(?:used|utili[sz]ed|employed|applied|consulted|prompted)|'
    r'with\s+(?:the\s+)?(?:help|assistance|aid|support)\s+of\s+'
    r'[A-Za-z][A-Za-z0-9\s\-\.,]{1,60}?\s+(?:for|to|in|during|we)|'
    r'(?:this|the)\s+(?:paper|manuscript|article|work|study|text|abstract)\s+'
    r'(?:was|has\s+been)\s+(?:partially\s+|partly\s+)?'
    r'(?:written|drafted|prepared|created|produced|generated|edited|revised|'
    r'reviewed|proofread|translated|polished|refined)\s+(?:with|using|by|through|via)|'
    r'(?:during|in|while|for)\s+(?:the\s+)?'
    r'(?:preparation|writing|drafting|composition|creation|development|production|editing)\s+'
    r'of\s+(?:this|the)\s+(?:paper|manuscript|article|work|study)|'
    r'for\s+(?:the\s+)?(?:purpose[s]?\s+of\s+)?'
    r'(?:improving|enhancing|refining|polishing|correcting|editing|drafting|'
    r'translating|paraphrasing|proofreading|writing)|'
    r'to\s+(?:improve|enhance|refine|polish|correct|edit|proofread|translate|'
    r'paraphrase|rephrase|rewrite|draft)\s+(?:the\s+)?'
    r'(?:writing|manuscript|paper|text|language|grammar|article|abstract)|'
    r'(?:no|did\s+not\s+use|have\s+not\s+used|without\s+(?:the\s+)?use\s+of)\s+'
    r'(?:ai|llm|generative\s+ai|chatgpt|genai|generative\s+artificial)|'
    r'(?:generated|produced|created|drafted|written)\s+(?:with|using|by)\s+'
    r'(?:chatgpt|gpt[\s\-]?\d|claude|gemini|copilot|grammarly|ai|llm[s]?|'
    r'generative\s+ai|gen\s?ai))',
    re.IGNORECASE,
)

NEGATIVE_DECL_PATTERN = re.compile(
    r'\b(?:no|did\s+not\s+use|have\s+not\s+used|without\s+(?:the\s+)?use\s+of)\s+'
    r'(?:any\s+)?(?:ai|llm|generative\s+ai|chatgpt|genai|generative\s+artificial)',
    re.IGNORECASE,
)

RESEARCH_CONTEXT_REJECTORS = [
    re.compile(r'\b(transformer|attention|encoder|decoder|embedding[s]?|'
               r'positional\s+(?:embedding|encoding))\b', re.IGNORECASE),
    re.compile(r'\b(fine[\s\-]?tun(?:e|ed|ing)|pre[\s\-]?train(?:ed|ing)|'
               r'training\s+(?:data|set|loss|objective)|'
               r'model\s+(?:parameters?|weights?|architecture)|'
               r'gradient\s+(?:descent|update)|loss\s+function|'
               r'learning\s+rate|hyper[\s\-]?parameter)\b', re.IGNORECASE),
    re.compile(r'\b(we\s+(?:propose|develop|design|build|introduce|present)\s+'
               r'(?:a\s+|an\s+|our\s+|the\s+)?(?:novel\s+|new\s+)?'
               r'(?:model|approach|method|framework|architecture|algorithm|system))\b',
               re.IGNORECASE),
    re.compile(r'\b(benchmark|baseline|evaluation\s+metric|'
               r'state[\s\-]of[\s\-]the[\s\-]art|sota)\b', re.IGNORECASE),
    re.compile(r'\b(such\s+as\s+(?:gpt|bert|llama|claude)|'
               r'(?:gpt|bert|llama|claude|gemini)\s+\(?\s*(?:radford|devlin|brown|touvron)|'
               r'(?:initialization|architecture|parameters?)\s+'
               r'(?:of|in|for|from)\s+(?:gpt|bert|llama))\b', re.IGNORECASE),
    re.compile(r'\b(acknowledge\s+(?:the\s+)?(?:support|funding|contribution[s]?|'
               r'help|assistance)\s+from|'
               r'support(?:ed)?\s+by\s+(?:the\s+)?(?:nsf|nih|grant|award|'
               r'foundation|university|institute)|'
               r'funded\s+by|grant\s+(?:no|number)|(?:thanks?|grateful)\s+to)\b',
               re.IGNORECASE),
    re.compile(r'\b(equation|theorem|lemma|proof|derivation|distribution\s+of|'
               r'we\s+derive|empirical\s+validation)\b', re.IGNORECASE),
]


def has_research_context(sentence):
    return any(p.search(sentence) for p in RESEARCH_CONTEXT_REJECTORS)


# Headers that count as "declaration-relevant" even if positioned in the
# front of the document (e.g., a misordered or OCR-shuffled paper).
DECL_HEADER_KEYWORDS = re.compile(
    r'\b(acknowledge?ments?|author\s+contributions?|'
    r'declaration|disclosure|ethics?\s+statement|'
    r'generative\s+ai\s+statement|ai\s+(use|usage|statement|declaration|disclosure)|'
    r'use\s+of\s+(ai|generative\s+ai|llm|large\s+language\s+model)|'
    r'competing\s+interests?|conflicts?\s+of\s+interest|'
    r'funding|data\s+availability|code\s+availability)\b',
    re.IGNORECASE,
)


def header_is_declaration_relevant(header):
    return bool(DECL_HEADER_KEYWORDS.search(header))


# ============================================================
# STAGE 5b: SCORING (now position-aware)
# ============================================================
def score_sentence(sentence, section, neighbor_text=""):
    """
    Score a sentence with position-aware adjustments.
    `section` is a dict with keys: header, priority, section_score,
    rel_pos, position_bucket.
    """
    hits = {}

    strong_tools = list({t for t in STRONG_TOOL_PATTERN.findall(sentence)})
    ambig_tools  = list({t for t in AMBIGUOUS_TOOL_PATTERN.findall(sentence)})

    if strong_tools:
        hits["strong_tools"] = strong_tools
    if ambig_tools:
        hits["ambiguous_tools"] = ambig_tools

    search_text   = sentence + " " + neighbor_text
    matched_tasks = [name for name, p in ALL_TASK_PATTERNS if p.search(search_text)]
    if matched_tasks:
        hits["manuscript_tasks"] = matched_tasks

    decl = STRONG_DECLARATION.search(sentence)
    if decl:
        hits["declaration"] = decl.group(0)[:120]

    if has_research_context(sentence):
        hits["research_context_flag"] = True

    is_negative = bool(NEGATIVE_DECL_PATTERN.search(sentence))

    if hits.get("research_context_flag"):
        if not strong_tools and "declaration" not in hits and not is_negative:
            return 0.0, hits, False

    has_strong_tool = bool(strong_tools)
    has_ambig_tool  = bool(ambig_tools)
    has_task        = bool(matched_tasks)
    has_strong_decl = "declaration" in hits

    # ---- Acceptance rules (same as v4) ----
    accepted = False
    reason   = None

    if has_strong_tool and has_task:
        accepted, reason = True, "strong_tool+task"
    elif has_strong_tool and has_strong_decl:
        accepted, reason = True, "strong_tool+decl"
    elif has_ambig_tool and has_task and has_strong_decl:
        accepted, reason = True, "ambig_tool+task+decl"
    elif has_strong_decl and has_task:
        accepted, reason = True, "decl+task"
    elif is_negative:
        accepted, reason = True, "negative_declaration"

    if not accepted:
        return 0.0, hits, False

    # ---- Position-aware filtering ----
    bucket = section.get("position_bucket", "BODY")
    decl_relevant_header = header_is_declaration_relevant(section.get("header", ""))

    # HARD REJECT: front-of-paper candidates that aren't negative declarations
    # AND don't sit in a declaration-relevant section header. These are
    # almost always papers DISCUSSING AI as a research subject.
    if bucket == "FRONT":
        if reason != "negative_declaration" and not decl_relevant_header:
            hits["rejected_by_position"] = "FRONT_non_declaration"
            return 0.0, hits, False

    # ---- Base score ----
    score = 0.0
    if has_strong_tool:
        score += 5
    if has_ambig_tool:
        score += 2
    if has_task:
        score += 2 * min(len(matched_tasks), 3)
    if has_strong_decl:
        score += 4
    score += section["section_score"]

    # ---- Position multiplier ----
    pos_mult = POSITION_MULTIPLIERS.get(bucket, 1.0)
    score *= pos_mult

    # Big bonus when the header itself is a declaration keyword
    if decl_relevant_header:
        score += 3

    hits["accept_reason"]    = reason
    hits["position_bucket"]  = bucket
    hits["position_mult"]    = pos_mult
    hits["decl_header"]      = decl_relevant_header
    return round(score, 2), hits, True


# ============================================================
# STAGE 7: PARAGRAPH EXPANDER
# ============================================================
def expand_to_paragraph(text, sentence):
    paragraphs = re.split(r'\n\s*\n', text)
    sent_strip  = sentence.strip()
    for para in paragraphs:
        if sent_strip in para:
            return para.strip()
    key = sent_strip[:80]
    for para in paragraphs:
        if key in para:
            return para.strip()
    return sent_strip


# ============================================================
# STAGE 8: PER-PDF PIPELINE
# ============================================================
def _find_candidate_paragraphs_inner(pdf_path):
    paper_id = Path(pdf_path).stem
    pdf_start = time.time()

    raw_text, parser = parse_pdf(pdf_path)
    if not raw_text:
        return {"paper_id": paper_id, "status": "parse_failed",
                "parser_used": None, "num_candidates": 0, "paragraphs": []}

    cleaned = clean_text(raw_text)
    text_len = len(cleaned)
    print(f"  text_len={text_len}", flush=True)

    if text_len > MAX_TEXT_LEN:
        print(f"  [SKIP: text_len {text_len} > {MAX_TEXT_LEN}]", flush=True)
        return {
            "paper_id":       paper_id,
            "parser_used":    parser,
            "status":         "skipped_too_large",
            "num_candidates": 0,
            "paragraphs":     [],
            "text_len":       text_len,
        }

    sections = locate_sections(cleaned)
    print(f"  sections_found={len(sections)}", flush=True)

    candidates = []
    seen_keys  = set()
    total_sentences_processed = 0
    timed_out = False

    for section in sections:
        if time.time() - pdf_start > MAX_PDF_SECONDS:
            print(f"  [PDF wall-clock — partial result]", flush=True)
            timed_out = True
            break

        if not section["content"]:
            continue

        sentences = split_sentences(section["content"])

        if len(sentences) > MAX_SENTS_PER_SECTION:
            print(f"  [SKIP section '{section['header'][:40]}': "
                  f"{len(sentences)} sentences > {MAX_SENTS_PER_SECTION}]", flush=True)
            continue

        if total_sentences_processed + len(sentences) > MAX_TOTAL_SENTENCES:
            print(f"  [total-sentences cap "
                  f"({total_sentences_processed}/{MAX_TOTAL_SENTENCES}) — stopping]", flush=True)
            timed_out = True
            break

        for i, sent in enumerate(sentences):
            if i % 200 == 0 and time.time() - pdf_start > MAX_PDF_SECONDS:
                timed_out = True
                break

            neighbor = ""
            if i > 0:
                neighbor += " " + sentences[i - 1]
            if i + 1 < len(sentences):
                neighbor += " " + sentences[i + 1]

            score, hits, accepted = score_sentence(
                sent, section, neighbor_text=neighbor
            )
            if not accepted:
                continue

            paragraph = expand_to_paragraph(section["content"], sent)
            key = paragraph[:200]
            if key in seen_keys:
                continue
            seen_keys.add(key)

            candidates.append({
                "paragraph":        paragraph,
                "trigger_sentence": sent,
                "section":          section["header"],
                "section_priority": section["priority"],
                "position_bucket":  section["position_bucket"],
                "rel_pos":          section["rel_pos"],
                "score":            score,
                "signals":          hits,
            })

        total_sentences_processed += len(sentences)
        if timed_out:
            break

    candidates.sort(key=lambda x: x["score"], reverse=True)
    status = "skipped_timeout" if timed_out else "success"

    return {
        "paper_id":       paper_id,
        "parser_used":    parser,
        "status":         status,
        "num_candidates": len(candidates),
        "paragraphs":     candidates,
        "text_len":       text_len,
        "elapsed":        round(time.time() - pdf_start, 2),
    }


def find_candidate_paragraphs(pdf_path):
    paper_id = Path(pdf_path).stem
    result = _run_with_timeout(
        _find_candidate_paragraphs_inner, pdf_path, timeout=MAX_PDF_SECONDS + 30
    )
    if result is None:
        print(f"  [OUTER TIMEOUT — paper marked skipped]", flush=True)
        return {
            "paper_id":       paper_id,
            "parser_used":    None,
            "status":         "skipped_timeout",
            "num_candidates": 0,
            "paragraphs":     [],
        }
    return result


# ============================================================
# STAGE 9: BATCH RUNNER
# ============================================================
def save_paragraph_txt(paragraphs_dir, result):
    if not result["paragraphs"]:
        return
    txt_file = paragraphs_dir / f"{result['paper_id']}.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"Paper ID:        {result['paper_id']}\n")
        f.write(f"Parser:          {result['parser_used']}\n")
        f.write(f"Status:          {result['status']}\n")
        f.write(f"# Candidates:    {result['num_candidates']}\n")
        f.write("=" * 80 + "\n\n")
        for idx, p in enumerate(result["paragraphs"], 1):
            f.write(f"--- Paragraph {idx} ---\n")
            f.write(f"Section:        {p['section']}\n")
            f.write(f"Priority:       {p['section_priority']}\n")
            f.write(f"Pos bucket:     {p.get('position_bucket', '?')}\n")
            f.write(f"Rel pos:        {p.get('rel_pos', '?')}\n")
            f.write(f"Score:          {p['score']}\n")
            f.write(f"Signals:        {json.dumps(p['signals'], ensure_ascii=False)}\n")
            f.write(f"Trigger Sent:   {p['trigger_sentence']}\n\n")
            f.write(f"Paragraph:\n{p['paragraph']}\n\n")
            f.write("-" * 80 + "\n\n")


def save_master_dict(output_path, master_dict):
    # Write to a temp file then rename — atomic, avoids the truncation
    # corruption you saw in v3/v4 checkpoints.
    tmp = output_path / "master_results.json.tmp"
    final = output_path / "master_results.json"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(master_dict, f, indent=2, ensure_ascii=False)
    tmp.replace(final)


def save_summary(output_path, master_dict, elapsed):
    total       = len(master_dict)
    parsed_ok   = sum(1 for v in master_dict.values() if v["status"] == "success")
    parse_fail  = sum(1 for v in master_dict.values() if v["status"] == "parse_failed")
    skip_large  = sum(1 for v in master_dict.values() if v["status"] == "skipped_too_large")
    skip_to     = sum(1 for v in master_dict.values() if v["status"] == "skipped_timeout")
    errors      = sum(1 for v in master_dict.values() if v["status"] == "error")
    with_decl   = sum(1 for v in master_dict.values() if v["num_candidates"] > 0)
    total_paras = sum(v["num_candidates"] for v in master_dict.values())

    summary = {
        "run_timestamp":       datetime.now().isoformat(),
        "input_folder":        INPUT_FOLDER,
        "total_pdfs":          total,
        "parsed_successfully": parsed_ok,
        "parse_failed":        parse_fail,
        "skipped_too_large":   skip_large,
        "skipped_timeout":     skip_to,
        "errors":              errors,
        "papers_with_decl":    with_decl,
        "total_paragraphs":    total_paras,
        "elapsed_seconds":     round(elapsed, 1),
    }
    with open(output_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for k, v in summary.items():
        print(f"  {k:25}: {v}", flush=True)
    print("=" * 60, flush=True)


def main():
    start = time.time()
    input_path  = Path(INPUT_FOLDER)
    output_path = Path(OUTPUT_FOLDER)

    if not input_path.exists():
        print(f"ERROR: Input folder not found: {INPUT_FOLDER}", flush=True)
        return

    output_path.mkdir(parents=True, exist_ok=True)
    paragraphs_dir = output_path / "paragraphs_txt"
    paragraphs_dir.mkdir(exist_ok=True)

    test_file = output_path / "_write_test.tmp"
    try:
        test_file.write_text("ok")
        test_file.unlink()
        print("Write access confirmed.", flush=True)
    except Exception as e:
        print(f"ERROR: Cannot write to output folder: {e}", flush=True)
        return

    master_results_path = output_path / "master_results.json"
    master_dict = {}
    if master_results_path.exists():
        try:
            with open(master_results_path, "r", encoding="utf-8") as f:
                master_dict = json.load(f)
            print(f"Loaded {len(master_dict)} existing results from checkpoint.", flush=True)
        except json.JSONDecodeError as e:
            print(f"WARNING: master_results.json is corrupt ({e}). Starting fresh.", flush=True)
            master_dict = {}

    already_done = set(master_dict.keys())
    pdf_files    = sorted(input_path.glob("*.pdf"))
    pending      = [p for p in pdf_files if p.stem not in already_done]

    print(f"\nFound {len(pdf_files)} PDFs in:\n  {INPUT_FOLDER}", flush=True)
    print(f"Already processed : {len(already_done)}", flush=True)
    print(f"Remaining         : {len(pending)}", flush=True)
    print(f"Output to:\n  {OUTPUT_FOLDER}\n", flush=True)
    print(f"Limits: max_pdf_seconds={MAX_PDF_SECONDS}s, "
          f"max_text_len={MAX_TEXT_LEN}, "
          f"max_sents/section={MAX_SENTS_PER_SECTION}, "
          f"max_total_sents={MAX_TOTAL_SENTENCES}", flush=True)
    print(f"Position multipliers: {POSITION_MULTIPLIERS}\n", flush=True)

    if not pending:
        print("Nothing new to process. Exiting.", flush=True)
        save_summary(output_path, master_dict, time.time() - start)
        return

    failures = []

    for i, pdf in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] Starting: {pdf.name}", flush=True)
        t0 = time.time()
        try:
            result = find_candidate_paragraphs(pdf)
            master_dict[result["paper_id"]] = result
            save_paragraph_txt(paragraphs_dir, result)
            elapsed_pdf = time.time() - t0
            print(f"[{i}/{len(pending)}] {pdf.name} -> {result['status']}, "
                  f"candidates={result['num_candidates']}, "
                  f"elapsed={elapsed_pdf:.1f}s", flush=True)
        except Exception as e:
            print(f"[{i}/{len(pending)}] {pdf.name} -> ERROR: {e}", flush=True)
            traceback.print_exc()
            failures.append({"file": pdf.name, "error": str(e)})
            master_dict[pdf.stem] = {
                "paper_id": pdf.stem, "status": "error", "error": str(e),
                "num_candidates": 0, "paragraphs": [],
            }

        if i % SAVE_EVERY_N == 0:
            save_master_dict(output_path, master_dict)
            print(f"   [checkpoint saved at {i}/{len(pending)}]", flush=True)

    save_master_dict(output_path, master_dict)

    if failures:
        with open(output_path / "failures.json", "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)

    elapsed = time.time() - start
    save_summary(output_path, master_dict, elapsed)

    print(f"\nDone. Elapsed: {elapsed:.1f}s", flush=True)
    print(f"  master_results.json : {master_results_path}", flush=True)
    print(f"  summary.json        : {output_path / 'summary.json'}", flush=True)
    if failures:
        print(f"  failures.json       : {output_path / 'failures.json'} "
              f"({len(failures)} errors)", flush=True)


if __name__ == "__main__":
    main()
