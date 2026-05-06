
import os
import re
import json
import time
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime

# PDF parsing
import fitz
import pdfplumber

# Sentence embeddings
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FOLDER  = r"path/to/input/folder"  # <-- UPDATE THIS PATH
OUTPUT_FOLDER = r"path/to/output/folder" # <-- UPDATE THIS PATH

# Model choice: 'all-MiniLM-L6-v2' (fast) or 'BAAI/bge-small-en-v1.5' (better accuracy)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Similarity thresholds
PRIMARY_THRESHOLD   = 0.55   # top-level filter: paragraph vs seed similarity
SECONDARY_THRESHOLD = 0.45   # looser threshold for high-priority sections

SAVE_EVERY_N = 25
BATCH_SIZE   = 64   # for batch embedding



SEED_SENTENCES = {

    # === 1. Grammar & Spelling ===
    "grammar_spelling": [
        "We used ChatGPT for grammar and spelling correction.",
        "Grammarly was used to catch grammatical errors and typos.",
        "The manuscript was proofread using an AI grammar checker.",
        "We employed LanguageTool for proofreading and copy-editing.",
        "AI was used to check grammar, spelling, and punctuation errors.",
        "The paper was corrected for grammatical mistakes using Grammarly.",
        "We utilized an AI tool to identify and fix spelling errors.",
        "ChatGPT assisted in catching errors that we might have missed.",
        "The authors used AI for grammar correction purposes.",
        "Grammar checking and error correction was done using ChatGPT.",
        "We used AI for correcting and checking the language of the manuscript.",
    ],

    # === 2. Writing Style & Clarity ===
    "writing_style": [
        "We used ChatGPT to improve the writing style and clarity.",
        "AI tools were used to enhance the readability and flow of the text.",
        "Grammarly was employed to improve sentence structure and word choice.",
        "We utilized ChatGPT to polish the language and enhance writing quality.",
        "The manuscript was refined using AI to improve clarity and conciseness.",
        "AI assisted with improving the overall flow of the writing.",
        "We used AI to enhance the tone and stylistic quality of the paper.",
        "ChatGPT helped improve the language and readability of this manuscript.",
        "The writing style was improved with the assistance of AI tools.",
        "We relied on AI for language editing and English polishing.",
        "AI was used for adjusting the language and improving text quality.",
        "The text was adjusted and refined using ChatGPT for clarity.",
        "ChatGPT was used for adjusting and improving the writing.",
    ],

    # === 3. Paraphrasing & Rewording ===
    "paraphrase": [
        "We used QuillBot to paraphrase and reword several sections.",
        "ChatGPT was employed to rephrase sentences for clarity and conciseness.",
        "AI tools helped express ideas in different ways.",
        "The text was rewritten using ChatGPT to ensure clarity.",
        "We utilized AI to reformulate and restructure sentences.",
        "AI assisted in rewording paragraphs to ensure conciseness.",
        "QuillBot was used to express ideas in alternative phrasings.",
        "ChatGPT was used for rephrasing portions of the manuscript.",
        "The authors used AI for rephrasing and adjusting the text.",
        "AI was employed for rephrasing, adjusting, and rewording purposes.",
        "We used ChatGPT for rephrasing and adjusting the language.",
        "The manuscript was rephrased using ChatGPT for better readability.",
        "ChatGPT 4.o was used for rephrasing, adjusting, and formatting purposes.",
        "AI tools helped in rephrasing and adjusting the text of this article.",
    ],

    # === 4. Translation & Transcription ===
    "translation": [
        "We used DeepL to translate the abstract from German to English.",
        "AI was used to translate portions of the manuscript into English.",
        "Google Translate was employed to translate the paper to reach a broader audience.",
        "The text was translated from the original language using AI translation tools.",
        "We used AI for transcribing interview recordings into text.",
        "DeepL was utilized to translate the article into multiple languages.",
        "AI assisted with speech-to-text transcription of the interviews.",
        "The manuscript was translated using an AI translation service.",
        "The authors used AI for translation purposes during preparation.",
        "AI was used for translating and transcribing text in this study.",
    ],

    # === 5. Drafting & Content Generation ===
    "drafting": [
        "ChatGPT was used to help draft the introduction section.",
        "We used AI to write different sections of the paper including the abstract.",
        "AI tools assisted in drafting the methodology description.",
        "The initial draft of the paper was generated with the help of ChatGPT.",
        "We employed ChatGPT for drafting and enhancing content.",
        "AI was used to draft a concise abstract that captures the gist of the research.",
        "ChatGPT helped compose the first draft of the conclusion.",
        "We utilized AI to generate text for the literature review section.",
        "The manuscript was partially written using ChatGPT.",
        "AI assisted in writing different sections of our paper.",
        "ChatGPT was used for content generation and text production.",
        "This paper was drafted with the assistance of a large language model.",
        "AI was employed for drafting purposes during the preparation of this article.",
        "Portions of this manuscript were drafted using generative AI.",
    ],

    # === 6. Brainstorming & Ideation ===
    "brainstorming": [
        "We used ChatGPT for brainstorming research ideas and creating hypotheses.",
        "AI was employed as a thinking partner for generating research questions.",
        "ChatGPT helped us explore different research directions.",
        "We utilized AI tools for ideation and concept exploration.",
        "AI assisted in formulating hypotheses for our study.",
        "ChatGPT was used for brainstorming and idea generation.",
        "We consulted ChatGPT to generate potential research questions.",
        "AI was used for brainstorming purposes during the research phase.",
    ],

    # === 7. Literature Review ===
    "literature_review": [
        "We used Elicit for searching scholarly literature and writing the review.",
        "AI tools assisted in generating the literature review section.",
        "ChatGPT was used to find relevant papers and summarize the literature.",
        "We employed AI for conducting a systematic literature search.",
        "Semantic Scholar was used to discover related work and references.",
        "AI helped in drafting a literature review starting from relevant papers.",
        "We used ResearchRabbit for paper discovery and literature mapping.",
        "AI tools assisted in searching scholarly literature.",
        "ChatGPT was used for generating a literature review from a set of papers.",
        "AI was employed for literature search purposes.",
    ],

    # === 8. Plagiarism Detection ===
    "plagiarism": [
        "We used Grammarly for plagiarism detection in our own writing.",
        "AI tools helped identify potential plagiarism issues.",
        "The manuscript was checked for originality using AI plagiarism detection.",
        "We employed AI to run a similarity check on our content.",
        "AI was used to detect duplicate content and ensure originality.",
        "Plagiarism detection was performed using an AI-based tool.",
    ],

    # === 9. Citation Management ===
    "citation": [
        "We used Zotero to format citations and references in APA style.",
        "AI tools helped format citations according to specific styles.",
        "Mendeley was employed for citation management and reference formatting.",
        "We utilized AI to manage and format our bibliography.",
        "AI assisted in formatting references according to the journal style.",
        "We used EndNote for citation management throughout the paper.",
        "AI helped format citations and references according to MLA style.",
    ],

    # === 10. Formatting Assistance ===
    "formatting": [
        "ChatGPT was used for formatting assistance to ensure journal guidelines.",
        "AI tools helped ensure the paper adheres to specific formatting requirements.",
        "We used AI to format the manuscript according to conference guidelines.",
        "AI assisted in ensuring the document meets formatting standards.",
        "The paper was formatted with the help of AI tools.",
        "AI ensured our paper adheres to formatting guidelines required by the journal.",
        "ChatGPT was used for formatting purposes during the preparation of this article.",
        "AI was employed for formatting and adjusting the layout of the manuscript.",
        "We used AI for rephrasing, adjusting, and formatting purposes.",
        "The manuscript was formatted and adjusted using AI tools.",
    ],

    # === 11. Peer Review Simulation ===
    "peer_review": [
        "We used ChatGPT to simulate peer review of our manuscript.",
        "AI provided feedback on the strengths and weaknesses of the paper.",
        "ChatGPT was employed to critique our content and arguments.",
        "AI tools gave us a mock review with constructive feedback.",
        "We used AI to evaluate the strengths and weaknesses of our paper.",
        "ChatGPT simulated peer review by providing critical feedback.",
        "AI was used for reviewing and providing feedback on our research.",
    ],

    # === 12. Content Enhancement ===
    "content_enhancement": [
        "ChatGPT was used for content enhancement and suggesting additional research.",
        "AI suggested additional content that could strengthen our arguments.",
        "We used AI to offer alternative perspectives on our findings.",
        "ChatGPT helped enrich the content with additional insights.",
        "AI tools were used to strengthen the arguments in the paper.",
        "We used AI for content improvement and enrichment of the text.",
        "ChatGPT suggested additional research to support our claims.",
    ],

    # === 13. Image / Graphics / Media ===
    "images_media": [
        "We used DALL-E to generate images for our paper.",
        "Midjourney was used to create illustrations for the figures.",
        "AI generated the cover image and graphical abstract.",
        "Stable Diffusion was employed to produce figures and graphics.",
        "We used Adobe Firefly for creating graphics for the manuscript.",
        "AI-generated images were used for the figures in this paper.",
        "We created diagrams and illustrations using AI image generation.",
        "AI was used for creating graphics, pictures, and visual content.",
    ],

    # === 14. Coding & Programming ===
    "coding": [
        "We used GitHub Copilot for coding assistance in Python.",
        "ChatGPT was employed for code generation and debugging.",
        "AI tools assisted with programming and script development.",
        "We utilized Copilot for pair programming and code completion.",
        "AI was used for coding and programming tasks in the research.",
        "ChatGPT helped write and debug code for data processing.",
        "We used AI for code generation and algorithm implementation.",
    ],

    # === 15. Data Analysis / Collection / Visualization ===
    "data_analysis": [
        "ChatGPT was used for data analysis and visualization.",
        "AI tools helped in collecting, processing, and interpreting data.",
        "We used AI for organising, representing, and visualising data.",
        "ChatGPT assisted with statistical analysis of survey results.",
        "AI was employed for data collection and transcription.",
        "We used AI for modelling complex phenomena in our dataset.",
        "AI tools helped generate data for validation of hypotheses and models.",
        "ChatGPT was used for data interpretation and chart generation.",
    ],

    # === 16. Summarization ===
    "summarization": [
        "We used ChatGPT to summarize the research papers we reviewed.",
        "AI tools helped extract key findings from the literature.",
        "ChatGPT was used to condense text and generate summaries.",
        "We utilized AI to produce a concise summary of our findings.",
        "AI assisted in summarizing the content of related papers.",
        "ChatGPT helped capture the gist of each referenced paper.",
    ],

    # === GENERIC DECLARATION PATTERNS ===
    # Real-world phrasings from actual papers (diverse styles)
    "generic_declaration": [
        # "We used X" style
        "We used ChatGPT during the preparation of this manuscript.",
        "AI tools were used to assist in manuscript preparation.",
        "We utilized large language models during the writing process.",
        "ChatGPT was used in the preparation of this paper.",
        "This manuscript was prepared with the assistance of generative AI.",
        "We used an LLM to assist with the preparation of this article.",
        "During the preparation of this work, the authors used ChatGPT.",
        "With the help of ChatGPT, we edited portions of this manuscript.",

        # "This paper was..." style
        "This paper was written with AI assistance.",
        "This paper was partially written using ChatGPT.",
        "This article was prepared with the help of generative AI tools.",
        "This manuscript was edited and refined using AI tools.",

        # Acknowledge / declare / disclose style
        "We declare the use of generative AI tools in this paper.",
        "The authors acknowledge the use of AI tools.",
        "The authors disclose the use of ChatGPT in writing this paper.",
        "We acknowledge the use of AI tools in preparing this manuscript.",
        "They acknowledge the use of ChatGPT for rephrasing and formatting purposes.",
        "The authors acknowledge the use of ChatGPT during the preparation of this article.",
        "We hereby disclose the use of generative AI in this work.",
        "The authors would like to declare the use of AI tools.",

        # "during the preparation" style
        "During the preparation of this article, the authors used ChatGPT.",
        "Generative AI was used in the preparation of this work.",
        "ChatGPT was employed during the preparation of this manuscript.",
        "AI tools were utilized during the writing and preparation of this paper.",
        "During the preparation of this article, AI tools were used for various purposes.",

        # Negative declarations (also valid - declare NO AI was used)
        "No generative AI tools were used in the preparation of this paper.",
        "The authors did not use any AI tools for this manuscript.",
        "We hereby declare that no AI assistance was used.",
        "No AI or LLM tools were used in the writing of this article.",
        "The authors confirm that no generative AI was used.",

        # Multi-purpose combined
        "ChatGPT was used for rephrasing, adjusting, and formatting purposes during the preparation of this article.",
        "We used AI for grammar checking, paraphrasing, and improving the writing style.",
        "AI tools were used for proofreading, language editing, and formatting.",
        "The authors used ChatGPT for rephrasing and adjusting the text of this manuscript.",
        "ChatGPT was employed for various purposes including proofreading and rephrasing.",
        "We used Grammarly and ChatGPT for spell and grammar correction and improve writing style.",
        "AI was used for multiple purposes including drafting, editing, and formatting.",
    ],
}


# ============================================================
# SECTION PRIORITIES (same as regex pipeline)
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


# ============================================================
# STAGE 1: PDF PARSER
# ============================================================
def parse_pdf(pdf_path):
    """Try fitz first, fall back to pdfplumber."""
    text = ""
    parser_used = None

    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        if text.strip():
            parser_used = "fitz"
    except Exception as e:
        print(f"   [fitz failed]: {e}")

    if not text.strip():
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    pt = page.extract_text()
                    if pt:
                        text += pt + "\n"
            if text.strip():
                parser_used = "pdfplumber"
        except Exception as e:
            print(f"   [pdfplumber failed]: {e}")
            return None, None

    return (text if text.strip() else None), parser_used


# ============================================================
# STAGE 2: TEXT CLEANER
# ============================================================
def clean_text(text):
    """Remove references, equations, tables, URLs etc."""
    # Remove References section onwards
    text = re.sub(
        r'\n\s*(References|Bibliography|Works\s+Cited|REFERENCES)\s*\n.*',
        '\n', text, flags=re.DOTALL,
    )
    # Inline citations
    text = re.sub(r'\[\d+(?:[,\-\u2013]\s*\d+)*\]', '', text)
    text = re.sub(
        r'\(([A-Z][a-zA-Z\-]+(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][a-zA-Z\-]+)*)'
        r'\s*,?\s*\d{4}[a-z]?\)', '', text,
    )
    # Equations
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\$[^$]+\$', '', text)
    text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)
    text = re.sub(
        r'\\begin\{(equation|align|gather|math)\*?\}.*?\\end\{\1\*?\}',
        '', text, flags=re.DOTALL,
    )
    # URLs / DOIs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'doi:\s*\S+', '', text, flags=re.IGNORECASE)
    # Tables / dividers
    text = re.sub(r'(\|[^\n]*\|\s*){2,}', '', text)
    text = re.sub(r'[-_=]{4,}', '', text)
    text = re.sub(r'\.{4,}', '', text)
    # Page numbers
    text = re.sub(r'\n\s*\d{1,4}\s*\n', '\n', text)
    # Whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    # Join hyphenated line breaks and soft-wrapped lines
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'(?<=[a-z,])\n(?=[a-z])', ' ', text)
    return text.strip()


# ============================================================
# STAGE 3: SECTION LOCATOR + PARAGRAPH SPLITTER
# ============================================================
def get_section_priority(header_text):
    """Score a section header by priority."""
    for level, patterns in SECTION_PRIORITIES.items():
        for pat in patterns:
            if re.search(pat, header_text, re.IGNORECASE):
                return level
    return "none"


def split_into_paragraphs(text, min_length=30, max_length=2000):
    """
    Split text into paragraphs. Also detect section headers.
    Returns list of {text, section_header, section_priority, para_idx}
    """
    # Detect headers
    header_pattern = re.compile(
        r'\n\s*('
        r'\d+\.?\s*(?:\d+\.?)*\s+[A-Z][A-Za-z\s&\-]{2,60}|'
        r'[A-Z][A-Z\s&\-]{2,60}|'
        r'[A-Z][a-z]+(?:\s+[A-Za-z]+){0,5}'
        r')\s*\n',
        re.MULTILINE,
    )

    # Split on double newline
    raw_paragraphs = re.split(r'\n\s*\n', text)

    results = []
    current_section = "PREAMBLE"
    current_priority = "none"
    idx = 0

    for raw_para in raw_paragraphs:
        raw_para = raw_para.strip()
        if not raw_para:
            continue

        # Check if this is a section header
        if len(raw_para) < 80 and re.match(
            r'^(\d+\.?\s*(?:\d+\.?)*\s+)?[A-Z]', raw_para
        ):
            test_priority = get_section_priority(raw_para)
            if test_priority != "none" or len(raw_para.split()) <= 8:
                current_section = raw_para.strip()
                current_priority = test_priority
                continue

        # Skip very short or very long paragraphs
        if len(raw_para) < min_length:
            continue
        if len(raw_para) > max_length:
            raw_para = raw_para[:max_length]

        results.append({
            "text": raw_para,
            "section_header": current_section,
            "section_priority": current_priority,
            "para_idx": idx,
        })
        idx += 1

    return results


# ============================================================
# STAGE 4: EMBEDDING ENGINE
# ============================================================
class EmbeddingEngine:
    """Handles embedding of seed sentences and paper paragraphs."""

    def __init__(self, model_name=EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Model loaded.")

        # Pre-compute seed embeddings
        self.seed_sentences = []
        self.seed_categories = []
        self.seed_embeddings = None
        self._build_seed_embeddings()

    def _build_seed_embeddings(self):
        """Flatten and embed all seed sentences."""
        for category, sentences in SEED_SENTENCES.items():
            for s in sentences:
                self.seed_sentences.append(s)
                self.seed_categories.append(category)

        print(f"Embedding {len(self.seed_sentences)} seed sentences across "
              f"{len(SEED_SENTENCES)} categories...")
        self.seed_embeddings = self.model.encode(
            self.seed_sentences,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        print("Seed embeddings ready.")

    def embed_paragraphs(self, texts):
        """Embed a list of paragraph texts."""
        if not texts:
            return np.array([])
        return self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def compute_similarities(self, paragraph_embeddings):
        """
        Compute cosine similarity between each paragraph and each seed.
        Returns: np.array of shape (num_paragraphs, num_seeds)

        Since embeddings are normalized, cosine sim = dot product.
        """
        if len(paragraph_embeddings) == 0:
            return np.array([])
        return np.dot(paragraph_embeddings, self.seed_embeddings.T)

    def get_top_matches(self, similarity_row, top_k=5):
        """
        For a single paragraph's similarity scores, return top-k
        matching seeds with their categories and scores.
        """
        top_indices = np.argsort(similarity_row)[::-1][:top_k]
        matches = []
        for idx in top_indices:
            matches.append({
                "seed_sentence": self.seed_sentences[idx],
                "category": self.seed_categories[idx],
                "similarity": float(similarity_row[idx]),
            })
        return matches

    def get_category_max_scores(self, similarity_row):
        """
        For each category, get the maximum similarity score.
        Returns dict: {category: max_score}
        """
        category_scores = {}
        for i, (cat, score) in enumerate(
            zip(self.seed_categories, similarity_row)
        ):
            if cat not in category_scores or score > category_scores[cat]:
                category_scores[cat] = float(score)
        return category_scores


# ============================================================
# STAGE 5: RESEARCH-CONTEXT FILTER (same as regex pipeline)
# ============================================================
RESEARCH_CONTEXT_REJECTORS = [
    re.compile(r'\b(transformer\s+(?:model|architecture|layer)|'
               r'attention\s+(?:mechanism|head|layer)|'
               r'encoder|decoder|'
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
    re.compile(r'\b(acknowledge\s+(?:the\s+)?(?:support|funding|contribution[s]?)\s+from|'
               r'support(?:ed)?\s+by\s+(?:the\s+)?(?:nsf|nih|grant|award|'
               r'foundation|university|institute)|'
               r'funded\s+by|grant\s+(?:no|number)|'
               r'(?:thanks?|grateful)\s+to)\b', re.IGNORECASE),
    re.compile(r'\b(equation|theorem|lemma|proof|we\s+derive|'
               r'empirical\s+validation|convergence|'
               r'derivation\s+of)\b', re.IGNORECASE),
]


def has_research_context(text):
    """Check if paragraph has strong research-context signals."""
    count = sum(1 for p in RESEARCH_CONTEXT_REJECTORS if p.search(text))
    return count >= 2   # Need at least 2 research signals to reject
                        # (single match could be coincidental)


# ============================================================
# STAGE 6: CANDIDATE SCORING & FILTERING
# ============================================================
def score_paragraph(paragraph_info, similarity_row, engine):
    """
    Score a paragraph based on semantic similarity + section priority.
    Returns (final_score, details_dict, accepted_bool)
    """
    text     = paragraph_info["text"]
    priority = paragraph_info["section_priority"]

    # Get top matches and category scores
    top_matches    = engine.get_top_matches(similarity_row, top_k=5)
    category_scores = engine.get_category_max_scores(similarity_row)

    max_sim   = top_matches[0]["similarity"] if top_matches else 0.0
    top_cat   = top_matches[0]["category"] if top_matches else "none"
    top_seed  = top_matches[0]["seed_sentence"] if top_matches else ""

    # Get categories above threshold
    matched_categories = {
        cat: round(score, 4)
        for cat, score in category_scores.items()
        if score >= SECONDARY_THRESHOLD
    }

    # Threshold depends on section priority
    if priority == "high":
        threshold = SECONDARY_THRESHOLD
    elif priority == "medium":
        threshold = (PRIMARY_THRESHOLD + SECONDARY_THRESHOLD) / 2
    else:
        threshold = PRIMARY_THRESHOLD

    # Research context check
    research_flag = has_research_context(text)

    # If research context AND similarity is borderline, reject
    if research_flag and max_sim < PRIMARY_THRESHOLD + 0.10:
        return 0.0, {
            "max_similarity": round(max_sim, 4),
            "top_category": top_cat,
            "research_context_flag": True,
        }, False

    # Accept?
    accepted = max_sim >= threshold

    details = {
        "max_similarity": round(max_sim, 4),
        "top_category": top_cat,
        "top_seed_sentence": top_seed,
        "matched_categories": matched_categories,
        "threshold_used": round(threshold, 4),
        "section_priority": priority,
    }
    if research_flag:
        details["research_context_flag"] = True

    return round(max_sim, 4), details, accepted


# ============================================================
# STAGE 7: PER-PDF PIPELINE
# ============================================================
def process_single_pdf(pdf_path, engine):
    """Full pipeline for one PDF."""
    paper_id = Path(pdf_path).stem

    # 1. Parse
    raw_text, parser = parse_pdf(pdf_path)
    if not raw_text:
        return {
            "paper_id": paper_id,
            "status": "parse_failed",
            "parser_used": None,
            "num_candidates": 0,
            "paragraphs": [],
        }

    # 2. Clean
    cleaned = clean_text(raw_text)

    # 3. Split into paragraphs
    paragraphs = split_into_paragraphs(cleaned)
    if not paragraphs:
        return {
            "paper_id": paper_id,
            "status": "no_paragraphs",
            "parser_used": parser,
            "num_candidates": 0,
            "paragraphs": [],
        }

    # 4. Embed all paragraphs
    para_texts = [p["text"] for p in paragraphs]
    para_embeddings = engine.embed_paragraphs(para_texts)

    # 5. Compute similarities
    similarities = engine.compute_similarities(para_embeddings)

    # 6. Score and filter
    candidates = []
    for i, para_info in enumerate(paragraphs):
        score, details, accepted = score_paragraph(
            para_info, similarities[i], engine
        )
        if not accepted:
            continue

        candidates.append({
            "paragraph": para_info["text"],
            "section": para_info["section_header"],
            "section_priority": para_info["section_priority"],
            "para_idx": para_info["para_idx"],
            "score": score,
            "signals": details,
        })

    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)

    return {
        "paper_id": paper_id,
        "parser_used": parser,
        "status": "success",
        "total_paragraphs_scanned": len(paragraphs),
        "num_candidates": len(candidates),
        "paragraphs": candidates,
    }


# ============================================================
# STAGE 8: BATCH RUNNER (MAIN)
# ============================================================
def save_master_dict(output_path, master_dict):
    """Save the master dictionary to JSON."""
    with open(output_path / "semantic_master_results.json", "w", encoding="utf-8") as f:
        json.dump(master_dict, f, indent=2, ensure_ascii=False)


def save_summary(output_path, master_dict, elapsed, model_name):
    """Save run summary."""
    total       = len(master_dict)
    parsed_ok   = sum(1 for v in master_dict.values() if v["status"] == "success")
    parse_fail  = sum(1 for v in master_dict.values() if v["status"] == "parse_failed")
    with_decl   = sum(1 for v in master_dict.values() if v["num_candidates"] > 0)
    total_cands = sum(v["num_candidates"] for v in master_dict.values())
    total_scanned = sum(v.get("total_paragraphs_scanned", 0) for v in master_dict.values())

    # Collect all matched categories across papers
    all_categories = {}
    for paper in master_dict.values():
        for para in paper.get("paragraphs", []):
            cats = para.get("signals", {}).get("matched_categories", {})
            for cat in cats:
                all_categories[cat] = all_categories.get(cat, 0) + 1

    summary = {
        "run_timestamp":          datetime.now().isoformat(),
        "input_folder":           INPUT_FOLDER,
        "embedding_model":        model_name,
        "primary_threshold":      PRIMARY_THRESHOLD,
        "secondary_threshold":    SECONDARY_THRESHOLD,
        "total_pdfs":             total,
        "parsed_successfully":    parsed_ok,
        "parse_failed":           parse_fail,
        "papers_with_candidates": with_decl,
        "total_candidates":       total_cands,
        "total_paragraphs_scanned": total_scanned,
        "elapsed_seconds":        round(elapsed, 1),
        "category_hit_counts":    dict(sorted(all_categories.items(),
                                              key=lambda x: x[1], reverse=True)),
    }

    with open(output_path / "semantic_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k:30}:")
            for kk, vv in v.items():
                print(f"    {kk:25}: {vv}")
        else:
            print(f"  {k:30}: {v}")
    print("=" * 60)


def main():
    """Main entry point — resumes from existing results if present."""
    start = time.time()
    input_path  = Path(INPUT_FOLDER)
    output_path = Path(OUTPUT_FOLDER)

    if not input_path.exists():
        print(f"ERROR: Input folder not found: {INPUT_FOLDER}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    # ── RESUME: load any existing results ──────────────────────────────
    master_results_path = output_path / "semantic_master_results.json"
    master_dict = {}
    if master_results_path.exists():
        with open(master_results_path, "r", encoding="utf-8") as f:
            master_dict = json.load(f)
        print(f"Loaded {len(master_dict)} existing results from checkpoint.")

    already_done = set(master_dict.keys())   # paper_id = PDF stem
    # ───────────────────────────────────────────────────────────────────

    # Initialize embedding engine
    engine = EmbeddingEngine(EMBEDDING_MODEL)

    pdf_files = sorted(input_path.glob("*.pdf"))
    print(f"\nFound {len(pdf_files)} PDFs in:\n  {INPUT_FOLDER}")

    # Filter to only unprocessed files
    pending = [p for p in pdf_files if p.stem not in already_done]
    print(f"Already processed : {len(already_done)}")
    print(f"Remaining to process: {len(pending)}")
    print(f"Output to:\n  {OUTPUT_FOLDER}\n")

    if not pending:
        print("Nothing new to process. Exiting.")
        elapsed = time.time() - start
        save_summary(output_path, master_dict, elapsed, EMBEDDING_MODEL)
        return

    failures = []

    for i, pdf in enumerate(pending, 1):
        try:
            result = process_single_pdf(pdf, engine)
            master_dict[result["paper_id"]] = result
            n = result["num_candidates"]
#             print(f"[{i}/{len(pending)}] {pdf.name} -> {result['status']}, candidates={n}")

        except Exception as e:
            print(f"[{i}/{len(pending)}] {pdf.name} -> ERROR: {e}")
            traceback.print_exc()
            failures.append({"file": pdf.name, "error": str(e)})
            master_dict[pdf.stem] = {
                "paper_id": pdf.stem,
                "status": "error",
                "error": str(e),
                "num_candidates": 0,
                "paragraphs": [],
            }

        # Checkpoint — saves merged old + new results
        if i % SAVE_EVERY_N == 0:
            save_master_dict(output_path, master_dict)
            print(f"   [checkpoint at {i}/{len(pending)}]")

    # Final save
    save_master_dict(output_path, master_dict)

    if failures:
        with open(output_path / "semantic_failures.json", "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)

    elapsed = time.time() - start
    save_summary(output_path, master_dict, elapsed, EMBEDDING_MODEL)

    print(f"\nDone. Elapsed: {elapsed:.1f}s")
    print(f"Outputs:")
    print(f"  - {master_results_path}")
    print(f"  - {output_path / 'semantic_summary.json'}")
    if failures:
        print(f"  - {output_path / 'semantic_failures.json'} ({len(failures)} errors)")
        
if __name__ == "__main__":
    main()
