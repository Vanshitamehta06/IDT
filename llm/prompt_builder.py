"""Centralized prompt templates and defensive context bounding.

Query classification
--------------------
classify_query_type(query, evidence) returns one of:
  "research"   – answerable from retrieved research context
  "general"    – conceptual/background question the LLM can answer from training
  "realtime"   – requires live/external data (weather, prices, current events)
  "out_of_scope" – clearly unrelated to research, code, or language models
"""

from __future__ import annotations

import re

from config import MAX_CONTEXT_CHARS, MAX_HISTORY_CHARS, MAX_HISTORY_TURNS

# ── Patterns for query classification ────────────────────────────────────────

_REALTIME_RE = re.compile(
    r"\b(weather|today['s]?|right now|current price|stock price|forex|bitcoin price|"
    r"breaking news|latest news|live score|what time is it|current time|"
    r"exchange rate|cryptocurrency|crypto price)\b",
    re.I,
)

_GENERAL_CONCEPTS = (
    "what is ", "what are ", "what does ", "what do ",
    "explain ", "define ", "definition of ", "meaning of ",
    "how does ", "how do ", "how is ", "how are ",
    "describe ", "overview of ", "introduction to ",
    "difference between ", "distinguish between ",
    "give an example of ", "example of ",
)

_OFF_TOPIC_RE = re.compile(
    r"\b(recipe|cook|bake|sport|football|cricket|tennis|celebrity|gossip|"
    r"horoscope|lottery|movie review|song lyrics|poem about|joke|meme)\b",
    re.I,
)


def classify_query_type(query: str, evidence: str = "", retrieved_kinds: list[str] | None = None) -> str:
    """Classify query into 'research' | 'general' | 'realtime' | 'out_of_scope'.

    retrieved_kinds: list of source_kind values from retrieved chunks
      (e.g. ['local_pdf', 'code', 'code']).  When all chunks are 'code' the
      evidence is treated as absent for conceptual questions.
    """
    q = (query or "").strip().lower()

    if _OFF_TOPIC_RE.search(q):
        return "out_of_scope"

    if _REALTIME_RE.search(q):
        return "realtime"

    is_conceptual = any(q.startswith(pat) for pat in _GENERAL_CONCEPTS)

    # Determine whether we have *relevant* evidence.
    # Code chunks are not useful evidence for conceptual/definition questions.
    kinds = retrieved_kinds or []
    has_pdf_evidence = bool(
        evidence and evidence.strip() and evidence.strip() != "(no retrieved evidence)"
        and any(k in ("local_pdf", "arxiv") for k in kinds)
    )
    # If all retrieved chunks are code, evidence doesn't help answer "what is X"
    all_code = bool(kinds) and all(k == "code" for k in kinds)

    if has_pdf_evidence and not all_code:
        if is_conceptual:
            return "general"   # conceptual phrasing WITH real PDF evidence → hybrid label
        return "research"

    # No useful evidence (or only code chunks)
    if is_conceptual:
        return "general"

    # Short factual look-ups without evidence → treat as general
    if len(q.split()) <= 6:
        return "general"

    return "research"


# ── Core utilities ────────────────────────────────────────────────────────────

def compose_raw_prompt(system: str | None, user_prompt: str) -> str:
    """Exact string sent to the chat API (system + user), after bounding."""
    parts: list[str] = []
    if system:
        parts.append(f"[SYSTEM]\n{system}")
    parts.append(f"[USER]\n{user_prompt}")
    return "\n\n".join(parts)


def bound_text(text: str | None, limit: int) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "…"


def format_history(history: list[dict[str, str]] | None) -> str:
    if not history:
        return "(none)"
    recent = history[-MAX_HISTORY_TURNS:]
    lines: list[str] = []
    for turn in recent:
        role = turn.get("role", "user")
        content = bound_text(turn.get("content", ""), 400)
        lines.append(f"{role}: {content}")
    return bound_text("\n".join(lines), MAX_HISTORY_CHARS)


# ── Prompt templates ──────────────────────────────────────────────────────────

def planner_prompt(query: str, history_block: str) -> str:
    return f"""You are an academic research planner. Output ONLY valid JSON (no markdown).
Classify the user query into one intent and list tools to run.

Allowed intents: research_qa, discovery, analysis, comparison, gap_analysis, recommendation
Allowed tools: rag_search, arxiv_search, research_qa, analysis, comparison, gap_analysis, recommendation

JSON schema:
{{
  "intent": "...",
  "tools": ["..."],
  "arxiv_query": "short arXiv search string or empty",
  "requires_local_rag": true,
  "comparison_targets": [],
  "rationale": "one sentence"
}}

Rules:
- Use discovery + arxiv_search when the user wants new papers, literature, or arXiv results.
- Use comparison when two or more papers/approaches are compared.
- Use gap_analysis for open problems, limitations, underexplored areas.
- Use recommendation for what to read next.
- Use analysis for paper structure: problem, method, contributions, limitations.
- Otherwise research_qa with rag_search.
- requires_local_rag is true unless the query is purely external discovery with no local docs needed.

Recent conversation:
{history_block}

User query:
{query}
"""


def qa_prompt(query: str, context: str, history_block: str, query_type: str = "research") -> str:
    """Main QA prompt.  Adapts tone and instructions based on query_type.

    query_type values:
      'research'     – answer strictly from retrieved evidence
      'general'      – explain the concept from training, note it's a general explanation
      'realtime'     – tell the user the assistant cannot provide live data
      'out_of_scope' – politely decline
    """
    ctx = bound_text(context, MAX_CONTEXT_CHARS)
    has_evidence = bool(ctx and ctx.strip() and ctx.strip() != "(no retrieved evidence)")

    if query_type == "realtime":
        return f"""You are a research assistant. The user has asked a question that requires real-time or live data.

Respond clearly and helpfully. Do NOT leave the answer blank.

Tell the user:
1. That you cannot provide live/real-time information (weather, current prices, live scores, etc.).
2. What kind of source they should check instead (e.g. a weather service, financial data provider, news site).
3. If the question has a research angle (e.g. "how does weather forecasting work?"), offer to answer that instead.

Do NOT fabricate any real-time data. Do NOT say "N/A".

Question: {query}
"""

    if query_type == "out_of_scope":
        return f"""You are a research assistant specialising in academic papers, RAG systems, and codebase understanding.

The user asked: "{query}"

Respond politely and helpfully. Do NOT leave the answer blank.

1. Explain briefly that this question is outside the current scope of the Research Assistant.
2. State what the assistant IS designed to help with (research papers, RAG, ML/NLP, codebase questions).
3. If there is any tangential research angle to the question, mention it briefly.

Question: {query}
"""

    if query_type == "general" and not has_evidence:
        return f"""You are a knowledgeable research assistant. The user asked a conceptual question.

There is no retrieved evidence from the local research library for this specific question.

Instructions:
1. Give a clear, complete, and accurate general explanation from your training knowledge.
2. Use concrete examples where helpful.
3. At the end, add a short note: "Note: This is a general explanation based on training knowledge, not directly from the selected research papers. Upload relevant papers and re-ask for a source-grounded answer."
4. NEVER leave the answer blank. NEVER write "N/A" or "Not reported".

Conversation history:
{history_block}

Question: {query}
"""

    if query_type == "general" and has_evidence:
        return f"""Answer the user's question directly and clearly. Then cite supporting evidence from the paper.

Question: {query}

Write your response in this exact structure — two sections, nothing else:

### Answer
Give a clear, direct explanation of what the user asked. Write 2–4 sentences in plain language.
Do NOT start with "According to the paper" or "The paper says". Just answer the question.
Do NOT copy or repeat any sentence from the Evidence section below.

### Supporting evidence
In 2–3 sentences, summarise what the uploaded paper adds. Cite [Source N] / [Page X | Section Y].
If the paper does not directly address the question, write: "The uploaded paper does not directly address this question."

Rules (silent — do not output these):
- Each heading appears exactly once.
- Do not repeat any sentence between the two sections.
- Do not add extra headings.

Evidence from the paper:
{ctx}
"""

    # ── Default: research query ───────────────────────────────────────────────
    if has_evidence:
        return f"""Answer the user's research question directly. Then cite supporting evidence from the paper.

Question: {query}

Write your response in this exact structure:

### Answer
Give a clear, direct answer in 2–4 sentences. Do NOT start with "According to the paper".
Do NOT repeat sentences from the Evidence section.

### Supporting evidence
Cite specific passages: [Source N] / [Page X | Section Y].
Keep this to 2–3 sentences. If evidence does not directly support the answer, say so.

Rules (silent):
- Each heading once.
- No repeated sentences between sections.
- No extra headings.
- Do not invent content not in the evidence.

Conversation:
{history_block}

Evidence:
{ctx}
"""


    else:
        # Research question but NO evidence retrieved
        return f"""You are a research assistant. The user asked a research question but no relevant evidence was found in the local knowledge base.

Instructions:
1. Tell the user clearly that the selected research sources do not contain enough information to answer this question directly.
2. Give a brief general explanation of what the topic is (if you know it) so the response is still useful.
3. Suggest what the user could do: upload a relevant paper, try a broader question, or use the arXiv discovery feature.
4. NEVER write "N/A" or leave the answer blank.
5. Label your response with "Note: No supporting evidence found in the selected sources."

Conversation:
{history_block}

Question: {query}
"""


def analysis_prompt(query: str, context: str) -> str:
    ctx = bound_text(context, MAX_CONTEXT_CHARS)
    return f"""You are producing a structured academic summary of a single research paper.

═══ STRICT RULES ═══
1. Use ONLY information from the evidence below. Do NOT use outside knowledge.
2. Refer to "the paper" (singular, not "the papers" or "these papers").
3. If a chunk describes Python code, Streamlit, FastAPI, evaluation scripts, or
   software infrastructure — SKIP IT. Those are not part of the paper.
4. Each section must appear EXACTLY ONCE. Do not repeat information across sections.
5. For limitations: ONLY include what the paper explicitly states. If the paper
   does not explicitly state a limitation, write:
   "The paper does not explicitly state this in the available content."
6. For findings/results: include exact numbers, benchmarks, or dataset names
   where the evidence contains them. Do not write "state-of-the-art" alone.
7. Do NOT invent paper titles, author names, dataset names, or numeric results.
8. Do NOT add a "What the papers say" section — that heading is forbidden here.
═══════════════════

Write the summary using EXACTLY these seven headings in this order.
Do not add extra headings. Do not repeat any heading.

### Quick Take
One or two sentences: what is this paper about and what is its central claim?

### Research Problem
What specific problem does the paper address? Why does it matter?
Cite the relevant source: [Source N] / [Page X | Section Y].

### Methodology / Approach
Describe the actual system or method the authors propose — not generic background.
Explain the architecture, components, and how they work together.
Cite [Source N].

### Key Findings / Results
What did the paper demonstrate or measure?
Include specific numbers, benchmarks, or dataset names from the evidence.
If no specific numbers are available in the evidence, say so explicitly.
Cite [Source N].

### Contributions
What does the paper introduce, demonstrate, or establish that is novel?
List as bullet points. Each bullet must be supported by the evidence.

### Limitations
List only limitations the paper explicitly acknowledges.
If none are stated in the available evidence, write:
"The paper does not explicitly state limitations in the available content."
Cite [Source N] if applicable.

### Conclusion / Significance
Why does this work matter? What does it enable or advance?
Base this strictly on what the evidence says, not general knowledge.

Evidence from the paper:
{ctx}
"""


def comparison_prompt(query: str, context: str,
                      per_paper_evidence: dict[str, str] | None = None) -> str:
    """Build a comparison prompt.

    If per_paper_evidence is provided (filename → evidence text), each paper's
    content is presented in a clearly separated, labelled block so the LLM
    cannot cross-contaminate facts between papers.

    context is kept as fallback when per_paper_evidence is None (legacy path).
    """
    ctx = bound_text(context, MAX_CONTEXT_CHARS)

    if per_paper_evidence and len(per_paper_evidence) >= 2:
        # Build labelled, separated evidence blocks
        paper_labels: list[str] = []
        evidence_blocks: list[str] = []
        for idx, (fname, ev) in enumerate(per_paper_evidence.items(), start=1):
            label = f"PAPER {idx}: {fname}"
            paper_labels.append(label)
            block = f"{'='*60}\n{label}\n{'='*60}\n{ev}"
            evidence_blocks.append(block)

        papers_line = " vs. ".join(paper_labels)
        evidence_text = "\n\n".join(evidence_blocks)
        n = len(per_paper_evidence)
        col_headers = " | ".join(fname for fname in per_paper_evidence)
        sep = "|".join(["---"] * (n + 1))

        return f"""You are writing a side-by-side comparison of {n} research papers.
The evidence for each paper is in a STRICTLY SEPARATED block below.

ABSOLUTE RULES:
1. NEVER write "This paper..." or "The paper..." — always name which paper (e.g. "rag_survey.pdf says...").
2. NEVER mix facts between papers. PAPER 1 evidence → PAPER 1 column only. PAPER 2 evidence → PAPER 2 column only.
3. If a specific piece of information is genuinely not present in a paper's evidence block, write:
   "Not found in available content" — never leave the cell blank or write generic filler.
4. Use actual numbers, dataset names, metric values from the evidence wherever possible.
5. Do NOT use knowledge from your training. Use ONLY the evidence blocks below.
6. Every prose statement must name the paper it refers to.

---

First, write the comparison table:

### Comparison table

| Aspect | {col_headers} |
| {sep} |
| Research problem | | |
| Objective | | |
| Method / Approach | | |
| Model / Architecture | | |
| Datasets / Benchmarks | | |
| Evaluation metrics | | |
| Key results (with numbers) | | |
| Main contributions | | |
| Limitations | | |
| Future work | | |

Then write these analysis sections. In each section, always name the paper when referring to it:

### Key similarities between the papers

### Key differences in methods

### Datasets and benchmarks compared

### Results compared (use actual numbers from the evidence)

### Strengths and weaknesses of each approach
For each paper, name the paper, then state its strengths and weaknesses.

### Key trade-offs

### Research gap or opportunity revealed by comparing these papers

---
Evidence (each paper STRICTLY separated — do NOT mix):

{evidence_text}

Query: {query}
"""

    # ── Fallback: flat evidence (legacy / no paper_filter) ───────────────────
    import re as _re
    _raw_names = _re.findall(r"\]\s*(.*?\.pdf)", ctx, _re.IGNORECASE)
    _seen: set[str] = set()
    _paper_names: list[str] = []
    for n in _raw_names:
        base = n.split("\\")[-1].split("/")[-1]
        if base not in _seen:
            _seen.add(base)
            _paper_names.append(base)
    paper_list = (
        "\n".join(f"  - {n}" for n in _paper_names)
        if _paper_names else "  (paper names not detected)"
    )

    return f"""You are comparing research papers. Use ONLY the evidence below.

Papers detected in evidence:
{paper_list}

Rules (silent):
1. Use ONLY the evidence. Do NOT use outside knowledge.
2. If a chunk describes Python code, Streamlit, FastAPI, or application files — SKIP IT.
3. Never invent results, numbers, or paper titles.
4. Write "Not found in the available content" for missing info.

Write your response:

### Comparison table
| Aspect | {" | ".join(_paper_names) if _paper_names else "Paper A | Paper B"} |
Use rows: Research Problem, Methodology, Key Findings, Datasets, Contributions, Limitations

### Bottom line

### Where they agree

### Where they differ

### Remaining gap

Evidence:
{ctx}

Query: {query}
"""


def gap_prompt(query: str, context: str) -> str:
    ctx = bound_text(context, MAX_CONTEXT_CHARS)
    return f"""You are identifying research gaps from an uploaded academic paper.

STRICT RULES:
1. Use ONLY the evidence below. Do NOT invent gaps, citations, or paper content.
2. Every gap must reference specific evidence: cite [Source N] / [Page X | Section Y].
3. If a chunk describes Python code, Streamlit, FastAPI, or software infrastructure — SKIP IT.
4. Do not produce generic gaps (e.g. "more data needed", "scalability") unless the paper explicitly mentions them.
5. If the paper is thin on limitations, say so — do not pad with invented issues.
6. Use "the paper" (singular). Do not say "the papers".
7. Each section appears EXACTLY ONCE.

Write using EXACTLY these headings in this order:

### Quick Take
One sentence: what is the main research frontier the paper operates in?

### What the paper establishes
2–4 bullet points of what the paper explicitly claims to have solved or demonstrated.
Cite [Source N] for each bullet.

### What the paper does not address
Gaps explicitly stated or clearly implied by the paper's scope, experiments, or future-work section.
Cite [Source N] for each point. If nothing is explicit, write:
"The paper does not explicitly identify unaddressed areas in the available content."

### Gap 01
**Limitation:** [state it concisely]
**Evidence:** [Source N] / [Page X | Section Y] — quote or paraphrase the relevant sentence.
**Why it matters:** [one sentence]
**Possible direction:** [one sentence]

### Gap 02
**Limitation:** [state it concisely]
**Evidence:** [Source N] / [Page X | Section Y] — or "implied by scope of the paper".
**Why it matters:** [one sentence]
**Possible direction:** [one sentence]

Add Gap 03 only if a third distinct gap is clearly supported by the evidence.

Evidence from the paper:
{ctx}
"""


def recommendation_prompt(query: str, context: str) -> str:
    ctx = bound_text(context, MAX_CONTEXT_CHARS)
    return f"""You are recommending research papers to read next, based on an uploaded academic paper.

STRICT RULES:
1. Use ONLY the evidence below to understand the paper's topic, methods, and gaps.
2. If a chunk describes Python code, Streamlit, FastAPI, evaluation scripts, or software — SKIP IT entirely. Do NOT recommend source code files as papers.
3. A "paper" recommendation must be an actual academic paper or research area — not a Python file, a configuration file, or application code.
4. Base recommendations on the research domain, methods, datasets, and open problems found in the evidence.
5. If the evidence mentions specific related works, cite them. If not, suggest research directions clearly labelled as general suggestions.
6. Do NOT invent paper titles, authors, or results.
7. Each section appears EXACTLY ONCE.

Write using EXACTLY these headings:

### Quick Take
One sentence: what research area does the uploaded paper belong to, and what would a natural next read be?

### Recommended reading

For each recommendation (aim for 3, maximum 5):

**[Rank]. [Topic or paper title if named in evidence]**
- **Why relevant:** connect it explicitly to the uploaded paper's problem, method, or gap.
- **What to look for:** what specific aspect makes it worth reading next.
- **Source in evidence:** [Source N] if the paper is mentioned in the evidence, else write "General suggestion based on the paper's topic."

### Why it matters
One short paragraph on how this reading list would build on or extend the uploaded paper.

Evidence from the paper:
{ctx}
"""


def general_explanation_prompt(query: str) -> str:
    """Standalone prompt for clearly general conceptual questions with no evidence."""
    return f"""You are a knowledgeable assistant explaining a concept clearly.

Give a complete, accurate, well-structured explanation. Use a concrete example where it helps understanding.

At the end, add exactly this note on its own line:
"Note: General explanation — not directly from the selected research papers."

NEVER leave the answer blank. NEVER write "N/A".

Question: {query}
"""


def realtime_fallback_prompt(query: str) -> str:
    """Prompt for queries that require live/external data."""
    return f"""You are a research assistant. The user asked a question that requires real-time or live information.

Respond helpfully and clearly. Do NOT leave the answer blank.

1. Explain that this assistant works with research papers and cannot provide live data (weather, prices, live news, etc.).
2. Tell the user where to find the information they need (e.g. a weather service, search engine, news site).
3. If there is a related research or educational angle to the question, briefly offer to explain that instead.

Do NOT fabricate any real-time data.

Question: {query}
"""


def out_of_scope_prompt(query: str) -> str:
    """Prompt for questions clearly outside the research assistant's domain."""
    return f"""You are a research assistant specialising in academic papers, RAG systems, NLP, and machine learning.

The user asked: "{query}"

Respond politely. Do NOT leave the answer blank.

1. Briefly explain that this question is outside the current scope of the Research Assistant.
2. State what topics you CAN help with.
3. If there is any research or educational angle to this question, mention it briefly.

Question: {query}
"""


def prompt_for_intent(intent: str, query: str, context: str) -> str:
    """Same production templates used by the live app — required for fair model comparison."""
    history = "(none)"
    mapping = {
        "analysis": analysis_prompt,
        "comparison": comparison_prompt,
        "gap_analysis": gap_prompt,
        "recommendation": recommendation_prompt,
    }
    builder = mapping.get(intent)
    if builder:
        return builder(query, context)
    query_type = classify_query_type(query, context)
    return qa_prompt(query, context, history, query_type=query_type)


def synthesis_prompt(query: str, intent: str, tool_outputs: str, history_block: str) -> str:
    body = bound_text(tool_outputs, MAX_CONTEXT_CHARS)
    return f"""Synthesize a final answer for the researcher.
Intent: {intent}
Keep citations of the form [Source N]. Be precise and avoid speculation.
NEVER leave the answer blank.

Conversation:
{history_block}

Tool outputs:
{body}

Original query:
{query}
"""
