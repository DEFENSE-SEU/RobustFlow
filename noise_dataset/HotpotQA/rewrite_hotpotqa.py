"""
HotpotQA Dataset Multi-Hop Question Rewriting Module

This module provides functionality to rewrite multi-hop questions from the HotpotQA dataset using various
transformation modes including requirements augmentation, paraphrasing, and noise injection.
It supports multiple noise levels (light, moderate, heavy) to test model robustness on multi-hop reasoning tasks.
"""

# Standard library imports
import json          # For JSON file handling and data serialization
import os           # For file system operations and path handling
import sys          # For system-specific parameters and functions
import time         # For adding delays between API calls to prevent rate limiting
import re           # For regular expression pattern matching in text processing

# Third-party imports
import openai       # OpenAI API client for GPT model interactions
import yaml         # For parsing YAML configuration files
from tqdm import tqdm  # For displaying progress bars during batch processing

# Configuration file path - points to the YAML config containing API settings
config_path = "../../config/config2.yaml"

def load_config(config_path):
    """
    Load and parse YAML configuration file.
    
    This function reads a YAML configuration file and returns the parsed configuration
    dictionary. The configuration contains API keys, base URLs, and model settings
    required for OpenAI API interactions.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Parsed configuration dictionary containing API settings
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    # Open the configuration file in read mode with UTF-8 encoding
    with open(config_path, 'r', encoding='utf-8') as f:
        # Parse the YAML content into a Python dictionary
        config = yaml.safe_load(f)
    return config

def get_openai_client(config_path, model_name=None):
    """
    Initialize and return OpenAI client with specified model configuration.
    
    This function loads the configuration, selects the appropriate model settings,
    and creates an OpenAI client instance with the correct API credentials and base URL.
    
    Args:
        config_path (str): Path to the YAML configuration file
        model_name (str, optional): Name of the model to use. If None, uses the first
                                  available model in the configuration.
        
    Returns:
        tuple: A tuple containing:
            - client (openai.OpenAI): Configured OpenAI client instance
            - model_config (dict): Configuration dictionary for the selected model
            
    Raises:
        ValueError: If the specified model name is not found in the configuration
        FileNotFoundError: If the configuration file doesn't exist
    """
    # Load configuration from YAML file
    config = load_config(config_path)
    
    # If no model name specified, use the first available model
    if not model_name:
        model_name = list(config['models'].keys())[0]
    
    # Validate that the requested model exists in the configuration
    if model_name not in config['models']:
        raise ValueError(f"Model '{model_name}' not found in configuration")
    
    # Extract model-specific configuration
    model_config = config['models'][model_name]

    # Create and configure OpenAI client with API credentials
    client = openai.OpenAI(
        api_key=model_config['api_key'],    # API key for authentication
        base_url=model_config['base_url']   # Base URL for API endpoint
    )
    
    # Return both client and model configuration
    return client, model_config

# Initialize OpenAI client and model configuration at module level
client, model_config = get_openai_client(config_path)

def rewrite_prompt_with_openai(original_prompt, mode):
    """
    Rewrite a HotpotQA multi-hop question using OpenAI API based on specified transformation mode.
    
    This function takes an original multi-hop question and applies one of five transformation modes:
    - 'requirements': Adds constraint-based instructions for multi-hop reasoning
    - 'paraphrasing': Rewrites the question while preserving multi-hop intent and meaning
    - 'light_noise': Adds light colloquial noise and typos while keeping multi-hop structure
    - 'moderate_noise': Adds moderate colloquial noise and typos with increased complexity
    - 'heavy_noise': Adds heavy colloquial noise and typos while maintaining multi-hop recoverability
    
    The function uses different system prompts and temperature settings based on the mode.
    For noise modes, higher temperature (0.7) is used for more creative variations.
    For requirements and paraphrasing modes, temperature 0.0 ensures consistency.
    
    Args:
        original_prompt (str): The original multi-hop question to be rewritten
        mode (str): Transformation mode - one of 'requirements', 'paraphrasing',
                   'light_noise', 'moderate_noise', or 'heavy_noise'
        
    Returns:
        str: The rewritten question extracted from the API response
        
    Raises:
        Exception: If API calls fail after retries, returns the original prompt
    """
    # Define system prompt for requirements mode - adds constraint-based instructions for multi-hop reasoning
    system_prompt_requirements = """
You are a HotpotQA constraint rewriter. Given a single original question sentence **Q**, rewrite it into a **fluent constraint-led instruction** that:

1. begins with constraints (e.g., Use only the given Wikipedia paragraphs, ...),
2. then says **to represent {Q_without_trailing_question_mark}.** (embed Q verbatim except drop the final ?),
3. contains **no added facts** and **does not change the meaning** of Q,
4. uses **concise, declarative English** (prefer **2–3 sentences** total).

## How to construct the instruction (pick 3–8 items that fit HotpotQA):
* **Source scope:** Use only the given Wikipedia paragraphs (with titles and sentence indices); no external knowledge or guessing.
* **Multi-hop evidence:** Derive the answer by combining facts across **≥2 supporting sentences**; when applicable, these should come from **different titles** and be connected by a **bridge entity**.
* **Output format:** Return **one-line JSON**  
  `{"answer":"<string-or-yes/no-or-unknown>", "supporting_facts":[["<title1>", <sent_idx1>], ["<title2>", <sent_idx2>]]}`  
  (include up to **2–4** minimally sufficient supporting pairs).
* **Evidence policy:** Each supporting pair must **directly entail** the answer; choose the most specific sentences (avoid lead/infobox fluff).
* **Uncertainty policy:** If a consistent chain cannot be formed from the provided paragraphs, set `"answer":"unknown"` and keep `"supporting_facts":[]`.
* **Disambiguation:** Prefer the entity that **simultaneously satisfies all hops**; if multiple candidates exist, pick the one **explicitly linked via the bridge** in the context.
* **Normalization:** For **yes/no** use lowercase `"yes"`/`"no"`. For **acronym expansion** questions, return the **expanded phrase only**, preserve capitalization, no quotes or trailing punctuation.
* **Forbidden operations:** No chain-of-thought or extra text; output **only** the specified JSON.

## Formatting rules (strict):
* Return **exactly one line**, wrapped as:  
  `<answer>{final_instruction_sentence(s)}</answer>`
* Do **not** echo Q separately; do **not** add examples or labels like Constraints:.
* Use plain ASCII punctuation. Remove only Q’s trailing `?`; otherwise keep Q’s wording intact.

### Example (style only; do not echo at runtime)

From:  
VIVA Media AG changed it's name in 2004. What does their new acronym stand for?

To:  
<answer>Use only the given Wikipedia paragraphs (titles with sentence indices) and combine evidence from at least two supporting sentences, possibly across different titles, to represent VIVA Media AG changed it's name in 2004. What does their new acronym stand for. Return one-line JSON {"answer":"<string-or-unknown>","supporting_facts":[["<title1>",i],["<title2>",j]]}; if no consistent chain is available, set "answer":"unknown" and an empty list. For acronym expansion, output the expanded phrase only, preserving capitalization and without quotes.</answer>
    """

    # Define user prompt for requirements mode - template for constraint-based multi-hop augmentation
    user_prompt_requirements = """
You will receive a single original HotpotQA question sentence (Q).

Rewrite Q into a constraint-led instruction that:
1) begins with constraints (e.g., "Use only the given Wikipedia paragraphs (with titles and sentence indices), ..."),
2) then says "to represent {Q_without_trailing_question_mark}.",
3) preserves Q's meaning exactly (no new facts, no reinterpretation),
4) is concise (2–3 sentences total).

When selecting constraints (choose 4–8, strict but not excessive), prefer:
- Source scope: use only the provided Wikipedia paragraphs (titles + sentence indices); no external knowledge or guessing.
- Multi-hop evidence: combine facts from at least two supporting sentences, ideally from different titles via a bridge entity.
- Output format: one-line JSON {"answer":"<string-or-yes/no-or-unknown>","supporting_facts":[["<title1>",i],["<title2>",j]]} with 2–4 minimally sufficient pairs.
- Evidence policy: each supporting pair must directly entail the answer; choose the most specific sentences and avoid generic lead/infobox lines.
- Uncertainty policy: if no consistent multi-hop chain can be formed from the provided paragraphs, set "answer":"unknown" and use an empty "supporting_facts".
- Disambiguation: pick the entity that jointly satisfies all hops; prefer the one explicitly linked via the bridge in context.
- Normalization: for yes/no use lowercase "yes"/"no"; for acronym expansion return the expanded phrase only (no quotes, no trailing punctuation).
- Forbidden operations: no chain-of-thought or extra text; output only the specified JSON.

Hard rules:
- Return exactly one line wrapped as: <answer>{final_instruction}</answer>
- Do not include labels like "Constraints:" or any brackets.
- Use ASCII punctuation. Keep Q verbatim inside "to represent ...", but drop its trailing "?".
- Keep the language consistent with Q (English in -> English out; Chinese in -> Chinese out).
- No examples, no extra commentary, no leading/trailing whitespace.
- Do not restate Q elsewhere.

Output:
<answer><final_instruction_sentence></answer>

Original question (`original_prompt`):
{{original_prompt}}
    """

    # Define system prompt for paraphrasing mode - rewrites multi-hop questions while preserving meaning
    system_prompt_paraphrasing = """
You are a HotpotQA prompt rewriter. Given an input that contains a single **question sentence (or two-sentence pattern)** Q (optionally prefixed with labels like 'Question:'), **rewrite ONLY the question** by changing its *form* (e.g., voice, sentence mood, order, register) while **preserving language and meaning**. Then output the **rewritten question** wrapped inside `<answer>...</answer>` with nothing else.

## Strict preservation
* Keep the **language** unchanged (English→English, Chinese→Chinese).
* Preserve **named entities, acronyms, numbers, units, dates, math expressions, quoted titles**, and any **inline code/literals/special tokens** (e.g., `<...>`, `{...}`, `$...`, URLs, file paths) **exactly**.
* Preserve **HotpotQA multi-hop intent**: if Q references multiple entities/pages or a **bridge entity**, keep them all; do not collapse the question into single-hop.
* Preserve the **question type** (e.g., WH-question, yes/no) and the **target being asked about** (e.g., acronym expansion vs. definition).
* If Q contains dataset-style markers or scope notes, keep them verbatim and in place.

## Allowed transformations (light touch)
* Sentence mood: interrogative ↔ imperative/declarative (e.g., “What does … stand for?” → “State what … stands for.”).
* Information order and structure: combine or split into one/two sentences **without dropping any referenced entity**; prefer a single self-contained sentence if unambiguous.
* Register/tonality: slightly more formal or plain.
* Near-synonyms in the **same language** that do not alter specificity.
* Minor grammatical or punctuation fixes that do not change meaning (e.g., “it’s”→“its” in non-quoted text).
* Length roughly within **±20%** of the original question.

## Do NOT
* Do **not** translate or switch languages.
* Do **not** add/remove **constraints**, hints, options, examples, or evidence requirements.
* Do **not** change specificity or scope (e.g., don’t turn “acronym expansion” into “definition” or “origin”).
* Do **not** add chain-of-thought, explanations, or any extra lines.

## Feasibility & tie-breakers
* If a rewrite risks meaning drift, choose the closest paraphrase or keep the original wording with minimal cosmetic edits.
* Ensure the rewritten question still cues a **multi-hop** retrieval when the original did.

## Output format (mandatory)
Return **exactly**:
<answer><rewritten question here></answer>

## Example (positive)
From:
VIVA Media AG changed it's name in 2004. What does their new acronym stand for?

To:
<answer>State what the new acronym adopted by VIVA Media AG in 2004 stands for.</answer>
    """

    # Define user prompt for paraphrasing mode - template for multi-hop question paraphrasing
    user_prompt_paraphrasing = """
You will receive a single original HotpotQA question (Q), which may be a one- or two-sentence pattern.

Paraphrase ONLY the question’s form (voice, sentence mood, order, register) while preserving its language, meaning, and **multi-hop intent**. Output the rewritten question wrapped inside `<answer>...</answer>` and nothing else.

Strict requirements:
- Keep the language unchanged (English→English, Chinese→Chinese).
- Preserve named entities, **acronyms**, numbers, units, dates, math expressions, quoted titles, and any inline code/literals/special tokens (e.g., `<...>`, `{...}`, `$...`, URLs, file paths) exactly.
- **Preserve multi-hop cues**: if Q references multiple entities/pages or a bridge entity, keep them all; do not collapse it into a single-hop question.
- Preserve the **question type and target** (e.g., acronym expansion vs. definition; yes/no vs. wh- question).
- If labels/scope notes appear (e.g., `Question:`, `use only the passage`), keep them in place.
- Keep length within ±20% of the original question.
- Do not add/remove constraints, hints, options, or examples; do not introduce new facts.

Allowed transformations (light touch):
- Interrogative ↔ imperative/declarative phrasing.
- Reorder clauses; you may merge two sentences into one if unambiguous, or keep two if clearer—without dropping any referenced entity.
- Slight register adjustment (more formal or plainer).
- Near-synonyms that do not alter specificity.

Do NOT:
- Translate or switch languages.
- Add chain-of-thought, explanations, or extra lines.

Output:
<answer><paraphrased question only></answer>

Original question (`original_prompt`):
{{original_prompt}}
    """

    # Define system prompt for light noise mode - adds minimal colloquial noise to multi-hop questions
    system_prompt_light_noise = """
You are a HotpotQA prompt noiser. Given a single **HotpotQA question** Q (one- or two-sentence pattern, optionally prefixed with labels like "Question:"), inject **light, colloquial, low-noise** edits into the **question only**, with a **mild bias toward typos/misspellings**, while **preserving the original meaning, language, and multi-hop intent**. Then output the **noised question** wrapped inside `<answer>...</answer>` with nothing else.

## Style goal
Make Q look slightly chatty and a bit messy—small slang bits, a few soft typos, gentle punctuation/casing quirks—**yet fully recoverable** by a grader for multi-hop reasoning.

## Noise palette (colloquial, lighter — question ONLY)
* **Typos & misspellings (primary):** minor insert/delete/substitute/transpose; small letter doubling/drops. Target **≥50%** of edits from this class (e.g., “acronym”→“acronymm”, “stand for”→“stand fr”).  
* **Slang & IM speak:** uh/erm/lemme/gonna/wanna/tbh/ngl/low-key/high-key/lol/bruh/tho/bc/BTW, brief asides (use sparingly).  
* **Contractions & light drops:** natural contractions; at most **one** light auxiliary/article/prep drop if intent stays clear.  
* **Vowel stretching & stutter (very light):** a single mild case like “reaaally” or “y-yeah”.  
* **Hedges & fillers:** like, kinda, sorta, basically, idk?—used cautiously.  
* **Casing & punctuation quirks:** small anomalies (random caps, an extra “?”/“,”, or a missing space). Allow **1–2** anomalies total.  
* **Keyboard slips:** occasional adjacent-key slip, at most once.  
* **Leet/character swaps (rare):** 0↔o, 1↔l, i↔l **at most once**, only if still clear.  
* **Emojis & emoticons:** **≤1** per question.  
* **Random symbol run (optional):** one run like `&^%$#@~`, **≤6** chars, **≤1** total, only at a clause boundary.  
* **Fragments & run-ons:** generally avoid; keep grammar intact.  
* **Mild paraphrase:** tiny reordering or near-synonyms that don’t change specificity, **without collapsing multi-hop**.

## Hard rules (recovery, semantics, multi-hop)
* **Do not translate** or switch languages.
* **Preserve question type and multi-hop intent.** Include **one clean occurrence** of the interrogative cue (e.g., “How many”, “Did”, “Which”, “Where”, “Who”). If Q has two sentences (context + question), you may noise both, but keep the multi-hop linkage evident.
* **Digits, dates, and relational connectors stay exact:** never change numeric **values** (e.g., years like 2004) or connective words encoding the hops/logic such as **changed (its) name in**, **before/after**, **between**, **compared to**, **from … to …**.  
  – Common units/nouns may be lightly noised (“years”→“yeaars”) if recognizability remains.
* **Protect verbatim:** quoted titles, acronyms in quotes, URLs, file paths, inline code/literals, special tokens (`<...>`, `{...}`, `$...`) must be byte-for-byte unchanged.
* **Named entities & bridge entities:** keep **at least one clean mention** of **each** proper noun/page title involved in the hops (source entity, bridge, target). Other mentions may be lightly noised.  
  – **Do not alter co-reference**: keep pronouns like “their/its” if present; do not replace or drop them.
* **No extra content:** do not add hints/options/constraints/explanations or chain-of-thought; do not invent/remove labels. If “Question:” exists, you may keep it.
* Output **one line** inside the wrapper; no leading/trailing whitespace outside.

## Intensity & limits (light profile)
* Target **15–30%** tokens noised; allow up to **35%** if still easily readable.
* Use **2–3** noise types overall; you may stack **≤2 edits per token** (e.g., tiny typo + light casing).
* Keep length within **±15%** of the original.
* Avoid noising **every** instance of a critical term; ensure recognizability via the clean interrogative anchor and clean proper-noun mentions.

## Output format (mandatory)
Return **exactly**:
<answer><noised question here></answer>

### Calibrated example (style only; do not echo at runtime)
From:
VIVA Media AG changed its name in 2004. What does their new acronym stand for?
To:
<answer>VIVA Media AG changed its name in 2004, tbh — what does their new acronymm stand for?</answer>
    """

    # Define user prompt for light noise mode - template for light noise injection in multi-hop questions
    user_prompt_light_noise = """
You will receive a single HotpotQA question (Q). It may be one sentence or a two-sentence pattern (context + question).

Make Q lightly colloquial and a bit messy (small slang, mild typos, gentle punctuation/casing quirks) while preserving its meaning, language, and **multi-hop intent** (keep the bridge between entities/titles). Keep **one clean interrogative anchor** (“How many/Did/Which/Where/Who…”) and **one clean mention of each proper noun/page title** (including bridge entities). Output the noised question wrapped inside `<answer>...</answer>` and nothing else.

Targets (light):
- Intensity: noise **15–30%** of tokens (allow up to **35%** if still easily readable).
- Edit mix: use **2–3** noise types; you may stack **≤2 edits** on the same token.
- Typos bias: **≥50%** of edits are typos/misspellings/keyboard slips.
- Digits & temporal/connective words stay exact (e.g., **2004**, and connectors like **changed its name in / before / after / between / compared to / from…to…**).
- Units/common nouns may be lightly noised (e.g., **years→yeaars**), but keep at least one recognizable occurrence.
- Quoted titles/URLs/code/special tokens must remain **verbatim**.
- **Multi-hop anchors:** keep ≥1 clean occurrence of **each** proper noun/page title involved across hops (source, bridge, target). If pronouns (“its/their”) encode the link, keep them; do not drop or replace.
- Casing/punctuation quirks allowed (**1–2** anomalies); **≤1** emoji; **≤1** symbol run (**≤6** chars). No blank lines.
- Length within **±15%** of the original.
- Produce **one line** inside the wrapper; no extra commentary or whitespace.

Do NOT:
- Translate or change meaning/scope/question type; **do not collapse multi-hop into single-hop**.
- Alter digits/values, invent/remove labels, or add hints/options/constraints/explanations.
- Break quoted titles, URLs, file paths, inline code/literals, or special tokens.
- Include chain-of-thought.

Return exactly:
<answer><noised question here></answer>

Original question (`original_prompt`):
{{original_prompt}}
    """

    # Define system prompt for moderate noise mode - adds medium-level colloquial noise to multi-hop questions
    system_prompt_moderate_noise = """
You are a HotpotQA prompt noiser. Given a single **HotpotQA question** Q (one- or two-sentence pattern, optionally prefixed with labels like "Question:"), inject **colloquial, medium-noise** edits into the **question only**, with a **balanced bias toward typos/misspellings**, while **preserving the original meaning, language, and multi-hop intent**. Then output the **noised question** wrapped inside `<answer>...</answer>` with nothing else.

## Style goal
Make Q look casual and a bit messy—some slang, a few stutters/stretches, and light punctuation/casing quirks—**yet clearly recoverable** by a grader for multi-hop reasoning.

## Noise palette (colloquial, moderate — question ONLY)
* **Typos & misspellings (primary):** insert/delete/substitute/transpose, light letter doubling/drops. Target **≥60%** of edits from this class (e.g., “acronym”→“acronymm”, “stand for”→“stannd for”).
* **Slang & IM speak:** uh/erm/lemme/gonna/wanna/tbh/ngl/low-key/high-key/lol/bruh/tho/bc/BTW, brief asides.
* **Contractions & drop words:** drop light auxiliaries/articles/preps if intent stays clear; moderate contractions allowed.
* **Vowel stretching & stutter:** “reaaally”, “y-yeah”, “hiiigher”.
* **Hedges & fillers:** like, kinda, sorta, basically, idk?, quick asides.
* **Casing & punctuation chaos:** rANdoM caps, !!!??!.., duplicated/missing commas/spaces; allow **2–4** anomalies total.
* **Keyboard slips:** adjacent-key hits, stray shift.
* **Leet/character swaps:** 0↔o, 1↔l, i↔l, sparingly.
* **Emojis & emoticons:** 🤔🙂😅 etc., **≤2** per question.
* **Random symbol runs:** `&^%$#@~` style, each **≤8** chars, **≤2** runs total; only at clause boundaries.
* **Fragments & run-ons:** permit one mild fragment and/or a short run-on; keep **one clean clause** intact.
* **Mild paraphrase:** reorder words; near-synonyms that don’t change specificity, **without collapsing multi-hop**.

## Hard rules (recovery, semantics, multi-hop)
* **Do not translate** or switch languages.
* **Preserve question type and multi-hop intent.** Include **one clean occurrence** of the interrogative cue (e.g., “How many”, “Did”, “Which”, “Where”, “Who”). If Q has two sentences (context + question), you may noise both, but keep the multi-hop linkage evident.
* **Digits, dates, and relational connectors stay exact:** never change numeric **values** (e.g., years like 2004) or connective words encoding the hops/logic such as **changed (its) name in**, **before/after**, **between**, **compared to**, **from … to …**.  
  – Common units/nouns may be lightly noised (“years”→“yeaars”) if recognizability remains.
* **Protect verbatim:** quoted titles, acronyms in quotes, URLs, file paths, inline code/literals, special tokens (`<...>`, `{...}`, `$...`) must be byte-for-byte unchanged.
* **Named entities & bridge entities:** keep **at least one clean mention** of **each** proper noun/page title involved in the hops (source entity, bridge, target). Other mentions may be lightly noised.  
  – **Do not alter co-reference**: keep pronouns like “their/its” if present; do not replace or drop them.
* **No extra content:** do not add hints/options/constraints/explanations or chain-of-thought; do not invent/remove labels. If “Question:” exists, you may keep it.
* Output **one line** inside the wrapper; no leading/trailing whitespace outside.

## Intensity & limits (moderate profile)
* Target **35–55%** tokens noised; allow up to **60%** if still readable.
* Use **3–4** noise types overall; you may stack **≤2 edits per token** (e.g., typo + casing).
* Keep length within **±20%** of the original.
* Avoid noising **every** instance of a critical term; ensure recognizability via the clean interrogative anchor and clean proper-noun mentions.

## Output format (mandatory)
Return **exactly**:
<answer><noised question here></answer>

### Calibrated example (style only; do not echo at runtime)
From:
VIVA Media AG changed its name in 2004. What does their new acronym stand for?
To:
<answer>VIVA Media AG changed its name in 2004 — tbh, what does their new acronymm stand for?</answer>
    """

    # Define user prompt for moderate noise mode - template for moderate noise injection in multi-hop questions
    user_prompt_moderate_noise = """
You will receive a single HotpotQA question (Q). It may be one sentence or a two-sentence pattern (context + question).

Make Q colloquial and moderately messy (chatty slang, some typos, light vowel stretches, punctuation/casing quirks) while preserving its meaning, language, and **multi-hop intent** (keep the bridge between entities/titles). Keep **one clean interrogative anchor** (“How many/Did/Which/Where/Who…”) and **one clean mention of each proper noun/page title** (including bridge entities). Output the noised question wrapped inside `<answer>...</answer>` and nothing else.

Targets (moderate):
- Intensity: noise **35–55%** of tokens (allow up to **60%** if still readable).
- Edit mix: use **3–4** noise types; you may stack **≤2 edits** on the same token.
- Typos bias: **≥60%** of edits are typos/misspellings/keyboard slips.
- Digits & temporal/connective words stay exact (e.g., **2004**, and connectors like **changed its name in / before / after / between / compared to / from…to…**).
- Units/common nouns may be lightly noised (e.g., **years→yeaars**), but keep at least one recognizable occurrence.
- Quoted titles/URLs/code/special tokens must remain **verbatim**.
- **Multi-hop anchors:** keep ≥1 clean occurrence of **each** proper noun/page title involved across hops (source, bridge, target). If pronouns (“its/their”) encode the link, keep them; do not drop or replace.
- Casing/punctuation quirks allowed (**2–4** anomalies); **≤2** emojis; **≤2** symbol runs (**≤8** chars). No blank lines.
- Length within **±20%** of the original.
- Produce **one line** inside the wrapper; no extra commentary or whitespace.

Do NOT:
- Translate or change meaning/scope/question type; **do not collapse multi-hop into single-hop**.
- Alter digits/values, invent/remove labels, or add hints/options/constraints/explanations.
- Break quoted titles, URLs, file paths, inline code/literals, or special tokens.
- Include chain-of-thought.

Return exactly:
<answer><noised question here></answer>

Original question (`original_prompt`):
{{original_prompt}}
    """

    # Define system prompt for heavy noise mode - adds maximum colloquial noise to multi-hop questions
    system_prompt_heavy_noise = """
You are a HotpotQA prompt noiser. Given a single **HotpotQA question** Q (one- or two-sentence pattern, optionally prefixed with labels like "Question:"), inject **ultra-colloquial, high-noise** edits into the **question only**, with a **strong bias toward typos/misspellings**, while **preserving the original meaning, language, and multi-hop intent**. Then output the **noised question** wrapped inside `<answer>...</answer>` with nothing else.

## Style goal
Make Q look chatty, messy, and almost unrecognizable—slangy, stuttery, stretched vowels, punctuation chaos—**yet still recoverable** by a grader for multi-hop reasoning.

## Noise palette (ultra-colloquial, heavier — question ONLY)
* **Typos & misspellings (primary):** insert/delete/substitute/transpose, letter doubling/drops. Target **≥65%** of edits from this class (e.g., “acronym”→“acronymm”, “stand for”→“stannd for”).
* **Slang & IM speak:** uh/erm/lemme/gonna/wanna/tbh/ngl/low-key/high-key/lol/bruh/tho/bc/BTW, brief asides.
* **Contractions & drop words:** drop light auxiliaries/articles/preps if intent stays clear; heavy contractions allowed.
* **Vowel stretching & stutter:** “reaaally”, “y-yeah”, “hiiigher”.
* **Hedges & fillers:** like, kinda, sorta, basically, idk?, quick asides.
* **Casing & punctuation chaos:** rANdoM caps, !!!??!?.., duplicated/missing commas/spaces; allow **3–6** anomalies total.
* **Keyboard slips:** adjacent-key hits, stray shift.
* **Leet/character swaps:** 0↔o, 1↔l, i↔l, sparingly.
* **Emojis & emoticons:** 🤔🙂😅 etc., **≤3** per question.
* **Random symbol runs:** `&^%$#@~` style, each **≤10** chars, **≤2** runs total; only at clause boundaries.
* **Fragments & run-ons:** permit one fragment and/or a run-on; keep **one clean clause** intact.
* **Mild paraphrase:** reorder words; near-synonyms that don’t change specificity (e.g., “rank higher” → “place higher”), **without collapsing multi-hop**.

## Hard rules (recovery, semantics, multi-hop)
* **Do not translate** or switch languages.
* **Preserve question type and multi-hop intent.** Include **one clean occurrence** of the interrogative cue (e.g., “How many”, “Did”, “Which”, “Where”, “Who”). If Q has two sentences (context + question), you may noise both, but keep the multi-hop linkage evident.
* **Digits, dates, and relational connectors stay exact:** never change numeric **values** (e.g., years like 2004) or connective words encoding the hops/logic such as **changed (its) name in**, **before/after**, **between**, **compared to**, **from … to …**.  
  – Common units/nouns may be noised (“years”→“yeaars”) if recognizability remains.
* **Protect verbatim:** quoted titles, acronyms in quotes, URLs, file paths, inline code/literals, special tokens (`<...>`, `{...}`, `$...`) must be byte-for-byte unchanged.
* **Named entities & bridge entities:** keep **at least one clean mention** of **each** proper noun/page title involved in the hops (source entity, bridge, target). Other mentions may be noised.  
  – **Do not alter co-reference**: keep pronouns like “their/its” if present; do not replace or drop them.
* **No extra content:** do not add hints/options/constraints/explanations or chain-of-thought; do not invent/remove labels. If “Question:” exists, you may keep it.
* Output **one line** inside the wrapper; no leading/trailing whitespace outside.

## Intensity & limits (strong profile)
* Target **60–80%** tokens noised; allow up to **85%** if still readable.
* Use **4–6** noise types overall; you may stack **≤3 edits per token** (e.g., typo + casing + elongation).
* Keep length within **±30%** of the original.
* Avoid noising **every** instance of a critical term; ensure recognizability via the clean interrogative anchor and clean proper-noun mentions.

## Output format (mandatory)
Return **exactly**:
<answer><noised question here></answer>

### Calibrated example (style only; do not echo at runtime)
From:
VIVA Media AG changed it's name in 2004. What does their new acronym stand for?
To:
<answer>tbh VIVA Media AG changed its name in 2004 — sooo like, what does their new acronymm even stannd for ?? lol 🤔 what does … stand for</answer>
    """

    # Define user prompt for heavy noise mode - template for heavy noise injection in multi-hop questions
    user_prompt_heavy_noise = """
You will receive a single HotpotQA question (Q). It may be one sentence or a two-sentence pattern (context + question).

Make Q ultra-colloquial and messy (chatty slang, typos, stretched vowels, punctuation chaos) while preserving its meaning, language, and **multi-hop intent** (keep the bridge between entities/titles). Keep **one clean interrogative anchor** (“How many/Did/Which/Where/Who…”) and **one clean mention of each proper noun/page title** (including bridge entities). Output the noised question wrapped inside `<answer>...</answer>` and nothing else.

Targets (strong):
- Intensity: noise **60–80%** of tokens (allow up to **85%** if still readable).
- Edit mix: use **4–6** noise types; you may stack **≤3 edits** on the same token.
- Typos bias: **≥65%** of edits are typos/misspellings/keyboard slips.
- Digits & temporal/connective words stay exact (e.g., **2004**, and connectors like **changed its name in / before / after / between / compared to / from…to…**).
- Units/common nouns may be noised (e.g., **years→yEaars**), but keep at least one recognizable occurrence.
- Quoted titles/URLs/code/special tokens must remain **verbatim**.
- **Multi-hop anchors:** keep ≥1 clean occurrence of **each** proper noun/page title involved across hops (source, bridge, target). If pronouns (“its/their”) encode the link, keep them; do not drop or replace.
- Casing/punctuation chaos allowed (**3–6** anomalies); **≤3** emojis; **≤2** symbol runs (each **≤10** chars). No blank lines.
- Length within **±30%** of the original.
- Produce **one line** inside the wrapper; no extra commentary or whitespace.

Do NOT:
- Translate or change meaning/scope/question type; **do not collapse multi-hop into single-hop**.
- Alter digits/values, invent/remove labels, or add hints/options/constraints/explanations.
- Break quoted titles, URLs, file paths, inline code/literals, or special tokens.
- Include chain-of-thought.

Return exactly:
<answer><noised question here></answer>

Original question (`original_prompt`):
{{original_prompt}}
    """

    # Replace template placeholder with actual original prompt in all user prompts
    user_prompt_requirements = user_prompt_requirements.replace(
        "{{original_prompt}}", original_prompt
    )
    user_prompt_paraphrasing = user_prompt_paraphrasing.replace(
        "{{original_prompt}}", original_prompt
    )
    user_prompt_light_noise = user_prompt_light_noise.replace(
        "{{original_prompt}}", original_prompt
    )
    user_prompt_moderate_noise = user_prompt_moderate_noise.replace(
        "{{original_prompt}}", original_prompt
    )
    user_prompt_heavy_noise = user_prompt_heavy_noise.replace(
        "{{original_prompt}}", original_prompt
    )

    
    # Select appropriate system and user prompts based on transformation mode
    if mode == "requirements":
        system_prompt = system_prompt_requirements
        user_prompt = user_prompt_requirements
    elif mode == "paraphrasing":
        system_prompt = system_prompt_paraphrasing
        user_prompt = user_prompt_paraphrasing
    elif mode == "light_noise":
        system_prompt = system_prompt_light_noise
        user_prompt = user_prompt_light_noise
    elif mode == "moderate_noise":
        system_prompt = system_prompt_moderate_noise
        user_prompt = user_prompt_moderate_noise
    elif mode == "heavy_noise":
        system_prompt = system_prompt_heavy_noise
        user_prompt = user_prompt_heavy_noise

    # Determine if the mode is a noise injection mode for temperature setting
    is_noise = mode in {"light_noise", "moderate_noise", "heavy_noise"}
    # Use higher temperature (0.7) for noise modes to encourage creativity,
    # lower temperature (0.0) for deterministic modes like requirements and paraphrasing
    temperature = 0.7 if is_noise else 0.0

    # Attempt API call with retry mechanism (up to 2 attempts)
    for attempt in range(2):
        try:
            # Make API call to OpenAI GPT model with selected prompts and temperature
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use GPT-4o-mini model for cost efficiency
                messages=[
                    {"role": "system", "content": system_prompt},  # System instructions
                    {"role": "user", "content": user_prompt},      # User request with original prompt
                ],
                temperature=temperature,  # Temperature setting based on mode
            )
            # Extract and clean the rewritten prompt from API response
            rewritten_prompt = response.choices[0].message.content.strip()
            return rewritten_prompt
        except Exception as e:
            # Handle API errors with retry logic
            if attempt == 0:
                print(f"API call error, retrying: {e}")
            else:
                print(f"API call still failed after retry: {e}")
                # Return original prompt if all retries fail
                return original_prompt


def process_jsonl_file(input_file, output_file, mode):
    """
    Process a JSONL file containing HotpotQA multi-hop questions and rewrite them using specified mode.
    
    This function reads a JSONL file where each line contains a JSON object with a 'question' field.
    The question field contains a multi-hop question. The function extracts the question text,
    rewrites it using the specified transformation mode, and updates the JSON object with
    the rewritten question.
    
    Args:
        input_file (str): Path to the input JSONL file containing original multi-hop questions
        output_file (str): Path to the output JSONL file for rewritten questions
        mode (str): Transformation mode - one of 'requirements', 'paraphrasing',
                   'light_noise', 'moderate_noise', or 'heavy_noise'
        
    Returns:
        None: Writes processed data to output file
        
    Note:
        The function expects HotpotQA-style format with 'question' field containing the multi-hop question.
        Other formats are skipped with warnings.
    """
    # Check if input file exists before processing
    if not os.path.exists(input_file):
        print(f"Error: Input File {input_file} not exists.")
        return

    # Read all lines from the input JSONL file
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Initialize list to store processed data entries
    processed_data = []
    
    # Compile regex pattern for extracting rewritten content from API responses
    TAG = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)

    # Process each line in the input file with progress bar
    for i, line in enumerate(tqdm(lines, desc="Processing Progress")):
        # Strip whitespace from the current line
        s = line.strip()
        
        # Skip empty lines
        if not s:
            continue
            
        try:
            # Parse JSON object from current line
            data = json.loads(s)
        except json.JSONDecodeError as e:
            # Handle JSON parsing errors for individual lines
            print(f"[Skip] Line {i+1} JSON parsing failed: {e}")
            continue

        # Extract question field from the JSON data
        q = data.get("question", None)
        
        # Process the question if it exists and is a non-empty string
        if isinstance(q, str) and q.strip():
            # Clean the original question text
            original_question = q.strip()

            # Call the OpenAI rewriter with the extracted question and specified mode
            rewritten = rewrite_prompt_with_openai(original_question, mode)
            
            # Extract the rewritten question from the API response using regex
            m = TAG.search(rewritten)
            # Use extracted content if found, otherwise use the full response
            new_q = (m.group(1) if m else rewritten).strip()

            # Update the data object with the rewritten question
            data["question"] = new_q
        else:
            # Skip lines that don't have a processable 'question' field
            print(f"[Warning] Line {i+1} missing processable 'question' field, keeping original.")

        # Add processed data entry to the results list
        processed_data.append(data)
        
        # Add small delay to prevent API rate limiting
        time.sleep(0.1)

    # Write all processed data to the output JSONL file
    with open(output_file, "w", encoding="utf-8") as f:
        for obj in processed_data:
            # Write each processed data entry as a JSON line
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Print completion message with processing statistics
    print(f"Processing completed! Processed {len(processed_data)} data entries")


def main():
    """
    Main function to process HotpotQA dataset with all transformation modes.
    
    This function orchestrates the complete processing pipeline by:
    1. Defining input and output file paths for each transformation mode
    2. Processing the original HotpotQA dataset with each of the 5 transformation modes:
       - Requirements: Adds constraint-based instructions for multi-hop reasoning
       - Paraphrasing: Rewrites questions while preserving multi-hop intent
       - Light Noise: Adds minimal colloquial noise and typos
       - Moderate Noise: Adds medium-level colloquial noise and typos
       - Heavy Noise: Adds maximum colloquial noise and typos
    
    The function processes the same input file multiple times, generating different
    augmented versions for robustness testing of multi-hop question answering models.
    
    Returns:
        None: Creates multiple output files with transformed multi-hop questions
    """
    # Define input file path (original HotpotQA dataset)
    input_file = "hotpotqa_original.jsonl"
    
    # Define output file paths for each transformation mode
    requirements_output_file = "hotpotqa_requirements.jsonl"
    paraphrasing_output_file = "hotpotqa_paraphrasing.jsonl"
    light_noise_output_file = "hotpotqa_light_noise.jsonl"
    moderate_noise_output_file = "hotpotqa_moderate_noise.jsonl"
    heavy_noise_output_file = "hotpotqa_heavy_noise.jsonl"
    
    # Print input file information
    print(f"Input File: {input_file}")
    
    # Process with requirements augmentation mode
    print("Requirement Augmentation:")
    process_jsonl_file(input_file, requirements_output_file, "requirements")
    print(f"Output File: {requirements_output_file}")

    # Process with paraphrasing mode
    print("Paraphrasing:")
    process_jsonl_file(input_file, paraphrasing_output_file, "paraphrasing")
    print(f"Output File: {paraphrasing_output_file}")

    # Process with light noise injection mode
    print("Light Noise:")
    process_jsonl_file(input_file, light_noise_output_file, "light_noise")
    print(f"Output File: {light_noise_output_file}")

    # Process with moderate noise injection mode
    print("Moderate Noise:")
    process_jsonl_file(input_file, moderate_noise_output_file, "moderate_noise")
    print(f"Output File: {moderate_noise_output_file}")

    # Process with heavy noise injection mode
    print("Heavy Noise:")
    process_jsonl_file(input_file, heavy_noise_output_file, "heavy_noise")
    print(f"Output File: {heavy_noise_output_file}")


# Script entry point - execute main function when run directly
if __name__ == "__main__":
    main()
