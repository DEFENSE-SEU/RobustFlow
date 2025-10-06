import json
import os
import sys
import time
import re
import openai
import yaml
from tqdm import tqdm

config_path = "../../config/config2.yaml"

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_openai_client(config_path, model_name=None):
    config = load_config(config_path)
    
    if not model_name:
        model_name = list(config['models'].keys())[0]
    
    if model_name not in config['models']:
        raise ValueError(f"Model '{model_name}' not found in configuration")
    
    model_config = config['models'][model_name]

    client = openai.OpenAI(
        api_key=model_config['api_key'],
        base_url=model_config['base_url']
    )
    
    return client, model_config

client, model_config = get_openai_client(config_path)

def rewrite_prompt_with_openai(original_prompt, mode):
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

    is_noise = mode in {"light_noise", "moderate_noise", "heavy_noise"}
    temperature = 0.7 if is_noise else 0.0

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            rewritten_prompt = response.choices[0].message.content.strip()
            return rewritten_prompt
        except Exception as e:
            if attempt == 0:
                print(f"API call error, retrying: {e}")
            else:
                print(f"API call still failed after retry: {e}")
                return original_prompt


def process_jsonl_file(input_file, output_file, mode):
    if not os.path.exists(input_file):
        print(f"Error: Input File {input_file} not exists.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    processed_data = []
    
    TAG = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)

    for i, line in enumerate(tqdm(lines, desc="Processing Progress")):
        s = line.strip()
        
        if not s:
            continue
            
        try:
            data = json.loads(s)
        except json.JSONDecodeError as e:
            print(f"[Skip] Line {i+1} JSON parsing failed: {e}")
            continue

        q = data.get("question", None)
        
        if isinstance(q, str) and q.strip():
            original_question = q.strip()

            rewritten = rewrite_prompt_with_openai(original_question, mode)
            
            m = TAG.search(rewritten)
            new_q = (m.group(1) if m else rewritten).strip()

            data["question"] = new_q
        else:
            print(f"[Warning] Line {i+1} missing processable 'question' field, keeping original.")

        processed_data.append(data)
        
        time.sleep(0.1)

    with open(output_file, "w", encoding="utf-8") as f:
        for obj in processed_data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Processing completed! Processed {len(processed_data)} data entries")


def main():
    input_file = "hotpotqa_original.jsonl"
    
    requirements_output_file = "hotpotqa_requirements.jsonl"
    paraphrasing_output_file = "hotpotqa_paraphrasing.jsonl"
    light_noise_output_file = "hotpotqa_light_noise.jsonl"
    moderate_noise_output_file = "hotpotqa_moderate_noise.jsonl"
    heavy_noise_output_file = "hotpotqa_heavy_noise.jsonl"
    
    print(f"Input File: {input_file}")
    
    print("Requirement Augmentation:")
    process_jsonl_file(input_file, requirements_output_file, "requirements")
    print(f"Output File: {requirements_output_file}")

    print("Paraphrasing:")
    process_jsonl_file(input_file, paraphrasing_output_file, "paraphrasing")
    print(f"Output File: {paraphrasing_output_file}")

    print("Light Noise:")
    process_jsonl_file(input_file, light_noise_output_file, "light_noise")
    print(f"Output File: {light_noise_output_file}")

    print("Moderate Noise:")
    process_jsonl_file(input_file, moderate_noise_output_file, "moderate_noise")
    print(f"Output File: {moderate_noise_output_file}")

    print("Heavy Noise:")
    process_jsonl_file(input_file, heavy_noise_output_file, "heavy_noise")
    print(f"Output File: {heavy_noise_output_file}")


if __name__ == "__main__":
    main()
