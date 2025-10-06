import json
import os
import time
import re
import sys
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
You are a QA constraint rewriter. Given a single original question sentence **Q**, rewrite it into a **fluent constraint-led instruction** that:

1. begins with constraints (e.g., Use only the given passage, ...),
2. then says **to represent {Q\_without\_trailing\_question\_mark}.** (embed Q verbatim except drop the final ?),
3. contains **no added facts** and **does not change the meaning** of Q,
4. uses **concise, declarative English** (prefer 2~3 sentences total).

## How to construct the instruction (pick 3~7 items that fit Q):

* **Source scope:** Use only the given passage (if a passage/context is provided). Otherwise: Use only the supplied information; no external knowledge.
* **Output format:** e.g., output an integer only (no words), output a single uppercase letter A~D only, output JSON {"answer":"..."} only.
* **Uncertainty policy:** e.g., if insufficient evidence, output unknown.
* **Units & rounding (for numeric):** e.g., in years; integer only; round half up to 0 decimals.
* **Computation rules:** e.g., treat ‘until YYYY’ as YYYY exclusive; include the explicitly tied start year only.
* **Disambiguation:** e.g., choose values explicitly tied to the target entity/event mentioned in Q.
* **Forbidden operations:** e.g., no external knowledge; no fabrication; no chain-of-thought.

## Formatting rules (strict):

* Return **exactly one line**, wrapped as:
  `<answer>{final_instruction_sentence}</answer>`
* Do **not** echo Q separately, do **not** add examples or explanations, do **not** add labels like Constraints: or brackets.
* Use plain ASCII punctuation. Remove only Q’s trailing ?; otherwise keep Q’s wording intact.


**Example**

From:
How many years did the Yamethin rebellion last?

To: 
<answer>Use only the given passage, output an integer only (no words) to represent How many years did the Yamethin rebellion last. If start or end year is missing output unknown, treat "went on until YYYY" as YYYY exclusive, choose years explicitly tied to the Yamethin rebellion.</answer>
    """

    user_prompt_requirements = """
You will receive a single original QA question sentence (Q).

Rewrite Q into a constraint-led instruction that:
1) begins with constraints (e.g., "Use only the given passage, ..."),
2) then says "to represent {Q_without_trailing_question_mark}.",
3) preserves Q’s meaning exactly (no new facts, no reinterpretation),
4) is concise (2~3 sentences total).

When selecting constraints (choose **3~7**, strict but not excessive), prefer:
- Source scope: use only the given passage / supplied information; no external knowledge or guessing.
- Output format: e.g., integer only; single uppercase letter A–D only; JSON {"answer":"..."} only.
- Uncertainty policy: if insufficient evidence, output "unknown".
- Units & rounding (for numeric): specify unit, integer/decimal, rounding mode if needed.
- Computation rules (temporal/range): e.g., treat "until YYYY" as YYYY exclusive; include only explicitly stated start.
- Disambiguation: choose values explicitly tied to the target entity/event in Q.
- Forbidden operations: no fabrication, no chain-of-thought, no citations unless the passage is provided and required.

Hard rules:
- Return exactly one line wrapped as: <answer>{final_instruction}</answer>
- Do not include labels like "Constraints:" or any brackets.
- Use ASCII punctuation. Keep Q verbatim inside "to represent ...", but drop its trailing "?".
- Keep the language consistent with Q (English in → English out; Chinese in → Chinese out).
- No examples, no extra commentary, no leading/trailing whitespace.
- Do not restate Q elsewhere.

Output:
<answer><final_instruction_sentence></answer>

Original question (`original_prompt`):
{{original_prompt}}
    """

    system_prompt_paraphrasing = """
You are a QA prompt rewriter. Given an input that contains a single **question sentence** Q (optionally prefixed with labels like 'Question:'), **rewrite ONLY the question** by changing its *form* (e.g., voice, sentence mood, order, register) while **preserving language and meaning**. Then output the **rewritten question** wrapped inside `<answer>...</answer>` with nothing else.

## Strict preservation
* Keep the **language** unchanged (English→English, Chinese→Chinese).
* Preserve **named entities, numbers, units, dates, math expressions, quoted titles**, and any **inline code/literals/special tokens** (e.g., `<...>`, `{...}`, `$...`, URLs, file paths) **exactly**.
* If the original includes any **constraints, scope notes, or markers** (e.g., use only the passage), keep them verbatim and in place relative to Q’s content.
* Do not introduce or remove information, premises, or assumptions. Do not change the target being asked about.

## Allowed transformations (light touch)
* Sentence mood: interrogative ↔ imperative/declarative (e.g., What is …? → State …).
* Information order and sentence structure (simple/complex; one sentence preferred).
* Register/tonality: slightly more formal or plain.
* Near-synonyms in the **same language** that do not alter specificity.
* Length roughly within **±20%** of the original question.

## Do NOT
* Do **not** translate or switch languages.
* Do **not** add/remove **constraints**, hints, options, or examples.
* Do **not** change specificity or scope (e.g., don’t replace years with months).
* Do **not** add chain-of-thought, explanations, or any extra lines.

## Feasibility & tie-breakers
* If a rewrite risks meaning drift, choose the closest paraphrase or keep the original wording with minimal cosmetic edits.
* Prefer a single, self-contained sentence that a grader can parse unambiguously.

## Output format (mandatory)
Return **exactly**:
<answer><rewritten question here></answer>

### Example (positive)
From:
How many years did the Yamethin rebellion last?

To:
<answer>State the number of years that the Yamethin rebellion lasted.</answer>
    """

    user_prompt_paraphrasing = """
You will receive a single original QA question sentence (Q).

Paraphrase ONLY the question’s form (voice, sentence mood, order, register) while preserving language and meaning. Output the rewritten question wrapped inside `<answer>...</answer>` and nothing else.

Strict requirements:
- Keep the language unchanged (English→English, Chinese→Chinese).
- Preserve named entities, numbers, units, dates, math expressions, quoted titles, and any inline code/literals/special tokens (e.g., `<...>`, `{...}`, `$...`, URLs, file paths) exactly.
- If the original includes labels or scope notes (e.g., Question:, use only the passage), keep them in place.
- Keep length within ±20% of the original question.
- Do not add/remove constraints, hints, options, or examples.
- Do not change specificity or scope; do not introduce new facts.

Allowed transformations (light touch):
- Interrogative ↔ imperative/declarative phrasing.
- Reordering or simplifying/complexifying the sentence structure.
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
You are a QA prompt noiser. Given a single **question sentence** Q (optionally prefixed with labels like 'Question:'), inject **light, colloquial, low-noise** edits into the question **only**, with a **mild bias toward typos/misspellings**, while **preserving the original meaning and language**. Then output the **noised question** wrapped inside `<answer>...</answer>` with nothing else.

## Style goal
Make Q look slightly chatty and casual—with a few soft typos, small colloquial touches, and gentle punctuation quirks—**but fully readable** and easy to recover by a grader.

## Noise palette (colloquial, lighter)
* **Typos & misspellings (primary):** minor insert/delete/substitute/transpose; small letter doubling/drops. Target **≥50%** of edits from this class.  
  – Allow mild forms like “realy/longr/how many/years” etc.  
* **Slang & IM speak (sparingly):** uh/erm/okay/ kinda/tbh/tho/BTW — at most a couple of short bits.
* **Contractions & light drops:** natural contractions; at most **one** light function-word drop if intent stays clear.
* **Vowel stretching & stutter (very light):** a single mild case like “reaaally” or “y-yeah” at most once.
* **Hedges & fillers:** optional small aside (e.g., “kinda”, “idk” used cautiously).
* **Casing & punctuation quirks:** allow **1–2** anomalies total (e.g., one extra comma, a doubled “?”).
* **Keyboard slips:** occasional adjacent-key typo, at most once.
* **Leet/character swaps:** avoid unless minimal; at most once (e.g., 0↔o) and only if still clear.
* **Emojis & emoticons:** **≤1** per question.
* **Random symbol run (optional):** at most **1** run, **≤6** chars, only at a clause boundary.
* **Fragments & run-ons:** avoid; keep the sentence grammatical. A short parenthetical aside is okay.
* **Mild paraphrase:** tiny reordering or near-synonyms that don’t change specificity.

## Hard rules (recovery & semantics)
* **Keep language unchanged. Do not translate.**
* **Preserve question type and intent.** Keep an **anchor** for the interrogative form: include **one clean occurrence** of the core cue (e.g., “How many”, “Did”, “Which”, “Where”, “Who”). Other instances may be lightly noised.
* **Digits & comparatives stay exact:** never change numeric **values** or comparative/connective words like “longer/higher/compared to/than/between”.  
  – Common units/nouns may be lightly noised (“yrs”, “yard(s)”) **only if at least one clean, recognizable occurrence remains**.
* **Protect verbatim:** quoted titles, URLs, file paths, inline code/literals, and special tokens (`<...>`, `{...}`, `$...`) must be byte-for-byte unchanged.
* **Named entities:** keep **at least one clean mention** of each proper noun (e.g., “Yamethin”, “Belgium”, “Germany”). Other mentions may be lightly noised.
* **No extra content:** do not add hints/options/constraints/explanations or chain-of-thought; do not invent/remove labels. If “Question:” exists, you may keep it.
* Output **one line** inside the wrapper; no leading/trailing whitespace outside.

## Intensity & limits (light profile)
* Target **15–30%** tokens noised; allow up to **35%** if still easily readable.
* Use **2–3** noise types overall; you may stack **≤2 edits per token** (e.g., tiny typo + light casing).
* Keep length within **±15%** of the original.
* Avoid noising **every** instance of a critical term; ensure recognizability via the clean anchor & clean proper-noun mentions.

## Output format (mandatory)
Return **exactly**:
<answer><noised question here></answer>

### Calibrated examples (do not echo at runtime)
From:
Question: How many years did the Yamethin rebellion last?
To:
<answer>Question: How many yars did the Yamethin rebellion last, kinda?</answer>

From:
Question: How many yards longer was Rob Bironas' longest field goal compared to John Carney's only field goal?
To:
<answer>Question: How many yards long was Rob Bironas' longest field g0al, compared to John Carney's only field-goal tho?</answer>

From:
Question: Did "My Love is Your Love" rank higher in Belgium or Germany?
To:
<answer>Question: Do "My Love is Your Love" ranking highger in Belg1um or Germany, tbh?</answer>
    """

    user_prompt_light_noise = """
You will receive a single QA question sentence (Q).

Make Q lightly colloquial and a bit messy (small slang, mild typos, gentle punctuation quirks) while preserving its meaning and language. Keep one clean interrogative anchor (“How many/Did/Which/Where/Who…”) and one clean mention of each proper noun. Output the noised question wrapped inside `<answer>...</answer>` and nothing else.

Targets (light):
- Intensity: noise **15–30%** of tokens (allow up to **35%** if still easily readable).
- Edit mix: use **2–3** noise types; you may stack **≤2 edits** on the same token.
- Typos bias: **≥50%** of edits are typos/misspellings/keyboard slips.
- Digits & comparative/connective words stay exact (e.g., **longer/higher/compared to/than/between**).
- Units/common nouns may be lightly noised (e.g., **years→yrs**), but keep at least one recognizable occurrence.
- Quoted titles/URLs/code/special tokens must remain **verbatim**.
- Named entities: keep **≥1 clean** occurrence per entity.
- Casing/punctuation quirks allowed (**1–2** anomalies); **≤1** emoji; **≤1** short symbol run (**≤6** chars). No blank lines.
- Length within **±15%** of the original.
- Produce **one line** inside the wrapper; no extra commentary or whitespace.

Do NOT:
- Translate or change meaning/scope/question type.
- Alter digits/values, invent/remove labels, or add hints/options/constraints/explanations.
- Include chain-of-thought.

Return exactly:
<answer><noised question here></answer>

Original question (`original_prompt`):
{{original_prompt}}
    """

    system_prompt_moderate_noise = """
You are a QA prompt noiser. Given a single **question sentence** Q (optionally prefixed with labels like 'Question:'), inject **colloquial, medium-noise** edits into the question **only**, with a **bias toward typos/misspellings**, while **preserving the original meaning and language**. Then output the **noised question** wrapped inside <answer>...</answer> with nothing else.

## Style goal
Make Q look casual, chattier, and a bit messy—some slang, light stutters/stretches, and punctuation/casing quirks—**yet clearly recoverable** by a grader.

## Noise palette (colloquial, medium)
* **Typos & misspellings (primary):** insert/delete/substitute/transpose, light letter doubling/drops. Target **≥60%** of edits from this class.  
  – Allow “reallly/longr/how manyy/yeaars” etc.
* **Slang & IM speak:** uh/erm/lemme/gonna/wanna/tbh/ngl/low-key/high-key/lol/bruh/tho/bc/vs/BTW.
* **Contractions & drop words:** drop auxiliaries/articles/preps where intent stays clear; moderate use of contractions.
* **Vowel stretching & stutter:** “reaaally”, “y-yea”, “hiiigher”, “yarrds”.
* **Hedges & fillers:** like, kinda, sorta, basically, idk? (without changing meaning), quick asides.
* **Casing & punctuation chaos:** rANdoM caps, !!!??!?.., duplicated/missing commas/spaces; allow **2–4** anomalies.
* **Keyboard slips:** adjacent-key hits, stray shift.
* **Leet/character swaps:** 0↔o, 1↔l, i↔l, sparingly.
* **Emojis & emoticons:** 🤔🙂😅 etc., **≤2** per question.
* **Random symbol runs:** &^%$#@~ style, each **≤8** chars, **≤2** runs total; only at clause boundaries.
* **Fragments & run-ons:** permit one mild fragment and/or a short run-on; keep **one clean clause** intact.
* **Mild paraphrase:** reorder words; near-synonyms that don’t change specificity (e.g., “rank higher” → “place higher”).

## Hard rules (recovery & semantics)
* **Keep language unchanged. Do not translate.**
* **Preserve question type and intent.** Keep an **anchor** for the interrogative form: include **one clean occurrence** of the core cue (e.g., “How many”, “Did”, “Which”, “Where”, “Who”). Other instances may be noised.
* **Digits & comparatives stay exact:** never change numeric **values** or comparative/connective words like “longer/higher/compared to/than/between”.  
  – Common units/nouns (years, yards, field goal) **may** be noised/shortened (“yrs”, “yarrds”) **if at least one clean occurrence of the unit or a clearly recognizable form remains**.
* **Protect verbatim:** quoted titles, URLs, file paths, inline code/literals, and special tokens (<...>, {...}, $...) must be byte-for-byte unchanged.
* **Named entities:** keep **at least one clean mention** of each proper noun (e.g., “Yamethin”, “Belgium”, “Germany”). Other mentions may be noised.
* **No extra content:** do not add hints/options/constraints/explanations or chain-of-thought; do not invent/remove labels. If “Question:” exists, you may keep it.
* Output **one line** inside the wrapper; no leading/trailing whitespace outside.

## Intensity & limits (medium profile)
* Target **35–55%** tokens noised; allow up to **60%** if still readable.
* Use **3–5** noise types overall; you may stack **≤2 edits per token** (e.g., typo + casing).
* Keep length within **±20%** of the original.
* Avoid noising **every** instance of a critical term; ensure recognizability via the clean anchor & clean proper-noun mentions.

## Output format (mandatory)
Return **exactly**:
<answer><noised question here></answer>

### Calibrated examples (do not echo at runtime)
From:
Question: How many years did the Yamethin rebellion last?
To:
<answer>Question: uh How many years did the Yamethin rebellion last, kinda??</answer>

From:
Question: How many yards longer was Rob Bironas' longest field goal compared to John Carney's only field goal?
To:
<answer>Question: ngl how many yarrds longer was Rob Bironas' longest field goal vs John Carney's only field goal? 🙂</answer>

From:
Question: Did "My Love is Your Love" rank higher in Belgium or Germany?
To:
<answer>Question: tbh did "My Love is Your Love" rank higher in Belgium or Germany ??</answer>
    """

    user_prompt_moderate_noise = """
You will receive a single QA question sentence (Q).

Make Q colloquial and moderately messy (chatty slang, some typos, light vowel stretches, punctuation/casing quirks) while preserving its meaning and language. Keep one clean interrogative anchor (“How many/Did/Which/Where/Who…”) and one clean mention of each proper noun. Output the noised question wrapped inside <answer>...</answer> and nothing else.

Targets (medium):
- Intensity: noise **35–55%** of tokens (allow up to **60%** if still readable).
- Edit mix: use **3–5** noise types; you may stack **≤2 edits** on the same token.
- Typos bias: **≥60%** of edits are typos/misspellings/keyboard slips.
- Digits & comparative/connective words stay exact (e.g., **longer/higher/compared to/than/between**).
- Units/common nouns may be noised (e.g., **years→yeaars**), but keep at least one recognizable occurrence.
- Quoted titles/URLs/code/special tokens must remain **verbatim**.
- Named entities: keep **≥1 clean** occurrence per entity.
- Casing/punctuation chaos allowed (**2–4** anomalies); **≤2** emojis; **≤2** symbol runs (each **≤8** chars). No blank lines.
- Length within **±20%** of the original.
- Produce **one line** inside the wrapper; no extra commentary or whitespace.

Do NOT:
- Translate or change meaning/scope/question type.
- Alter digits/values, invent/remove labels, or add hints/options/constraints/explanations.
- Include chain-of-thought.

Return exactly:
<answer><noised question here></answer>

Original question (original_prompt):
{{original_prompt}}
    """

    system_prompt_heavy_noise = """
You are a QA prompt noiser. Given a single **question sentence** Q (optionally prefixed with labels like 'Question:'), inject **ultra-colloquial, high-noise** edits into the question **only**, with a **strong bias toward typos/misspellings**, while **preserving the original meaning and language**. Then output the **noised question** wrapped inside `<answer>...</answer>` with nothing else.

## Style goal
Make Q look chatty, messy, and almost unrecognizable at a glance—full of slang, stutters, stretched vowels, and punctuation chaos—**yet still recoverable** by a grader.

## Noise palette (ultra-colloquial, heavier)
* **Typos & misspellings (primary):** insert/delete/substitute/transpose, letter doubling/drops. Target **≥65%** of edits from this class.  
  – Allow “reallly/loonger/how manyy/yeaars” etc.  
* **Slang & IM speak:** uh/erm/lemme/gonna/wanna/tbh/ngl/low-key/high-key/lol/bruh/tho/bc/vs/BTW.  
* **Contractions & drop words:** drop auxiliaries/articles/preps where intent stays clear; heavy use of contractions.
* **Vowel stretching & stutter:** “reaaally”, “y-yea”, “hiiigher”, “yarrds”.
* **Hedges & fillers:** like, kinda, sorta, basically, idk? (without changing meaning), quick asides.
* **Casing & punctuation chaos:** rANdoM caps, !!!??!?.., duplicated/missing commas/spaces; allow **3–6** anomalies.
* **Keyboard slips:** adjacent-key hits, stray shift.
* **Leet/character swaps:** 0↔o, 1↔l, i↔l, sparingly.
* **Emojis & emoticons:** 🤔🙂😅 etc., **≤3** per question.
* **Random symbol runs:** `&^%$#@~` style, each **≤10** chars, **≤2** runs total; only at clause boundaries.
* **Fragments & run-ons:** permit one fragment and/or a run-on; keep **one clean clause** intact.
* **Mild paraphrase:** reorder words; near-synonyms that don’t change specificity (e.g., “rank higher” → “place higher”).

## Hard rules (recovery & semantics)
* **Keep language unchanged. Do not translate.**
* **Preserve question type and intent.** Keep an **anchor** for the interrogative form: include **one clean occurrence** of the core cue (e.g., “How many”, “Did”, “Which”, “Where”, “Who”). Other instances may be noised.
* **Digits & comparatives stay exact:** never change numeric **values** or comparative/connective words like “longer/higher/compared to/than/between”.  
  – Common units/nouns (years, yards, field goal) **may** be noised/shortened (“yrs”, “yarrds”) **if at least one clean occurrence of the unit or a clearly recognizable form remains**.
* **Protect verbatim:** quoted titles, URLs, file paths, inline code/literals, and special tokens (`<...>`, `{...}`, `$...`) must be byte-for-byte unchanged.
* **Named entities:** keep **at least one clean mention** of each proper noun (e.g., “Yamethin”, “Belgium”, “Germany”). Other mentions may be noised.
* **No extra content:** do not add hints/options/constraints/explanations or chain-of-thought; do not invent/remove labels. If “Question:” exists, you may keep it.
* Output **one line** inside the wrapper; no leading/trailing whitespace outside.

## Intensity & limits (strong profile)
* Target **60–80%** tokens noised; allow up to **85%** if still readable.
* Use **4–6** noise types overall; you may stack **≤3 edits per token** (e.g., typo + casing + elongation).
* Keep length within **±30%** of the original.
* Avoid noising **every** instance of a critical term; ensure recognizability via the clean anchor & clean proper-noun mentions.

## Output format (mandatory)
Return **exactly**:
<answer><noised question here></answer>

### Calibrated examples (do not echo at runtime)
From:
Question: How many years did the Yamethin rebellion last?
To:
<answer>Question: uh so like HOW manyy yeaars did the Yamethin rebellion last ?? lol &^%$ ... how many years tho</answer>

From:
Question: How many yards longer was Rob Bironas' longest field goal compared to John Carney's only field goal?
To:
<answer>Question: ngl how many yarrds longerr was Rob Bironas' longest field goaal vs John Carney's only field goal ?! 🤔 how many yards</answer>

From:
Question: Did "My Love is Your Love" rank higher in Belgium or Germany?
To:
<answer>Question: tbh did "My Love is Your Love" rank hiher in Belgum or Germany ??? 🙂 Did</answer>
    """

    user_prompt_heavy_noise = """
You will receive a single QA question sentence (Q).

Make Q ultra-colloquial and messy (chatty slang, typos, stretched vowels, punctuation chaos) while preserving its meaning and language. Keep one clean interrogative anchor (“How many/Did/Which/Where/Who…”) and one clean mention of each proper noun. Output the noised question wrapped inside `<answer>...</answer>` and nothing else.

Targets (strong):
- Intensity: noise **60–80%** of tokens (allow up to **85%** if still readable).
- Edit mix: use **4–6** noise types; you may stack **≤3 edits** on the same token.
- Typos bias: **≥65%** of edits are typos/misspellings/keyboard slips.
- Digits & comparative/connective words stay exact (e.g., **longer/higher/compared to/than/between**).
- Units/common nouns may be noised (e.g., **years→yEaars**), but keep at least one recognizable occurrence.
- Quoted titles/URLs/code/special tokens must remain **verbatim**.
- Named entities: keep **≥1 clean** occurrence per entity.
- Casing/punctuation chaos allowed (**3–6** anomalies); **≤3** emojis; **≤2** symbol runs (each **≤10** chars). No blank lines.
- Length within **±30%** of the original.
- Produce **one line** inside the wrapper; no extra commentary or whitespace.

Do NOT:
- Translate or change meaning/scope/question type.
- Alter digits/values, invent/remove labels, or add hints/options/constraints/explanations.
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
    
    for i, line in enumerate(tqdm(lines, desc="Processing Progress")):
        try:
            data = json.loads(line.strip())

            if (
                "context" in data
                and isinstance(data["context"], str)
                and "Question:" in data["context"]
            ):
                ctx = data["context"]

                m = re.search(
                    r"(?s)(.*?\bQuestion:\s*)(.*?)(\s*(?:\n\s*Answer:|$))", ctx
                )
                if not m:
                    print(
                        f"Warning: Line {i+1} failed to locate Question segment, skipping rewrite (keeping original)"
                    )
                    processed_data.append(data)
                    continue

                prefix, q_text, suffix = m.group(1), m.group(2), m.group(3)
                original_question = q_text.strip()

                rewritten = rewrite_prompt_with_openai(original_question, mode)
                
                match = re.search(
                    r"<answer>(.*?)</answer>", rewritten, re.DOTALL | re.IGNORECASE
                )
                new_question = (match.group(1) if match else rewritten).strip()

                data["context"] = f"{prefix}{new_question}{suffix}"

            else:
                print(f"Warning: Line {i+1} has neither 'context' nor 'prompt', skipping rewrite")

            processed_data.append(data)
            
            time.sleep(0.1)

        except json.JSONDecodeError as e:
            print(f"Error: Line {i+1} JSON parsing failed: {e}")
            continue
        except Exception as e:
            print(f"Error: Exception occurred while processing line {i+1}: {e}")
            continue

    with open(output_file, "w", encoding="utf-8") as f:
        for data in processed_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Processing completed! Processed {len(processed_data)} data entries")


def main():
    input_file = "drop_original.jsonl"
    
    requirements_output_file = "drop_requirements.jsonl"
    paraphrasing_output_file = "drop_paraphrasing.jsonl"
    light_noise_output_file = "drop_light_noise.jsonl"
    moderate_noise_output_file = "drop_moderate_noise.jsonl"
    heavy_noise_output_file = "drop_heavy_noise.jsonl"

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