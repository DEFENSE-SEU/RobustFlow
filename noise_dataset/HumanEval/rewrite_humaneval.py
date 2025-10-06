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
    system_prompt_requirements = r'''
You are a prompt refiner for HumanEval-style coding tasks. Each item is formatted as:

* a Python function signature and body stub first (e.g., `def foo(...):`),
* immediately followed by a triple-quoted docstring `""" ... """`.

**Rewrite ONLY the text inside each docstring’s triple quotes** to strengthen constraints while keeping it concise and feasible. Do not touch anything outside the quotes. Then output the **modified prompt** (the whole snippet with updated docstrings and all original code otherwise unchanged) wrapped inside `<answer>...</answer>` with nothing else.

## What to strengthen (pick 3–7 items max; be strict but not excessive)

* **Input domain & validation**: define valid types/ranges and how to handle invalid inputs (e.g., `TypeError`/`ValueError`), consistent with the original intent.
* **Output contract**: exact return type/format/invariants (e.g., no leading zeros, pure string vs. number).
* **Complexity/resource bounds**: prefer O(n)/O(n log n) time and O(1)/O(n) extra space when reasonable.
* **Allowed/forbidden operations**: may require or forbid specific standard ops that align with the original (no new heavy deps; don’t add imports).
* **Edge cases**: enumerate representative tricky cases (empties, boundaries, duplicates, negatives, float NaN/inf).
* **Determinism & side-effects**: pure function; no I/O, no randomness, no global state mutation.
* **Precision rules** (when floats involved): tolerance (e.g., `abs(a-b) < 1e-9`), rounding/truncation definition.

## Hard rules

* **Edit scope**: Only modify characters strictly between the matching `"""` delimiters of each docstring.

  * Keep the opening/closing `"""` and their indentation **byte-for-byte**.
  * Do **not** alter function signatures, defaults, decorators, comments, or any code outside docstrings.
* **Examples/Doctests**: If the docstring already contains examples/doctests, keep those lines **verbatim and in the same order**. Add constraints as prose sections (e.g., a short “Constraints:” block) without changing existing examples.
* **Language**: Keep the docstring language consistent with the original (English in → English out; Chinese in → Chinese out).
* **Length**: Keep the rewritten docstring clear and brief (typically ≤ 120–150 words or original length + \~30%).
* **Feasibility first**: Only add constraints solvable under the given stub and standard library already implied; do not add imports.
* **Deduplicate**: If constraints already exist, refine/clarify rather than repeat.
* **No extra content**: Output only the modified prompt wrapped in `<answer>...</answer>`.

## Output format (mandatory)

Output exactly:

```
<answer>
<the original code with only the docstring texts rewritten; all code outside docstrings unchanged byte-for-byte>
</answer>
```

If there is nothing meaningful to tighten, minimally clarify the task (still only within docstrings) and follow the format above.

## Example (positive)

From:
def incr_list(l: list):
    """Return list with elements incremented by 1.
    >>> incr_list([1, 2, 3])
    [2, 3, 4]
    """

To:
<answer>
def incr_list(l: list):
    """Return a new list where each element equals the corresponding input element plus 1.

    Constraints:
    - Input: l is a list of integers; non-integer elements should raise TypeError.
    - Behavior: do not modify the input list; no I/O or randomness.
    - Complexity: O(n) time and O(n) extra space (for the returned list).

    >>> incr_list([1, 2, 3])
    [2, 3, 4]
    """
</answer>
    '''

    user_prompt_requirements = """
You are given a HumanEval-style snippet where each function is immediately followed by a triple-quoted docstring.

Task: Rewrite ONLY the natural-language text inside each docstring’s triple quotes to strengthen constraints while keeping them concise and feasible. Everything outside the quotes must remain byte-for-byte identical (function signatures, indentation, quotes, and any code).

Strengthen 3–7 aspects (be strict but not excessive): input domain & validation; output contract; complexity/resource bounds; allowed/forbidden operations (no new imports/heavy deps); edge cases; determinism & side-effects; float precision rules if applicable. If examples/doctests are present in the docstring, keep those lines verbatim and in the same order.

Language must match the original. Keep docstrings brief (typically ≤120–150 words or original length + ~30%). Do not add content beyond the docstrings. Do not modify any code.

Output format (mandatory):
<answer>
<the original snippet with only docstring texts rewritten; all code outside docstrings unchanged>
</answer>

Original prompt:
{{original_prompt}}
    """

    system_prompt_paraphrasing = r'''
You are a prompt rewriter for HumanEval-style items. Each item is a Python function signature/stub followed by a triple-quoted docstring.

Task
- Rewrite ONLY the natural-language prose inside each docstring, i.e., characters strictly between the matching triple quotes `""" ... """`.
- Preserve language and meaning; change only the *form* (voice, sentence mood, order, register, nominalization, near-synonyms).
- Keep length roughly within ±20% of the original prose.

Strict preservation
- Do NOT modify anything outside the docstring text: function names, parameters, annotations, defaults, decorators, comments, whitespace, blank lines, and order must remain byte-for-byte.
- Inside the docstring:
  - Keep existing examples/doctests **verbatim and in the same order**.
  - Preserve identifiers and inline code/literals exactly (function/parameter names, numbers, regexes, paths, URLs, special tokens like `<...>`, `{...}`, `$...`).
  - Keep the opening/closing `"""` and indentation unchanged.

Allowed transformations (light touch)
- Voice (active ↔ passive), sentence mood (imperative/declarative/interrogative).
- Information order and sentence structure (prose ↔ brief bullets).
- Register/tonality (slightly more formal or plain).
- Nominalization vs. verbal phrasing.
- Strict near-synonyms in the **same language**.

Do NOT
- Do not translate or switch languages.
- Do not add/remove constraints, examples, tests, or requirements.
- Do not change specificity, scope, difficulty, or semantics.

Feasibility & tie-breakers
- If a rewrite risks meaning drift, prefer the closest paraphrase or leave the sentence unchanged.
- If the description is already minimal/clear, make only cosmetic edits.

## Output format (mandatory):
<answer>
<the original code with only the docstring prose rewritten; all code and doctests unchanged byte-for-byte>
</answer>

## Example (positive)

From:
def incr\_list(l: list):
"""Return list with elements incremented by 1.
\>>> incr\_list(\[1, 2, 3])
\[2, 3, 4]
\>>> incr\_list(\[5, 3, 5, 2, 3, 3, 9, 0, 123])
\[6, 4, 6, 3, 4, 4, 10, 1, 124]
"""

To:
<answer>
def incr\_list(l: list):
"""Produce a list where each element equals the corresponding input value plus 1.
\>>> incr\_list(\[1, 2, 3])
\[2, 3, 4]
\>>> incr\_list(\[5, 3, 5, 2, 3, 3, 9, 0, 123])
\[6, 4, 6, 3, 4, 4, 10, 1, 124]
""" 
</answer>
    '''

    user_prompt_paraphrasing = r'''
You are given a HumanEval-style Python snippet where each function is immediately followed by a triple-quoted docstring.

Task: Rewrite ONLY the natural-language prose inside each docstring’s triple quotes `""" ... """` to change its form (voice, sentence mood, order, register, nominalization), while preserving language and meaning. Keep length within ±20% of the original prose. If rewriting risks meaning drift or there is no paraphrasable prose, keep the original prose.

Strict preservation:
- Do not modify anything outside the docstrings: code, signatures, annotations, defaults, decorators, comments, whitespace, blank lines, and order must remain byte-for-byte.
- Inside docstrings, keep examples/doctests **verbatim and in the same order**.
- Preserve identifiers and inline code/literals exactly (function/parameter names, numbers, regexes, paths, URLs, special tokens like `<...>`, `{...}`, `$...`).
- Do not add/remove constraints, examples, or requirements. Do not translate.

Output format (mandatory):
<answer>
<the original snippet with only the docstring prose rewritten; doctests and all code outside docstrings unchanged>
</answer>

Original prompt:
{{original_prompt}}
    '''

    system_prompt_light_noise = r'''
You are an ultra-colloquial prompt noiser for HumanEval-style items. Each item is a Python function signature/stub followed by a triple-quoted docstring.

Task
- Inject **mild noise, human-like** edits into the docstring’s **natural-language prose only**, i.e., characters strictly between the matching triple quotes `""" ... """`.
- **Do not modify doctest/example lines** inside the docstring (e.g., lines starting with `>>>` and their output lines) — keep them verbatim and in the same order.
- Do not touch anything outside the docstring. Output the full snippet with only the prose noised, wrapped in `<answer>...</answer>`.

Style goal
- Make the prose casual and slightly informal, but still readable and clear — minor spelling errors, contractions, and slight rearrangements of words. Avoid heavy slang, and aim for a conversational tone.
- **Noise palette (mild):** 
    - Minor **typos & misspellings**: Insert, delete, or transpose a few letters.
    - **Contractions**: Use contractions like “you’re” instead of “you are”.
    - **Slang & IM speak**: Light instances of colloquial terms, like “kinda”, “like”, “tho”.
    - **Vowel stretching**: Limited, e.g., "sooo", "really".
    - **Minor punctuation changes**: Slight modifications like unnecessary commas or extra periods.
    - **Fragments**: A single mild fragment that doesn’t compromise clarity.
    - **Hedges & fillers**: Light fillers like “you know”, “well”, “idk”.

Hard rules (recovery & semantics)
- **Edit scope:** Only prose within `""" ... """`. Keep the opening/closing `"""` and indentation **byte-for-byte**.
- **Doctests/examples:** preserve **verbatim** (no noise in lines starting with `>>>` nor their outputs).
- **Language unchanged:** do not translate or switch languages.
- **Preserve meaning, scope, and specificity.** Do **not** add/remove constraints, examples, or new technical content.
- **Recoverability:** Include at least **one clean, unambiguous full sentence** summarizing the task and at least **one clean mention** of every critical concept/identifier used by the task.

Intensity & limits (mild profile)
- Target **20–40%** tokens noised.
- Use **2–4** noise types overall; ensure **≥50%** of edits are typos/misspellings/keyboard slips.
- Keep length within **±20%** of the original prose.
- Avoid noising **every** instance of a critical term.
- Do **not** insert or remove blank lines **between the docstring and the code**, nor break code fences/indentation.
- If any noise risks semantic drift, tone it down or revert.

## Output format (mandatory)
<answer>
<the original snippet with only the docstring prose noised; doctests/examples and all code outside docstrings unchanged byte-for-byte>
</answer>

## Calibrated example (do not echo at runtime)

From:
def incr\_list(l: list):
"""Return list with elements incremented by 1.
\>>> incr\_list(\[1, 2, 3])
\[2, 3, 4]
\>>> incr\_list(\[5, 3, 5, 2, 3, 3, 9, 0, 123])
\[6, 4, 6, 3, 4, 4, 10, 1, 124]
"""

To: 
<answer>
def incr\_list(l: list):
"""So, uh, just return a list where every element’s bumped by 1, okay? You know, nothing crazy. It’s really simple.
\>>> incr\_list(\[1, 2, 3])
\[2, 3, 4]
\>>> incr\_list(\[5, 3, 5, 2, 3, 3, 9, 0, 123])
\[6, 4, 6, 3, 4, 4, 10, 1, 124]
""" 
</answer>
    '''

    user_prompt_light_noise = r'''
You are given a HumanEval-style Python snippet: a function signature/stub followed by a triple-quoted docstring.

Task: Inject **mild noise, ultra-colloquial** edits into the docstring’s **natural-language prose only** (strictly within the `""" ... """` bounds), while preserving the original **language and meaning**. Do **not** modify anything outside the docstring.

Strict preservation:
- Keep the opening/closing `"""` and indentation **byte-for-byte**.
- **Do not noise doctests/examples** inside the docstring (e.g., lines starting with `>>>` and their output lines) — keep them verbatim and in the same order.
- Preserve technical literals exactly in the prose: inline code/backticks, identifiers (function/parameter names), regexes, numbers/units, file paths, URLs, and special tokens like `<...>`, `{...}`, `$...`.
- Do not translate or change scope/specificity; do not add/remove constraints or examples.

Noise profile (mild):
- Intensity: target **20–40%** tokens noised (≤45% if still readable).
- Mix **2–4** noise types; ensure **≥50%** of edits are typos/misspellings/keyboard slips.
- Allowed noise (prose only): slang/IM speak, contractions, dropped/duplicated short words, vowel stretching & stutters, hedges/fillers, random casing & punctuation chaos (aim 3–5 anomalies/paragraph), sparse leet swaps, emojis ≤3/docstring, symbol runs `&^%$#@~` ≤10 chars (≤2/paragraph, clause boundaries only), one fragment and/or a run-on.
- Digits & comparatives remain exact (“longer/higher/compared to/than/between” unchanged).
- Keep **at least one clean, unambiguous full sentence** summarizing the task and **one clean mention** of each critical concept/identifier.
- Keep length within **±20%** of the original prose.

Output format (mandatory):
<answer>
<the original snippet with only the docstring prose noised; doctests/examples and all code outside docstrings unchanged byte-for-byte>
</answer>

Original prompt:
{original_prompt}
    '''

    system_prompt_moderate_noise = r'''
You are an ultra-colloquial prompt noiser for HumanEval-style items. Each item is a Python function signature/stub followed by a triple-quoted docstring.

Task
- Inject **moderate noise, human-like** edits into the docstring’s **natural-language prose only**, i.e., characters strictly between the matching triple quotes `""" ... """`.
- **Do not modify doctest/example lines** inside the docstring (e.g., lines starting with `>>>` and their output lines) — keep them verbatim and in the same order.
- Do not touch anything outside the docstring. Output the full snippet with only the prose noised, wrapped in `<answer>...</answer>`.

Style goal
- Make the prose informal and conversational with moderate changes — light slang, occasional stutters, mild punctuation anomalies — **still recoverable by a grader**.
- **Noise palette (moderate):**
    - **Typos & misspellings**: Insert or transpose letters, but not too chaotic.
    - **Contractions**: Use contractions like “didn’t”, “wouldn’t”, “you’re”.
    - **Slang & IM speak**: Light usage of colloquial terms like “kinda”, “you know”, “basically”.
    - **Vowel stretching & stutter**: Mild instances, e.g., "reaally", "sooo", "y-yeah".
    - **Punctuation changes**: Minor punctuation changes, like redundant commas or periods.
    - **Fragments & fillers**: A few light fillers like “um”, “you know”, but keep overall clarity.
    - **Random casing**: Occasional shifts, like "Random", "rANdom", etc.

Hard rules (recovery & semantics)
- **Edit scope:** Only prose within `""" ... """`. Keep the opening/closing `"""` and indentation **byte-for-byte**.
- **Doctests/examples:** preserve **verbatim** (no noise in lines starting with `>>>` nor their outputs).
- **Language unchanged:** do not translate or switch languages.
- **Preserve meaning, scope, and specificity.** Do **not** add/remove constraints, examples, or new technical content.
- **Digits & comparatives stay exact** (“longer/higher/compared to/than/between” unchanged).
- **Protect verbatim technical literals in prose**: inline code/backticks, identifiers (function/parameter names), regexes, numbers/units, file paths, URLs, and special tokens like `<...>`, `{...}`, `$...` must be byte-for-byte unchanged.
- **Recoverability:** Include at least **one clean, unambiguous full sentence** summarizing the task and at least **one clean mention** of every critical concept/identifier used by the task.

Intensity & limits (moderate profile)
- Target **40–60%** tokens noised.
- Use **3–5** noise types overall; ensure **≥60%** of edits are typos/misspellings/keyboard slips.
- Keep length within **±20%** of the original prose.
- Avoid noising **every** instance of a critical term.
- Do **not** insert or remove blank lines **between the docstring and the code**, nor break code fences/indentation.
- If any noise risks semantic drift, tone it down or revert.

## Output format (mandatory)
<answer>
<the original snippet with only the docstring prose noised; doctests/examples and all code outside docstrings unchanged byte-for-byte>
</answer>

## Calibrated example (do not echo at runtime)

From:
def incr\_list(l: list):
"""Return list with elements incremented by 1.
\>>> incr\_list(\[1, 2, 3])
\[2, 3, 4]
\>>> incr\_list(\[5, 3, 5, 2, 3, 3, 9, 0, 123])
\[6, 4, 6, 3, 4, 4, 10, 1, 124]
"""

To: 
<answer>
def incr\_list(l: list):
"""Okay, so like, return a list where everything’s just +1, simple enough, right? Don’t mess with the original list tho! 🙏 Just make a new one. One clean sentence: Return a list with each element incremented by 1.
\>>> incr\_list(\[1, 2, 3])
\[2, 3, 4]
\>>> incr\_list(\[5, 3, 5, 2, 3, 3, 9, 0, 123])
\[6, 4, 6, 3, 4, 4, 10, 1, 124]
""" 
</answer>
    '''

    user_prompt_moderate_noise = r'''
You are given a HumanEval-style Python snippet: a function signature/stub followed by a triple-quoted docstring.

Task: Inject **moderate noise, ultra-colloquial** edits into the docstring’s **natural-language prose only** (strictly within the `""" ... """` bounds), while preserving the original **language and meaning**. Do **not** modify anything outside the docstring.

Strict preservation:
- Keep the opening/closing `"""` and indentation **byte-for-byte**.
- **Do not noise doctests/examples** inside the docstring (e.g., lines starting with `>>>` and their output lines) — keep them verbatim and in the same order.
- Preserve technical literals exactly in the prose: inline code/backticks, identifiers (function/parameter names), regexes, numbers/units, file paths, URLs, and special tokens like `<...>`, `{...}`, `$...`.
- Do not translate or change scope/specificity; do not add/remove constraints or examples.

Noise profile (moderate):
- Intensity: target **40–60%** tokens noised (≤65% if still readable).
- Use **3–5** noise types overall; ensure **≥60%** of edits are typos/misspellings/keyboard slips.
- Allowed noise (prose only): slang/IM speak, contractions, dropped/duplicated short words, vowel stretching & stutters, hedges/fillers, random casing & punctuation chaos (aim 3–5 anomalies/paragraph), sparse leet swaps, emojis ≤3/docstring, symbol runs `&^%$#@~` ≤10 chars (≤2/paragraph, clause boundaries only), one fragment and/or a run-on.
- Digits & comparatives remain exact (“longer/higher/compared to/than/between” unchanged).
- Keep **at least one clean, unambiguous full sentence** summarizing the task and **one clean mention** of each critical concept/identifier.
- Keep length within **±20%** of the original prose.

Output format (mandatory):
<answer>
<the original snippet with only the docstring prose noised; doctests/examples and all code outside docstrings unchanged byte-for-byte>
</answer>

Original prompt:
{original_prompt}
    '''

    system_prompt_heavy_noise = r'''
You are an ultra-colloquial prompt noiser for HumanEval-style items. Each item is a Python function signature/stub followed by a triple-quoted docstring.

Task
- Inject **high-noise, human-like** edits into the docstring’s **natural-language prose only**, i.e., characters strictly between the matching triple quotes `""" ... """`.
- **Do not modify doctest/example lines** inside the docstring (e.g., lines starting with `>>>` and their output lines) — keep them verbatim and in the same order.
- Do not touch anything outside the docstring. Output the full snippet with only the prose noised, wrapped in `<answer>...</answer>`.

Style goal
- Make the prose chatty, messy, and almost unrecognizable at a glance — slang, stutters, stretched vowels, punctuation chaos — **yet still recoverable** by a grader.

Noise palette (ultra-colloquial, heavier)
- **Typos & misspellings (primary):** insert/delete/substitute/transpose; letter doubling/drops; adjacent-key slips. **≥65% of all edits** should be from this class. Keep at least **one clean mention** of each critical concept.
- **Slang & IM speak:** uh/erm/lemme/gonna/wanna/tbh/ngl/low-key/high-key/lol/bruh/tho/bc/vs/BTW.
- **Contractions & drop words:** drop light auxiliaries/articles/preps if meaning stays clear; heavy contractions allowed.
- **Vowel stretching & stutter:** “reaaally”, “y-yeah”, “hiiigher”.
- **Hedges & fillers:** like, kinda, sorta, basically, idk?, brief asides.
- **Casing & punctuation chaos:** rANdoM caps; !!!??!?..; duplicated/missing commas/spaces; aim for **3–6 anomalies** per paragraph.
- **Leet/character swaps:** 0↔o, 1↔l, i↔l, sparingly.
- **Emojis & emoticons:** ≤3 per docstring.
- **Random symbol runs:** `&^%$#@~` etc., each **≤10 chars**, **≤2 runs** per paragraph; only at clause boundaries.
- **Fragments & run-ons:** allow one fragment and/or a run-on; keep **one clean full sentence** intact.
- **Mild paraphrase:** reorder words; near-synonyms that don’t change specificity.

Hard rules (recovery & semantics)
- **Edit scope:** Only prose within `""" ... """`. Keep the opening/closing `"""` and indentation **byte-for-byte**.
- **Doctests/examples:** preserve **verbatim** (no noise in lines starting with `>>>` nor their outputs).
- **Language unchanged:** do not translate or switch languages.
- **Preserve meaning, scope, and specificity.** Do **not** add/remove constraints, examples, or new technical content.
- **Digits & comparatives stay exact:** never change numeric values or words like “longer/higher/compared to/than/between”.
- **Protect verbatim technical literals in prose:** inline code/backticks, identifiers (function/parameter names), regexes, numbers/units, file paths, URLs, and special tokens (`<...>`, `{...}`, `$...`) must be byte-for-byte unchanged.
- **Recoverability:** include at least **one clean, unambiguous full sentence** summarizing the task and at least **one clean mention** of every critical concept/identifier used by the task.

Intensity & limits (strong profile)
- Target **60–80%** tokens noised; allow up to **85%** if still readable.
- Use **4–6** noise types overall; you may stack **≤3 edits per token** (e.g., typo + casing + elongation).
- Keep length within **±30%** of the original prose.
- Avoid noising **every** instance of a critical term.
- Do **not** insert or remove blank lines **between the docstring and the code**, nor break code fences/indentation.
- If any noise risks semantic drift, tone it down or revert.

## Output format (mandatory)
<answer>
<the original snippet with only the docstring prose noised; doctests/examples and all code outside docstrings unchanged byte-for-byte>
</answer>

## Calibrated example (do not echo at runtime)

From:
def incr\_list(l: list):
"""Return list with elements incremented by 1.
\>>> incr\_list(\[1, 2, 3])
\[2, 3, 4]
\>>> incr\_list(\[5, 3, 5, 2, 3, 3, 9, 0, 123])
\[6, 4, 6, 3, 4, 4, 10, 1, 124]
"""

To: 
<answer>
def incr\_list(l: list):
"""uhh sooo like, make a list that is the same order but each value is +1, ok? reaaally basic lol — keep it a NEW list, not messin’ the input 🙏 One clean sentence: Return a list with each element incremented by 1.
\>>> incr\_list(\[1, 2, 3])
\[2, 3, 4]
\>>> incr\_list(\[5, 3, 5, 2, 3, 3, 9, 0, 123])
\[6, 4, 6, 3, 4, 4, 10, 1, 124]
""" 
</answer>
    '''

    user_prompt_heavy_noise = r'''
You are given a HumanEval-style Python snippet: a function signature/stub followed by a triple-quoted docstring.

Task: Inject **high-noise, ultra-colloquial** edits into the docstring’s **natural-language prose only** (strictly within the `""" ... """` bounds), while preserving the original **language and meaning**. Do **not** modify anything outside the docstring.

Strict preservation:
- Keep the opening/closing `"""` and indentation **byte-for-byte**.
- **Do not noise doctests/examples** inside the docstring (e.g., lines starting with `>>>` and their output lines) — keep them verbatim and in the same order.
- Preserve technical literals exactly in the prose: inline code/backticks, identifiers (function/parameter names), regexes, numbers/units, file paths, URLs, and special tokens like `<...>`, `{...}`, `$...`.
- Do not translate or change scope/specificity; do not add/remove constraints or examples.

Noise profile (strong):
- Intensity: target **60–80%** tokens noised (≤85% if still readable).
- Mix **4–6** noise types; ensure **≥65%** of edits are typos/misspellings/keyboard slips.
- Allowed noise (prose only): slang/IM speak, contractions, dropped/duplicated short words, vowel stretching & stutters, hedges/fillers, random casing & punctuation chaos (aim 3–6 anomalies/paragraph), sparse leet swaps, emojis ≤3/docstring, symbol runs `&^%$#@~` ≤10 chars (≤2/paragraph, clause boundaries only), one fragment and/or a run-on.
- Digits & comparatives remain exact (“longer/higher/compared to/than/between” unchanged).
- Keep **at least one clean, unambiguous full sentence** summarizing the task and **one clean mention** of each critical concept/identifier.
- Keep length within **±30%** of the original prose.

Output format (mandatory):
<answer>
<the original snippet with only the docstring prose noised; doctests/examples and all code outside docstrings unchanged byte-for-byte>
</answer>

Original prompt:
{original_prompt}
    '''

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

        p = data.get("prompt", None)

        if isinstance(p, str) and p:
            original_prompt = p

            rewritten = rewrite_prompt_with_openai(original_prompt, mode)
            
            m = TAG.search(rewritten)
            new_prompt = m.group(1) if m else rewritten

            data["prompt"] = new_prompt
        else:
            print(f"[Warning] Line {i+1} missing processable 'prompt' field, keeping original.")

        processed_data.append(data)
        
        time.sleep(0.1)

    with open(output_file, "w", encoding="utf-8") as f:
        for obj in processed_data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Processing completed! Processed {len(processed_data)} data entries")


def main():
    input_file = "humaneval_original.jsonl"
    
    requirements_output_file = "humaneval_requirements.jsonl"
    paraphrasing_output_file = "humaneval_paraphrasing.jsonl"
    light_noise_output_file = "humaneval_light_noise.jsonl"
    moderate_noise_output_file = "humaneval_moderate_noise.jsonl"
    heavy_noise_output_file = "humaneval_heavy_noise.jsonl"

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
