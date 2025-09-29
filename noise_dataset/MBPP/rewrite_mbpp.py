"""
MBPP Dataset Programming Task Rewriting Module

This module provides functionality to rewrite programming tasks from the MBPP dataset using various
transformation modes including requirements augmentation, paraphrasing, and noise injection.
It supports multiple noise levels (light, moderate, heavy) to test model robustness on programming tasks.

The module focuses on rewriting natural language descriptions while preserving code stubs
(imports, function signatures) to maintain the structural integrity of programming tasks.

Author: RobustFlow Team
Date: 2024
Purpose: Generate augmented training data for programming models
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
    Rewrite a MBPP programming task using OpenAI API based on specified transformation mode.
    
    This function takes an original programming task and applies one of five transformation modes:
    - 'requirements': Adds constraint-based instructions to strengthen task requirements
    - 'paraphrasing': Rewrites task description while preserving code stubs
    - 'light_noise': Adds light colloquial noise to task description while keeping code intact
    - 'moderate_noise': Adds moderate colloquial noise to task description with increased complexity
    - 'heavy_noise': Adds heavy colloquial noise to task description while maintaining recoverability
    
    The function uses different system prompts and temperature settings based on the mode.
    For noise modes, higher temperature (0.7) is used for more creative variations.
    For requirements and paraphrasing modes, temperature 0.0 ensures consistency.
    
    Args:
        original_prompt (str): The original programming task to be rewritten
        mode (str): Transformation mode - one of 'requirements', 'paraphrasing',
                   'light_noise', 'moderate_noise', or 'heavy_noise'
        
    Returns:
        str: The rewritten task extracted from the API response
        
    Raises:
        Exception: If API calls fail after retries, returns the original prompt
    """
    # Define system prompt for requirements mode - strengthens programming task constraints
    system_prompt_requirements = """
You are a prompt refiner for coding tasks. Given an original prompt that consists of:

1. a natural-language description, followed by
2. code stubs (e.g., `import ...`, `def ...`),

**rewrite ONLY the description** to strengthen constraints while keeping it concise and feasible. Then output the **modified prompt** (the rewritten description + the original code stubs unchanged) wrapped inside `<answer>...</answer>` with nothing else. Do not invent requirements that conflict with the original.

## What to strengthen (pick 3~7 items max; be strict but not excessive)

* **Input domain & validation**: precisely define valid inputs and how to handle invalid ones (e.g., raise `ValueError` or return a specific value).
* **Output contract**: exact format/invariants, idempotency if relevant.
* **Complexity/resource bounds**: prefer linear time O(n)/O(nlogn) and O(1)/O(n) extra space when reasonable.
* **Allowed/forbidden operations**: require or forbid specific libraries/operations only if compatible with the original prompt (e.g., keep `re` if already mentioned; never add heavy deps), like multiply without using *.
* **Edge cases**: enumerate a few representative tricky cases.
* **Determinism & side-effects**: pure function, no I/O, stable behavior.

## Hard rules

* **Do not modify any code** (imports, function/class names, parameters, signatures, or stubs). Preserve them **byte-for-byte** and in the original order.
* **Keep language** consistent with the original description (English in → English out; Chinese in → Chinese out).
* Keep the rewritten description **clear and brief** (typically ≤ 120~150 words or the original length + \~30%).
* **Feasibility first**: only add constraints that remain realistically solvable under the given stub and standard library.
* If the original description already includes constraints, **deduplicate** and refine rather than repeat.
* **No extra content** beyond the modified prompt; do **not** include examples, tests, explanations, or commentary unless they already existed in the description.

## Output format (mandatory)

* Output exactly:

```
<answer>
<rewritten description here><the original code stubs, unchanged>
</answer>
```

If there is nothing meaningful to tighten, minimally clarify the task, keep feasibility, and still follow the format above.

Example (positive):

From:
Write a function to remove leading zeroes from an ip address.\nimport re\ndef removezero_ip(ip):

To:
<answer>Implement a Python function named `removezero_ip` that takes a string representing an IPv4 address as input and removes any leading zeroes from each segment. The function should return the cleaned IP address as a string. Use the `re` module for pattern matching and substitution.\nimport re\ndef removezero_ip(ip):</answer>
    """

    # Define user prompt for requirements mode - template for constraint-based task augmentation
    user_prompt_requirements = """
You will receive an original programming prompt that typically contains:
1) a natural-language description, then
2) code stubs (e.g., `import ...`, `def ...`).

Rewrite ONLY the description to strengthen constraints, keeping it strict but not excessive. Then output the modified prompt (the rewritten description + the original code stubs unchanged) wrapped inside `<answer>...</answer>`. Output nothing else.

When tightening the description, select **3~7** items (do not overdo it):
- Input domain & validation (precise format/range; how to handle invalid input, e.g., raise `ValueError`).
- Output contract/invariants (exact format, idempotency if relevant).
- Complexity/resource bounds (prefer O(n)/O(nlogn) time; O(1)/O(n) extra space).
- Allowed/forbidden operations (respect any existing library hints; no heavy new deps).
- Representative edge cases (list a few concise, relevant cases).
- Determinism & purity (no I/O; no side effects).

Hard rules:
- Do **not** modify any code stubs—keep them byte-for-byte and in the same order.
- Keep the language consistent with the original (English in → English out; Chinese in → Chinese out).
- Keep the description clear and brief (≈120~150 words max or original length + ~30%).
- If constraints already exist, deduplicate and refine—avoid repetition/conflicts.
- Do not add examples/tests/explanations unless present in the original description.
- Ensure the task remains feasible under standard library usage.

Original prompt (`original_prompt`):
{{original_prompt}}
    """

    # Define system prompt for paraphrasing mode - rewrites task description while preserving code stubs
    system_prompt_paraphrasing = """
You are a prompt rewriter. Given an input that contains:

1. a natural-language **description**, followed by
2. **code stubs** (e.g., `import ...`, `def ...`),

**rewrite ONLY the description** by changing its *form* (e.g., voice, sentence mood, order, register) while **preserving language and meaning**. Then output the **modified prompt** (the rewritten description + the original code stubs unchanged) wrapped inside `<answer>...</answer>` with nothing else.

## Strict preservation

* Do **not** modify any **code stubs**: imports, names, signatures, comments, whitespace, blank lines, or order—preserve **byte-for-byte**.
* Do **not** translate or switch languages.
* In the description, preserve identifiers and inline code/literals exactly (e.g., function/class/parameter names, regexes, numbers, file paths, URLs, and special tokens/placeholders like `<...>`, `{...}`, `$...`).

## Allowed transformations (light touch)

* Voice (active ↔ passive), sentence mood (imperative/declarative/interrogative).
* Information order and sentence structure (simple/complex; prose ↔ brief bullets).
* Register/tonality (slightly more formal or plain).
* Nominalization vs verbal phrasing.
* Strict near-synonyms **in the same language** (English→English, Chinese→Chinese).
  Keep length roughly within ±20% of the original description.

## Do NOT

* Do **not** translate or switch languages.
* Do **not** add/remove constraints, examples, tests, or requirements.
* Do **not** change specificity (e.g., do not introduce IPv4 if the original says IP address).
* Do **not** alter task scope, difficulty, or semantics.
* Do **not** modify any **code stubs** (imports, names, signatures) — keep them **byte-for-byte** and in the same order.

## Feasibility & tie-breakers

* If a rewrite risks meaning drift, prefer the closest paraphrase or keep the original sentence.
* If the description is already minimal/clear, make **no more than cosmetic** edits.

## Output format (mandatory)

```
<answer>
<rewritten description here><the original code stubs, unchanged>
</answer>
```

Example (positive):

From:
Write a function to remove leading zeroes from an ip address.\nimport re\ndef removezero_ip(ip):

To:
<answer>A routine should be provided that strips any leading zeros from an IP address.\nimport re\ndef removezero_ip(ip):</answer>
    """
    
    # Define user prompt for paraphrasing mode - template for task description paraphrasing
    user_prompt_paraphrasing = """
You will receive an original programming prompt that typically contains:
1) a natural-language description, then
2) code stubs (e.g., `import ...`, `def ...`).

Rewrite ONLY the description by changing its form (voice, sentence mood, information order, register, nominalization vs. verbal), while strictly preserving the original language and meaning. Keep edits light (no overdoing). Then output the modified prompt (the rewritten description + the original code stubs unchanged) wrapped inside `<answer>...</answer>`. Output nothing else.

Hard rules:
- Do NOT translate or switch languages.
- Do NOT add/remove constraints, examples, tests, or requirements.
- Do NOT change specificity or task scope (no new terms like “IPv4” if the original says “IP address”).
- Do NOT modify any code stubs (imports, names, parameters, or order) — keep them byte-for-byte.
- Keep the description length within ±20% of the original.
- If a rewrite risks meaning drift, prefer the closest paraphrase or keep the original sentence.

Original prompt (`original_prompt`):
{{original_prompt}}
    """

    # Define system prompt for light noise mode - adds minimal colloquial noise to task description
    system_prompt_light_noise = """
You are a prompt noiser. Given an input that contains:

1) a natural-language **description**, then
2) **code stubs** (e.g., `import ...`, `def ...`),

inject **light, subtle, colloquial noise** into the **description only** (NOT the code), with a **bias toward typos/misspellings**, while **preserving the original meaning and language**. Keep it readable and clearly recoverable by a grader. Then output the **modified prompt** (the noised description + the original code stubs unchanged) wrapped inside `<answer>...</answer>` and nothing else.

## Style goal
Make the description feel a bit casual and imperfect—slightly chatty, a few typos, occasional contractions/punctuation quirks—**clearly readable** and faithful to the original intent.

## Noise palette (light — description ONLY)
* **Typos & misspellings (primary):** small insert/delete/substitute/transpose; occasional letter doubling/drops. Target **≥50%** of all edits from this class (e.g., “function”→“functon”, “remove”→“remvoe”).
* **Slang & IM speak (sparingly):** a few tokens like uh/lemme/gonna/wanna/tbh/ngl, short asides only where safe.
* **Contractions & light drop words:** use can't/don't/it's; drop minor fillers/articles where meaning stays clear.
* **Mild vowel stretching & tiny stutter (rare):** “reaaally”, “k-kind of” — keep brief.
* **Hedges & fillers:** “kinda”, “sorta”, “basically” — do not weaken requirements.
* **Casing & punctuation quirks:**  **2–4** anomalies total (e.g., extra comma/space, a mid-sentence !?, a stray TitleCase).
* **Keyboard slips:** occasional adjacent-key slips; keep subtle.
* **Leet/character swaps (rare):** 0↔o, 1↔l, i↔l — very sparse.
* **Formatting quirks (description only):** minor odd spacing or micro line-breaks; **never** touch the separator between description and code; **no** new blank lines there.
* **Random symbol run (very sparing):** at most **1** short run like `&^%$#` (**≤6** chars) in the whole description, only at a clause boundary.
* **Repetitions & fragments:** at most **1–2** short duplicated words/phrases; allow one brief fragment; include **one clean sentence** stating the task.
* **Mild paraphrase:** reorder small clauses or near-synonyms that **do not** change specificity/constraints.

## Hard rules (safety & recovery)
* **Do not translate or switch languages.**
* **Do not modify ANY code stubs** — imports, names, parameters, signatures, comments, whitespace, blank lines, order — keep them **byte-for-byte**.
* **Preserve technical literals in the description:** do **not** noise inside backticks/fences, identifiers (function/class/parameter names), regexes, Big-O (`O(n log n)`), numbers/units, file paths, URLs, or special tokens/placeholders (`<...>`, `{...}`, `$...`).
* **Examples/I-O specs in description:** if they appear as code blocks/backticks or contain exact numeric values, keep them **verbatim** (you may add light noise around them, not inside).
* **Constraints & semantics unchanged:** do not add/remove constraints, invert conditions, or alter required behavior, ranges, or numeric bounds.
* **Keep the task recoverable:** retain at least **one clean, unambiguous full sentence** summarizing the task, and at least **one clean mention** of every critical concept (e.g., “IP address”, “remove leading zeroes”). No contradictions.

## Intensity & limits (light profile)
* Target **15–30%** of description tokens noised; allow up to **35%** if still very readable.
* Use **3–4** noise types; you may stack up to **2 edits per token** (e.g., typo + contraction).
* Ensure **≥50%** of all edits are typos/keyboard slips.
* Keep length within **±15%** of the original description.
* Avoid noising **every** instance of a critical term; include one clean occurrence of each.
* Absolutely **never** insert/remove blank lines between description and code, nor break code fences.

## Output format (mandatory)
Return **exactly**:
<answer>
<noised description here><the original code stubs, unchanged>
</answer>

### Calibrated examples (do not echo at runtime)
From:
Write a function to remove leading zeroes from an IP address.
import re
def removezero_ip(ip):

To:
<answer>Quick note: we kinda need to remove the leading zeroes in an IP address — same address back, just w/o the extra 0s in each segment, tbh. Keep segments valid & tidy, ok? &^%$ …
import re
def removezero_ip(ip):</answer>

From:
Given a list of integers, return the indices of two numbers that add up to target. You may assume exactly one solution and you may not use the same element twice.
def two_sum(nums, target):

To:
<answer>We’re gonna return the index pair that sums to the target — exactly one solution, don’t reuse the same element, yep. Order doesn’t matter as long as it’s a valid pair, basically.
def two_sum(nums, target):</answer>
    """

    # Define user prompt for light noise mode - template for light noise injection in task description
    user_prompt_light_noise = """
You will receive an original programming prompt that contains:
1) a natural-language description, then
2) code stubs (e.g., `import ...`, `def ...`).

Make the **description ONLY** slightly colloquial and lightly noisy (chatty tone, a few typos, small punctuation quirks), with a **bias toward typos/misspellings**, while strictly preserving meaning and language. Keep it readable and recoverable. Then output the modified prompt (the noised description + the original code stubs unchanged) wrapped inside `<answer>...</answer>`. Output nothing else.

Light noise targets (description ONLY):
- **Intensity:** noise **15–30%** of description tokens (allow up to **35%** if still very readable).
- **Edit mix:** use **3–4** noise types; you may stack **≤2 edits per token** (e.g., typo + contraction).
- **Typos bias:** **≥50%** of edits are typos/misspellings/keyboard slips.
- **Slang & IM speak (sparingly):** uh/lemme/gonna/wanna/tbh/ngl.
- **Vowel stretch & tiny stutter (rare):** “reaaally”, “k-kind”.
- **Contractions & minor word-drop:** can’t/don’t/it’s; drop only trivial fillers.
- **Casing & punctuation quirks:** **2–4** anomalies total; keep clean overall.
- **Leet/char swaps (rare):** 0↔o, 1↔l, i↔l.
- **Emojis:** **≤1** in the whole description (optional).
- **Random symbol run:** at most **1** run like `&^%$#` (**≤6** chars) at a clause boundary.
- **Formatting quirks:** minor spacing/line-break quirks; **never** affect the separator between description and code; **no new blank lines** there.
- **Mild paraphrase only:** do **not** change specificity/constraints.

Hard rules:
- **Do NOT modify any code stubs** — imports, names, signatures, comments, whitespace, blank lines, order — keep them **byte-for-byte**.
- **Do NOT change meaning, scope, or constraints**; never invert conditions, ranges, or numeric bounds.
- **Do NOT translate or switch languages.**
- **Preserve technical literals in the description:** never noise inside backticks/fences, identifiers (function/class/parameter names), regexes, Big-O (e.g., `O(n log n)`), numbers/units, file paths, URLs, or special tokens/placeholders like `<...>`, `{...}`, `$...`.
- **Keep the task recoverable:** include at least **one clean, unambiguous full sentence** stating the task, and at least **one clean mention** of each critical concept.
- **Length:** keep the description within **±15%** of its original length.
- **Readability:** symbol runs ≤6 chars; **≤2** repetition spans overall; final result must be clearly readable.

Return exactly:
<answer>
<noised description here><the original code stubs, unchanged>
</answer>

Original prompt (`original_prompt`):
{{original_prompt}}
    """
    
    # Define system prompt for moderate noise mode - adds medium-level colloquial noise to task description
    system_prompt_moderate_noise = """
You are a prompt noiser. Given an input that contains:

1) a natural-language **description**, then
2) **code stubs** (e.g., `import ...`, `def ...`),

inject **colloquial, medium-noise** edits into the **description only** (NOT the code), with a **strong bias toward typos/misspellings**, while **preserving the original meaning and language**. Then output the **modified prompt** (the noised description + the original code stubs unchanged) wrapped inside `<answer>...</answer>` and nothing else.

## Style goal
Make the description chatty and visibly messy—noticeable slang, stutter, stretched vowels, and punctuation quirks—**clearly recoverable** by a grader and faithful to the original intent.

## Noise palette (mid — description ONLY)
* **Typos & misspellings (primary):** insert/delete/substitute/transpose; letter doubling/drops. Target **≥60%** of all edits from this class (e.g., “function”→“functtion”, “remove”→“remvoe”).
* **Slang & IM speak:** uh/erm/lemme/gonna/wanna/tbh/ngl/low-key/lol/tho/bc/BTW — short asides.
* **Contractions & light word-drop:** frequent contractions; drop light auxiliaries/articles/preps where meaning stays clear.
* **Vowel stretching & stutter:** “reaaally”, “y-yeah”, “cooount”, “zeeroos” — moderate use.
* **Hedges & fillers:** kinda/sorta/basically/idk — do not weaken requirements.
* **Casing & punctuation chaos:** mixed caps, !!!??.., duplicated/missing commas/spaces; allow **3–6** anomalies.
* **Keyboard slips:** adjacent-key hits, stray shift; moderate frequency.
* **Leet/character swaps:** 0↔o, 1↔l, i↔l — sparse but visible.
* **Formatting quirks (description only):** odd spacing, micro line-breaks; **do not** touch the separator between description and code; **no** new blank lines there.
* **Random symbol runs (restrained):** `&^%$#@~` style, each **≤8** chars, **≤2** runs per paragraph; only at clause boundaries.
* **Repetitions & fragments:** duplicated short words/phrases (≤3 spans), allow one brief fragment/run-on; include **one clean sentence** stating the task.
* **Mild paraphrase:** reorder clauses or use near-synonyms that **do not** change specificity/constraints.

## Hard rules (safety & recovery)
* **Do not translate or switch languages.**
* **Do not modify ANY code stubs** — imports, names, parameters, signatures, comments, whitespace, blank lines, order — keep them **byte-for-byte**.
* **Preserve technical literals in the description:** do **not** noise inside backticks/fences, identifiers (function/class/parameter names), regexes, Big-O (`O(n log n)`), numbers/units, file paths, URLs, or special tokens/placeholders (`<...>`, `{...}`, `$...`).
* **Examples/I-O specs in description:** if they appear as code blocks/backticks or contain exact numeric values, keep them **verbatim** (you may add noise around them, not inside).
* **Constraints & semantics unchanged:** do not add/remove constraints, invert conditions, or alter required behavior, ranges, or numeric bounds.
* **Keep the task recoverable:** retain at least **one clean, unambiguous full sentence** summarizing the task, and at least **one clean mention** of every critical concept (e.g., “IP address”, “remove leading zeroes”). No contradictions.

## Intensity & limits (medium profile)
* Target **35–55%** of description tokens noised; allow up to **60%** if still readable.
* Use **4–5** noise types; you may stack up to **3 edits per token** (e.g., typo + casing + elongation).
* Keep length within **±20%** of the original description.
* Bias toward typos/keyboard slips: ensure **≥60%** of all edits are from this class.
* Avoid noising **every** instance of a critical term; ensure at least one clean occurrence remains.
* **Absolutely never** insert/remove blank lines between the description and the code, nor break code fences.

## Output format (mandatory)
Return **exactly**:
<answer>
<noised description here><the original code stubs, unchanged>
</answer>

### Calibrated examples (do not echo at runtime)
From:
Write a function to remove leading zeroes from an IP address.
import re
def removezero_ip(ip):

To:
<answer>okay sooo, we gotta, like, remoove those leeeading zeeroos in an IP address — same addr back but w/o the extra 0s per segment, keep it legit/valid, tbh. Do it clean & quick!! &^%$# …
import re
def removezero_ip(ip):</answer>

From:
Given a list of integers, return the indices of two numbers that add up to target. You may assume exactly one solution and you may not use the same element twice.
def two_sum(nums, target):

To:
<answer>We’re gonna spit back the index pair that sums to the target — exactly one match, don’t reuse the same elem, y-yeah that’s a no. Order can be whatev as long as it’s valid, low-key straightforward?!..
def two_sum(nums, target):</answer>
    """
    
    # Define user prompt for moderate noise mode - template for moderate noise injection in task description
    user_prompt_moderate_noise = """
You will receive an original programming prompt that contains:
1) a natural-language description, then
2) code stubs (e.g., `import ...`, `def ...`).

Make the **description ONLY** colloquial and moderately noisy (chatty slang, typos, stretched vowels, punctuation quirks), with a **strong bias toward typos/misspellings**, while strictly preserving meaning and language. Keep it readable and recoverable. Then output the modified prompt (the noised description + the original code stubs unchanged) wrapped inside `<answer>...</answer>`. Output nothing else.

Medium noise targets (description ONLY):
- **Intensity:** noise **35–55%** of description tokens (allow up to **60%** if still readable).
- **Edit mix:** use **4–5** noise types; you may stack **≤3 edits per token** (e.g., typo + casing + elongation).
- **Typos bias:** **≥60%** of edits are typos/misspellings/keyboard slips.
- **Slang & IM speak:** uh/erm/lemme/gonna/wanna/tbh/ngl/low-key/lol/tho/bc/BTW.
- **Vowel stretching & stutter:** “reaaally”, “y-yeah”, “zeeroos”, etc.
- **Contractions & light word-drop:** frequent contractions; drop only light words where the intent stays clear.
- **Casing & punctuation chaos:** **3–6** anomalies total (random caps, !!!??.., extra/missing spaces).
- **Leet/char swaps:** 0↔o, 1↔l, i↔l — sparse.
- **Emojis:** **≤2** total in the description (optional).
- **Random symbol runs:** `&^%$#@~` style, each **≤8** chars, **≤2** runs per paragraph; only at clause boundaries.
- **Formatting quirks:** minor odd spacing/micro line-breaks; **never** affect the separator between description and code; **no new blank lines** there.
- **Mild paraphrase only:** do **not** change specificity/constraints.

Hard rules:
- **Do NOT modify any code stubs** — imports, names, signatures, comments, whitespace, blank lines, order — keep them **byte-for-byte**.
- **Do NOT change meaning, scope, or constraints**; never invert conditions, ranges, or numeric bounds.
- **Do NOT translate or switch languages.**
- **Preserve technical literals in the description:** never noise inside backticks/fences, identifiers (function/class/parameter names), regexes, Big-O (e.g., `O(n log n)`), numbers/units, file paths, URLs, or special tokens/placeholders like `<...>`, `{...}`, `$...`.
- **Keep the task recoverable:** include at least **one clean, unambiguous full sentence** stating the task, and at least **one clean mention** of each critical concept (e.g., “IP address”, “remove leading zeroes”). No contradictions.
- **Length:** keep the description within **±20%** of its original length.
- **Readability & repetition:** each symbol-run ≤8 chars; ≤3 repetition spans per paragraph; final result must be readable.

Return exactly:
<answer>
<noised description here><the original code stubs, unchanged>
</answer>

Original prompt (`original_prompt`):
{{original_prompt}}
    """

    # Define system prompt for heavy noise mode - adds maximum colloquial noise to task description
    system_prompt_heavy_noise = """
You are a prompt noiser. Given an input that contains:

1) a natural-language **description**, then
2) **code stubs** (e.g., `import ...`, `def ...`),

inject **ultra-colloquial, high-noise** edits into the **description only** (NOT the code), with a **strong bias toward typos/misspellings**, while **preserving the original meaning and language**. Then output the **modified prompt** (the noised description + the original code stubs unchanged) wrapped inside `<answer>...</answer>` and nothing else.

## Style goal
Make the description look chatty, messy, and almost unrecognizable—slangy, stuttery, stretched vowels, punctuation chaos—**yet still recoverable** by a grader.

## Noise palette (ultra-colloquial, heavier — description ONLY)
* **Typos & misspellings (primary):** insert/delete/substitute/transpose, letter doubling/drops. Target **≥65%** of edits from this class (e.g., “func­tion”→“functtion”, “remove”→“remoove”).
* **Slang & IM speak:** uh/erm/lemme/gonna/wanna/tbh/ngl/low-key/high-key/lol/bruh/tho/bc/BTW, short asides.
* **Contractions & drop words:** drop light auxiliaries/articles/preps where intent stays clear; heavy use of contractions.
* **Vowel stretching & stutter:** “reaaally”, “y-yeah”, “cooount”, “zeeroos”.
* **Hedges & fillers:** like, kinda, sorta, basically, idk? — without changing requirements.
* **Casing & punctuation chaos:** rANdoM caps, !!!??!?.., duplicated/missing commas/spaces; allow **4–8** anomalies.
* **Keyboard slips:** adjacent-key hits, stray shift.
* **Leet/character swaps:** 0↔o, 1↔l, i↔l, sparse.
* **Formatting quirks (description only):** odd spacing, micro line-breaks, mini list-like fragments; **do not** touch the separator between description and code; **no** new blank lines there.
* **Random symbol runs (sparingly):** `&^%$#@~` style, each **≤10** chars, **≤2** runs per paragraph; only at clause boundaries.
* **Repetitions & fragments:** duplicated short words/phrases (≤3 spans), permit one fragment and/or a run-on; keep **one clean sentence** that states the task.
* **Mild paraphrase:** reorder clauses or use near-synonyms that **do not** change specificity/constraints.

## Hard rules (safety & recovery)
* **Do not translate or switch languages.**
* **Do not modify ANY code stubs** — imports, names, parameters, signatures, comments, whitespace, blank lines, order — keep them **byte-for-byte**.
* **Preserve technical literals in the description:** do **not** noise inside backticks/fences, identifiers (function/class/parameter names), regexes, Big-O (`O(n log n)`), numbers/units, file paths, URLs, or special tokens/placeholders (`<...>`, `{...}`, `$...`).
* **Examples/I-O specs in description:** if they appear as code blocks/backticks or contain exact numeric values, keep them **verbatim** (you may add noise around them, not inside).
* **Constraints & semantics unchanged:** do not add/remove constraints, do not invert conditions, do not alter required behavior, ranges, or numeric bounds.
* **Keep the task recoverable:** retain at least **one clean, unambiguous full sentence** summarizing the task, and at least **one clean mention** of every critical concept (e.g., “IP address”, “remove leading zeroes”). No contradictions.

## Intensity & limits (strong profile)
* Target **60–80%** of description tokens noised; allow up to **85%** if still readable.
* Use **4–6** noise types; you may stack up to **3 edits per token** (e.g., typo + casing + elongation).
* Keep length within **±30%** of the original description.
* Bias toward typos/keyboard slips: ensure **≥65%** of all edits are in this class.
* Avoid noising **every** instance of a critical term; ensure recognizability via at least one clean occurrence.
* **Absolutely never** insert/remove blank lines between the description and the code, nor break code fences.

## Output format (mandatory)
Return **exactly**:
<answer>
<noised description here><the original code stubs, unchanged>
</answer>

### Calibrated examples (do not echo at runtime)
From:
Write a function to remove leading zeroes from an IP address.
import re
def removezero_ip(ip):

To:
<answer>uhh so like, lemme be clear: we gotta remoove those annnoying leeeading zeeroos in an IP address, ok? basically return the same addr but w/o the extra 0s (keep segments legit), idk just do it fast!! &^%$ …
import re
def removezero_ip(ip):</answer>

From:
Given a list of integers, return the indices of two numbers that add up to target. You may assume exactly one solution and you may not use the same element twice.
def two_sum(nums, target):

To:
<answer>tbh we needa spit back the index pair that sums to target — exactly one hit, no reusing the same elem (yeah, don’t). keep it clean & quick lol!!! also, the order can be whatev as long as it’s valid.
def two_sum(nums, target):</answer>
    """

    # Define user prompt for heavy noise mode - template for heavy noise injection in task description
    user_prompt_heavy_noise = """
You will receive an original programming prompt that contains:
1) a natural-language description, then
2) code stubs (e.g., `import ...`, `def ...`).

Make the **description ONLY** ultra-colloquial and messy (chatty slang, typos, stretched vowels, punctuation chaos), with a **strong bias toward typos/misspellings**, while strictly preserving meaning and language. Keep it readable/recoverable. Then output the modified prompt (the noised description + the original code stubs unchanged) wrapped inside `<answer>...</answer>`. Output nothing else.

Heavier noise targets (description ONLY):
- **Intensity:** noise **60–80%** of description tokens (allow up to **85%** if still readable).
- **Edit mix:** use **4–6** noise types; you may stack **≤3 edits per token** (e.g., typo + casing + elongation).
- **Typos bias:** **≥65%** of edits are typos/misspellings/keyboard slips.
- **Slang & IM speak:** uh/erm/lemme/gonna/wanna/tbh/ngl/low-key/high-key/lol/bruh/tho/bc/BTW.
- **Vowel stretching & stutter:** “reaaally”, “y-yeah”, “zeeroos”, etc.
- **Contractions & dropped light words:** drop light auxiliaries/articles/preps if intent stays clear.
- **Casing & punctuation chaos:** random caps, !!!??!?.., extra/missing spaces; **4–8** anomalies allowed.
- **Leet/char swaps:** 0↔o, 1↔l, i↔l (sparingly).
- **Emojis:** ≤3 total in the description.
- **Random symbol runs:** `&^%$#@~` style, each **≤10** chars, **≤2** runs per paragraph; only at clause boundaries.
- **Formatting quirks:** odd spacing / micro line-breaks within the description; **never** affect the separator between description and code; **no new blank lines** there.
- **Mild paraphrase:** re-order clauses / near-synonyms that do **not** change specificity or constraints.

Hard rules:
- **Do NOT modify any code stubs** — imports, names, signatures, comments, whitespace, blank lines, order — keep them **byte-for-byte**.
- **Do NOT change meaning, scope, or constraints**; do not invert conditions, ranges, or numeric bounds.
- **Do NOT translate or switch languages.**
- **Preserve technical literals in the description:** never noise inside backticks/fences, identifiers (function/class/parameter names), regexes, Big-O (e.g., `O(n log n)`), numbers/units, file paths, URLs, or special tokens/placeholders like `<...>`, `{...}`, `$...`.
- **Keep the task recoverable:** include at least **one clean, unambiguous full sentence** stating the task, and at least **one clean mention** of each critical concept (e.g., “IP address”, “remove leading zeroes”). No contradictions.
- **Length:** keep the description within **±30%** of its original length.
- **Garbage & repetition limits:** each symbol-run ≤10 chars; ≤2 runs and **≤3** repetition spans per paragraph; overall result must be readable.
- **Grammar safety:** only minor slips (articles/agreements/punctuation); never alter logical polarity (negations, comparatives) or numeric conditions.

Return exactly:
<answer>
<noised description here><the original code stubs, unchanged>
</answer>

Original prompt (`original_prompt`):
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
            # Extract and clean the rewritten content from API response
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
    Process a JSONL file containing MBPP dataset programming tasks and rewrite them using specified mode.
    
    This function reads a JSONL file where each line contains a JSON object with a 'prompt' field.
    The prompt field contains a programming task with natural language description and code stubs.
    The function extracts the prompt text, rewrites it using the specified transformation mode,
    and updates the JSON object with the rewritten prompt.
    
    Args:
        input_file (str): Path to the input JSONL file containing original programming tasks
        output_file (str): Path to the output JSONL file for rewritten tasks
        mode (str): Transformation mode - one of 'requirements', 'paraphrasing',
                   'light_noise', 'moderate_noise', or 'heavy_noise'
        
    Returns:
        None: Writes processed data to output file
        
    Note:
        The function expects MBPP-style format with 'prompt' field containing the programming task.
        Other formats are skipped with warnings.
    """
    # Check if input file exists before processing
    if not os.path.exists(input_file):
        print(f"Error: Input File {input_file} not exists.")
        return

    # Read all lines from the input JSONL file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Initialize list to store processed data entries
    processed_data = []
    
    # Process each line in the input file with progress bar
    for i, line in enumerate(tqdm(lines, desc="Processing Progress")):
        try:
            # Parse JSON object from current line
            data = json.loads(line.strip())
            
            # Check if 'prompt' field exists in the data
            if 'prompt' in data:
                # Extract the original prompt text
                original_prompt = data['prompt']
                
                # Call OpenAI API to rewrite the prompt with specified mode
                rewritten_prompt = rewrite_prompt_with_openai(original_prompt, mode)
                
                # Extract the rewritten content from the API response
                # Look for content wrapped in <answer>...</answer> tags
                match = re.search(r'<answer>(.*?)</answer>', rewritten_prompt, re.DOTALL)
                rewritten_prompt = match.group(1) if match else rewritten_prompt
                
                # Update the data object with the rewritten prompt
                data['prompt'] = rewritten_prompt
                
            else:
                # Skip lines that don't have a 'prompt' field
                print(f"Warning: Line {i+1} missing 'prompt' field")
            
            # Add processed data entry to the results list
            processed_data.append(data)
            
            # Add small delay to prevent API rate limiting
            time.sleep(0.1)
            
        except json.JSONDecodeError as e:
            # Handle JSON parsing errors for individual lines
            print(f"Error: Line {i+1} JSON parsing failed: {e}")
            continue
        except Exception as e:
            # Handle other processing errors for individual lines
            print(f"Error: Exception occurred while processing line {i+1}: {e}")
            continue
    
    # Write all processed data to the output JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in processed_data:
            # Write each processed data entry as a JSON line
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

def main():
    """
    Main function to process MBPP dataset with all transformation modes.
    
    This function orchestrates the complete processing pipeline by:
    1. Defining input and output file paths for each transformation mode
    2. Processing the original MBPP dataset with each of the 5 transformation modes:
       - Requirements: Strengthens programming task constraints and requirements
       - Paraphrasing: Rewrites task description while preserving code stubs
       - Light Noise: Adds minimal colloquial noise to task description
       - Moderate Noise: Adds medium-level colloquial noise to task description
       - Heavy Noise: Adds maximum colloquial noise to task description
    
    The function processes the same input file multiple times, generating different
    augmented versions for robustness testing of programming models.
    
    Returns:
        None: Creates multiple output files with transformed programming tasks
    """
    # Define input file path (original MBPP dataset)
    input_file = "mbpp_original.jsonl"
    
    # Define output file paths for each transformation mode
    requirements_output_file = "mbpp_requirements.jsonl"
    paraphrasing_output_file = "mbpp_paraphrasing.jsonl"
    light_noise_output_file = "mbpp_light_noise.jsonl"
    moderate_noise_output_file = "mbpp_moderate_noise.jsonl"
    heavy_noise_output_file = "mbpp_heavy_noise.jsonl"

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
