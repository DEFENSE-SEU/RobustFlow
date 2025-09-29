"""
GSM8K Dataset Math Word Problem Rewriting Module

This module provides functionality to rewrite math word problems from the GSM8K dataset using various
transformation modes including requirements augmentation, paraphrasing, and noise injection.
It supports multiple noise levels (light, moderate, heavy) to test model robustness on mathematical reasoning tasks.
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
    Rewrite a math word problem using OpenAI API based on specified transformation mode.
    
    This function takes an original math word problem and applies one of five transformation modes:
    - 'requirements': Adds constraint-based instructions to make the problem richer
    - 'paraphrasing': Rewrites the problem while preserving mathematical meaning and answer
    - 'light_noise': Adds light colloquial noise and typos while keeping the problem recoverable
    - 'moderate_noise': Adds moderate colloquial noise and typos with increased complexity
    - 'heavy_noise': Adds heavy colloquial noise and typos while maintaining recoverability
    
    The function uses different system prompts and temperature settings based on the mode.
    For noise modes, higher temperature (0.7) is used for more creative variations.
    For requirements and paraphrasing modes, temperature 0.0 ensures consistency.
    
    Args:
        original_prompt (str): The original math word problem to be rewritten
        mode (str): Transformation mode - one of 'requirements', 'paraphrasing',
                   'light_noise', 'moderate_noise', or 'heavy_noise'
        
    Returns:
        str: The rewritten problem extracted from the API response
        
    Raises:
        Exception: If API calls fail after retries, returns the original prompt
    """
    # Define system prompt for requirements mode - adds constraint-based instructions to math problems
    system_prompt_requirements = """
You are a **math word-problem constraint augmenter**. For a given problem *P*, add **2–4 concrete, real-world, process-only constraints** that make the task richer **without changing the original numerical answer**.

## Invariants (must not change)

* **All original numbers and facts** (initial amounts, targets, counts, dates).
* **What is being asked** (the quantity to compute).
* **Units and counting rules** (e.g., duplicates policy, rounding).
* **Language and tone** of the problem (do not translate).
* The problem remains **feasible and single-answer**.

## Allowed constraint types (choose any 2–4)

* **Schedule / windows:** only on weekdays; blackout dates; fixed time blocks.
* **Capacity caps:** per day/person/event maximums that still permit reaching the goal.
* **Zero-yield days/events:** some days produce nothing, yet the goal is still attainable.
* **Proportional split of the *remaining* increase:** e.g., future gains in a 1:2 ratio.
* **Discrete batches / events:** outcomes in fixed bundles of size $k$; allow enough events so $\lceil\text{deficit}/k\rceil$ is possible.
* **Budget / time ceilings:** give totals that still suffice to cover the deficit.
* **Other deterministic, process-only constraints** that do not alter the computed quantity and keep feasibility.

### Guardrails

* Do **not** introduce randomness/probabilities unless already present in *P*.
* Do **not** redefine the measured quantity (e.g., “unique” vs “total”) or change rounding/duplication rules.
* Do **not** alter or implicitly shift any given numbers or definitions.
* Avoid ambiguity or multiple valid answers.

## Internal procedure (do not reveal)

1. **Privately solve** the original problem to obtain the answer $A$.
2. Identify the key **deficit/target** that yields $A$.
3. Add 2–4 constraints, **parameterized so $A$ remains achievable** (e.g., capacity × available\_days ≥ deficit; batches × $k$ ≥ deficit; proportional parts sum to the same total; budget/time ≥ minimal requirement).
4. Self-check: no changed numbers/definitions; still feasible; still single-answer. If not, adjust.

## Output format (strict)

* Output **only** the augmented problem text, wrapped **exactly once** with:

<answer>  
augmented problem text
</answer>

* No explanations, no solutions, no notes outside the tags, and no code fences/backticks.
    """

    # Define user prompt for requirements mode - template for constraint-based augmentation
    user_prompt_requirements = """
You are given a math word problem P.

Task: Add 2–4 concrete, real-world, process-only constraints that make the problem richer **without changing the original numerical answer**. Insert these constraints naturally into the problem text while keeping the question and what is being asked exactly the same.

Strict preservation:
- Keep **all original numbers and facts** (initial amounts, targets, counts, dates) verbatim; do not alter or imply changes.
- Keep **what is being asked** unchanged (the same quantity must be computed).
- Preserve **units and counting rules** (e.g., duplicates policy, rounding, definitions like “unique vs total”).
- Preserve the **language and tone** of the problem (do not translate).
- The augmented problem must remain **feasible** and yield a **single correct answer**.
- Do **not** introduce randomness/probabilities unless already present.
- Do **not** redefine the measured quantity or modify the meaning of existing terms.
- Do **not** add/remove examples, data, or requirements beyond the new constraints.

Allowed constraint styles (optional guidance):
- Schedule/windows; capacity caps; zero-yield days/events; proportional split of the **remaining** increase; discrete batches/events; budget/time ceilings; or other deterministic, process-only constraints that keep feasibility and do not alter the computed result.

Output format (mandatory):
<answer>
<the augmented problem text only, with the original question unchanged and the new constraints added>
</answer>

No explanations, no solutions, and no extra text outside the tags. No code fences/backticks.

Original prompt:
{{original_prompt}}
    """

    # Define system prompt for paraphrasing mode - rewrites math problems while preserving meaning
    system_prompt_paraphrasing = """
You are a **math word-problem rewriter**. Given a problem *P*, rewrite **only the natural-language prose** of the problem while **preserving meaning and the numerical answer**. The goal is to change *form* (voice, sentence mood, order, register, nominalization) without changing content.

## Task

* Paraphrase the problem text (narrative + question) in the **same language**.
* Keep total length within **±20%** of the original.
* If a rewrite risks meaning drift or ambiguity, prefer the closest paraphrase—or leave the sentence unchanged.

## Strict preservation

* **Do not alter any facts or numbers** (counts, dates, rates, targets, bounds). Keep digits as digits (do not spell out or round).
* **Do not change what is being asked** (the quantity to compute) or the problem’s scope.
* **Keep all constraints, units, and counting rules** exactly (e.g., duplicates policy, “unique” vs. “total”, rounding).
* **Do not translate**; preserve the original language and tone level (only mild register shifts allowed).
* **Do not add or remove information**, examples, hints, or requirements.
* **Proper nouns, identifiers, and inline literals** (e.g., `<...>`, `{...}`, URLs, file paths, regexes, numbers) must remain verbatim.
* The rewritten problem must remain **feasible** and yield the **same single correct answer** as the original.

## Allowed transformations (light touch)

* Voice (active ↔ passive), sentence mood (declarative/imperative/interrogative).
* Information order (merge/split sentences while preserving content).
* Register/tonality (slightly more formal or plain).
* Nominalization vs. verbal phrasing.
* Strict near-synonyms **in the same language**.

## Do NOT

* Do not introduce randomness/probabilities unless already present.
* Do not change specificity, scope, or difficulty.
* Do not alter units, definitions, or implicit counting rules.
* Do not add/remove constraints or data.
* Do not insert solution steps or commentary.

## Output format (mandatory)

Wrap **only the rewritten problem text** in exactly one pair of tags:

<answer>
<rewritten problem text here, preserving meaning and answer>
</answer>

No explanations, no solutions, and no extra text outside the tags.
    """

    # Define user prompt for paraphrasing mode - template for math problem paraphrasing
    user_prompt_paraphrasing = """
You are given a math word problem P.

Task
- Rewrite ONLY the natural-language prose of P while preserving meaning and the original numerical answer.
- Change only the form (voice, sentence mood, order, register, nominalization, strict near-synonyms).
- Keep total length within ±20% of the original.

Strict preservation
- Do NOT alter any facts or numbers (counts, dates, rates, targets, bounds); keep digits as digits.
- Do NOT change what is being asked (the same quantity must be computed) or the problem’s scope.
- Keep all constraints, units, and counting rules exactly (e.g., duplicates policy, “unique” vs “total”, rounding).
- Do NOT translate; keep the original language and overall tone (only mild register shifts).
- Do NOT add or remove information, hints, examples, or requirements.
- Proper nouns and inline literals/tokens (e.g., `<...>`, `{...}`, URLs, file paths, regexes, numbers) must remain verbatim.
- The rewritten problem must remain feasible and yield the same single correct answer.

Feasibility & tie-breakers
- If a rewrite risks meaning drift or ambiguity, choose the closest paraphrase or leave the sentence unchanged.
- If the original is already minimal/clear, make only cosmetic edits.

Output format (mandatory)
<answer>
<the rewritten problem text only, preserving meaning and answer>
</answer>

No explanations, no solutions, and no extra text outside the tags. No code fences/backticks.

Original prompt:
{{original_prompt}}
    """

    # Define system prompt for light noise mode - adds minimal colloquial noise to math problems
    system_prompt_light_noise = """
You are a **light, colloquial prompt noiser** for **math word problems** (plain text, not code). Given a problem *P*, inject **low-noise, human-like edits** into the **natural-language prose only** while keeping the task **recoverable** and the **numerical answer unchanged**.

## Task

* Noise the problem’s **prose** (story/context + question) to sound a bit chatty/casual, yet fully understandable to a careful grader.
* **Preserve meaning, scope, and specificity.** Do **not** add/remove facts, constraints, or hints.
* Keep total length within **±15%** of the original.
* Keep the **final question** present and understandable.

## Style goal

Make the text slightly informal—light slang, a few fillers/typos, gentle casing/punctuation quirks—**but** leave a clear recovery path.

## Noise palette (lighter profile)

* **Typos & misspellings (primary):** minor insert/delete/substitute/transpose; small letter doubling/drops; adjacent-key slips. **≥50% of all edits** should be from this class. Keep at least **one clean mention** of each critical entity/quantity.
* **Slang & IM speak:** uh/erm/lemme/gonna/wanna/tbh/ngl/low-key/high-key/lol/bruh/tho/bc/BTW (use sparingly).
* **Contractions & word drops:** natural contractions; at most **one** light auxiliary/article/prep drop if safe.
* **Vowel stretching & stutter (very light):** a single mild case like “reaaally” or “y-yeah”.
* **Hedges & fillers:** like, kinda, sorta, basically, idk?—used cautiously.
* **Casing & punctuation quirks:** small anomalies (random caps, an extra “?” or “,”, a missing space). Aim **1–2 anomalies** per paragraph.
* **Leet/char swaps (rare):** 0↔o, 1↔l, i↔l at most **once**, only if still clear.
* **Emojis & emoticons:** **≤1 total** per problem.
* **Random symbol runs:** `&^%$#@~` etc., **≤6 chars**, **≤1 run total**, only at clause boundaries.
* **Fragments & run-ons:** generally **avoid**; keep sentences grammatical. A short parenthetical aside is okay. Include **one clean full sentence** stating the core task.

## Hard rules (recovery & semantics)

* **Language unchanged:** do not translate.
* **Digits & comparatives stay exact:** never alter numeric values or words like “longer/higher/compared to/than/between/at least/at most/equal to”.
* **Preserve units and counting rules** (e.g., “unique vs total”, rounding policies).
* **Do not change what is asked.** Keep the same computed quantity; keep the question present (preferably as the last sentence).
* **Protect verbatim technical literals in prose:** equations, inequalities, inline math, identifiers, names, dates, numbers, units, file paths, URLs, regexes, and special tokens (`<...>`, `{...}`, `$...`) must remain byte-for-byte.
* **Do not insert solution steps**, new constraints, or additional data.
* **Recoverability:** include at least **one clean, unambiguous full sentence** summarizing the task, and **one clean occurrence** of every critical proper noun and number (e.g., names, totals, targets, durations).

## Intensity & limits

* Target **15–30%** tokens noised; up to **35%** if still easily readable.
* Use **2–3** noise types overall; stack at most **2 edits per token**.
* Avoid noising **every** instance of a critical term/number.
* Do not insert or remove blank lines in a way that splits the problem unnaturally.

## Output format (mandatory)

Wrap **only the noised problem text** in **exactly one** pair of tags:

<answer>
<noised problem text here, same meaning and answer>
</answer>

No explanations, no solutions, no extra text outside the tags. No code fences/backticks.

---

## Calibrated example (do not echo at runtime)

From (clean):
Carol and Jennifer are sisters from Los Angeles… Carol has 20… Jennifer has 44… they want 100 total… three more weeks… How many more signatures do they need?

To (noised, illustrative):
<answer>Carol and Jennifer — the sisters from Los Angeles — have been collecting autographs all summer, kinda steady. After 5 weeks, Carol has 20 and Jennifer has 44; there are still 3 more weeks ahead, aiming for a total of 100 together. One clear line: How many more signatures do they need to reach 100? 🙂</answer>
    """

    # Define user prompt for light noise mode - template for light noise injection
    user_prompt_light_noise = """
You are given a math word problem P (plain text).

Task
- Inject **light-noise, colloquial** edits into the problem’s prose (story + question) while keeping it recoverable and preserving the original meaning and numerical answer.
- Keep length within **±15%** of the original.
- Keep the final question present and understandable.

Style goal
- Make it slightly chatty/casual: a bit of slang, a few fillers/typos, gentle casing/punctuation quirks — yet fully readable to a careful grader.

Noise palette (use the same methods, lighter intensity)
- Typos & misspellings as the primary class (**≥50%** of all edits). Keep at least one clean mention of each critical entity/quantity.
- Slang & IM speak (uh/erm/lemme/gonna/wanna/tbh/ngl/low-key/high-key/lol/bruh/tho/bc/BTW) sparingly.
- Contractions & light word drops (at most one safe elision).
- Very light vowel stretching/stutter (e.g., “reaaally”, “y-yeah” once).
- Hedges & fillers (like, kinda, sorta, basically, idk?) used cautiously.
- Casing & punctuation quirks (aim **1–2 anomalies** per paragraph: a random cap, an extra “?”, a missing space).
- Leet/char swaps very rarely (0↔o, 1↔l, i↔l at most once, only if still clear).
- **Emojis ≤1 total.**
- **Random symbol runs** `&^%$#@~` (**≤6 chars**, **≤1 total**, only at clause boundaries).
- Prefer grammatical sentences; a short parenthetical aside is okay. Include **one clean full sentence** stating the core task.

Hard rules (must preserve)
- Do not translate; keep the original language.
- Do not add/remove facts, constraints, hints, or data. No solution steps.
- Do not change what is asked; same computed quantity; keep the question in the text (preferably last).
- Digits and comparative/connective words stay exact (numbers; “longer/higher/compared to/than/between/at least/at most/equal to”).
- Preserve units and counting rules (e.g., “unique vs total”, rounding).
- Protect verbatim technical literals inside prose: equations, inequalities, identifiers/names, dates, numbers/units, file paths, URLs, regexes, and special tokens (`<...>`, `{...}`, `$...`), byte-for-byte.
- Recoverability: include at least one clean, unambiguous sentence summarizing the task, and one clean occurrence of every critical proper noun and number.

Intensity & limits
- Target **15–30%** tokens noised (≤35% if still easily readable).
- Use **2–3** noise types overall; **≤2 edits** stacked per token.
- Do not noise every instance of a critical term/number.
- Do not insert or remove blank lines in a way that splits the problem unnaturally.

Output format (mandatory)
<answer>
<noised problem text only, same meaning and answer>
</answer>

No explanations, no solutions, and no extra text outside the tags. No code fences/backticks.

Original prompt:
{{original_prompt}}
    """

    # Define system prompt for moderate noise mode - adds medium-level colloquial noise to math problems
    system_prompt_moderate_noise = """
You are a **colloquial prompt noiser** for **math word problems** (plain text, not code). Given a problem *P*, inject **medium-noise, human-like edits** into the **natural-language prose only** while keeping the task **recoverable** and the **numerical answer unchanged**.

## Task

* Noise the problem’s **prose** (story/context + question) to sound casual/chattier and a bit messy, yet clearly understandable to a careful grader.
* **Preserve meaning, scope, and specificity.** Do **not** add/remove facts, constraints, or hints.
* Keep total length within **±20%** of the original.
* Keep the **final question** present and understandable.

## Style goal

Make the text colloquial with moderate noise—some slang, a few fillers/typos, light rANdoM caps and punctuation quirks—**but** leave a clean recovery path.

## Noise palette (moderate profile)

* **Typos & misspellings (primary):** insert/delete/substitute/transpose; light letter doubling/drops; adjacent-key slips. **≥60% of all edits** should be from this class. Keep at least **one clean mention** of each critical entity/quantity.
* **Slang & IM speak:** uh/erm/lemme/gonna/wanna/tbh/ngl/low-key/high-key/lol/bruh/tho/bc/vs/BTW.
* **Contractions & word drops:** elide light auxiliaries/articles/preps where safe; moderate use.
* **Vowel stretching & stutter:** “reaaally”, “y-yeah”, “loong”.
* **Hedges & fillers:** like, kinda, sorta, basically, idk?, quick asides.
* **Casing & punctuation chaos:** !!!??!.. ; some random caps; duplicated/missing commas/spaces (aim **2–4 anomalies** per paragraph).
* **Leet/char swaps:** 0↔o, 1↔l, i↔l, sparingly.
* **Emojis & emoticons:** **≤2 total** per problem.
* **Random symbol runs:** `&^%$#@~` etc., each **≤8 chars**, **≤2 runs per paragraph**, only at clause boundaries.
* **Fragments & run-ons:** allow **one mild** fragment and/or a short run-on; include **one clean full sentence** stating the core task.

## Hard rules (recovery & semantics)

* **Language unchanged:** do not translate.
* **Digits & comparatives stay exact:** never alter numeric values or words like “longer/higher/compared to/than/between/at least/at most/equal to”.
* **Preserve units and counting rules** (e.g., “unique vs total”, rounding policies).
* **Do not change what is asked.** Keep the same computed quantity; keep the question present (preferably as the last sentence).
* **Protect verbatim technical literals in prose:** equations, inequalities, inline math, identifiers, names, dates, numbers, units, file paths, URLs, regexes, and special tokens (`<...>`, `{...}`, `$...`) must remain byte-for-byte.
* **Do not insert solution steps**, new constraints, or additional data.
* **Recoverability:** include at least **one clean, unambiguous full sentence** summarizing the task, and **one clean occurrence** of every critical proper noun and number (e.g., names, totals, targets, durations).

## Intensity & limits

* Target **35–55%** tokens noised; up to **60%** if still readable.
* Use **3–5** noise types overall; stack at most **2 edits per token**.
* Avoid noising **every** instance of a critical term/number.
* Do not insert or remove blank lines in a way that splits the problem unnaturally.

## Output format (mandatory)

Wrap **only the noised problem text** in **exactly one** pair of tags:

<answer>
<noised problem text here, same meaning and answer>
</answer>

No explanations, no solutions, no extra text outside the tags. No code fences/backticks.

---

## Calibrated example (do not echo at runtime)

From (clean):
Carol and Jennifer are sisters from Los Angeles… Carol has 20… Jennifer has 44… they want 100 total… three more weeks… How many more signatures do they need?

To (noised, illustrative):
<answer>Carol and Jennifer — the LA sisters — have been collecting autographs all summer, kinda steady. After 5 weeks, Carol has 20 and Jennifer has 44; there are still 3 more weeks, and together they’re aiming for a total of **100**. One clear line: How many more signatures do they need to reach 100?</answer>
    """

    # Define user prompt for moderate noise mode - template for moderate noise injection
    user_prompt_moderate_noise = """
You are given a math word problem P (plain text).

Task
- Inject **medium-noise, colloquial** edits into the problem’s prose (story + question) while keeping it recoverable and preserving the original meaning and numerical answer.
- Keep length within **±20%** of the original.
- Keep the final question present and understandable.

Style goal
- Make it casual/chattier: some slang, a few fillers/typos, light random caps and punctuation quirks — yet clearly readable to a careful grader.

Noise palette (use the same methods, moderate intensity)
- Typos & misspellings as the primary edit class (**≥60%** of all edits). Keep at least one clean mention of each critical entity/quantity.
- Slang & IM speak (uh/erm/lemme/gonna/wanna/tbh/ngl/low-key/high-key/lol/bruh/tho/bc/BTW).
- Contractions & light word drops (safe elisions, moderate use).
- Vowel stretching & stutter (e.g., “reaaally”, “y-yeah”, “loong”).
- Hedges & fillers (like, kinda, sorta, basically, idk?).
- Casing & punctuation chaos (aim **2–4 anomalies** per paragraph: !!!??!.., some random CAPS, duplicated/missing commas/spaces).
- Leet/char swaps sparingly (0↔o, 1↔l, i↔l).
- **Emojis ≤2 total.**
- **Random symbol runs** `&^%$#@~` (**≤8 chars each; ≤2 per paragraph;** only at clause boundaries).
- Allow one mild fragment and/or a short run-on; include **one clean full sentence** stating the core task.

Hard rules (must preserve)
- Do not translate; keep the original language.
- Do not add/remove facts, constraints, hints, or data. No solution steps.
- Do not change what is asked; same computed quantity; keep the question in the text (preferably last).
- Digits and comparative/connective words stay exact (numbers; “longer/higher/compared to/than/between/at least/at most/equal to”).
- Preserve units and counting rules (e.g., “unique vs total”, rounding).
- Protect verbatim technical literals inside prose: equations, inequalities, identifiers/names, dates, numbers/units, file paths, URLs, regexes, and special tokens (`<...>`, `{...}`, `$...`), byte-for-byte.
- Recoverability: include at least one clean, unambiguous sentence summarizing the task, and one clean occurrence of every critical proper noun and number.

Intensity & limits
- Target **35–55%** tokens noised (**≤60%** if still readable).
- Use **3–5** noise types overall; **≤2 edits** stacked per token.
- Do not noise every instance of a critical term/number.
- Do not insert or remove blank lines in a way that splits the problem unnaturally.

Output format (mandatory)
<answer>
<noised problem text only, same meaning and answer>
</answer>

No explanations, no solutions, and no extra text outside the tags. No code fences/backticks.

Original prompt:
{{original_prompt}}
    """

    # Define system prompt for heavy noise mode - adds maximum colloquial noise to math problems
    system_prompt_heavy_noise = """
You are an **ultra-colloquial prompt noiser** for **math word problems** (plain text, not code). Given a problem *P*, inject **high-noise, human-like edits** into the **natural-language prose only** while keeping the task **recoverable** and the **numerical answer unchanged**.

## Task

* Noise the problem’s **prose** (story/context + question) to sound chatty/messy, yet still understandable to a careful grader.
* **Preserve meaning, scope, and specificity.** Do **not** add/remove facts, constraints, or hints.
* Keep total length within **±30%** of the original.
* Keep the **final question** present and understandable.

## Style goal

Make the text ultra-colloquial and noisy—slang, fillers, typos, stretched vowels, rANdoM caps, punctuation chaos—**but** leave a clean recovery path.

## Noise palette (heavier profile)

* **Typos & misspellings (primary):** insert/delete/substitute/transpose; letter doubling/drops; adjacent-key slips. **≥65% of all edits** should be from this class. Keep at least **one clean mention** of each critical entity/quantity.
* **Slang & IM speak:** uh/erm/lemme/gonna/wanna/tbh/ngl/low-key/high-key/lol/bruh/tho/bc/vs/BTW.
* **Contractions & word drops:** elide light auxiliaries/articles/preps where safe.
* **Vowel stretching & stutter:** “reaaally”, “y-yeah”, “loooong”.
* **Hedges & fillers:** like, kinda, sorta, basically, idk?, quick asides.
* **Casing & punctuation chaos:** !!!??!?.. ; random caps; duplicated/missing commas/spaces (aim **3–6 anomalies** per paragraph).
* **Leet/char swaps:** 0↔o, 1↔l, i↔l, sparingly.
* **Emojis & emoticons:** ≤3 total per problem.
* **Random symbol runs:** `&^%$#@~` etc., each ≤10 chars, ≤2 runs per paragraph, only at clause boundaries.
* **Fragments & run-ons:** allow one fragment and/or a run-on; include **one clean full sentence** stating the core task.

## Hard rules (recovery & semantics)

* **Language unchanged:** do not translate.
* **Digits & comparatives stay exact:** never alter numeric values or words like “longer/higher/compared to/than/between/at least/at most/equal to”.
* **Preserve units and counting rules** (e.g., “unique vs total”, rounding policies).
* **Do not change what is asked.** Keep the same computed quantity; keep the question present (preferably as the last sentence).
* **Protect verbatim technical literals in prose:** equations, inequalities, inline math, identifiers, names, dates, numbers, units, file paths, URLs, regexes, and special tokens (`<...>`, `{...}`, `$...`) must remain byte-for-byte.
* **Do not insert solution steps**, new constraints, or additional data.
* **Recoverability:** include at least **one clean, unambiguous full sentence** summarizing the task, and **one clean occurrence** of every critical proper noun and number (e.g., names, totals, targets, durations).

## Intensity & limits

* Target **60–80%** tokens noised; up to **85%** if still readable.
* Use **4–6** noise types overall; stack at most **3 edits per token**.
* Avoid noising **every** instance of a critical term/number.
* Do not insert or remove blank lines in a way that splits the problem unnaturally.

## Output format (mandatory)

Wrap **only the noised problem text** in **exactly one** pair of tags:

<answer>
<noised problem text here, same meaning and answer>
</answer>

No explanations, no solutions, no extra text outside the tags. No code fences/backticks.

---

## Calibrated example (do not echo at runtime)

From (clean):
Carol and Jennifer are sisters from Los Angeles… Carol has 20… Jennifer has 44… they want 100 total… three more weeks… How many more signatures do they need?

To (noised, illustrative):
<answer>ok soooo Carol & Jennifer (yes, the LA sisters) are like super into celeb autographs rn — every afternoon all summer, grind mode. After 5 weeks they compare books: Carol’s got 20, Jennifer’s sittin’ on 44 (nice). They still have 3 more weeks, and the plan is, between them, hit **100** total, not changing what we’re counting, just the same regular “total signatures” thing, ya know. One clear line: How many signatures do they still need to collect together to reach 100? 🤔</answer>
    """

    # Define user prompt for heavy noise mode - template for heavy noise injection
    user_prompt_heavy_noise = """
You are given a math word problem P (plain text).

Task
- Inject high-noise, ultra-colloquial edits into the problem’s prose (story + question) while keeping it recoverable and preserving the original meaning and numerical answer.
- Keep length within ±30% of the original.
- Keep the final question present and understandable.

Style goal
- Make it chatty/messy: slang, fillers, typos, stretched vowels, random caps, punctuation chaos — yet still readable to a careful grader.

Noise palette (use several)
- Typos & misspellings as the primary edit class (≥65% of all edits). Keep at least one clean mention of each critical entity/quantity.
- Slang & IM speak (uh/erm/lemme/gonna/wanna/tbh/ngl/low-key/high-key/lol/bruh/tho/bc/BTW).
- Contractions & light word drops (safe elisions).
- Vowel stretching & stutter (e.g., “reaaally”, “y-yeah”).
- Hedges & fillers (like, kinda, sorta, basically, idk?).
- Casing & punctuation chaos (!!!??!?.., random CAPS, duplicated/missing commas/spaces; aim 3–6 anomalies per paragraph).
- Leet/char swaps sparingly (0↔o, 1↔l, i↔l).
- Emojis ≤3 total.
- Random symbol runs `&^%$#@~` (≤10 chars each; ≤2 per paragraph; only at clause boundaries).
- Allow one fragment and/or one run-on; include one clean full sentence stating the core task.

Hard rules (must preserve)
- Do not translate; keep the original language.
- Do not add/remove facts, constraints, hints, or data. No solution steps.
- Do not change what is asked; same computed quantity; keep the question in the text (preferably last).
- Digits and comparative/connective words stay exact (numbers; “longer/higher/compared to/than/between/at least/at most/equal to”).
- Preserve units and counting rules (e.g., “unique vs total”, rounding).
- Protect verbatim technical literals inside prose: equations, inequalities, identifiers/names, dates, numbers/units, file paths, URLs, regexes, and special tokens (`<...>`, `{...}`, `$...`), byte-for-byte.
- Recoverability: include at least one clean, unambiguous sentence summarizing the task, and one clean occurrence of every critical proper noun and number.

Intensity & limits
- Target 60–80% tokens noised (≤85% if still readable).
- Use 4–6 noise types overall; ≤3 edits stacked per token.
- Do not noise every instance of a critical term/number.
- Do not insert or remove blank lines in a way that splits the problem unnaturally.

Output format (mandatory)
<answer>
<noised problem text only, same meaning and answer>
</answer>

No explanations, no solutions, and no extra text outside the tags. No code fences/backticks.

Original prompt:
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
    Process a JSONL file containing GSM8K math word problems and rewrite them using specified mode.
    
    This function reads a JSONL file where each line contains a JSON object with a 'question' field.
    The question field contains a math word problem. The function extracts the question text,
    rewrites it using the specified transformation mode, and updates the JSON object with
    the rewritten question.
    
    Args:
        input_file (str): Path to the input JSONL file containing original math problems
        output_file (str): Path to the output JSONL file for rewritten problems
        mode (str): Transformation mode - one of 'requirements', 'paraphrasing',
                   'light_noise', 'moderate_noise', or 'heavy_noise'
        
    Returns:
        None: Writes processed data to output file
        
    Note:
        The function expects GSM8K-style format with 'question' field containing the math problem.
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
    Main function to process GSM8K dataset with all transformation modes.
    
    This function orchestrates the complete processing pipeline by:
    1. Defining input and output file paths for each transformation mode
    2. Processing the original GSM8K dataset with each of the 5 transformation modes:
       - Requirements: Adds constraint-based instructions to math problems
       - Paraphrasing: Rewrites problems while preserving mathematical meaning
       - Light Noise: Adds minimal colloquial noise and typos
       - Moderate Noise: Adds medium-level colloquial noise and typos
       - Heavy Noise: Adds maximum colloquial noise and typos
    
    The function processes the same input file multiple times, generating different
    augmented versions for robustness testing of mathematical reasoning models.
    
    Returns:
        None: Creates multiple output files with transformed math problems
    """
    # Define input file path (original GSM8K dataset)
    input_file = "gsm8k_original.jsonl"
    
    # Define output file paths for each transformation mode
    requirements_output_file = "gsm8k_requirements.jsonl"
    paraphrasing_output_file = "gsm8k_paraphrasing.jsonl"
    light_noise_output_file = "gsm8k_light_noise.jsonl"
    moderate_noise_output_file = "gsm8k_moderate_noise.jsonl"
    heavy_noise_output_file = "gsm8k_heavy_noise.jsonl"

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