"""
MATH Dataset Mathematical Problem Rewriting Module

This module provides functionality to rewrite mathematical problems from the MATH dataset using various
transformation modes including requirements augmentation, paraphrasing, and noise injection.
It supports multiple noise levels (light, moderate, heavy) to test model robustness on mathematical reasoning tasks.

The module includes sophisticated protection mechanisms for mathematical content including LaTeX expressions,
asy diagrams, and code blocks to ensure mathematical integrity during transformations.

Author: RobustFlow Team
Date: 2024
Purpose: Generate augmented training data for mathematical reasoning models
"""

# Standard library imports
import json          # For JSON file handling and data serialization
import os           # For file system operations and path handling
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


# Mathematical content protection patterns - regex patterns to identify and protect mathematical expressions
# These patterns ensure that LaTeX, asy diagrams, and code blocks are not modified during transformations

# Pattern for code fences (```...```)
FENCE_CODE = re.compile(r"```.*?```", re.S)

# Pattern for asy diagram blocks ([asy]...[/asy])
ASY_BLOCK  = re.compile(r"\[asy\].*?\[/asy\]", re.S | re.I)

# Pattern for double dollar LaTeX expressions ($$...$$)
LATEX_DOLLAR_DBL = re.compile(r"\$\$(?:\\.|[^$])*\$\$", re.S)

# Pattern for LaTeX expressions in parentheses (\(...\))
LATEX_PAREN      = re.compile(r"\\\((?:\\.|[^)])*?\\\)", re.S)

# Pattern for LaTeX expressions in brackets (\[...\])
LATEX_BRACKET    = re.compile(r"\\\[(?:\\.|[^\]])*?\\\]", re.S)

# Pattern for single dollar LaTeX expressions ($...$) - matched last to avoid conflicts
LATEX_DOLLAR     = re.compile(r"\$(?:\\.|[^\$])*\$", re.S)

# Pattern for placeholder tokens used to mask protected content
_PLACEHOLDER_RE = re.compile(r"<<<CB(\d+)>>>")

# List of all protected patterns in order of precedence
PROTECTED_PATTERNS = [
    FENCE_CODE,        # Code fences first (highest priority)
    ASY_BLOCK,         # Asy diagrams second
    LATEX_DOLLAR_DBL,  # Double dollar LaTeX third
    LATEX_BRACKET,     # Bracket LaTeX fourth
    LATEX_PAREN,       # Parenthesis LaTeX fifth
    LATEX_DOLLAR,      # Single dollar LaTeX last (lowest priority)
]

def _collect_spans(text, patterns):
    """
    Collect and merge overlapping spans from multiple regex patterns.
    
    This function finds all matches from the given patterns in the text and merges
    overlapping spans to create a list of non-overlapping protected regions.
    
    Args:
        text (str): The input text to search for patterns
        patterns (list): List of compiled regex patterns to search for
        
    Returns:
        list: List of tuples (start, end) representing non-overlapping spans
    """
    # Initialize list to store all found spans
    spans = []
    
    # Find all matches from each pattern
    for pat in patterns:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    
    # Sort spans by start position for merging
    spans.sort(key=lambda x: x[0])
    
    # Merge overlapping spans
    merged = []
    for s, e in spans:
        # If no merged spans yet or current span doesn't overlap with last merged span
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            # Extend the last merged span to include current span
            merged[-1][1] = max(merged[-1][1], e)
    
    # Convert back to tuples
    return [(s, e) for s, e in merged]

def mask_protected(text):
    """
    Mask protected mathematical content with placeholder tokens.
    
    This function identifies mathematical expressions (LaTeX, asy diagrams, code blocks)
    in the text and replaces them with placeholder tokens to protect them from
    modification during text transformations.
    
    Args:
        text (str): The input text containing mathematical expressions
        
    Returns:
        tuple: A tuple containing:
            - masked_text (str): Text with protected content replaced by placeholders
            - masked_parts (list): List of original protected content chunks
    """
    # Find all protected spans in the text
    spans = _collect_spans(text, PROTECTED_PATTERNS)
    
    # If no protected content found, return original text
    if not spans:
        return text, []
    
    # Initialize lists for building masked text and storing original parts
    masked_parts = []
    out = []
    idx = 0
    
    # Replace each protected span with a placeholder
    for i, (s, e) in enumerate(spans):
        # Add text before the protected span
        out.append(text[idx:s])
        
        # Create placeholder token
        placeholder = f"<<<CB{i}>>>"
        out.append(placeholder)
        
        # Store the original protected content
        masked_parts.append(text[s:e])
        
        # Update index to continue from end of current span
        idx = e
    
    # Add remaining text after last protected span
    out.append(text[idx:])
    
    # Join all parts and return
    return "".join(out), masked_parts

def restore_protected(masked_text: str, masked_parts: list[str]) -> str:
    """
    Restore protected mathematical content from placeholder tokens.
    
    This function takes masked text with placeholder tokens and restores the original
    mathematical expressions by replacing placeholders with their corresponding
    protected content.
    
    Args:
        masked_text (str): Text containing placeholder tokens like <<<CB0>>>
        masked_parts (list[str]): List of original protected content chunks
        
    Returns:
        str: Text with placeholders replaced by original protected content
    """
    def _repl(m: re.Match) -> str:
        """Helper function to replace placeholder with original content."""
        # Extract the index from the placeholder token
        idx = int(m.group(1))
        
        # Check if index is valid and return corresponding protected content
        if 0 <= idx < len(masked_parts):
            return masked_parts[idx]
        
        # Invalid placeholder: return original placeholder (shouldn't happen with proper masking)
        return m.group(0)
    
    # Replace all placeholders with their original content
    return _PLACEHOLDER_RE.sub(_repl, masked_text)

def placeholders_ok(s: str, n: int) -> bool:
    """
    Validate that placeholder tokens are correctly formatted and complete.
    
    This function checks that all placeholder tokens in the text are properly formatted
    and that the expected number of placeholders are present with consecutive indices.
    
    Args:
        s (str): Text containing placeholder tokens to validate
        n (int): Expected number of placeholders
        
    Returns:
        bool: True if placeholders are valid, False otherwise
    """
    # Find all placeholder tokens in the text
    ids = _PLACEHOLDER_RE.findall(s)
    
    try:
        # Convert string IDs to integers
        ids = list(map(int, ids))
    except ValueError:
        # Invalid placeholder format
        return False
    
    # Check that we have the expected number of placeholders
    # and that they form a consecutive sequence starting from 0
    return len(ids) == n and sorted(ids) == list(range(n))

def rewrite_prompt_with_openai(original_prompt, mode):
    """
    Rewrite a MATH dataset mathematical problem using OpenAI API based on specified transformation mode.
    
    This function takes an original mathematical problem and applies one of five transformation modes:
    - 'requirements': Adds constraint-based instructions to strengthen mathematical problem requirements
    - 'paraphrasing': Rewrites problem prose while preserving mathematical meaning and expressions
    - 'light_noise': Adds light colloquial noise to problem prose while keeping math intact
    - 'moderate_noise': Adds moderate colloquial noise to problem prose with increased complexity
    - 'heavy_noise': Adds heavy colloquial noise to problem prose while maintaining recoverability
    
    The function uses sophisticated mathematical content protection to ensure LaTeX expressions,
    asy diagrams, and code blocks remain unchanged during transformations.
    
    Args:
        original_prompt (str): The original mathematical problem to be rewritten
        mode (str): Transformation mode - one of 'requirements', 'paraphrasing',
                   'light_noise', 'moderate_noise', or 'heavy_noise'
        
    Returns:
        str: The rewritten problem with protected mathematical content restored
        
    Raises:
        Exception: If API calls fail after retries, returns the original prompt
    """
    # Mask protected mathematical content (LaTeX, asy, code blocks) with placeholders
    masked_text, protected_chunks = mask_protected(original_prompt)

    # Define system prompt for requirements mode - strengthens mathematical problem constraints
    system_prompt_requirements = r"""
You are a **math problem constraint augmenter (MATH dataset version)**. For a given problem *P* (algebra/number theory/geometry/combinatorics), add **2–4 deterministic, process-only constraints** that enrich the reasoning **without changing the original answer**.

## Protected placeholders & formatting (critical)

* The input may contain protected placeholders like `<<<CB0>>>`, `<<<CB1>>>`, … which stand for LaTeX, asy diagrams, or code blocks.
* **Copy these placeholders verbatim, keep them in the same positions, and do not edit around them.** Do **not** introduce, remove, or reorder placeholders.
* Do **not** evaluate, simplify, or alter any LaTeX/asy/code content (it will be restored later).
* Keep overall length within **±20%** of the original and do **not** translate the language or change tone.

## Invariants (must not change)

* **All original numbers, symbols, relations, and quantifiers** (e.g., equalities/inequalities, “acute,” “convex,” base of logs if given, units like degrees vs. radians).
* **What is being asked** (quantity/form of the final result: exact/approximate, rounding rule, units/base).
* **Domains and counting rules** already present (e.g., integers vs. reals, uniqueness/duplication policies).
* Single-answer feasibility must be preserved.

## Allowed constraint types (choose any 2–4; only make explicit what is already standard or implied)

* **Explicit conventions/units already implied:** e.g., “angles are measured in degrees,” “work over real numbers,” “logs use the base already specified by the problem.”
* **Benign variable/label introductions:** name existing quantities without adding new relations (e.g., “Let the two congruent acute angles each be $\alpha$ with $0<\alpha<90^\circ$,” “Let $x$ denote the common measure”).
* **Neutral auxiliary constructions:** add objects commonly used in standard solutions **without asserting new equalities** (e.g., “Label the pentagon $A,B,C,D,E$ clockwise; draw diagonal $AC$,” “Introduce $S$ to denote the sum $<<<CB0>>>$”).
* **Nondegeneracy and domain bounds already entailed by *P*:** e.g., positivity/acuity/convexity intervals that are logical consequences of the given statements.
* **Method/process guidance that doesn’t change the value computed:** e.g., “Use the interior-angle-sum identity,” “Convert repeating decimals to fractions before inversion,” “Round only at the final step to the required place.”

> If uncertain whether a constraint is safely implied by *P*, **omit it**.

## Guardrails

* Do **not** assert new numerical relations, special positions, coprimality/parity/primality, or additional equalities/congruences that narrow the solution space.
* Do **not** pin labels to specific locations unless already fixed by *P* (e.g., don’t declare which vertices carry the acute angles if *P* didn’t).
* Do **not** alter rounding rules, bases, units, requested form, or introduce randomness/probabilities.
* Do **not** change variable domains (e.g., reals → integers) or add side conditions that could create multiple answers or infeasibility.
* **Placeholders must remain exactly as received** (same tokens, same order, same count).

## Internal procedure (do not reveal)

1. **Privately solve** the original problem to obtain the answer $A$.
2. Identify the key target/quantity and the structural facts that lead to $A$.
3. Add 2–4 constraints chosen above that **only scaffold the reasoning** (naming, units, neutral constructions, method guidance) so that $A$ is still the unique answer.
4. Self-check: numbers/symbols/quantifiers unchanged; placeholders intact; still feasible and single-answer; language and length preserved.

## Output format (strict)

Output **only** the augmented problem text, wrapped **exactly once** with:

<answer>  
augmented problem text (including placeholders unchanged)  
</answer>
    """

    # Define user prompt for requirements mode - template for constraint-based mathematical problem augmentation
    user_prompt_requirements = r"""
You are given a **MATH dataset problem** $P$ (algebra / number theory / geometry / combinatorics).
**Note:** $P$ may include protected placeholders such as `<<<CB0>>>`, `<<<CB1>>>`, … representing LaTeX/asy/code. **Copy these placeholders verbatim, keep them in the same positions/order/count, and do not modify text inside or immediately around them.**

**Task:** Add **2–4 deterministic, process-only constraints** that make the problem richer **without changing the original answer**. Insert these constraints naturally into the problem text while keeping the original question (what is being asked) exactly the same.

**Strict preservation:**

* Keep **all original numbers, symbols, relations, and quantifiers** (e.g., equalities/inequalities, “acute,” “convex,” bases of logs).
* Keep **the requested quantity and form** unchanged (e.g., exact value vs. approximation, rounding place, representation, units like degrees/radians, base of logarithms).
* **Do not evaluate, simplify, alter, or move** any LaTeX/asy/code content; **do not add/remove/reorder** placeholder tokens.
* Preserve the **language and tone** of the problem (do not translate); keep overall length within **±20%** of the original.
* The augmented problem must remain **feasible** and yield a **single correct answer**.
* Do **not** introduce randomness/probabilities or change domains/definitions.

**Allowed constraint styles (pick any 2–4):**

* **Explicit conventions/units already implied** (e.g., “angles measured in degrees,” “work over real numbers”).
* **Benign naming/labeling** without new relations (e.g., “Label the pentagon $A,B,C,D,E$ clockwise,” “Let the two congruent acute angles each be $\alpha$ with $0<\alpha<90^\circ$”).
* **Neutral auxiliary constructions** commonly used in solutions **without asserting new equalities/special positions** (e.g., “draw diagonal $AC$,” “introduce $S$ to denote the sum $<<<CB0>>>$”).
* **Nondegeneracy/domain bounds** already entailed by $P$ (e.g., positivity, convexity intervals).
* **Method/process guidance** that does not alter the computed value (e.g., “use the interior-angle-sum identity,” “convert repeating decimals to fractions before inversion,” “round only at the final step to the required place”).

**Prohibited changes (guardrails):**

* Do **not** assert new numerical relations, special placements, coprimality/parity/primality, additional equalities/congruences, or figure-specific constraints that narrow the solution set.
* Do **not** alter rounding rules, bases, units, requested form, or what quantity is asked.
* Do **not** add/remove examples, data, or requirements beyond the new constraints.

**Output format (mandatory):** <answer>
\<the augmented problem text only, with placeholders unchanged and the original question preserved> </answer>

No explanations, no solutions, and no extra text outside the tags. No code fences/backticks.

**Original prompt:**
{{original_prompt}}
    """

    # Define system prompt for paraphrasing mode - rewrites mathematical problem prose while preserving math
    system_prompt_paraphrasing = r"""
You are a **MATH dataset problem rewriter**. Given a problem *P* (algebra / number theory / geometry / combinatorics), rewrite **only the natural-language prose** while **preserving the exact mathematical meaning and the numerical answer**. The goal is to change *form* (voice, sentence mood, order, register, nominalization) **without changing content**.

## Protected placeholders & formatting (critical)

* *P* may include protected placeholders like `<<<CB0>>>`, `<<<CB1>>>`, … representing LaTeX/asy/code.
* **Copy these placeholders verbatim, keep the same positions/order/count, and do not modify text inside or immediately around them.**
* Do **not** evaluate, simplify, or alter any LaTeX/asy/code content (it is restored later).

## Task

* Paraphrase the problem text (narrative + question) in the **same language** as the input.
* Keep total length within **±20%** of the original.
* If a rewrite risks meaning drift or ambiguity, choose the closest paraphrase—or leave that sentence unchanged.

## Strict preservation

* **Do not alter any facts, numbers, symbols, relations, or quantifiers** (e.g., equalities/inequalities, “acute,” “convex,” domain specifications).
* **Do not change what is being asked** (the quantity/form of the final result), including **exact vs. approximate**, rounding place, degree/radian units, bases of logs, or representation requirements.
* **Digits stay digits** (do not spell out or round). Repeating-decimal notation (e.g., `.\overline{2}`) and any given variable names must remain as is.
* **Keep all constraints, units, definitions, and counting rules** exactly (e.g., uniqueness vs. total, inclusivity of bounds).
* **Do not translate**; preserve the original language and general tone (only mild register shifts allowed).
* **Do not add or remove information**, hints, examples, data, or requirements.
* **Proper nouns, identifiers, inline literals, and placeholders** must remain **verbatim**.
* The rewritten problem must remain **feasible** and yield the **same single correct answer**.

## Allowed transformations (light touch)

* Voice (active ↔ passive), sentence mood (declarative/imperative/interrogative).
* Information order (merge/split sentences while preserving content).
* Register/tonality (slightly more formal or plain).
* Nominalization vs. verbal phrasing.
* Strict near-synonyms **in the same language**.
* Benign labeling phrases that **do not introduce new relations** (e.g., “label the figure” wording) are allowed only if already implicit and **do not change content**.

## Do NOT

* Do not introduce randomness/probabilities unless already present.
* Do not change specificity, scope, difficulty, domains, or add side conditions.
* Do not alter units, definitions, bases, rounding rules, or implicit counting conventions.
* Do not add/remove constraints or data, or assert new equalities/congruences/positions.

## Output format (mandatory)

Wrap **only** the rewritten problem text in exactly one pair of tags:

<answer>
rewritten problem text here, preserving meaning and answer; placeholders unchanged
</answer>

No explanations, no solutions, and no extra text outside the tags. No code fences/backticks.
    """

    # Define user prompt for paraphrasing mode - template for mathematical problem prose paraphrasing
    user_prompt_paraphrasing = r"""
You are given a **MATH dataset problem** $P$ (algebra / number theory / geometry / combinatorics).

**Placeholder notice:** $P$ may include protected placeholders such as `<<<CB0>>>`, `<<<CB1>>>`, … which represent LaTeX/asy/code. **Copy these placeholders verbatim, keep the same positions/order/count, and do not modify text inside or immediately around them.** Do not evaluate/simplify any LaTeX/asy/code.

**Task**

* Rewrite **only the natural-language prose** of $P$ while preserving meaning and the original numerical answer.
* Change only the **form** (voice, sentence mood, order, register, nominalization, strict near-synonyms).
* Keep total length within **±20%** of the original and keep the **same language** as the input.

**Strict preservation**

* Do **not** alter any **facts, numbers, symbols, relations, or quantifiers**; keep digits as digits.
* Do **not** change **what is being asked** (the same quantity/form must be computed), including exact vs. approximate form, rounding place, representation, degree/radian units, bases of logs, etc.
* Keep all **constraints, units, and counting rules** exactly (e.g., uniqueness vs. total, inclusivity of bounds).
* Do **not** translate; keep the original language and overall tone (only mild register shifts).
* Do **not** add or remove information, hints, examples, or requirements.
* **Proper nouns, identifiers, inline literals, and placeholders** (e.g., `<...>`, `{...}`, URLs, file paths, regexes, numbers, `<<<CBi>>>`) must remain **verbatim**.
* The rewritten problem must remain **feasible** and yield the **same single correct answer**.

**Feasibility & tie-breakers**

* If a rewrite risks meaning drift or ambiguity, choose the **closest paraphrase** or leave that sentence unchanged.
* If the original is already minimal/clear, make only **cosmetic edits**.

**Output format (mandatory)** <answer>
\<the rewritten problem text only, preserving meaning and answer; placeholders unchanged> </answer>

No explanations, no solutions, and no extra text outside the tags. No code fences/backticks.

**Original prompt:**
{{original_prompt}}
    """

    # Define system prompt for light noise mode - adds minimal colloquial noise to mathematical problem prose
    system_prompt_light_noise = r"""
You are a **lightly-colloquial prompt noiser** for **MATH dataset problems** (algebra / number theory / geometry / combinatorics). Given a problem *P*, inject **light-noise, human-like edits** into the **natural-language prose only** while keeping the task **recoverable** and the **numerical answer unchanged**.

## Protected placeholders & technical literals (critical)

* *P* may include protected placeholders like `<<<CB0>>>`, `<<<CB1>>>`, … standing for LaTeX/asy/code blocks. **Copy these placeholders verbatim, keep the same positions/order/count, and do not modify text inside or immediately around them.**
* Do **not** evaluate/simplify/alter any LaTeX/asy/code content. Treat inline math and literals as **verbatim**: `$...$`, `\(...\)`, `\[...\]`, `$$...$$`, `[asy]...[/asy]`, and code fences `...`.
* **Never** add/remove/reorder placeholder tokens.

## Task

* Noise the problem’s **prose** (story/context + question) to sound gently chatty and slightly messy, yet fully understandable to a careful grader.
* **Preserve meaning, scope, and specificity.** Do **not** add/remove facts, constraints, hints, or steps.
* Keep total length within **±20%** of the original.
* Keep the **final question** present and understandable.
* Keep the **same language** as the input.

## Style goal

Make the text lightly colloquial—occasional slang, a few fillers, mild typos, small punctuation quirks—**but** leave a very clean recovery path.

## Noise palette (lighter profile)

* **Typos & misspellings (primary):** small insert/delete/substitute/transpose; mild letter doubling/drops. **≈50–60% of all edits** should be from this class. Keep at least **one clean mention** of each critical symbol/quantity (variables like $x,\alpha$, units like “degrees,” and all numbers).
* **Soft slang & IM speak** (sparingly): uh/lemme/wanna/BTW/tho.
* **Contractions & light word drops:** only where safe and natural.
* **Gentle vowel stretching & light stutter:** “reaaally”, “kinda”.
* **Mild casing & punctuation variety:** 1–2 anomalies per paragraph (e.g., extra comma/space, a single !!! or ?? once).
* **Leet/char swaps:** extremely sparing (maybe once).
* **Emojis:** ≤1 total, optional.
* **Tiny symbol runs:** `&^%$#` (≤6 chars; ≤1 per paragraph; only at clause boundaries).
* **Fragments & run-ons:** at most **one** minor fragment **or** one short run-on; include **one clean full sentence** stating the core task.

## Hard rules (recovery & semantics)

* **Language unchanged:** do not translate.
* **Digits, symbols, and comparatives stay exact:** never alter numeric values, variables/symbols, or words like “longer/higher/compared to/than/between/at least/at most/equal to”.
* **Preserve units, bases, and forms** (e.g., degrees vs. radians, base of logarithms, exact vs. approximate, rounding place, representation requirements like “decimal”).
* **Do not change what is asked.** Keep the same computed quantity; keep the question present (preferably last).
* **Protect verbatim technical literals in prose:** equations, inequalities, inline math, identifiers, names, dates, numbers, units, file paths, URLs, regexes, special tokens (`<...>`, `{...}`, `$...`), and **all** `<<<CBi>>>` placeholders must remain byte-for-byte.
* **No new constraints/data** and **no solution steps**.
* **Recoverability:** include at least **one clean, unambiguous full sentence** summarizing the task, and **one clean occurrence** of every critical symbol and number (e.g., $\alpha$, “degrees”, targets, totals).

## Intensity & limits

* Target **25–40%** tokens noised; up to **45%** if still clearly readable.
* Use **3–4** noise types overall; stack at most **2 edits per token**.
* Avoid noising **every** instance of a critical term/symbol/number.
* Do not insert or remove blank lines in a way that splits the problem unnaturally.

## Output format (mandatory)

Wrap **only** the noised problem text in **exactly one** pair of tags:

<answer>
<noised problem text here, same meaning and answer; placeholders unchanged>
</answer>

No explanations, no solutions, and no extra text outside the tags. No code fences/backticks.

---

## Calibrated examples (do not echo at runtime)

**Geometry (angles):**
From: “A particular convex pentagon has two congruent, acute angles… What is the common measure of the large angles, in degrees?”
To (light-noised): <answer>So, picture a convex pentagon: two angles are congruent and actually acute. For each of the other interior angles, its measure equals the **sum** of those two acute ones—yep, that exact rule. We’re working in degrees. One clean ask: What is the common measure of the larger angles, in degrees?</answer>

**Estimation/rounding:**
From: “Estimate $14.7923412^2$ to the nearest hundred.”
To (light-noised): <answer>Quick one: square $14.7923412$ (don’t change those digits), then give an **estimate** rounded to the nearest hundred. What value do we get to the nearest hundred?</answer>

**Repeating decimals:**
From: “Let $a = .\overline{2} + .\overline{6}$. Find the reciprocal of $a$, expressed as a decimal.”
To (light-noised): <answer>Let $a = .\overline{2} + .\overline{6}$ exactly as written. Now find the reciprocal $1/a$ and write it as a decimal—give that result.</answer>
    """

    # Define user prompt for light noise mode - template for light noise injection in mathematical problem prose
    user_prompt_light_noise = r"""
You are given a **MATH dataset problem** $P$ (algebra / number theory / geometry / combinatorics, plain text).

**Placeholder notice:** $P$ may include protected placeholders such as `<<<CB0>>>`, `<<<CB1>>>`, … representing LaTeX/asy/code. **Copy these placeholders verbatim, keep the same positions/order/count, and do not modify text inside or immediately around them.** Do **not** evaluate/simplify any LaTeX/asy/code. Treat inline/block math and literals as **verbatim**: `$...$`, `\(...\)`, `\[...\]`, `$$...$$`, `[asy]...[/asy]`, and code fences `...`.

**Task**

* Inject **light-noise, lightly-colloquial edits** into the problem’s **prose** (story/context + question) while keeping it **recoverable** and preserving the original meaning and **numerical answer**.
* Keep total length within **±20%** of the original.
* Keep the **final question** present and understandable.
* Keep the **same language** as the input.

**Style goal**

* Make it gently chatty: a bit of slang, a few fillers, mild typos, small punctuation quirks — still easy for a careful grader to read.

**Noise palette (use several, lightly)**

* **Typos & misspellings as the primary edit class (≈50–60% of all edits).** Keep at least **one clean mention** of each critical symbol/quantity (variables like $x,\alpha$, numbers, and units like “degrees”).
* Soft slang & IM speak (sparingly): uh/lemme/wanna/BTW/tho.
* Contractions & safe word drops.
* Mild vowel stretching & light stutter (“reaaally”, “kinda”).
* Gentle casing & punctuation variety: 1–2 anomalies per paragraph.
* Leet/char swaps: very rare.
* Emojis ≤1 total (optional).
* Tiny symbol runs `&^%$#` (≤6 chars each; ≤1 per paragraph; only at clause boundaries).
* Allow at most **one** minor fragment **or** one short run-on; include **one clean full sentence** stating the core task.

**Hard rules (must preserve)**

* **Do not translate**; keep the original language.
* **Do not add/remove facts, constraints, hints, data, or solution steps.**
* **Do not change what is asked**; the same quantity/form must be computed; keep the question in the text (preferably last).
* **Digits, symbols, and comparative/connective words stay exact** (numbers; variables; “longer/higher/compared to/than/between/at least/at most/equal to”).
* **Preserve units, bases, and representation rules** (degrees vs. radians, base of logs, exact vs. approximate, rounding place, “express as a decimal/fraction,” etc.).
* **Protect verbatim technical literals inside prose:** equations, inequalities, identifiers, names, dates, numbers/units, file paths, URLs, regexes, special tokens (`<...>`, `{...}`, `$...`), and **all** `<<<CBi>>>` placeholders **byte-for-byte**.
* **Recoverability:** include at least **one clean, unambiguous full sentence** summarizing the task, and **one clean occurrence** of every critical symbol and number.

**Intensity & limits**

* Target **25–40%** tokens noised (≤45% if still clearly readable).
* Use **3–4** noise types overall; ≤2 edits stacked per token.
* Do not noise **every** instance of a critical term/symbol/number.
* Do not insert or remove blank lines in a way that splits the problem unnaturally.

**Output format (mandatory)** <answer>
<noised problem text only, same meaning and answer; placeholders unchanged> </answer>

No explanations, no solutions, and no extra text outside the tags. No code fences/backticks.

**Original prompt:**
{{original_prompt}}
    """
    
    # Define system prompt for moderate noise mode - adds medium-level colloquial noise to mathematical problem prose
    system_prompt_moderate_noise = r"""
You are a **moderately-colloquial prompt noiser** for **MATH dataset problems** (algebra / number theory / geometry / combinatorics). Given a problem *P*, inject **moderate-noise, human-like edits** into the **natural-language prose only** while keeping the task **recoverable** and the **numerical answer unchanged**.

## Protected placeholders & technical literals (critical)

* *P* may include protected placeholders like `<<<CB0>>>`, `<<<CB1>>>`, … standing for LaTeX/asy/code blocks. **Copy these placeholders verbatim, keep the same positions/order/count, and do not modify text inside or immediately around them.**
* Do **not** evaluate/simplify/alter any LaTeX/asy/code content. Treat inline math and literals as **verbatim**: `$...$`, `\(...\)`, `\[...\]`, `$$...$$`, `[asy]...[/asy]`, and code fences `...`.
* **Never** add/remove/reorder placeholder tokens.

## Task

* Noise the problem’s **prose** (story/context + question) so it sounds casual and a bit messy, yet clearly understandable to a careful grader.
* **Preserve meaning, scope, and specificity.** Do **not** add/remove facts, constraints, hints, or steps.
* Keep total length within **±25%** of the original.
* Keep the **final question** present and understandable.
* Keep the **same language** as the input.

## Style goal

Make the text moderately colloquial—slang/fillers/typos sprinkled in, some punctuation variety—**but** leave an obvious recovery path.

## Noise palette (moderate profile)

* **Typos & misspellings (primary):** insert/delete/substitute/transpose; light letter doubling/drops; adjacent-key slips. **≥60% of all edits** should be from this class. Keep at least **one clean mention** of each critical symbol/quantity (variables like $x,\alpha$, units like “degrees,” and all numbers).
* **Slang & IM speak:** uh/lemme/gonna/wanna/tbh/ngl/low-key/BTW/tho (use moderately).
* **Contractions & word drops:** elide minor auxiliaries/articles/preps where safe.
* **Vowel stretching & stutter:** occasional “reaaally”, “y-yeah”.
* **Hedges & fillers:** like, kinda, sorta, basically, idk? short asides.
* **Casing & punctuation chaos:** 2–4 anomalies per paragraph (e.g., !!!, ??, random CAPS, duplicated/missing comma/space).
* **Leet/char swaps:** 0↔o, 1↔l, i↔l, sparingly.
* **Emojis & emoticons:** ≤2 total per problem.
* **Random symbol runs:** `&^%$#@~` etc., each ≤8 chars, ≤1–2 runs per paragraph, only at clause boundaries.
* **Fragments & run-ons:** allow **one** fragment **or** **one** mild run-on; include **one clean full sentence** stating the core task.

## Hard rules (recovery & semantics)

* **Language unchanged:** do not translate.
* **Digits, symbols, and comparatives stay exact:** never alter numeric values, variables/symbols, or words like “longer/higher/compared to/than/between/at least/at most/equal to”.
* **Preserve units, bases, and forms** (e.g., degrees vs. radians, base of logarithms, exact vs. approximate, rounding place, representation requirements like “decimal”).
* **Do not change what is asked.** Keep the same computed quantity; keep the question present (preferably last).
* **Protect verbatim technical literals in prose:** equations, inequalities, inline math, identifiers, names, dates, numbers, units, file paths, URLs, regexes, special tokens (`<...>`, `{...}`, `$...`), and **all** `<<<CBi>>>` placeholders must remain byte-for-byte.
* **No new constraints/data** and **no solution steps**.
* **Recoverability:** include at least **one clean, unambiguous full sentence** summarizing the task, and **one clean occurrence** of every critical symbol and number (e.g., $\alpha$, “degrees”, targets, totals).

## Intensity & limits

* Target **40–60%** tokens noised; up to **65%** if still readable.
* Use **4–5** noise types overall; stack at most **2–3 edits per token**.
* Avoid noising **every** instance of a critical term/symbol/number.
* Do not insert or remove blank lines in a way that splits the problem unnaturally.

## Output format (mandatory)

Wrap **only** the noised problem text in **exactly one** pair of tags:

<answer>
<noised problem text here, same meaning and answer; placeholders unchanged>
</answer>

No explanations, no solutions, and no extra text outside the tags. No code fences/backticks.

---

## Calibrated examples (do not echo at runtime)

**Geometry (angles):**
From: “A particular convex pentagon has two congruent, acute angles… What is the common measure of the large angles, in degrees?”
To (moderate-noised): <answer>Okay, picture a convex pentagon: two angles are congruent and they’re actually acute. For each of the other interior angles, its measure equals the **sum** of those two acute ones—yep, that rule stays. We’re working in degrees, BTW. Clean ask: What is the common measure of the larger angles, in degrees?</answer>

**Estimation/rounding:**
From: “Estimate $14.7923412^2$ to the nearest hundred.”
To (moderate-noised): <answer>Quick task: square $14.7923412$ (don’t touch the digits), then give an **estimate** rounded to the nearest hundred—no extra tricks. What’s that value to the nearest hundred?</answer>

**Repeating decimals:**
From: “Let $a = .\overline{2} + .\overline{6}$. Find the reciprocal of $a$, expressed as a decimal.”
To (moderate-noised): <answer>Set $a = .\overline{2} + .\overline{6}$ exactly as written. Now, find the reciprocal $1/a$ and write it as a decimal—report that number.</answer>
    """
    
    # Define user prompt for moderate noise mode - template for moderate noise injection in mathematical problem prose
    user_prompt_moderate_noise = r"""
You are given a **MATH dataset problem** $P$ (algebra / number theory / geometry / combinatorics, plain text).

**Placeholder notice:** $P$ may include protected placeholders such as `<<<CB0>>>`, `<<<CB1>>>`, … representing LaTeX/asy/code. **Copy these placeholders verbatim, keep the same positions/order/count, and do not modify text inside or immediately around them.** Do **not** evaluate/simplify any LaTeX/asy/code. Treat inline/block math and literals as **verbatim**: `$...$`, `\(...\)`, `\[...\]`, `$$...$$`, `[asy]...[/asy]`, and code fences `...`.

**Task**

* Inject **moderate-noise, colloquial edits** into the problem’s **prose** (story/context + question) while keeping it **recoverable** and preserving the original meaning and **numerical answer**.
* Keep total length within **±25%** of the original.
* Keep the **final question** present and understandable.
* Keep the **same language** as the input.

**Style goal**

* Make it casually chatty with visible but not overwhelming noise: some slang/fillers/typos and punctuation variety—still readable to a careful grader.

**Noise palette (use several)**

* **Typos & misspellings as the primary edit class (≥60% of all edits).** Keep at least **one clean mention** of each critical symbol/quantity (variables like $x,\alpha$, numbers, and units like “degrees”).
* Slang & IM speak in moderation: uh/lemme/gonna/wanna/tbh/ngl/low-key/BTW/tho.
* Contractions & safe word drops.
* Occasional vowel stretching & light stutter (“reaaally”, “y-yeah”).
* Casing & punctuation chaos: aim **2–4 anomalies** per paragraph.
* Leet/char swaps: sparing (0↔o, 1↔l, i↔l).
* Emojis ≤2 total.
* Random symbol runs `&^%$#@~` (≤8 chars each; ≤1–2 per paragraph; only at clause boundaries).
* Allow **one** minor fragment **or** **one** mild run-on; include **one clean full sentence** stating the core task.

**Hard rules (must preserve)**

* **Do not translate**; keep the original language.
* **Do not add/remove facts, constraints, hints, data, or solution steps.**
* **Do not change what is asked**; the same quantity/form must be computed; keep the question in the text (preferably last).
* **Digits, symbols, and comparative/connective words stay exact** (numbers; variables; “longer/higher/compared to/than/between/at least/at most/equal to”).
* **Preserve units, bases, and representation rules** (degrees vs. radians, base of logs, exact vs. approximate, rounding place, “express as a decimal/fraction,” etc.).
* **Protect verbatim technical literals inside prose:** equations, inequalities, identifiers, names, dates, numbers/units, file paths, URLs, regexes, special tokens (`<...>`, `{...}`, `$...`), and **all** `<<<CBi>>>` placeholders **byte-for-byte**.
* **Recoverability:** include at least **one clean, unambiguous full sentence** summarizing the task, and **one clean occurrence** of every critical symbol and number.

**Intensity & limits**

* Target **40–60%** tokens noised (≤65% if still readable).
* Use **4–5** noise types overall; ≤2–3 edits stacked per token.
* Do not noise **every** instance of a critical term/symbol/number.
* Do not insert or remove blank lines in a way that splits the problem unnaturally.

**Output format (mandatory)** <answer>
\<noised problem text only, same meaning and answer; placeholders unchanged> </answer>

No explanations, no solutions, and no extra text outside the tags. No code fences/backticks.

**Original prompt:**
{{original_prompt}}
    """

    # Define system prompt for heavy noise mode - adds maximum colloquial noise to mathematical problem prose
    system_prompt_heavy_noise = r"""
You are an **ultra-colloquial prompt noiser** for **MATH dataset problems** (algebra / number theory / geometry / combinatorics). Given a problem *P*, inject **high-noise, human-like edits** into the **natural-language prose only** while keeping the task **recoverable** and the **numerical answer unchanged**.

## Protected placeholders & technical literals (critical)

* *P* may include protected placeholders like `<<<CB0>>>`, `<<<CB1>>>`, … standing for LaTeX/asy/code blocks. **Copy these placeholders verbatim, keep the same positions/order/count, and do not modify text inside or immediately around them.**
* Do **not** evaluate/simplify/alter any LaTeX/asy/code content. Treat inline math and literals as **verbatim**: `$...$`, `\(...\)`, `\[...\]`, `$$...$$`, `[asy]...[/asy]`, and code fences `...`.
* **Never** add/remove/reorder placeholder tokens.

## Task

* Noise the problem’s **prose** (story/context + question) to sound chatty/messy yet still understandable to a careful grader.
* **Preserve meaning, scope, and specificity.** Do **not** add/remove facts, constraints, hints, or steps.
* Keep total length within **±30%** of the original.
* Keep the **final question** present and understandable.
* Keep the **same language** as the input.

## Style goal

Make the text ultra-colloquial and noisy—slang, fillers, typos, stretched vowels, rANdoM caps, punctuation chaos—**but** leave a clean recovery path.

## Noise palette (heavier profile)

* **Typos & misspellings (primary):** insert/delete/substitute/transpose; letter doubling/drops; adjacent-key slips. **≥65% of all edits** should be from this class. Keep at least **one clean mention** of each critical symbol/quantity (variables like $x,\alpha$, units like “degrees,” and all numbers).
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
* **Digits, symbols, and comparatives stay exact:** never alter numeric values, variables/symbols, or words like “longer/higher/compared to/than/between/at least/at most/equal to”.
* **Preserve units, bases, and forms** (e.g., degrees vs. radians, base of logarithms, exact vs. approximate, rounding place, representation requirements like “decimal”).
* **Do not change what is asked.** Keep the same computed quantity; keep the question present (preferably last).
* **Protect verbatim technical literals in prose:** equations, inequalities, inline math, identifiers, names, dates, numbers, units, file paths, URLs, regexes, special tokens (`<...>`, `{...}`, `$...`), and **all** `<<<CBi>>>` placeholders must remain byte-for-byte.
* **No new constraints/data** and **no solution steps**.
* **Recoverability:** include at least **one clean, unambiguous full sentence** summarizing the task, and **one clean occurrence** of every critical symbol and number (e.g., $\alpha$, “degrees”, targets, totals).

## Intensity & limits

* Target **60–80%** tokens noised; up to **85%** if still readable.
* Use **4–6** noise types overall; stack at most **3 edits per token**.
* Avoid noising **every** instance of a critical term/symbol/number.
* Do not insert or remove blank lines in a way that splits the problem unnaturally.

## Output format (mandatory)

Wrap **only** the noised problem text in **exactly one** pair of tags:

<answer>
<noised problem text here, same meaning and answer; placeholders unchanged>
</answer>

No explanations, no solutions, and no extra text outside the tags. No code fences/backticks.

---

## Calibrated examples (do not echo at runtime)

**Geometry (angles):**
From: “A particular convex pentagon has two congruent, acute angles… What is the common measure of the large angles, in degrees?”
To (noised): <answer>ok sooo there’s this convex pentagon, right, two angles are congruent *and* acute (yeah, actually acute). For every other interior angle, its measure = the **sum** of those two acute ones — like, that’s the rule, not changing it. We’re still working in **degrees**, BTW. One clean line: What’s the common measure of the larger angles, in degrees? 🤔</answer>

**Estimation/rounding:**
From: “Estimate \$14.7923412^2\$ to the nearest hundred.”
To (noised): <answer>lemme be quick: you gotta square \$14.7923412\$ (don’t touch the digits!), then give an **estimate** rounded to the nearest hundred — same policy as stated, no funny business. So, what’s that value to the nearest hundred??</answer>

**Repeating decimals:**
From: “Let \$a = .\overline{2} + .\overline{6}\$. Find the reciprocal of \$a\$, expressed as a decimal.”
To (noised): <answer>ngl this is tiny but neat: set $a = .\overline{2} + .\overline{6}$ (yep, exactly that form). Don’t change the notation. Then, what’s the reciprocal $1/a$ written as a decimal—give that, clean and simple. 🙂</answer>
    """

    # Define user prompt for heavy noise mode - template for heavy noise injection in mathematical problem prose
    user_prompt_heavy_noise = r"""
You are given a **MATH dataset problem** $P$ (algebra / number theory / geometry / combinatorics, plain text).

**Placeholder notice:** $P$ may include protected placeholders such as `<<<CB0>>>`, `<<<CB1>>>`, … representing LaTeX/asy/code. **Copy these placeholders verbatim, keep the same positions/order/count, and do not modify text inside or immediately around them.** Do **not** evaluate/simplify any LaTeX/asy/code. Treat inline/block math and literals as **verbatim**: `$...$`, `\(...\)`, `\[...\]`, `$$...$$`, `[asy]...[/asy]`, and code fences `...`.

**Task**

* Inject **high-noise, ultra-colloquial edits** into the problem’s **prose** (story/context + question) while keeping it **recoverable** and preserving the original meaning and **numerical answer**.
* Keep total length within **±30%** of the original.
* Keep the **final question** present and understandable.
* Keep the **same language** as the input.

**Style goal**

* Make it chatty/messy: slang, fillers, typos, stretched vowels, rANdoM caps, punctuation chaos — yet still readable to a careful grader.

**Noise palette (use several)**

* **Typos & misspellings as the primary edit class (≥65% of all edits).** Keep at least **one clean mention** of each critical symbol/quantity (variables like $x,\alpha$, numbers, and units like “degrees”).
* Slang & IM speak (uh/erm/lemme/gonna/wanna/tbh/ngl/low-key/high-key/lol/bruh/tho/bc/BTW).
* Contractions & light word drops (safe elisions).
* Vowel stretching & stutter (e.g., “reaaally”, “y-yeah”).
* Hedges & fillers (like, kinda, sorta, basically, idk?).
* Casing & punctuation chaos (!!!??!?.., random CAPS, duplicated/missing commas/spaces; aim **3–6 anomalies** per paragraph).
* Leet/char swaps sparingly (0↔o, 1↔l, i↔l).
* Emojis ≤3 total.
* Random symbol runs `&^%$#@~` (≤10 chars each; ≤2 per paragraph; only at clause boundaries).
* Allow one fragment and/or one run-on; include **one clean full sentence** stating the core task.

**Hard rules (must preserve)**

* **Do not translate**; keep the original language.
* **Do not add/remove facts, constraints, hints, data, or solution steps.**
* **Do not change what is asked**; the same quantity/form must be computed; keep the question in the text (preferably last).
* **Digits, symbols, and comparative/connective words stay exact** (numbers; variables; “longer/higher/compared to/than/between/at least/at most/equal to”).
* **Preserve units, bases, and representation rules** (degrees vs. radians, base of logs, exact vs. approximate, rounding place, “express as a decimal/fraction,” etc.).
* **Protect verbatim technical literals inside prose:** equations, inequalities, identifiers, names, dates, numbers/units, file paths, URLs, regexes, special tokens (`<...>`, `{...}`, `$...`), and **all** `<<<CBi>>>` placeholders **byte-for-byte**.
* **Recoverability:** include at least **one clean, unambiguous full sentence** summarizing the task, and **one clean occurrence** of every critical symbol and number.

**Intensity & limits**

* Target **60–80%** tokens noised (≤85% if still readable).
* Use **4–6** noise types overall; ≤3 edits stacked per token.
* Do not noise **every** instance of a critical term/symbol/number.
* Do not insert or remove blank lines in a way that splits the problem unnaturally.

**Output format (mandatory)** <answer>
\<noised problem text only, same meaning and answer; placeholders unchanged> </answer>

No explanations, no solutions, and no extra text outside the tags. No code fences/backticks.

**Original prompt:**
{{original_prompt}}
    """
    
    # Replace template placeholder with actual masked text in all user prompts
    user_prompt_requirements = user_prompt_requirements.replace(
        "{{original_prompt}}", masked_text
    )
    user_prompt_paraphrasing = user_prompt_paraphrasing.replace(
        "{{original_prompt}}", masked_text
    )
    user_prompt_light_noise = user_prompt_light_noise.replace(
        "{{original_prompt}}", masked_text
    )
    user_prompt_moderate_noise = user_prompt_moderate_noise.replace(
        "{{original_prompt}}", masked_text
    )
    user_prompt_heavy_noise = user_prompt_heavy_noise.replace(
        "{{original_prompt}}", masked_text
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
                    {"role": "user", "content": user_prompt},      # User request with masked text
                ],
                temperature=temperature,  # Temperature setting based on mode
            )
            # Extract and clean the rewritten content from API response
            content = response.choices[0].message.content.strip()
        except Exception as e:
            # Handle API errors with retry logic
            if attempt == 0:
                print(f"API call error, retrying: {e}")
            else:
                print(f"API call still failed after retry: {e}")
                # Return original prompt if all retries fail
                return original_prompt

    # Extract rewritten content from API response using regex
    m = re.search(r"<answer>(.*?)</answer>", content, re.S | re.I)
    rewritten_masked = (m.group(1) if m else content).strip()

    # Validate that all placeholders are correctly preserved
    if not placeholders_ok(rewritten_masked, len(protected_chunks)):
        return original_prompt

    # Restore protected mathematical content from placeholders
    restored = restore_protected(rewritten_masked, protected_chunks)
    return restored


def process_jsonl_file(input_file, output_file, mode):
    """
    Process a JSONL file containing MATH dataset mathematical problems and rewrite them using specified mode.
    
    This function reads a JSONL file where each line contains a JSON object with a 'problem' field.
    The problem field contains a mathematical problem with LaTeX expressions, asy diagrams, and code blocks.
    The function extracts the problem text, rewrites it using the specified transformation mode,
    and updates the JSON object with the rewritten problem.
    
    Args:
        input_file (str): Path to the input JSONL file containing original mathematical problems
        output_file (str): Path to the output JSONL file for rewritten problems
        mode (str): Transformation mode - one of 'requirements', 'paraphrasing',
                   'light_noise', 'moderate_noise', or 'heavy_noise'
        
    Returns:
        None: Writes processed data to output file
        
    Note:
        The function expects MATH-style format with 'problem' field containing the mathematical problem.
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

        # Extract problem field from the JSON data
        p = data.get("problem", None)
        
        # Process the problem if it exists and is a non-empty string
        if isinstance(p, str) and p.strip():
            # Use the original problem text directly
            original_problem = p
            
            try:
                # Call the OpenAI rewriter with the extracted problem and specified mode
                new_problem = rewrite_prompt_with_openai(original_problem, mode)
            except Exception as e:
                # Handle API call errors for individual problems
                print(f"[Error] Line {i+1} API call failed: {e}")
                # Use original problem if API call fails
                new_problem = original_problem

            # Update the data object with the rewritten problem
            data["problem"] = new_problem
        else:
            # Skip lines that don't have a processable 'problem' field
            print(f"[Warning] Line {i+1} missing processable 'problem' field, keeping original.")

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
    Main function to process MATH dataset with all transformation modes.
    
    This function orchestrates the complete processing pipeline by:
    1. Defining input and output file paths for each transformation mode
    2. Processing the original MATH dataset with each of the 5 transformation modes:
       - Requirements: Strengthens mathematical problem constraints and requirements
       - Paraphrasing: Rewrites problem prose while preserving mathematical expressions
       - Light Noise: Adds minimal colloquial noise to problem prose
       - Moderate Noise: Adds medium-level colloquial noise to problem prose
       - Heavy Noise: Adds maximum colloquial noise to problem prose
    
    The function processes the same input file multiple times, generating different
    augmented versions for robustness testing of mathematical reasoning models.
    
    Returns:
        None: Creates multiple output files with transformed mathematical problems
    """
    # Define input file path (original MATH dataset)
    input_file = "math_original.jsonl"
    
    # Define output file paths for each transformation mode
    requirements_output_file = "math_requirements.jsonl"
    paraphrasing_output_file = "math_paraphrasing.jsonl"
    light_noise_output_file = "math_light_noise.jsonl"
    moderate_noise_output_file = "math_moderate_noise.jsonl"
    heavy_noise_output_file = "math_heavy_noise.jsonl"

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
