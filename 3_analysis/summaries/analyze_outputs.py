"""Score one raw experiment run into an `analysis/summary.txt` report.

**Step 0 of the analysis pipeline.** Every other script in `3_analysis/` sits
downstream of the `summary.txt` files this produces — run it once per run directory
before `generate_noknowledge_results_summary.py` / `generate_rag_results_summary.py`,
which parse those reports into the paper's tables.

Given a run directory containing `raw_outputs.csv` (and optionally `config.json`),
it decides for each model response whether the true author was named, and aggregates
accuracy and author-omission rates by prompt type and by demographic subgroup.

Author matching is the substantive logic here, in `AuthorMatcher`: name
canonicalization, last-name extraction with a stopword list, and non-person-author
detection. `3_analysis/stats/compute_noknowledge_quote_metrics.py` imports these
helpers directly so that quote-level metrics use identical matching rules.

The section headers written into summary.txt are a parsing contract — the
`generate_*_results_summary.py` scripts match on those exact strings, so changing a
header silently empties the downstream tables.

Usage
-----
  python analyze_outputs.py <run_directory>     # dir holding raw_outputs.csv

Set OUTPUT_ALL_ANALYSES = True (below) to also emit the per-analysis CSVs; by
default only `analysis/summary.txt` is written.
"""
import json
import os
import pandas as pd
import sys
import re
import unicodedata
from collections import Counter
from datetime import datetime

# When False, only write analysis/summary.txt.
# When True, write all CSV analysis artifacts as well.
OUTPUT_ALL_ANALYSES = False

LAST_NAME_STOPWORDS = {
    # Common nouns/titles that should not trigger last-name matches
    'poet', 'writer', 'author', 'artist', 'musician', 'singer', 'rapper', 'actor',
    'actress', 'producer', 'director', 'professor', 'doctor', 'dr', 'mr', 'mrs',
    'ms', 'miss', 'sir', 'lord', 'lady', 'jr', 'sr',
    # Common words that appear frequently in normal prose.
    'love', 'truth', 'still', 'wisdom', 'better', 'bond', 'legacy', 'foster', 'pride',
    'battle', 'early', 'silver', 'virtue', 'bonds', 'innocent', 'seven', 'cope',
    'fields', 'quick', 'page', 'flash', 'rather', 'hope', 'gates', 'sign', 'good', "black",
    'young', 'french', 'great', 'deep', 'books', 'holiday', 'prior', 'chance', 'block',
    'white', 'rock'
}

ALIASES = {
    # Canonical name -> list of acceptable aliases/short forms
    "martin luther king jr": ["dr king", "mlk", "mlk jr", "king jr"],
    "malcolm x": ["malcolm little"],
    "w e b dubois": ["web dubois", "w e b du bois", "w e b du bois"],
    "j r r tolkien": ["jrr tolkien"],
    "t i": ["ti", "t.i."],
}

NON_PERSON_AUTHORS = {
    # Known titles/entities that appear in author fields but are not people.
    "the sound of serendipity",
}


def normalize_alias(text: str) -> str:
    return normalize_for_match(text)


def build_alias_map() -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        canon_norm = normalize_alias(canonical)
        for alias in aliases:
            alias_norm = normalize_alias(alias)
            if alias_norm:
                alias_map[alias_norm] = canon_norm
    return alias_map
def canonicalize_author(name):
    """Normalize author string and drop nicknames/works."""
    if not isinstance(name, str):
        return name
    # Remove leading punctuation/dashes
    name = re.sub(r"^[\s\-–—•]+", "", name)
    # Drop explicit AKA indicators
    name = re.split(r"\baka\b|\ba\.k\.a\.\b", name, flags=re.IGNORECASE)[0]
    # Strip nicknames in parentheses or angle/quote wrappers
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"«[^»]*»", "", name)
    name = re.sub(r"\"[^\"]*\"", "", name)
    name = re.sub(r"'[^']*'", "", name)
    # Drop trailing dash-separated work titles/slogans
    name = re.split(r"\s[-–—]\s", name)[0]
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    # If a trailing dash remains (e.g., 'Name- Work'), strip it
    name = re.sub(r"[-–—]\s*$", "", name).strip()
    return name


def is_non_person_author(name: str) -> bool:
    if not isinstance(name, str):
        return False
    return normalize_for_match(name) in NON_PERSON_AUTHORS

def normalize_for_match(text):
    """Lowercase, strip punctuation, and collapse whitespace for fuzzy matching."""
    # Normalize accents/diacritics (e.g., "Pelé" -> "Pele")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def contains_author_norm(norm_output, norm_author):
    """Check full-name match using normalized strings."""
    if not norm_output or not norm_author:
        return False
    tokens = norm_author.split()
    if not tokens:
        return False
    pattern = r"\b" + r"\s+".join(re.escape(t) for t in tokens) + r"\b"
    if re.search(pattern, norm_output):
        return True
    if len(tokens) >= 2:
        rev_pattern = r"\b" + r"\s+".join(re.escape(t) for t in reversed(tokens)) + r"\b"
        return re.search(rev_pattern, norm_output) is not None
    return False

def compile_name_pattern(names):
    """Compile a word-boundary regex for a list of normalized names."""
    if not names:
        return None
    escaped = [re.escape(name).replace(r"\ ", r"\s+") for name in names]
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern)

def extract_last_name(author):
    if not author:
        return None
    parts = author.split()
    if len(parts) <= 1:
        return None
    last_name = normalize_for_match(parts[-1])
    if len(last_name) < 4 or last_name in LAST_NAME_STOPWORDS:
        return None
    return last_name

def build_last_name_index(authors):
    candidates = []
    for author in authors:
        last_name = extract_last_name(author)
        if last_name and isinstance(last_name, str):
            candidates.append((len(author), author, last_name))
    candidates.sort(key=lambda x: x[0], reverse=True)
    last_name_to_author = {}
    for _, author, last_name in candidates:
        last_name_to_author.setdefault(last_name, author)
    pattern = compile_name_pattern(list(last_name_to_author.keys()))
    return last_name_to_author, pattern

class AuthorMatcher:
    def __init__(self, authors):
        self.alias_map = build_alias_map()
        self.norm_authors = {
            normalize_for_match(author): author
            for author in authors
            if author
        }
        self.reversed_to_author = {}
        for norm_name, author in self.norm_authors.items():
            parts = norm_name.split()
            if len(parts) >= 2:
                reversed_name = " ".join(reversed(parts))
                self.reversed_to_author.setdefault(reversed_name, author)
        self.alias_to_author = {}
        self.author_aliases = {}
        for alias_norm, canon_norm in self.alias_map.items():
            author = self.norm_authors.get(canon_norm)
            if author:
                self.alias_to_author[alias_norm] = author
                self.author_aliases.setdefault(canon_norm, []).append(alias_norm)
        alias_names = sorted(self.alias_to_author.keys(), key=len, reverse=True)
        full_names = sorted(self.norm_authors.keys(), key=len, reverse=True)
        reversed_names = sorted(self.reversed_to_author.keys(), key=len, reverse=True)
        self.alias_pattern = compile_name_pattern(alias_names)
        self.full_name_pattern = compile_name_pattern(full_names)
        self.reversed_name_pattern = compile_name_pattern(reversed_names)
        self.last_name_to_author, self.last_name_pattern = build_last_name_index(authors)

    def extract_mentioned_author(self, norm_output):
        if not norm_output:
            return None
        if self.alias_pattern:
            match = self.alias_pattern.search(norm_output)
            if match:
                key = normalize_for_match(match.group(0))
                author = self.alias_to_author.get(key)
                if author:
                    return author
        if self.full_name_pattern:
            match = self.full_name_pattern.search(norm_output)
            if match:
                key = normalize_for_match(match.group(0))
                author = self.norm_authors.get(key)
                if author:
                    return author
        if self.reversed_name_pattern:
            match = self.reversed_name_pattern.search(norm_output)
            if match:
                key = normalize_for_match(match.group(0))
                author = self.reversed_to_author.get(key)
                if author:
                    return author
        if self.last_name_pattern:
            match = self.last_name_pattern.search(norm_output)
            if match:
                key = normalize_for_match(match.group(0))
                return self.last_name_to_author.get(key)
        return None

    def check_author_match(self, norm_output, norm_author, last_name):
        if not norm_output or not norm_author:
            return False
        if contains_author_norm(norm_output, norm_author):
            return True
        for alias_norm in self.author_aliases.get(norm_author, []):
            pattern = r"\b" + re.escape(alias_norm) + r"\b"
            if re.search(pattern, norm_output):
                return True
        if last_name and isinstance(last_name, str):
            pattern = r"\b" + re.escape(last_name) + r"\b"
            return re.search(pattern, norm_output) is not None
        return False
def contains_author(norm_output, author):
    """Check full-name match using word boundaries to avoid substring false positives."""
    if not norm_output or not author:
        return False
    norm_author = normalize_for_match(author)
    tokens = norm_author.split()
    if not tokens:
        return False
    pattern = r"\b" + r"\s+".join(re.escape(t) for t in tokens) + r"\b"
    return re.search(pattern, norm_output) is not None


def check_author_match(output, author):
    """Check if the output correctly identifies the author (punctuation-insensitive)."""
    if not output or not author:
        return False
    norm_output = normalize_for_match(output)
    if contains_author(norm_output, author):
        return True
    # Alias match
    alias_map = build_alias_map()
    norm_author = normalize_for_match(author)
    for alias_norm, canon_norm in alias_map.items():
        if canon_norm == norm_author:
            pattern = r"\b" + re.escape(alias_norm) + r"\b"
            if re.search(pattern, norm_output):
                return True
        return True
    parts = author.split()
    if len(parts) > 1:
        last_name = normalize_for_match(parts[-1])
        if len(last_name) >= 4 and last_name not in LAST_NAME_STOPWORDS:
            pattern = r"\b" + re.escape(last_name) + r"\b"
            return re.search(pattern, norm_output) is not None
    return False


def mentions_any_name(output: str) -> bool:
    """Heuristic: detect any person-like name in free-form output."""
    if not output or not isinstance(output, str):
        return False
    # Look for sequences like "Jane Doe", "J. K. Rowling", "Jean-Paul Sartre"
    pattern = re.compile(
        r"\b([A-Z][a-z]+|[A-Z]\.)"
        r"(?:[-'][A-Z][a-z]+)?"
        r"(?:\s+([A-Z][a-z]+|[A-Z]\.)"
        r"(?:[-'][A-Z][a-z]+)?)+\b"
    )
    return bool(pattern.search(output))

def extract_mentioned_author(output, all_authors):
    """Extract which author was mentioned in the output, if any.
    
    FIXED: Uses word boundaries to avoid false matches like 'king' in 'thinking'.
    """
    if not output:
        return None
    norm_output = normalize_for_match(output)
    
    # Alias pass: map known aliases to canonical names if present
    alias_map = build_alias_map()
    for alias_norm, canon_norm in alias_map.items():
        pattern = r"\b" + re.escape(alias_norm) + r"\b"
        if re.search(pattern, norm_output):
            # return the author from all_authors that matches canonical
            for author in all_authors:
                if normalize_for_match(author) == canon_norm:
                    return author

    # First pass: match full names using word boundaries (avoid substring false matches)
    for author in all_authors:
        if not author:
            continue
        if contains_author(norm_output, author):
            return author
    
    # Second pass: match last names only with word boundaries
    # Sort by length (longest first) to match more specific names first
    sorted_authors = sorted(all_authors, key=len, reverse=True)
    for author in sorted_authors:
        parts = author.split()
        if len(parts) > 1:
            last_name = normalize_for_match(parts[-1])
            # Only match last names that are at least 4 characters post-normalization
            if len(last_name) >= 4 and last_name not in LAST_NAME_STOPWORDS:
                # Use word boundaries to avoid matching "King" in "thinking" or "working"
                pattern = r'\b' + re.escape(last_name) + r'\b'
                if re.search(pattern, norm_output):
                    return author
    return None

def check_no_answer(output):
    """Check if output indicates inability to identify author."""
    if not output:
        return True
    no_answer = [
        "don't know", "do not know", "cannot identify", "can't identify", 
        "unable to determine", "not enough information", "insufficient information", 
        "cannot say", "can't say", "unknown author", "anonymous"
    ]
    return any(phrase in output.lower() for phrase in no_answer)

def analyze_errors(results_df, author_demo):
    """Analyze demographic patterns in misattributions."""
    error_records = []
    
    for _, row in results_df.iterrows():
        if not row['identified_correct_author'] and row['mentioned_author']:
            wrong_author = row['mentioned_author']
            correct_author = row['author']
            if wrong_author in author_demo and correct_author in author_demo:
                correct_demo = author_demo[correct_author]
                wrong_demo = author_demo[wrong_author]
                error_records.append({
                    'quote_id': row['quote_id'], 
                    'prompt_type': row['prompt_type'], 
                    'run': row['run'],
                    'correct_author': correct_author, 
                    'correct_gender': correct_demo['gender'], 
                    'correct_race': correct_demo['race_ethnicity'],
                    'wrong_author': wrong_author, 
                    'wrong_gender': wrong_demo['gender'], 
                    'wrong_race': wrong_demo['race_ethnicity'],
                    'same_gender': correct_demo['gender'] == wrong_demo['gender'],
                    'same_race': correct_demo['race_ethnicity'] == wrong_demo['race_ethnicity'],
                    'same_both': (correct_demo['gender'] == wrong_demo['gender']) and 
                                (correct_demo['race_ethnicity'] == wrong_demo['race_ethnicity'])
                })
    return pd.DataFrame(error_records) if error_records else None

def create_confusion_matrix(results_df, all_authors, author_demo, top_n=50):
    """Create confusion matrices for author misattributions."""
    confusion_pairs = [
        (row['author'], row['mentioned_author']) 
        for _, row in results_df.iterrows() 
        if not row['identified_correct_author'] and 
           row['mentioned_author'] and 
           row['mentioned_author'] in all_authors
    ]
    
    if not confusion_pairs:
        return None, None
    
    confusion_counts = Counter(confusion_pairs)
    top_confusions = pd.DataFrame([
        {'correct_author': c, 'wrong_author': w, 'count': cnt} 
        for (c, w), cnt in confusion_counts.most_common(top_n)
    ])
    
    demo_confusion = [
        {
            'correct_demo': f"{author_demo[c]['gender']}_{author_demo[c]['race_ethnicity']}",
            'wrong_demo': f"{author_demo[w]['gender']}_{author_demo[w]['race_ethnicity']}"
        } 
        for c, w in confusion_pairs 
        if c in author_demo and w in author_demo
    ]
    
    demo_confusion_df = pd.DataFrame(demo_confusion)
    demo_confusion_summary = (demo_confusion_df
                             .groupby(['correct_demo', 'wrong_demo'])
                             .size()
                             .reset_index(name='count')
                             .sort_values('count', ascending=False))
    return top_confusions, demo_confusion_summary

def create_summary(
    quote_level,
    error_df,
    top_confusions,
    demo_confusion,
    results_df,
    config,
    prompt_types,
    reasoning_conditional_stats=None,
    requested_conditionals_by_race_gender=None,
    requested_conditionals_by_race=None,
):
    """Generate human-readable summary report."""
    lines = [
        "="*80,
        "PROMPT TYPE ATTRIBUTION ANALYSIS",
        "="*80,
        f"\nAnalysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Generation timestamp: {config.get('timestamp', 'Unknown')}",
        f"Model: {config.get('model', 'Unknown')}",
        f"Total quotes: {config.get('total_quotes', 'Unknown')}",
        f"Runs per prompt: {config.get('runs', 'Unknown')}\n"
    ]

    def _label_text(value):
        if isinstance(value, tuple):
            return "_".join(str(v) for v in value)
        return str(value)

    def _correct_given_author_mentioned(subset):
        mentioned_subset = subset[subset['mentioned_any_author']]
        n_author_mentioned = len(mentioned_subset)
        if n_author_mentioned == 0:
            return None, 0
        return mentioned_subset['identified_correct_author'].mean(), n_author_mentioned
    
    lines.append("="*80 + "\nOVERALL ACCURACY BY PROMPT TYPE (THREE-WAY BREAKDOWN)\n" + "="*80)
    
    for prompt in prompt_types:
        prompt_data = quote_level[quote_level['prompt_type'] == prompt]
        prompt_results = results_df[results_df['prompt_type'] == prompt]
        n_prompt_rows = len(prompt_results)
        n_no_author = int((~prompt_results['mentioned_any_author']).sum()) if n_prompt_rows > 0 else 0
        
        # Calculate three-way breakdown
        correct_author_rate = prompt_data['author_accuracy'].mean()
        no_author_rate = (~prompt_results['mentioned_any_author']).mean()
        wrong_author_rate = ((~prompt_results['identified_correct_author']) & 
                           (prompt_results['mentioned_any_author'])).mean()
        
        lines.append(f"\n{prompt.upper()}:")
        lines.append(f"  ✓ Correct Author Identified: {correct_author_rate*100:.1f}%")
        lines.append(f"  ✗ Wrong Author Mentioned:     {wrong_author_rate*100:.1f}%")
        lines.append(f"  ∅ No Author Mentioned:        {no_author_rate*100:.1f}%")
        lines.append(f"  ∅ No Author Mentioned (count): {n_no_author}/{n_prompt_rows}")
        lines.append(f"  Total (should be ~100%):      {(correct_author_rate + wrong_author_rate + no_author_rate)*100:.1f}%")
        if prompt in ['indirect', 'indirect_overt']:
            cond_rate, n_author_mentioned = _correct_given_author_mentioned(prompt_results)
            if cond_rate is None:
                lines.append("  P(Correct | Author Given):    N/A (n_author_given=0)")
            else:
                lines.append(
                    f"  P(Correct | Author Given):    {cond_rate*100:.1f}% "
                    f"(n_author_given={n_author_mentioned})"
                )
        lines.append(f"  Quotes analyzed: {len(prompt_data)}")
        if config.get('runs', 1) > 1:
            lines.append(f"  Variance (std): {prompt_data['author_std'].mean():.3f}")

    # Conditional breakdowns given DIRECT correct
    lines.append("\n" + "="*80 + "\nCONDITIONAL (GIVEN DIRECT CORRECT)\n" + "="*80)
    direct_quotes = quote_level[
        (quote_level['prompt_type'] == 'direct') &
        (quote_level['author_accuracy'] == 1)
    ]['quote_id'].unique().tolist()
    lines.append(f"\nDirect-correct quotes: {len(direct_quotes)}")
    if direct_quotes:
        for prompt in ['indirect', 'indirect_overt']:
            if prompt not in prompt_types:
                continue
            subset = results_df[
                (results_df['quote_id'].isin(direct_quotes)) &
                (results_df['prompt_type'] == prompt)
            ]
            if len(subset) == 0:
                continue
            correct_rate = subset['identified_correct_author'].mean()
            no_author_rate = (~subset['mentioned_any_author']).mean()
            wrong_author_rate = ((~subset['identified_correct_author']) &
                                 (subset['mentioned_any_author'])).mean()
            lines.append(f"\n{prompt.upper()}:")
            lines.append(f"  ✓ Correct Author Identified: {correct_rate*100:.1f}%")
            lines.append(f"  ✗ Wrong Author Mentioned:     {wrong_author_rate*100:.1f}%")
            lines.append(f"  ∅ No Author Mentioned:        {no_author_rate*100:.1f}%")
            lines.append(f"  Total (should be ~100%):      {(correct_rate + wrong_author_rate + no_author_rate)*100:.1f}%")
            cond_rate, n_author_mentioned = _correct_given_author_mentioned(subset)
            if cond_rate is None:
                lines.append("  P(Correct | Author Given):    N/A (n_author_given=0)")
            else:
                lines.append(
                    f"  P(Correct | Author Given):    {cond_rate*100:.1f}% "
                    f"(n_author_given={n_author_mentioned})"
                )

            # Race-only conditional detail for the same slice.
            lines.append("  By race:")
            race_rows = []
            race_grouped = subset.groupby(['race_ethnicity'])
            for race, g in race_grouped:
                race_rows.append({
                    'race_ethnicity': race,
                    'correct_rate': g['identified_correct_author'].mean(),
                    'wrong_rate': ((~g['identified_correct_author']) & (g['mentioned_any_author'])).mean(),
                    'no_author_rate': (~g['mentioned_any_author']).mean(),
                    'n': len(g),
                })
            race_df = pd.DataFrame(race_rows).sort_values(
                ['correct_rate', 'n'], ascending=[False, False]
            )
            for _, row in race_df.iterrows():
                race_label = _label_text(row['race_ethnicity'])
                lines.append(
                    f"    {race_label:<28} "
                    f"✓ {row['correct_rate']*100:>5.1f}%  "
                    f"✗ {row['wrong_rate']*100:>5.1f}%  "
                    f"∅ {row['no_author_rate']*100:>5.1f}%  "
                    f"(n={int(row['n'])})"
                )

            # Subgroup detail for the same conditional slice.
            lines.append("  By subgroup (race x gender):")
            subgroup_rows = []
            grouped = subset.groupby(['race_ethnicity', 'gender'])
            for (race, gender), g in grouped:
                subgroup_rows.append({
                    'race_ethnicity': race,
                    'gender': gender,
                    'correct_rate': g['identified_correct_author'].mean(),
                    'wrong_rate': ((~g['identified_correct_author']) & (g['mentioned_any_author'])).mean(),
                    'no_author_rate': (~g['mentioned_any_author']).mean(),
                    'n': len(g),
                })
            subgroup_df = pd.DataFrame(subgroup_rows).sort_values(
                ['correct_rate', 'n'], ascending=[False, False]
            )
            for _, row in subgroup_df.iterrows():
                subgroup_name = f"{_label_text(row['race_ethnicity'])}_{_label_text(row['gender'])}"
                lines.append(
                    f"    {subgroup_name:<28} "
                    f"✓ {row['correct_rate']*100:>5.1f}%  "
                    f"✗ {row['wrong_rate']*100:>5.1f}%  "
                    f"∅ {row['no_author_rate']*100:>5.1f}%  "
                    f"(n={int(row['n'])})"
                )

    lines.append("\n\n" + "="*80 + "\nACCURACY BY DEMOGRAPHIC GROUP\n" + "="*80)
    for prompt in prompt_types:
        prompt_data = quote_level[quote_level['prompt_type'] == prompt]
        
        # Gender - sorted by accuracy descending
        lines.append(f"\n{prompt.upper()} - By Gender:")
        gender_stats = []
        for g in prompt_data['gender'].unique():
            d = prompt_data[prompt_data['gender'] == g]
            gender_stats.append((g, d['author_accuracy'].mean(), len(d)))
        gender_stats.sort(key=lambda x: x[1], reverse=True)
        for g, acc, n in gender_stats:
            lines.append(f"  {g}: {acc*100:.1f}% (n={n})")
        
        # Race - sorted by accuracy descending
        lines.append(f"\n{prompt.upper()} - By Race:")
        race_stats = []
        for r in prompt_data['race_ethnicity'].unique():
            d = prompt_data[prompt_data['race_ethnicity'] == r]
            race_stats.append((r, d['author_accuracy'].mean(), len(d)))
        race_stats.sort(key=lambda x: x[1], reverse=True)
        for r, acc, n in race_stats:
            lines.append(f"  {r}: {acc*100:.1f}% (n={n})")
    
    lines.append("\n\n" + "="*80 + "\nINTERSECTIONAL ANALYSIS (Gender × Race)\n" + "="*80)
    for prompt in prompt_types:
        prompt_data = quote_level[quote_level['prompt_type'] == prompt].copy()
        prompt_data['intersectional'] = prompt_data['gender'] + '_' + prompt_data['race_ethnicity']
        inter = (prompt_data
                .groupby('intersectional')
                .agg({'author_accuracy': 'mean', 'quote_id': 'count'})
                .reset_index()
                .sort_values('author_accuracy', ascending=False))
        lines.append(f"\n{prompt.upper()}:")
        for _, row in inter.iterrows():
            lines.append(f"  {row['intersectional']:<30} {row['author_accuracy']*100:>5.1f}% (n={int(row['quote_id'])})")

    lines.append("\n\n" + "="*80 + "\nNO AUTHOR MENTIONED BY SUBGROUP (RACE x GENDER)\n" + "="*80)
    for prompt in prompt_types:
        prompt_results = results_df[results_df['prompt_type'] == prompt]
        if len(prompt_results) == 0:
            continue
        lines.append(f"\n{prompt.upper()}:")
        subgroup_no_author_rows = []
        for (race, gender), g in prompt_results.groupby(['race_ethnicity', 'gender']):
            n_rows = len(g)
            n_no_author = int((~g['mentioned_any_author']).sum())
            subgroup_no_author_rows.append({
                'race_ethnicity': race,
                'gender': gender,
                'n_rows': n_rows,
                'n_no_author_given': n_no_author,
                'no_author_rate': (n_no_author / n_rows) if n_rows > 0 else 0.0,
            })
        subgroup_no_author_df = pd.DataFrame(subgroup_no_author_rows).sort_values(
            ['no_author_rate', 'n_rows'], ascending=[False, False]
        )
        for _, row in subgroup_no_author_df.iterrows():
            subgroup_name = f"{_label_text(row['race_ethnicity'])}_{_label_text(row['gender'])}"
            lines.append(
                f"  {subgroup_name:<28} "
                f"{row['n_no_author_given']}/{int(row['n_rows'])} "
                f"({row['no_author_rate']*100:.1f}%)"
            )

    lines.append("\n\n" + "="*80 + "\nNO AUTHOR MENTIONED BY RACE\n" + "="*80)
    for prompt in prompt_types:
        prompt_results = results_df[results_df['prompt_type'] == prompt]
        if len(prompt_results) == 0:
            continue
        lines.append(f"\n{prompt.upper()}:")
        race_no_author_rows = []
        for race, g in prompt_results.groupby('race_ethnicity'):
            n_rows = len(g)
            n_no_author = int((~g['mentioned_any_author']).sum())
            race_no_author_rows.append({
                'race_ethnicity': race,
                'n_rows': n_rows,
                'n_no_author_given': n_no_author,
                'no_author_rate': (n_no_author / n_rows) if n_rows > 0 else 0.0,
            })
        race_no_author_df = pd.DataFrame(race_no_author_rows).sort_values(
            ['no_author_rate', 'n_rows'], ascending=[False, False]
        )
        for _, row in race_no_author_df.iterrows():
            race_name = _label_text(row['race_ethnicity'])
            lines.append(
                f"  {race_name:<28} "
                f"{row['n_no_author_given']}/{int(row['n_rows'])} "
                f"({row['no_author_rate']*100:.1f}%)"
            )
    
    lines.append("\n\n" + "="*80 + "\nRESPONSE CHARACTERISTICS\n" + "="*80)
    for prompt in ['direct', 'indirect', 'neutral']:
        prompt_results = results_df[results_df['prompt_type'] == prompt]
        no_author_rate = (~prompt_results['mentioned_any_author']).mean()
        wrong_author_rate = ((~prompt_results['identified_correct_author']) & 
                           (prompt_results['mentioned_any_author'])).mean()
        lines.append(f"\n{prompt.upper()}:")
        lines.append(f"  Avg response length: {prompt_results['response_length'].mean():.0f} chars")
        lines.append(f"  No author mentioned: {no_author_rate*100:.1f}%")
        lines.append(f"  Wrong author mentioned: {wrong_author_rate*100:.1f}%")
        lines.append(f"  No answer rate: {prompt_results['no_answer_given'].mean()*100:.1f}%")

    lines.append("\n\n" + "="*80 + "\nREASONING CONDITIONED STATS\n" + "="*80)
    if reasoning_conditional_stats is not None and len(reasoning_conditional_stats) > 0:
        lines.append("\nP(final mentions correct author | reasoning mentions correct author), by race x gender:")
        for _, row in reasoning_conditional_stats.iterrows():
            lines.append(
                f"  {row['race_ethnicity']}_{row['gender']:<24} "
                f"{row['p_final_mentions_author_given_reasoning_mentions_author']*100:>5.1f}% "
                f"(n_reasoning={int(row['n_reasoning_mentions_author'])}, "
                f"n_final={int(row['n_final_mentions_author'])})"
            )
    elif reasoning_conditional_stats is not None:
        lines.append("\nNo rows where reasoning mentions the correct author.")
    else:
        lines.append("\nReasoning summaries not available in raw outputs; metric skipped.")
    
    if error_df is not None and len(error_df) > 0:
        lines.append("\n\n" + "="*80 + "\nERROR ANALYSIS\n" + "="*80)
        lines.append(f"\nTotal errors analyzed: {len(error_df)}")
        lines.append(f"Same gender misattribution: {error_df['same_gender'].mean()*100:.1f}%")
        lines.append(f"Same race misattribution: {error_df['same_race'].mean()*100:.1f}%")
        lines.append(f"Same gender AND race: {error_df['same_both'].mean()*100:.1f}%")
        lines.append("\nError patterns by prompt type:")
        for prompt in prompt_types:
            prompt_errors = error_df[error_df['prompt_type'] == prompt]
            if len(prompt_errors) > 0:
                lines.append(f"  {prompt}: {len(prompt_errors)} errors "
                           f"(Same gender: {prompt_errors['same_gender'].mean()*100:.1f}%, "
                           f"Same race: {prompt_errors['same_race'].mean()*100:.1f}%)")
    
    if top_confusions is not None:
        lines.append("\n\n" + "="*80 + "\nTOP 20 MOST COMMON CONFUSIONS\n" + "="*80)
        for _, row in top_confusions.head(20).iterrows():
            lines.append(f"{row['correct_author']:<30} → {row['wrong_author']:<30} ({int(row['count'])} times)")
    
    if demo_confusion is not None:
        lines.append("\n\n" + "="*80 + "\nDEMOGRAPHIC-LEVEL CONFUSION PATTERNS\n" + "="*80)
        for _, row in demo_confusion.head(15).iterrows():
            lines.append(f"{row['correct_demo']:<30} → {row['wrong_demo']:<30} ({int(row['count'])} times)")

    if requested_conditionals_by_race_gender is not None and len(requested_conditionals_by_race_gender) > 0:
        lines.append("\n\n" + "="*80 + "\nREQUESTED SUBGROUP CONDITIONALS (RACE x GENDER)\n" + "="*80)
        metric_order = [
            "p_correct_indirect_given_direct_correct",
            "p_correct_indirect_overt_given_direct_correct",
            "p_correct_indirect_given_author_given_indirect",
            "p_correct_indirect_overt_given_author_given_indirect_overt",
        ]
        for metric_name in metric_order:
            metric_df = requested_conditionals_by_race_gender[
                requested_conditionals_by_race_gender['metric'] == metric_name
            ].copy()
            if len(metric_df) == 0:
                continue
            metric_df = metric_df.sort_values(['probability', 'n_condition'], ascending=[False, False])
            lines.append(f"\n{metric_name}:")
            for _, row in metric_df.iterrows():
                subgroup_name = f"{_label_text(row['race_ethnicity'])}_{_label_text(row['gender'])}"
                lines.append(
                    f"  {subgroup_name:<28} {row['probability']*100:>5.1f}% "
                    f"(n={int(row['n_correct'])}/{int(row['n_condition'])})"
                )

    if requested_conditionals_by_race is not None and len(requested_conditionals_by_race) > 0:
        lines.append("\n\n" + "="*80 + "\nREQUESTED SUBGROUP CONDITIONALS (RACE ONLY)\n" + "="*80)
        metric_order = [
            "p_correct_indirect_given_direct_correct",
            "p_correct_indirect_overt_given_direct_correct",
            "p_correct_indirect_given_author_given_indirect",
            "p_correct_indirect_overt_given_author_given_indirect_overt",
        ]
        for metric_name in metric_order:
            metric_df = requested_conditionals_by_race[
                requested_conditionals_by_race['metric'] == metric_name
            ].copy()
            if len(metric_df) == 0:
                continue
            metric_df = metric_df.sort_values(['probability', 'n_condition'], ascending=[False, False])
            lines.append(f"\n{metric_name}:")
            for _, row in metric_df.iterrows():
                race_name = _label_text(row['race_ethnicity'])
                lines.append(
                    f"  {race_name:<28} {row['probability']*100:>5.1f}% "
                    f"(n={int(row['n_correct'])}/{int(row['n_condition'])})"
                )
    
    lines.append("\n\n" + "="*80 + "\nKEY FINDINGS\n" + "="*80)
    accuracies = {
        prompt: quote_level[quote_level['prompt_type'] == prompt]['author_accuracy'].mean() 
        for prompt in prompt_types
    }
    best_prompt = max(accuracies, key=accuracies.get)
    worst_prompt = min(accuracies, key=accuracies.get)
    lines.append(f"\n1. Best performing prompt: {best_prompt.upper()} ({accuracies[best_prompt]*100:.1f}%)")
    lines.append(f"   Worst performing prompt: {worst_prompt.upper()} ({accuracies[worst_prompt]*100:.1f}%)")
    lines.append(f"   Difference: {(accuracies[best_prompt] - accuracies[worst_prompt])*100:.1f} percentage points")
    
    for prompt in prompt_types:
        prompt_data = quote_level[quote_level['prompt_type'] == prompt]
        gender_rates = prompt_data.groupby('gender')['author_accuracy'].mean()
        race_rates = prompt_data.groupby('race_ethnicity')['author_accuracy'].mean()
        if len(gender_rates) > 1:
            lines.append(f"\n2. {prompt.upper()} gender gap: {(gender_rates.max() - gender_rates.min())*100:.1f} percentage points")
        if len(race_rates) > 1:
            lines.append(f"   {prompt.upper()} race gap: {(race_rates.max() - race_rates.min())*100:.1f} percentage points")
    
    lines.append("\n" + "="*80)
    lines.append("\nSee analysis/author_outcome_by_race_gender.csv for subgroup right/wrong/no rates and no-author counts.")
    lines.append("See analysis/conditional_author_outcomes_by_race_gender.csv for conditional subgroup rates.")
    lines.append("See analysis/final_given_reasoning_by_race_gender.csv for reasoning-conditioned subgroup rates.")
    lines.append("See analysis/requested_conditionals_by_race.csv for requested race-only conditionals.")
    lines.append("See analysis/requested_conditionals_by_race_gender.csv for requested race x gender conditionals.")
    return '\n'.join(lines)

def analyze_outputs(output_dir):
    """
    Analyze raw outputs from generation script.
    
    Args:
        output_dir: Path to directory containing raw_outputs.csv and config.json
    
    Returns:
        Path to analysis output directory
    """
    print(f" Loading outputs from: {output_dir}")
    
    # Load raw outputs
    raw_outputs_path = os.path.join(output_dir, "raw_outputs.csv")
    if not os.path.exists(raw_outputs_path):
        raise FileNotFoundError(f"Cannot find raw_outputs.csv in {output_dir}")
    
    results_df = pd.read_csv(raw_outputs_path)
    print(f" Loaded {len(results_df)} outputs")
    for demo_col in ['gender', 'race_ethnicity']:
        if demo_col in results_df.columns:
            results_df[demo_col] = (
                results_df[demo_col]
                .fillna('unknown')
                .astype(str)
                .str.strip()
                .replace({'': 'unknown', 'nan': 'unknown', 'None': 'unknown'})
            )
    if 'prompt_type' in results_df.columns:
        # Normalize RAG prompt variants back to base prompt names so existing
        # analysis logic (which expects direct/indirect/neutral/indirect_overt)
        # works without downstream changes.
        results_df['prompt_type_raw'] = results_df['prompt_type']
        results_df['prompt_type'] = (
            results_df['prompt_type']
            .astype(str)
            .str.replace(r'_(labeled|unlabeled)$', '', regex=True)
        )
    results_df['author_canonical'] = results_df['author'].apply(canonicalize_author)
    results_df['author_norm'] = results_df['author_canonical'].apply(normalize_for_match)
    results_df['author_last_name'] = results_df['author_canonical'].apply(extract_last_name)
    results_df['norm_output'] = results_df['llm_output'].fillna("").apply(normalize_for_match)
    
    # Load config
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Get unique authors and their demographics
    all_authors = [
        a for a in pd.unique(results_df['author_canonical'])
        if isinstance(a, str) and not is_non_person_author(a)
    ]
    author_demo = (results_df
                  .groupby('author_canonical')
                  .agg({'gender': 'first', 'race_ethnicity': 'first'})
                  .to_dict('index'))
    author_demo = {
        author: demo
        for author, demo in author_demo.items()
        if isinstance(author, str) and not is_non_person_author(author)
    }
    matcher = AuthorMatcher(all_authors)
    
    print(" Analyzing outputs...")
    
    # Add analysis columns
    norm_outputs = results_df['norm_output'].tolist()
    norm_authors = results_df['author_norm'].tolist()
    last_names = results_df['author_last_name'].tolist()
    raw_outputs = results_df['llm_output'].tolist()
    identified = []
    mentioned = []
    mentioned_any = []
    response_lengths = []
    no_answer = []
    for norm_output, norm_author, last_name, raw_output in zip(
        norm_outputs, norm_authors, last_names, raw_outputs
    ):
        identified.append(matcher.check_author_match(norm_output, norm_author, last_name))
        mentioned_author = matcher.extract_mentioned_author(norm_output)
        mentioned.append(mentioned_author)
        mentioned_any.append(bool(mentioned_author) or mentions_any_name(raw_output))
        if isinstance(raw_output, str):
            response_lengths.append(len(raw_output))
            no_answer.append(check_no_answer(raw_output))
        else:
            response_lengths.append(0)
            no_answer.append(True)
    results_df['identified_correct_author'] = identified
    results_df['mentioned_author'] = mentioned
    results_df['mentioned_any_author'] = mentioned_any
    results_df['response_length'] = response_lengths
    results_df['no_answer_given'] = no_answer
    results_df['author_in_final_answer'] = results_df['identified_correct_author']

    reasoning_conditional_stats = None
    if 'resp_reasoning_summary' in results_df.columns:
        reasoning_norm = results_df['resp_reasoning_summary'].fillna("").apply(normalize_for_match).tolist()
        author_in_reasoning = []
        for norm_reasoning, norm_author, last_name in zip(reasoning_norm, norm_authors, last_names):
            author_in_reasoning.append(
                matcher.check_author_match(norm_reasoning, norm_author, last_name)
            )
        results_df['author_in_reasoning'] = author_in_reasoning

        reasoning_subset = results_df[results_df['author_in_reasoning']]
        if len(reasoning_subset) > 0:
            grouped = reasoning_subset.groupby(['race_ethnicity', 'gender'])
            rows = []
            for (race, gender), g in grouped:
                rows.append({
                    'race_ethnicity': race,
                    'gender': gender,
                    'n_reasoning_mentions_author': len(g),
                    'n_final_mentions_author': int(g['author_in_final_answer'].sum()),
                    'p_final_mentions_author_given_reasoning_mentions_author': g['author_in_final_answer'].mean(),
                })
            reasoning_conditional_stats = pd.DataFrame(rows).sort_values(
                'p_final_mentions_author_given_reasoning_mentions_author',
                ascending=False
            )
        else:
            reasoning_conditional_stats = pd.DataFrame(
                columns=[
                    'race_ethnicity',
                    'gender',
                    'n_reasoning_mentions_author',
                    'n_final_mentions_author',
                    'p_final_mentions_author_given_reasoning_mentions_author',
                ]
            )
    else:
        print(" No resp_reasoning_summary column found; skipping reasoning-conditioned stats.")
    
    # Create quote-level aggregation
    runs = config.get('runs', 1)
    if runs > 1:
        agg_dict = {
            'identified_correct_author': ['mean', 'std'], 
            'response_length': 'mean', 
            'no_answer_given': 'mean'
        }
        quote_level = (results_df
                      .groupby(['quote_id', 'prompt_type', 'author_canonical', 'gender', 'race_ethnicity'])
                      .agg(agg_dict)
                      .reset_index())
        quote_level.columns = [
            'quote_id', 'prompt_type', 'author', 'gender', 'race_ethnicity', 
            'author_accuracy', 'author_std', 'avg_response_length', 'no_answer_rate'
        ]
        quote_level['author_std'] = quote_level['author_std'].fillna(0)
    else:
        quote_level = results_df[[
            'quote_id', 'prompt_type', 'author_canonical', 'gender', 'race_ethnicity', 
            'identified_correct_author', 'response_length', 'no_answer_given'
        ]].copy()
        quote_level.columns = [
            'quote_id', 'prompt_type', 'author', 'gender', 'race_ethnicity', 
            'author_accuracy', 'avg_response_length', 'no_answer_rate'
        ]
        quote_level['author_std'] = 0.0
    
    print("  Running error analysis...")
    error_df = analyze_errors(results_df, author_demo)
    print(f"   Error records found: {len(error_df) if error_df is not None else 0}")
    
    print("  Creating confusion matrices...")
    top_confusions, demo_confusion = create_confusion_matrix(results_df, all_authors, author_demo, top_n=50)
    print(f"   Confusion pairs found: {len(top_confusions) if top_confusions is not None else 0}")
    
    prompt_types = list(results_df['prompt_type'].unique())
    
    # Conditional breakdowns: given direct correct, report indirect outcomes
    direct_quotes = quote_level[
        (quote_level['prompt_type'] == 'direct') &
        (quote_level['author_accuracy'] == 1)
    ]['quote_id'].unique().tolist()
    conditional_rows = []
    if direct_quotes:
        for prompt in ['indirect', 'indirect_overt']:
            if prompt not in prompt_types:
                continue
            subset = results_df[
                (results_df['quote_id'].isin(direct_quotes)) &
                (results_df['prompt_type'] == prompt)
            ]
            if len(subset) == 0:
                continue
            correct_rate = subset['identified_correct_author'].mean()
            no_author_rate = (~subset['mentioned_any_author']).mean()
            wrong_author_rate = ((~subset['identified_correct_author']) &
                                 (subset['mentioned_any_author'])).mean()
            mentioned_subset = subset[subset['mentioned_any_author']]
            n_author_given = len(mentioned_subset)
            correct_given_author_given = (
                mentioned_subset['identified_correct_author'].mean()
                if n_author_given > 0 else pd.NA
            )
            conditional_rows.append({
                'prompt_type': prompt,
                'n_rows': len(subset),
                'n_quotes_conditioned': len(set(direct_quotes)),
                'correct_rate': correct_rate,
                'wrong_author_rate': wrong_author_rate,
                'no_author_rate': no_author_rate,
                'n_author_given': n_author_given,
                'correct_given_author_given': correct_given_author_given,
            })
    # Subgroup breakdowns (race x gender)
    subgroup_rows = []
    for prompt in prompt_types:
        subset = results_df[results_df['prompt_type'] == prompt]
        if len(subset) == 0:
            continue
        grouped = subset.groupby(['race_ethnicity', 'gender'])
        for (race, gender), g in grouped:
            correct_rate = g['identified_correct_author'].mean()
            no_author_rate = (~g['mentioned_any_author']).mean()
            wrong_author_rate = ((~g['identified_correct_author']) &
                                 (g['mentioned_any_author'])).mean()
            mentioned_g = g[g['mentioned_any_author']]
            n_author_given = len(mentioned_g)
            correct_given_author_given = (
                mentioned_g['identified_correct_author'].mean()
                if n_author_given > 0 else pd.NA
            )
            subgroup_rows.append({
                'prompt_type': prompt,
                'race_ethnicity': race,
                'gender': gender,
                'n_rows': len(g),
                'n_no_author_given': int((~g['mentioned_any_author']).sum()),
                'correct_rate': correct_rate,
                'wrong_author_rate': wrong_author_rate,
                'no_author_rate': no_author_rate,
                'n_author_given': n_author_given,
                'correct_given_author_given': correct_given_author_given,
            })

    # Per-prompt no-author counts (explicit count + rate).
    no_author_prompt_rows = []
    for prompt in prompt_types:
        prompt_subset = results_df[results_df['prompt_type'] == prompt]
        n_rows = len(prompt_subset)
        if n_rows == 0:
            continue
        n_no_author = int((~prompt_subset['mentioned_any_author']).sum())
        no_author_prompt_rows.append({
            'prompt_type': prompt,
            'n_rows': n_rows,
            'n_no_author_given': n_no_author,
            'no_author_rate': n_no_author / n_rows,
        })

    conditional_subgroup_rows = []
    if direct_quotes:
        cond_subset = results_df[results_df['quote_id'].isin(direct_quotes)]
        for prompt in ['indirect', 'indirect_overt']:
            if prompt not in prompt_types:
                continue
            prompt_subset = cond_subset[cond_subset['prompt_type'] == prompt]
            if len(prompt_subset) == 0:
                continue
            grouped = prompt_subset.groupby(['race_ethnicity', 'gender'])
            for (race, gender), g in grouped:
                correct_rate = g['identified_correct_author'].mean()
                no_author_rate = (~g['mentioned_any_author']).mean()
                wrong_author_rate = ((~g['identified_correct_author']) &
                                     (g['mentioned_any_author'])).mean()
                mentioned_g = g[g['mentioned_any_author']]
                n_author_given = len(mentioned_g)
                correct_given_author_given = (
                    mentioned_g['identified_correct_author'].mean()
                    if n_author_given > 0 else pd.NA
                )
                conditional_subgroup_rows.append({
                    'prompt_type': prompt,
                    'race_ethnicity': race,
                    'gender': gender,
                    'n_rows': len(g),
                    'n_quotes_conditioned': len(set(direct_quotes)),
                    'n_no_author_given': int((~g['mentioned_any_author']).sum()),
                    'correct_rate': correct_rate,
                    'wrong_author_rate': wrong_author_rate,
                    'no_author_rate': no_author_rate,
                    'n_author_given': n_author_given,
                    'correct_given_author_given': correct_given_author_given,
                })

    # Explicit requested conditionals (overall):
    # 1) P(correct in indirect | author given in indirect)
    # 2) P(correct in indirect_overt | author given in indirect_overt)
    # 3) P(correct in indirect | direct correct)
    # 4) P(correct in indirect_overt | direct correct)
    requested_conditionals = []
    requested_conditionals_by_race_gender = []
    requested_conditionals_by_race = []
    for prompt in ['indirect', 'indirect_overt']:
        if prompt not in prompt_types:
            continue
        prompt_subset = results_df[results_df['prompt_type'] == prompt]
        author_given_subset = prompt_subset[prompt_subset['mentioned_any_author']]
        n_author_given = len(author_given_subset)
        n_correct_given_author_given = int(author_given_subset['identified_correct_author'].sum())
        p_correct_given_author_given = (
            author_given_subset['identified_correct_author'].mean()
            if n_author_given > 0 else pd.NA
        )
        requested_conditionals.append({
            'metric': f"p_correct_{prompt}_given_author_given_{prompt}",
            'prompt_type': prompt,
            'condition': 'author_given_same_prompt',
            'n_condition': n_author_given,
            'n_correct': n_correct_given_author_given,
            'probability': p_correct_given_author_given,
        })

        direct_condition_subset = results_df[
            (results_df['quote_id'].isin(direct_quotes)) &
            (results_df['prompt_type'] == prompt)
        ]
        n_direct_correct = len(direct_condition_subset)
        n_correct_given_direct_correct = int(direct_condition_subset['identified_correct_author'].sum())
        p_correct_given_direct_correct = (
            direct_condition_subset['identified_correct_author'].mean()
            if n_direct_correct > 0 else pd.NA
        )
        requested_conditionals.append({
            'metric': f"p_correct_{prompt}_given_direct_correct",
            'prompt_type': prompt,
            'condition': 'direct_correct',
            'n_condition': n_direct_correct,
            'n_correct': n_correct_given_direct_correct,
            'probability': p_correct_given_direct_correct,
        })

        # Subgroup breakdown: P(correct in prompt | author given in prompt)
        for (race, gender), g in author_given_subset.groupby(['race_ethnicity', 'gender']):
            requested_conditionals_by_race_gender.append({
                'metric': f"p_correct_{prompt}_given_author_given_{prompt}",
                'prompt_type': prompt,
                'condition': 'author_given_same_prompt',
                'race_ethnicity': race,
                'gender': gender,
                'n_condition': len(g),
                'n_correct': int(g['identified_correct_author'].sum()),
                'probability': g['identified_correct_author'].mean(),
            })
        for race, g in author_given_subset.groupby('race_ethnicity'):
            requested_conditionals_by_race.append({
                'metric': f"p_correct_{prompt}_given_author_given_{prompt}",
                'prompt_type': prompt,
                'condition': 'author_given_same_prompt',
                'race_ethnicity': race,
                'n_condition': len(g),
                'n_correct': int(g['identified_correct_author'].sum()),
                'probability': g['identified_correct_author'].mean(),
            })

        # Subgroup breakdown: P(correct in prompt | direct correct)
        for (race, gender), g in direct_condition_subset.groupby(['race_ethnicity', 'gender']):
            requested_conditionals_by_race_gender.append({
                'metric': f"p_correct_{prompt}_given_direct_correct",
                'prompt_type': prompt,
                'condition': 'direct_correct',
                'race_ethnicity': race,
                'gender': gender,
                'n_condition': len(g),
                'n_correct': int(g['identified_correct_author'].sum()),
                'probability': g['identified_correct_author'].mean(),
            })
        for race, g in direct_condition_subset.groupby('race_ethnicity'):
            requested_conditionals_by_race.append({
                'metric': f"p_correct_{prompt}_given_direct_correct",
                'prompt_type': prompt,
                'condition': 'direct_correct',
                'race_ethnicity': race,
                'n_condition': len(g),
                'n_correct': int(g['identified_correct_author'].sum()),
                'probability': g['identified_correct_author'].mean(),
            })

    # Save analysis outputs
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    if OUTPUT_ALL_ANALYSES:
        results_df.to_csv(os.path.join(analysis_dir, "analyzed_outputs.csv"), index=False)
        quote_level.to_csv(os.path.join(analysis_dir, "quote_level_analysis.csv"), index=False)
        
        if error_df is not None and len(error_df) > 0:
            error_df.to_csv(os.path.join(analysis_dir, "error_analysis.csv"), index=False)
        
        if top_confusions is not None:
            top_confusions.to_csv(os.path.join(analysis_dir, "top_confusions.csv"), index=False)
        
        if demo_confusion is not None:
            demo_confusion.to_csv(os.path.join(analysis_dir, "demographic_confusion.csv"), index=False)

        if conditional_rows:
            pd.DataFrame(conditional_rows).to_csv(
                os.path.join(analysis_dir, "conditional_author_outcomes.csv"),
                index=False,
            )

        if subgroup_rows:
            pd.DataFrame(subgroup_rows).to_csv(
                os.path.join(analysis_dir, "author_outcome_by_race_gender.csv"),
                index=False,
            )

        if no_author_prompt_rows:
            pd.DataFrame(no_author_prompt_rows).to_csv(
                os.path.join(analysis_dir, "no_author_given_by_prompt.csv"),
                index=False,
            )

        if conditional_subgroup_rows:
            pd.DataFrame(conditional_subgroup_rows).to_csv(
                os.path.join(analysis_dir, "conditional_author_outcomes_by_race_gender.csv"),
                index=False,
            )

        if reasoning_conditional_stats is not None:
            reasoning_conditional_stats.to_csv(
                os.path.join(analysis_dir, "final_given_reasoning_by_race_gender.csv"),
                index=False,
            )

        if requested_conditionals:
            pd.DataFrame(requested_conditionals).to_csv(
                os.path.join(analysis_dir, "requested_conditionals.csv"),
                index=False,
            )

        if requested_conditionals_by_race_gender:
            pd.DataFrame(requested_conditionals_by_race_gender).to_csv(
                os.path.join(analysis_dir, "requested_conditionals_by_race_gender.csv"),
                index=False,
            )
        if requested_conditionals_by_race:
            pd.DataFrame(requested_conditionals_by_race).to_csv(
                os.path.join(analysis_dir, "requested_conditionals_by_race.csv"),
                index=False,
            )
    else:
        print("  OUTPUT_ALL_ANALYSES=False -> writing summary.txt only")

    print("  Generating summary...")
    summary_text = create_summary(
        quote_level,
        error_df,
        top_confusions,
        demo_confusion,
        results_df,
        config,
        prompt_types,
        reasoning_conditional_stats=reasoning_conditional_stats,
        requested_conditionals_by_race_gender=(
            pd.DataFrame(requested_conditionals_by_race_gender)
            if requested_conditionals_by_race_gender else None
        ),
        requested_conditionals_by_race=(
            pd.DataFrame(requested_conditionals_by_race)
            if requested_conditionals_by_race else None
        ),
    )
    
    with open(os.path.join(analysis_dir, "summary.txt"), 'w') as f:
        f.write(summary_text)
    
    print(f"\n Analysis complete! Results in {analysis_dir}/")
    print("\n" + summary_text)
    
    return analysis_dir

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_outputs.py <output_directory>")
        print("\nExample:")
        print("  python analyze_outputs.py results/prompt_attribution_20240120_143022")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    analyze_outputs(output_dir)
