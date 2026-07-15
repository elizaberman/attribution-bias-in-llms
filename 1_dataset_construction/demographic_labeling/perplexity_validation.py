"""Tier-3 demographic validation via Perplexity's sonar model.

Stage (b) of dataset construction, the final labeling tier. Authors whose race/gender
came from Wikidata (`wikidata_demographics.py`) are treated as authoritative and
skipped; only authors labeled by an LLM (`openai_mc_labeling.py`) are re-queried
here, and agreement between the two LLM sources forms the consensus label.

Queries run concurrently with checkpointing, so a rerun resumes from the checkpoint
rather than repeating paid calls. Authors whose sources disagree are written to a
separate manual-review file rather than being silently resolved.

Usage
-----
  python perplexity_validation.py --input-file authors_to_validate.csv

  reads  $ATTRIBENCH_RESULTS/demographic_labeling/authors_to_validate.csv (--input-file)
  writes $ATTRIBENCH_RESULTS/demographic_labeling/ (--outdir)
           perplexity_output.csv         (validated labels)
           perplexity_manual_review.csv  (source disagreements)
           perplexity_checkpoint.csv     (resume state)

Pass --query-gender to validate gender in addition to race.

REFERENCE-ONLY: calls a paid API. The released consensus labels ship in
`datasets/*_with_quotes.csv` (`race` / `gender` with their `*_source` columns).
Requires PERPLEXITY_API_KEY.
"""
import argparse
import os
import time
import random
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv(override=True)

# --- AttriBench path resolution ---
_ATTRIBENCH_ROOT = os.environ.get(
    "ATTRIBENCH_ROOT", str(Path(__file__).resolve().parents[2])
)
_ATTRIBENCH_RESULTS = os.environ.get(
    "ATTRIBENCH_RESULTS", os.path.join(_ATTRIBENCH_ROOT, "results")
)
_LABELING_DIR = os.path.join(_ATTRIBENCH_RESULTS, "demographic_labeling")
# ----------------------------------

# Custom exception for rate limiting
class RateLimitException(Exception):
    """Raised when API rate limit is exceeded"""
    pass

# API client
perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")

if not perplexity_api_key:
    raise SystemExit("PERPLEXITY_API_KEY not set in .env file")

perplexity_client = OpenAI(api_key=perplexity_api_key, base_url="https://api.perplexity.ai")

# Categories
RACE_CATEGORIES = ["black", "white", "asian", "middle eastern", "indigenous", "latino", "other"]
GENDER_CATEGORIES = ["male", "female", "non-binary", "other"]

# System instruction for MCQ
SYSTEM_INSTRUCTION = "You are an evaluator. Respond with ONLY a single letter (A, B, C, D, E, F, or G) - nothing else. Do not include periods, explanations, or any other characters. Just the letter."

def build_mcq_race(author):
    """Build MCQ for race/ethnicity prediction"""
    options = RACE_CATEGORIES.copy()
    random.shuffle(options)

    mcq = f"What is the race/ethnicity of {author}?\n\n"
    for i, opt in enumerate(options):
        letter = chr(ord('A') + i)
        mcq += f"{letter}. {opt}\n"
    mcq += "\nRespond with only the correct letter."

    return mcq, options

def build_mcq_gender(author):
    """Build MCQ for gender prediction"""
    options = GENDER_CATEGORIES.copy()
    random.shuffle(options)

    mcq = f"What is the gender of {author}?\n\n"
    for i, opt in enumerate(options):
        letter = chr(ord('A') + i)
        mcq += f"{letter}. {opt}\n"
    mcq += "\nRespond with only the correct letter."

    return mcq, options

def query_perplexity_mcq(question_text, max_retries=3, max_reprompts=3):
    """Query Perplexity with MCQ format, reprompting up to 3 times if response is not a single letter

    Args:
        question_text: The MCQ question to ask
        max_retries: Number of retries for API errors (default: 3)
        max_reprompts: Number of attempts to get a single letter response (default: 3)

    Returns:
        Single letter response (A-G) or invalid response after max_reprompts attempts
    """
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(2)

            # Try up to max_reprompts times to get a valid single letter response
            for reprompt_attempt in range(max_reprompts):
                if reprompt_attempt > 0:
                    print(f"  🔄 Reprompting Perplexity (attempt {reprompt_attempt + 1}/{max_reprompts}) - previous response was not a single letter")
                    time.sleep(1)

                response = perplexity_client.chat.completions.create(
                    model="sonar",
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTION},
                        {"role": "user", "content": question_text}
                    ],
                    temperature=0.0
                )

                answer = response.choices[0].message.content.strip().upper()

                # Check if it's a valid single letter response
                if len(answer) == 1 and answer.isalpha():
                    print(f"  ✓ Perplexity returned valid single letter: {answer}")
                    return answer
                else:
                    print(f"  ⚠️ Perplexity returned invalid response: {answer} (expected single letter)")
                    if reprompt_attempt == max_reprompts - 1:
                        # Tried 3 times, still invalid - return what we got
                        print(f"  ❌ Perplexity failed to return single letter after {max_reprompts} attempts")
                        return answer

            return answer

        except Exception as e:
            error_msg = str(e).lower()
            # Check if it's a rate limit error
            if 'rate limit' in error_msg or 'rate_limit' in error_msg or '429' in error_msg or 'too many requests' in error_msg:
                print(f"  🚨 Perplexity RATE LIMIT EXCEEDED: {e}")
                raise RateLimitException(f"Perplexity rate limit exceeded: {e}")

            print(f"  ⚠️ Perplexity error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                continue
            return None

    return None

def predict_perplexity(author, demographic_type="race"):
    """Predict demographics using Perplexity only

    Args:
        author: Author name
        demographic_type: "race" or "gender"

    Returns:
        Predicted category (str) or None
    """
    if demographic_type == "race":
        mcq_text, options = build_mcq_race(author)
    else:  # gender
        mcq_text, options = build_mcq_gender(author)

    print(f"  🤖 Querying Perplexity for {demographic_type}...")
    perplexity_letter = query_perplexity_mcq(mcq_text)
    time.sleep(2.0)  # Respect 50 RPM rate limit

    # Decode prediction
    perplexity_pred = None

    if perplexity_letter:
        # Check if it's a single letter response
        if len(perplexity_letter) == 1 and perplexity_letter.isalpha():
            index = ord(perplexity_letter) - ord('A')
            if 0 <= index < len(options):
                perplexity_pred = options[index]
                print(f"  💬 Perplexity predicted: {perplexity_letter} -> {perplexity_pred}")
        else:
            # Check if it's a direct category response
            normalized_response = perplexity_letter.lower().strip()
            if normalized_response in options:
                perplexity_pred = normalized_response
                print(f"  💬 Perplexity predicted (direct): {perplexity_pred}")
            else:
                print(f"  ⚠️ Perplexity returned invalid response: {perplexity_letter}")

    return perplexity_pred

def predict_author_perplexity(author, need_race, need_gender):
    """Predict race and/or gender for a single author using Perplexity

    Args:
        author: Author name
        need_race: Whether to query for race
        need_gender: Whether to query for gender

    Returns:
        Tuple of (author, predictions_dict, error)
    """
    try:
        print(f"\n{'='*60}")
        print(f"Validating with Perplexity: {author}")
        print(f"  Need race: {need_race}, Need gender: {need_gender}")
        print(f"{'='*60}")

        predictions = {}

        # Predict race only if needed
        if need_race:
            perplexity_race = predict_perplexity(author, "race")
            predictions['perplexity_race'] = perplexity_race
            time.sleep(0.5)
        else:
            print(f"  ⏭️  Skipping race (source is wikidata)")
            predictions['perplexity_race'] = None

        # Predict gender only if needed
        if need_gender:
            perplexity_gender = predict_perplexity(author, "gender")
            predictions['perplexity_gender'] = perplexity_gender
            time.sleep(0.5)
        else:
            print(f"  ⏭️  Skipping gender (source is wikidata)")
            predictions['perplexity_gender'] = None

        return author, predictions, None

    except RateLimitException as e:
        print(f"  ❌ Rate limit error for {author}: {e}")
        return author, None, str(e)
    except Exception as e:
        print(f"  ❌ Error processing {author}: {e}")
        return author, None, str(e)

def run_perplexity_validation(
    input_file = os.path.join(_LABELING_DIR, 'authors_to_validate.csv'),
    output_file = os.path.join(_LABELING_DIR, 'perplexity_output.csv'),
    manual_review_file = os.path.join(_LABELING_DIR, 'perplexity_manual_review.csv'),
    checkpoint_file = os.path.join(_LABELING_DIR, 'perplexity_checkpoint.csv'),
    max_workers = 2,
    query_gender = False
):
    """
    Run Perplexity validation over the author list in `input_file`.
    Only queries Perplexity when source is "llm" (not "wikidata").
    Set query_gender=False to skip gender queries even if gender_source == llm.

    Args:
        input_file: Path to input CSV file
        output_file: Path to save all results with Perplexity predictions
        manual_review_file: Path to save rows where ChatGPT and Perplexity disagree
        max_workers: Number of concurrent workers (default: 2, conservative for rate limits)
        checkpoint_file: Path to save checkpoint after each author
    """
    print(f"\n{'='*60}")
    print(f"PERPLEXITY VALIDATION SCRIPT")
    print(f"{'='*60}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Manual Review: {manual_review_file}")
    print(f"Concurrent workers: {max_workers}")
    print(f"{'='*60}\n")

    # Load dataset
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")

    # Group by author to determine what needs to be queried
    author_query_needs = {}
    for _, row in df.iterrows():
        author = row['author']
        if author not in author_query_needs:
            # Determine if we need to query Perplexity for this author
            need_race = str(row['race_source']).lower() == 'llm'
            need_gender = query_gender and str(row['gender_source']).lower() == 'llm'
            author_query_needs[author] = {
                'need_race': need_race,
                'need_gender': need_gender
            }

    print(f"Unique authors: {len(author_query_needs)}")

    # Count how many need queries
    need_race_count = sum(1 for v in author_query_needs.values() if v['need_race'])
    need_gender_count = sum(1 for v in author_query_needs.values() if v['need_gender'])
    print(f"Authors needing race query: {need_race_count}")
    print(f"Authors needing gender query: {need_gender_count} (query_gender={query_gender})")

    # Check for existing checkpoint
    author_predictions_cache = {}
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"\n📂 Loading checkpoint from {checkpoint_file}...")
        checkpoint_df = pd.read_csv(checkpoint_file)
        for _, row in checkpoint_df.iterrows():
            author = row['author']
            author_predictions_cache[author] = {
                'perplexity_race': row.get('perplexity_race'),
                'perplexity_gender': row.get('perplexity_gender')
            }
        print(f"✓ Loaded {len(author_predictions_cache)} authors from checkpoint")

        # Filter out already processed authors
        authors_to_process = [a for a in author_query_needs.keys() if a not in author_predictions_cache]
        print(f"Remaining authors to process: {len(authors_to_process)}\n")
    else:
        authors_to_process = list(author_query_needs.keys())

    if len(authors_to_process) == 0:
        print("✓ All authors already processed!")
    else:
        # Process authors with concurrent workers
        print(f"{'='*60}")
        print(f"Processing {len(authors_to_process)} authors with {max_workers} concurrent workers...")
        print(f"{'='*60}\n")

        rate_limit_errors = []
        rate_limit_threshold = 5

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_author = {
                executor.submit(
                    predict_author_perplexity,
                    author,
                    author_query_needs[author]['need_race'],
                    author_query_needs[author]['need_gender']
                ): author
                for author in authors_to_process
            }

            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_author):
                author, predictions, error = future.result()
                completed += 1

                if error:
                    # Check if it's a rate limit error
                    if 'rate limit' in error.lower():
                        rate_limit_errors.append(author)
                        print(f"[{completed}/{len(authors_to_process)}] 🚨 RATE LIMIT: {author}")
                        print(f"  Rate limit errors so far: {len(rate_limit_errors)}/{rate_limit_threshold}")

                        # Stop if we hit the threshold
                        if len(rate_limit_errors) >= rate_limit_threshold:
                            print(f"\n{'='*60}")
                            print(f"🛑 STOPPING: Hit {len(rate_limit_errors)} rate limit errors!")
                            print(f"{'='*60}")
                            print(f"Processed: {completed}/{len(authors_to_process)} authors")
                            print(f"Successful: {len(author_predictions_cache)}")
                            print(f"\n⚠️  RECOMMENDATION:")
                            print(f"  1. Reduce max_workers (currently {max_workers})")
                            print(f"  2. Wait a few minutes before restarting")
                            print(f"  3. Your progress is saved in the checkpoint")
                            print(f"{'='*60}\n")

                            # Cancel remaining futures
                            for f in future_to_author:
                                f.cancel()

                            break
                    else:
                        print(f"[{completed}/{len(authors_to_process)}] ❌ Failed: {author} - {error}")
                else:
                    author_predictions_cache[author] = predictions
                    print(f"[{completed}/{len(authors_to_process)}] ✓ Completed: {author}")

                    # Save checkpoint
                    if checkpoint_file:
                        # Create temporary results for checkpoint
                        temp_results = []
                        for a, preds in author_predictions_cache.items():
                            temp_results.append({
                                'author': a,
                                'perplexity_race': preds.get('perplexity_race'),
                                'perplexity_gender': preds.get('perplexity_gender')
                            })
                        checkpoint_df = pd.DataFrame(temp_results)
                        checkpoint_df.to_csv(checkpoint_file, index=False)

    # Create final results dataframe with updated perplexity predictions
    results = []
    manual_review_rows = []

    for _, row in df.iterrows():
        author = row['author']

        # Create result row starting with original data
        result_row = row.to_dict()

        # Get the query needs for this author
        need_race = author_query_needs[author]['need_race']
        need_gender = author_query_needs[author]['need_gender']

        # If source is wikidata, set perplexity value to chatgpt value (no query needed)
        if not need_race:
            # race_source is wikidata, so set perplexity_race = chatgpt_race
            result_row['perplexity_race'] = row['chatgpt_race']

        if not need_gender:
            # gender_source is wikidata, so set perplexity_gender = chatgpt_gender
            result_row['perplexity_gender'] = row['chatgpt_gender']

        # Update perplexity predictions if we queried for them
        if author in author_predictions_cache:
            cached = author_predictions_cache[author]

            # Update race if we queried for it
            if need_race and cached.get('perplexity_race') is not None:
                result_row['perplexity_race'] = cached['perplexity_race']

            # Update gender if we queried for it
            if need_gender and cached.get('perplexity_gender') is not None:
                result_row['perplexity_gender'] = cached['perplexity_gender']

        # Check for disagreement
        race_agree = (
            str(result_row['chatgpt_race']).strip().lower() ==
            str(result_row['perplexity_race']).strip().lower()
        )
        gender_agree = (
            str(result_row['chatgpt_gender']).strip().lower() ==
            str(result_row['perplexity_gender']).strip().lower()
        )

        result_row['race_agree'] = race_agree
        result_row['gender_agree'] = gender_agree

        # Add to results
        results.append(result_row)

        # Add to manual review if there's disagreement
        if not race_agree or not gender_agree:
            manual_review_rows.append(result_row)

    # Create results dataframes
    results_df = pd.DataFrame(results)
    manual_review_df = pd.DataFrame(manual_review_rows)

    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ All results saved to: {output_file} ({len(results_df)} rows)")

    if len(manual_review_df) > 0:
        manual_review_df.to_csv(manual_review_file, index=False)
        print(f"⚠️  Manual review needed saved to: {manual_review_file} ({len(manual_review_df)} rows)")
    else:
        print(f"✅ No disagreements - no manual review needed!")

    # Calculate and display statistics
    print(f"\n{'='*60}")
    print(f"📊 AGREEMENT STATISTICS")
    print(f"{'='*60}")

    # Race agreement
    race_agree_count = results_df['race_agree'].sum()
    race_total = len(results_df)
    race_agreement_rate = (race_agree_count / race_total * 100) if race_total > 0 else 0
    print(f"Race Agreement: {race_agreement_rate:.2f}% ({race_agree_count}/{race_total} agree)")

    # Gender agreement
    gender_agree_count = results_df['gender_agree'].sum()
    gender_total = len(results_df)
    gender_agreement_rate = (gender_agree_count / gender_total * 100) if gender_total > 0 else 0
    print(f"Gender Agreement: {gender_agreement_rate:.2f}% ({gender_agree_count}/{gender_total} agree)")

    # Both agree
    both_agree = (results_df['race_agree'] & results_df['gender_agree']).sum()
    both_rate = (both_agree / len(results_df) * 100) if len(results_df) > 0 else 0
    print(f"Both Agree: {both_rate:.2f}% ({both_agree}/{len(results_df)} agree)")

    # Manual review needed
    print(f"\nManual Review Needed: {len(manual_review_df)} rows ({len(manual_review_df)/len(results_df)*100:.1f}%)")

    print(f"{'='*60}\n")

    return results_df, manual_review_df

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Perplexity-sonar consensus validation of author race/gender labels."
    )
    ap.add_argument("--input-file", default=os.path.join(_LABELING_DIR, "authors_to_validate.csv"),
                    help="CSV with columns author, race_source, gender_source, chatgpt_race, chatgpt_gender.")
    ap.add_argument("--outdir", default=_LABELING_DIR,
                    help="Directory for perplexity_output.csv / _manual_review.csv / _checkpoint.csv.")
    ap.add_argument("--max-workers", type=int, default=2)
    ap.add_argument("--query-gender", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    run_perplexity_validation(
        input_file=args.input_file,
        output_file=os.path.join(args.outdir, "perplexity_output.csv"),
        manual_review_file=os.path.join(args.outdir, "perplexity_manual_review.csv"),
        checkpoint_file=os.path.join(args.outdir, "perplexity_checkpoint.csv"),
        max_workers=args.max_workers,
        query_gender=args.query_gender,
    )
