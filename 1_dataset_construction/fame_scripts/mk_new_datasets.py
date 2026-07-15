#!/usr/bin/env python3
"""Fame- and quote-balanced author matching — AttriBench's fame-balancing stage.

Implements the paper's fame-balancing procedure (Section 4, "Fame Balancing", and
Appendix Algorithm 1 "Fame- and quote-balanced subsampling"):

  1. Filter authors to fame log10_hits >= 3.
  2. For each dataset, designate the smallest subgroup as the reference group
     (Black female for the intersectional dataset, Latino for the multirace dataset).
  3. Single run: shuffle the reference group; sort each comparison group ascending by
     log10_hits; greedily match each reference author to one author per comparison
     group without replacement. Matches align on fame via binary search at
     target h_r - Delta_j (running per-group offset), constrained to the same quote
     count bin b(x) = floor(log2(count)), descending b_r, b_r-1, ..., 0 on failure.
     A tuple is kept when its squared fame discrepancy E = sum_j (h_j - h_r)^2 < lambda*M
     (lambda = 1); each kept author contributes to_sample = min quote count across the tuple.
  4. Repeat the single run over 100 randomized runs.
  5. Select one best run via unweighted rank aggregation over five criteria:
     authors/subgroup (higher better), quotes/subgroup (higher better),
     mean fame (higher better), fame range (lower better), RMS error (lower better).

Input
-----
``universe_authors.csv`` — one row per candidate author with columns:
    race, gender, log10_hits, count   (count = number of available quotes)

Outputs (under --out-dir, per dataset)
--------------------------------------
    <dataset>_all_runs.csv        every run's matched authors (with run_id, to_sample, match_idx)
    <dataset>_run_stats.csv       per-run metrics used for selection
    <dataset>_best_run.csv        the rank-aggregation-selected matched dataset

Reference-only: the released AttriBench CSVs in ``1_dataset_construction/datasets/``
are the canonical benchmark.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LOWER_FAME_BOUND = 3
SELECTION_CRITERIA = ["authors_per_group", "quotes_per_group", "avg_fame", "fame_range", "rms_error"]
# ascending=False -> higher is better; ascending=True -> lower is better
SELECTION_ASCENDING = [False, False, False, True, True]


def process(reference_group, other_groups, reference_group_name, other_group_names, squared_error_limit=1):
    reference_group = reference_group.sample(frac=1).reset_index(drop=True)  # shuffle reference group
    other_groups = [x.sort_values(by="log10_hits").reset_index(drop=True) for x in other_groups]  # sort non-reference groups
    num_other_groups = len(other_groups)

    # initial values
    print(reference_group_name, reference_group['log10_hits'].count(), reference_group['log10_hits'].mean())
    for i in range(num_other_groups):
        print(other_group_names[i], other_groups[i]['log10_hits'].count(), other_groups[i]['log10_hits'].mean())

    # used to match means between groups
    offsets = np.zeros(num_other_groups)

    # keep track of total squared error from best matches
    squared_error = 0

    # these will be used to store the records that we are keeping from each group
    reference_group_keep = pd.DataFrame(columns=reference_group.columns)
    other_groups_keep = [pd.DataFrame(columns=x.columns) for x in other_groups]

    # for each row in the reference group
    match_idx_count = 0
    for _, reference_row in reference_group.iterrows():
        reference_hits = reference_row.loc['log10_hits']
        reference_quote_bin = int(np.log(reference_row.loc['count'])/np.log(2))

        # initialize
        theindex = [0]*num_other_groups
        therow = [0]*num_other_groups

        # find match for reference_hits in each other group: note that we are targeting reference_hits - current difference between group means
        for i in range(num_other_groups):
            start_idx = other_groups[i].loc[:, 'log10_hits'].searchsorted(reference_hits - offsets[i])
            match_found = False
            # try bins in order: ref_bin, ref_bin-1, ..., 0
            for b in range(reference_quote_bin, -1, -1):
                idx = start_idx
                while idx < len(other_groups[i]):
                    if (idx in other_groups[i].index) and (int(np.log((other_groups[i].loc[idx, 'count']))/np.log(2)) == b):
                        theindex[i] = idx
                        match_found = True
                        break
                    idx += 1
                if match_found:
                    break
            # last attempt of no bin constraint
            if not match_found:
                idx = start_idx
                while idx not in other_groups[i].index:
                    idx += 1
                theindex[i] = idx
            therow[i] = other_groups[i].loc[theindex[i]]

        # only keep the row and its matches if the squared error is sufficiently small
        temp_squared_error = np.sum([(x['log10_hits']-reference_hits)*(x['log10_hits']-reference_hits) for x in therow])
        if (temp_squared_error < num_other_groups*squared_error_limit):
            min_count = int(np.min([reference_row['count']] + [therow[i]['count'] for i in range(num_other_groups)]))

            reference_row2 = reference_row.copy()
            reference_row2["to_sample"] = min_count
            reference_row2["match_idx"] = match_idx_count
            reference_group_keep.loc[len(reference_group_keep)] = reference_row2

            for i in range(num_other_groups):
                row2 = therow[i].copy()
                row2["to_sample"] = min_count
                row2["match_idx"] = match_idx_count
                other_groups_keep[i].loc[len(other_groups_keep[i])] = row2

                other_groups[i].drop(index=[theindex[i]], inplace=True)  # remove from the original data
                offsets[i] += therow[i]['log10_hits']-reference_hits  # keep track of difference between group means
            squared_error += temp_squared_error
            match_idx_count += 1

    print()
    authors_per_group = reference_group_keep['log10_hits'].count()
    print('Count', authors_per_group)  # how many records do we have per group
    all_fames = [reference_group_keep['log10_hits'].mean()]+[x['log10_hits'].mean() for x in other_groups_keep]
    all_names = [reference_group_name]+other_group_names
    for i in range(num_other_groups+1):  # average fame for each group
        print(all_names[i], all_fames[i])
    fame_range = np.max(all_fames)-np.min(all_fames)
    print('range', fame_range)  # max fame - min fame
    avg = np.mean(all_fames)
    print('average', avg)  # avg fame
    rms_error = np.sqrt(squared_error / (num_other_groups*reference_group_keep['log10_hits'].count()))
    print('RMS error', rms_error)  # root-mean-squared matching error
    print()

    num_matches = reference_group_keep['log10_hits'].count()
    reference_group_keep = reference_group_keep.reset_index(drop=True)
    other_groups_keep = [x.reset_index(drop=True) for x in other_groups_keep]
    total_quotes_per_group = 0
    for i in range(num_matches):
        temp_counts = [reference_group_keep.loc[i, 'count']]+[x.iloc[i].loc['count'] for x in other_groups_keep]
        total_quotes_per_group += np.min(temp_counts)
    print("Total quotes:", total_quotes_per_group)

    # sort each group from max to min fame and concatenate together
    reference_group_keep = reference_group_keep.sort_values(by="log10_hits", ascending=False)
    other_groups_keep = [x.sort_values(by="log10_hits", ascending=False) for x in other_groups_keep]
    return pd.concat([reference_group_keep]+other_groups_keep), authors_per_group, total_quotes_per_group, fame_range, avg, rms_error


# Dataset specifications: (reference subgroup, comparison subgroups) — see paper Section 4.
def intersectional_groups(data):
    reference_group = data[(data['race'] == 'black') & (data['gender'] == 'female')]
    other_groups = [
        data[(data['race'] == 'black') & (data['gender'] == 'male')],
        data[(data['race'] == 'white') & (data['gender'] == 'male')],
        data[(data['race'] == 'white') & (data['gender'] == 'female')],
    ]
    return reference_group, 'black_female', other_groups, ['black_male', 'white_male', 'white_female']


def multirace_groups(data):
    reference_group = data[(data['race'] == 'latino')]
    other_groups = [
        data[(data['race'] == 'black')],
        data[(data['race'] == 'white')],
        data[(data['race'] == 'asian')],
    ]
    return reference_group, 'latino', other_groups, ['black', 'white', 'asian']


DATASETS = {
    "intersectional": intersectional_groups,
    "multirace": multirace_groups,
}


def run_dataset(data, group_fn, n_runs, squared_error_limit):
    """Run the single-run matcher n_runs times; return (all_runs_df, run_stats_df)."""
    run_frames = []
    run_stats = []
    for i in range(n_runs):
        print("run: ", i)
        reference_group, reference_group_name, other_groups, other_group_names = group_fn(data)
        df_i, authors_per_group, quotes_per_group, fame_range, avg, rms_error = process(
            reference_group, other_groups, reference_group_name, other_group_names,
            squared_error_limit=squared_error_limit,
        )
        df_i = df_i.copy()
        df_i["run_id"] = i
        run_frames.append(df_i)
        run_stats.append({
            "run_id": i,
            "authors_per_group": authors_per_group,
            "quotes_per_group": quotes_per_group,
            "avg_fame": avg,
            "fame_range": fame_range,
            "rms_error": rms_error,
            "squared_error_limit": squared_error_limit,
        })
    return pd.concat(run_frames, ignore_index=True), pd.DataFrame(run_stats)


def select_best_run(stats_df):
    """Unweighted 5-criterion rank aggregation; return the winning run_id (paper Section 4)."""
    stats = stats_df.copy()
    rank_cols = []
    for crit, ascending in zip(SELECTION_CRITERIA, SELECTION_ASCENDING):
        col = f"rank_{crit}"
        stats[col] = stats[crit].rank(ascending=ascending)
        rank_cols.append(col)
    stats["rank_sum"] = stats[rank_cols].sum(axis=1)

    min_sum = stats["rank_sum"].min()
    cands = stats[stats["rank_sum"] == min_sum]
    if len(cands) == 1:
        return int(cands.iloc[0]["run_id"])
    print("ties: ", len(cands))
    return int(cands.sort_values(SELECTION_CRITERIA, ascending=SELECTION_ASCENDING).iloc[0]["run_id"])


def main():
    parser = argparse.ArgumentParser(description="Fame- and quote-balanced author matching (AttriBench).")
    parser.add_argument("--universe-csv", default="universe_authors.csv",
                        help="Author-level CSV with columns: race, gender, log10_hits, count.")
    parser.add_argument("--out-dir", default="outputs_fame_balanced",
                        help="Directory to write per-run outputs and the selected best run.")
    parser.add_argument("--runs", type=int, default=100,
                        help="Randomized runs per dataset (paper uses 100).")
    parser.add_argument("--squared-error-limit", type=float, default=1.0,
                        help="Match acceptance threshold lambda (paper: lambda = 1).")
    args = parser.parse_args()

    data = pd.read_csv(args.universe_csv)
    data['to_sample'] = 0
    data['match_idx'] = None
    data = data[data["log10_hits"] >= LOWER_FAME_BOUND]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, group_fn in DATASETS.items():
        print(f"\n========== {dataset_name} ==========")
        all_runs, run_stats = run_dataset(data, group_fn, args.runs, args.squared_error_limit)
        all_runs.to_csv(out_dir / f"{dataset_name}_all_runs.csv", index=False)
        run_stats.to_csv(out_dir / f"{dataset_name}_run_stats.csv", index=False)

        best_run_id = select_best_run(run_stats)
        best_run = all_runs[all_runs["run_id"] == best_run_id]
        best_run.to_csv(out_dir / f"{dataset_name}_best_run.csv", index=False)
        print(f"{dataset_name}: selected best run_id={best_run_id} "
              f"({len(best_run)} authors) -> {out_dir / f'{dataset_name}_best_run.csv'}")


if __name__ == "__main__":
    main()
