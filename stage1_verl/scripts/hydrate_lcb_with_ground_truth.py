"""Hydrate LCB oracle parquet with the full `reward_model.ground_truth` from the source.

Inputs:
  --hf_repo      HF dataset (e.g. violetxi/livecodebench_v6_gemini-3-pro) that has
                 oracle solutions in extra_info but reward_model.ground_truth=""
  --source       Local source parquet with the full reward_model.ground_truth
                 (e.g. data/instruct/livecodebench_v6/test.parquet)
  --output       Output parquet path (will be ~4GB for LCB v6)

The two are matched on extra_info["index"] (a row's position in the original source).

Streaming pyarrow read/write so memory stays bounded even for 4GB outputs.
"""
import argparse
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_repo", default="violetxi/livecodebench_v6_gemini-3-pro",
                    help="HF dataset repo_id containing the oracle parquet.")
    ap.add_argument("--hf_filename", default="train.parquet")
    ap.add_argument("--source", default="data/instruct/livecodebench_v6/test.parquet",
                    help="Local source parquet with full reward_model.ground_truth.")
    ap.add_argument("--output", default="data/instruct/livecodebench_v6_with_oracle/train_hydrated.parquet")
    ap.add_argument("--rg_size", type=int, default=64,
                    help="Row-group size for streaming I/O (smaller = less memory).")
    args = ap.parse_args()

    t0 = time.time()

    # Step 1: get the oracle parquet (small, ~3MB). Download from HF.
    print(f"[{time.time()-t0:.1f}s] downloading {args.hf_repo}/{args.hf_filename} ...", flush=True)
    oracle_path = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_filename,
                                   repo_type="dataset")
    print(f"[{time.time()-t0:.1f}s]   {oracle_path}", flush=True)

    # Step 2: build index → extra_info lookup from oracle (small, load all).
    oracle_pf = pq.ParquetFile(oracle_path)
    oracle_tbl = oracle_pf.read()
    oracle_dict = oracle_tbl.to_pydict()
    oracle_by_index = {}
    for i, ei in enumerate(oracle_dict["extra_info"]):
        idx = ei.get("index")
        if idx is None:
            raise ValueError(f"oracle row {i}: extra_info has no 'index' field — cannot match")
        oracle_by_index[idx] = ei
    print(f"[{time.time()-t0:.1f}s] loaded {len(oracle_by_index)} oracle rows", flush=True)

    # Step 3: stream source parquet, swap extra_info, write hydrated output.
    src_pf = pq.ParquetFile(args.source)
    print(f"[{time.time()-t0:.1f}s] source: {src_pf.metadata.num_rows} rows, "
          f"{src_pf.num_row_groups} row groups", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    n_matched = n_unmatched = 0
    row_idx = 0
    batch_num = 0

    # iter_batches (not read_row_group) — pyarrow can't read nested columns into a
    # chunked Table; small single-chunk RecordBatches sidestep that limit.
    for batch in src_pf.iter_batches(batch_size=args.rg_size):
        cols = batch.to_pydict()

        # Only keep source rows that have an oracle match — drop unmatched.
        # This ensures the hydrated output mirrors the (filtered) oracle's row set,
        # so every row in the final parquet has both full ground_truth AND an oracle
        # solution in extra_info.
        keep_indices = []
        new_extra_info = []
        for i in range(batch.num_rows):
            ei_oracle = oracle_by_index.get(row_idx)
            if ei_oracle is None:
                n_unmatched += 1
            else:
                keep_indices.append(i)
                new_extra_info.append(ei_oracle)
                n_matched += 1
            row_idx += 1

        if not keep_indices:
            del batch, cols
            continue

        new_tbl = pa.table({
            "data_source":  [cols["data_source"][i]  for i in keep_indices],
            "prompt":       [cols["prompt"][i]       for i in keep_indices],
            "ability":      [cols["ability"][i]      for i in keep_indices],
            "reward_model": [cols["reward_model"][i] for i in keep_indices],  # full ground_truth preserved
            "extra_info":   new_extra_info,
        })
        if writer is None:
            writer = pq.ParquetWriter(args.output, new_tbl.schema, compression="snappy")
        writer.write_table(new_tbl)
        del batch, cols, new_tbl
        batch_num += 1
        if batch_num % 10 == 0:
            print(f"[{time.time()-t0:.1f}s] processed {row_idx}/{src_pf.metadata.num_rows} rows "
                  f"(matched {n_matched}, unmatched-dropped {n_unmatched})", flush=True)

    if writer is not None:
        writer.close()
    out_size_mb = Path(args.output).stat().st_size / 1e6
    print(f"[{time.time()-t0:.1f}s] ✓ wrote {args.output} ({out_size_mb:.1f} MB)", flush=True)
    print(f"  matched: {n_matched}, unmatched: {n_unmatched}", flush=True)

    # Sanity check: re-read and confirm a row has both ground_truth and oracle solution.
    print(f"[{time.time()-t0:.1f}s] verifying...", flush=True)
    verify_pf = pq.ParquetFile(args.output)
    sample = next(verify_pf.iter_batches(batch_size=1)).to_pydict()
    ei0 = sample["extra_info"][0]
    rm0 = sample["reward_model"][0]
    print(f"  row 0 extra_info keys: {sorted(ei0.keys())}")
    print(f"  row 0 has oracle solution: {bool(ei0.get('solution'))}")
    print(f"  row 0 has oracle solution_code_only: {bool(ei0.get('solution_code_only'))}")
    print(f"  row 0 reward_model.ground_truth length: {len(rm0.get('ground_truth') or '')}")
    print(f"  row 0 reward_model.style: {rm0.get('style')}")


if __name__ == "__main__":
    main()
