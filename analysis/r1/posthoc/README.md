# Post-hoc analyses of the published data (first revision, D1 to D3)

These scripts reanalyse the published result data (data deposit version 1.1.1) and
Archive S1 of the submission. They make no model call and start no service. Each
analysis first reproduces the published figures it starts from and reports every
mismatch.

| Script | What it does |
|---|---|
| `posthoc_common.py` | Loads the 180 benchmark windows (the deposited own-data export), every configuration of Tables 6, 8 and S13 as per-window verdicts, and the archived statistics code (area A, B, C and D scripts of the deposit, loaded function by function without running them; the re-implemented Kayenta classifier; the judge service's series guards, Mann-Kendall test and summary-packet code; `tools/ci_utils.py`). |
| `extract_published_tables.py` | Reads Tables 1, 6, 8, S13 and S38 and the text of Sections 4.2, 4.6 and 5.1 out of the submitted manuscript and Supplementary Materials (Word-document reader, read only). |
| `d1_cost_sensitivity.py` | D1: expected loss r × p × FNR + (1 − p) × FPR (caption of Figure S17); reproduces the crossovers 9/64 and 5/6 of the comparison set and the boundaries of Figure S17; the loss-minimising configuration on the grid of cost ratios and prevalences, with and without healed_transient among the benign windows. Exact rational arithmetic. |
| `d2_label_policies.py` | D2: detection, false alarms, balanced accuracy and MCC for every configuration of Tables 6 and 8 under labelling policies P0 to P7; reproduces Table S13 (P1, P2) and Tables 6 and 8 (P0); label changes per family under P6 and P7 from the realised series; the order of the configurations per policy; the statements of Sections 4.2, 4.6 and 5.1 evaluated under each policy. |
| `d3_rationale_config.json` | D3: claim patterns and tolerances, fixed before the check was run on any rationale. |
| `d3_rationale_check.py` | D3: claims of full rationales (level, spread, trend, tail, quoted numbers) checked against the window series and the summary packet the model saw. Runs on the name-attribution arms of the deposit (`--archive`) or on long-format records of new runs (`--records`, with `--series-jsonl` or `--generate`). |
| `test_d3_extraction.py` | Tests of the D3 extraction and checks on hand-written sentences and synthetic windows. |
| `build_facts_d.py` | Merges the fact files of D1 to D3 into one list of facts (ids D-001 onwards). |

Run from the repository root with Python 3.12, numpy 2.3.5, pandas 3.0.1 and scipy
1.18.1 (D1 to D3) and a Word-document reader (the table extraction), always with `-I -B`:

```bash
python -I -B analysis/r1/posthoc/extract_published_tables.py <manuscript.docx> <supplementary.docx> <out>/published_tables.json
python -I -B analysis/r1/posthoc/d1_cost_sensitivity.py <data-archive> <archive-s1-dir> <out>
python -I -B analysis/r1/posthoc/d2_label_policies.py <data-archive> <archive-s1-dir> <out>/published_tables.json <out>
python -I -B analysis/r1/posthoc/test_d3_extraction.py
python -I -B analysis/r1/posthoc/d3_rationale_check.py --archive <data-archive> --out <out>
python -I -B analysis/r1/posthoc/build_facts_d.py <out>
```

`<archive-s1-dir>` is the unpacked Archive S1 (06_Supplementary_Reanalysis.zip). No
script overwrites an existing output file.

The D3 check on new runs reads their long-format records and the window series:

```bash
python -I -B analysis/r1/posthoc/d3_rationale_check.py --records <records>.csv ... \
    --generate <dataset-id> ... --master-seed 20260621 --n-per-family 20 --out <out>
```

or with `--series-jsonl <series>.jsonl ...` in place of `--generate`. The judge service
caps a rationale at 500 characters; the run record counts rationales of exactly that
length.

Limits of D3, stated with every result: extraction is lexical and also matches
negated, hedged and malformed text (Text S8 of the Supplementary Materials warns of
the same for the lexical audit); a claim is assigned to a metric only by kind words;
a sentence about the control is read as a claim about the canary against the
control; the trend claim is checked on canary minus control, not on the canary's own
slope that the summary packet prints.
