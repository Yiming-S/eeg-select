# eeg-select

Minimal code and derived results for **Small calibration cohorts do not reliably select motor-imagery EEG systems for new participants**, by Yiming Shen.

The study compares CSP-LDA, Riemannian tangent-space LDA, and EEGNet, used alone, in directed confidence routes, or in two-decoder ensembles. Training, calibration, and test participants are separate.

## Run

Use Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python reproduce.py
```

The script replays selection among the 12 configurations using calibration accuracy and a cost limit of 0.75 times solo EEGNet. It checks the choices against the recorded decisions, rebuilds the configuration and policy summaries, and recomputes the paired confidence intervals and fixed-use sensitivity. Results are written to `results/`. The default uses 10,000 bootstrap draws and checks the numerical outputs against `reference/`.

## Files and scope

- `reproduce.py`: configuration selection, summary statistics, and bootstrap analysis.
- `data/`: compressed calibration and participant-level results, recorded decisions, and timing summaries.
- `reference/`: the original numerical results used to check reproduction.
- `provenance.json`: source records and SHA-256 checksums.

This release reanalyses saved results. EEG preprocessing, decoder fitting, and hardware timing are not rerun. Each participant receives equal weight; EEGNet initialization seeds were averaged before these tables were saved. The bootstrap resamples participants and outer fits for PhysioNet and Cho2017, and participants and repeats for BNCI2014-004. It conditions on the fitted models already evaluated in the study.

The fixed reference and the wider use-rate analyses were chosen after earlier results were inspected. Costs at the 20% request come from direct complete-path timing; other use rates in the fixed-use table use the study's anchored cost model. The separate direct-timing summary retains the later measurements at 30%, 40%, and 50%.

## Public EEG sources

- [PhysioNet EEG Motor Movement/Imagery](https://doi.org/10.13026/C28G6P)
- [BNCI2014-004](https://bnci-horizon-2020.eu/database/data-sets)
- [Cho2017](https://doi.org/10.5524/100295)

Raw EEG, fitted model weights, and submission correspondence are excluded. The participant numbers in the tables are the public datasets' identifiers. Code and included derived tables are released under the MIT license.
