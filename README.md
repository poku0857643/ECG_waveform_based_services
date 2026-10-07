# ECG waveform based services

ECG studies and a watch-ECG recommendation service built on the **PTB-XL** 12-lead ECG dataset.
Each feature or study is added on its own branch (stacked in order, one pull request each).

| # | Branch | Adds |
|---|---|---|
| 1 | `feature/data-loading-hrv` | 100 Hz signal loading, patient metadata, diagnostic superclasses, recording dates, HRV of repeat ECGs |
| 2 | `study/ecg-similarity` | similarity between a patient's two ECGs: raw signal vs median beat vs HRV |
| 3 | `study/cleaning-clustering` | data-cleaning test (beat detection, quality filter) and median-beat clustering of all patients |
| 4 | `feature/watch-step1-groups` | watch (lead I) recommendation, step 1: ECG group + profile, change between two recordings |
| 5 | `feature/watch-step2-actions` | step 2: health / lifestyle action rules |
| 6 | `feature/watch-step3-risk` | step 3: diagnosis-risk scores |
| 7 | `feature/final-model-api` | final train / tune / test, saved model, FastAPI endpoint + tests |
| 8 | `study/qt-interval` | QT / corrected QT study in `qt-interval/`: correlation with age and groups, effect on grouping |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download PTB-XL 1.0.3 from PhysioNet (https://physionet.org/content/ptb-xl/1.0.3/) and put the folder
`ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3` in the project root (git-ignored), or link it:

```bash
ln -s /path/to/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 .
```

## Data and licence

PTB-XL: Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly
available electrocardiography dataset (version 1.0.3). PhysioNet. https://doi.org/10.13026/kfzx-aw45 -
licensed under CC BY 4.0. The dataset itself is not included in this repository.

**Not a medical device.** Outputs describe similarity to PTB-XL patients; they are not a diagnosis.
