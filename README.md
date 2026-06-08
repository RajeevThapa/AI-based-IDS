# AI Based Intrusion Detection for IoT

## Missing Files

Large datasets and trained models are not included in the repository.

### Missing in v0.01

```
dataset/x86/sar.csv
dataset/x86/pcap.csv

outputs/models/rf_fused_v2.pkl
outputs/models/rf_pcap_only.pkl
outputs/models/rf_sar_only.pkl

outputs/fused_pcap_test.csv
outputs/pcap_test.csv
outputs/pcap_train.csv
outputs/sar_test.csv
outputs/sar_train.csv
outputs/scenario_comparision.csv
```

### Missing in v0.02

```
outputs/models/rf_fused_v2.pkl
outputs/fused_pcap_test.csv
```

---

## Dataset Requirements

Place the required dataset files in:

```
dataset/x86/
├── pcap.csv
└── sar.csv
```

---

## Setup

### 1. Copy Existing Model Artifacts

Copy the outputs generated from v0.01 into the v0.02 directory:

```
V0.02/
└── outputs/
    ├── models/
    │   └── rf_fused_v2.pkl
    ├── scalers/
    │   ├── fused_scaler_v2.pkl
    │   └── fused_imputer.pkl
    ├── label_encoder.json
    └── pcap_test.csv
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training Pipeline

Run the scripts in the following order:

```bash
python phase1_load_explore.py
python phase1_preprocess.py
python phase1_eda.py
python phase2_models.py
python phase2_fused_fix.py
python create_fused_test_csv.py
```

These steps will:

1. Load and inspect the dataset
2. Preprocess PCAP and SAR data
3. Perform exploratory data analysis
4. Train baseline models
5. Train the fused model
6. Generate `fused_pcap_test.csv` for inference

---

## Running the Pipeline

### Default Batch Simulation

```bash
python pipeline.py
```

Runs inference on the default sample size (~300 samples).

### Custom Sample Count

```bash
python pipeline.py --samples 1000
```

### Demonstration Mode

```bash
python pipeline.py --demo
```

Runs a demonstration showing all supported response actions.

### Feedback Adaptation Demo

```bash
python pipeline.py --feedback
```

Demonstrates adaptive threshold tuning (Research Question 2.1).

### Single-Sample Inference

```bash
python infer.py
```

Runs prediction on an individual sample.

---

## Project Structure


## Notes

- The repository does not include trained models or datasets due to size constraints.
- `fused_pcap_test.csv` must be generated using `create_fused_test_csv.py` before running the inference pipeline.
- The fused model (`rf_fused_v2.pkl`) provides the highest detection accuracy by combining PCAP and SAR features.

---

## License

