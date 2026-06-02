import numpy as np
import pandas as pd

TARGET_COLUMN = "recurrence_status"
GROUP_COLUMN = "poiseid"

CLINICAL_FEATURES = [
    "cancerstage_cat",
    "er_cat",
    "pr_cat",
    "her2_cat",
    "chemo",
    "cancertype",
]

CATEGORICAL_FEATURES = CLINICAL_FEATURES.copy()


def load_recurrence_data(expression_threshold=0.1, min_sample_fraction=0.20):
    tpm = pd.read_csv("data/reads/pnas_tpm_96_nodup.txt", sep="\t", header=None)
    patient_info = pd.read_csv("data/pnas_patient_info.csv")

    if tpm.shape[1] - 1 != len(patient_info):
        raise ValueError(
            "TPM sample count does not match patient metadata rows: "
            f"{tpm.shape[1] - 1} TPM samples vs {len(patient_info)} metadata rows"
        )

    expression = tpm.set_index(tpm.columns[0]).T
    expression.index = patient_info["sample_id"].astype(str).to_numpy()
    expression.index.name = "sample_id"
    expression = expression.apply(pd.to_numeric, errors="coerce")

    min_samples = int(np.ceil(len(expression) * min_sample_fraction))
    gene_mask = (expression > expression_threshold).sum(axis=0) >= min_samples
    expression = expression.loc[:, gene_mask]

    metadata = patient_info.copy()
    metadata.index = expression.index

    metadata[TARGET_COLUMN] = metadata["recurStatus"].map({"N": 0, "R": 1})
    if metadata[TARGET_COLUMN].isna().any():
        bad_labels = metadata.loc[metadata[TARGET_COLUMN].isna(), "recurStatus"].unique()
        raise ValueError(f"Unexpected recurrence labels: {bad_labels}")

    missing_clinical = [c for c in CLINICAL_FEATURES if c not in metadata.columns]
    if missing_clinical:
        raise ValueError(f"Missing clinical feature columns: {missing_clinical}")

    clinical = metadata[CLINICAL_FEATURES].copy()
    clinical_encoded = pd.get_dummies(
        clinical,
        columns=CATEGORICAL_FEATURES,
        drop_first=False,
        dummy_na=False,
        dtype=float,
    )

    data = pd.concat(
        [expression, clinical_encoded, metadata[[TARGET_COLUMN, GROUP_COLUMN]]],
        axis=1,
    )

    expression_features = expression.columns.tolist()
    encoded_clinical_features = clinical_encoded.columns.tolist()
    numeric_features = expression_features + encoded_clinical_features

    return {
        "data": data,
        "metadata": metadata,
        "expression_features": expression_features,
        "encoded_clinical_features": encoded_clinical_features,
        "numeric_features": numeric_features,
        "categorical_features": CATEGORICAL_FEATURES,
        "clinical_features": CLINICAL_FEATURES,
    }

def load_validation_rec_data(
    tpm_path="data/reads/validation_exon_tpm",
    meta_path="data/validation_bc_meta.xlsx",
    train_expression_features=None,
):
    import pandas as pd

    target_column = "recurrence_status"
    sample_id_column = "Mapping ID"
    label_column = "Recurrence Staus at the time of collection"

    if train_expression_features is None:
        raise ValueError("train_expression_features must be provided.")

    val_tpm = pd.read_csv(tpm_path, sep="\t")
    print("Raw validation TPM shape:", val_tpm.shape)
    print("First 5 raw validation columns:", val_tpm.columns[:5].tolist())

    val_expression = val_tpm.set_index(val_tpm.columns[0]).T
    val_expression.index = val_expression.index.astype(str)
    val_expression.index.name = sample_id_column
    val_expression.columns = val_expression.columns.astype(str).str.strip()
    val_expression = val_expression.apply(pd.to_numeric, errors="coerce")

    print("Validation expression shape after transpose:", val_expression.shape)
    print("First 10 validation feature columns:", val_expression.columns[:10].tolist())

    if meta_path.lower().endswith((".xlsx", ".xls")):
        val_meta = pd.read_excel(meta_path)
    else:
        val_meta = pd.read_csv(meta_path)

    if sample_id_column not in val_meta.columns:
        raise ValueError(f"Missing sample ID column in metadata: {sample_id_column}")
    if label_column not in val_meta.columns:
        raise ValueError(f"Missing label column in metadata: {label_column}")

    val_meta[sample_id_column] = val_meta[sample_id_column].astype(str).str.strip()

    shared_samples = [sid for sid in val_meta[sample_id_column] if sid in val_expression.index]
    print("Shared samples:", len(shared_samples))

    if not shared_samples:
        raise ValueError("No overlapping sample IDs between validation TPM and metadata.")

    val_meta = val_meta[val_meta[sample_id_column].isin(shared_samples)].copy()
    val_meta = val_meta.drop_duplicates(subset=[sample_id_column])
    val_meta = val_meta.set_index(sample_id_column).loc[shared_samples]
    val_expression = val_expression.loc[shared_samples]

    label_map = {
        "Nonrecurrent": 0,
        "Recurrent": 1,
        "Non-Recurrent": 0,
        "Recurrent ": 1,
    }
    val_meta[target_column] = val_meta[label_column].astype(str).str.strip().map(label_map)

    if val_meta[target_column].isna().any():
        bad_labels = val_meta.loc[val_meta[target_column].isna(), label_column].unique()
        raise ValueError(f"Unexpected validation recurrence labels: {bad_labels}")

    train_expression_features = [str(f).strip() for f in train_expression_features]

    common_expression_features = [
        f for f in train_expression_features if f in val_expression.columns
    ]
    missing_expression_features = [
        f for f in train_expression_features if f not in val_expression.columns
    ]

    print("Training features:", len(train_expression_features))
    print("Validation parsed features:", len(val_expression.columns))
    print("Common features:", len(common_expression_features))
    print("First 10 common features:", common_expression_features[:10])

    if not common_expression_features:
        raise ValueError(
            "No overlapping expression features between training and validation data."
        )

    X = val_expression.reindex(columns=train_expression_features, fill_value=0.0)
    y = val_meta[target_column].astype(int)

    return {
        "X": X,
        "y": y,
        "metadata": val_meta,
        "sample_ids": val_meta.index.to_list(),
        "common_expression_features": common_expression_features,
        "missing_expression_features": missing_expression_features,
    }


if __name__ == "__main__":
    recurrence_data = load_recurrence_data()
    data = recurrence_data["data"]

    X = data.drop(columns=[TARGET_COLUMN, GROUP_COLUMN])
    y = data[TARGET_COLUMN]
    groups = data[GROUP_COLUMN]

    print(f"Samples: {len(data)}")
    print(f"Total columns: {data.shape[1]}")
    print(f"Expression features after filtering: {len(recurrence_data['expression_features'])}")
    print(f"Encoded clinical features: {len(recurrence_data['encoded_clinical_features'])}")
    for col in recurrence_data["encoded_clinical_features"]:
        print(f" - {col}")
    print("Label counts:")
    print(y.value_counts().rename(index={0: "N", 1: "R"}))
    print(f"Unique donors: {groups.nunique()}")