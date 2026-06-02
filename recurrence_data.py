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
    print("Label counts:")
    print(y.value_counts().rename(index={0: "N", 1: "R"}))
    print(f"Unique donors: {groups.nunique()}")