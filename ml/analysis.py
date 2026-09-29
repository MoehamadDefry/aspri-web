import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

FEATURES = [
    "kat_smt1",
    "kat_smt2",
    "kat_kehadiran",
    "kat_ekskul",
    "kat_perilaku",
]
LABELS = ["Tinggi", "Sedang", "Rendah"]


def kategori_nilai(nilai):
    if nilai >= 90:
        return "Tinggi"
    elif nilai >= 80:
        return "Sedang"
    return "Rendah"


def kategori_kehadiran(persen):
    if persen >= 95:
        return "Sangat Baik"
    elif persen >= 90:
        return "Baik"
    return "Cukup"


def kategori_ekskul(jumlah):
    if jumlah >= 3:
        return "Tinggi"
    elif jumlah >= 1:
        return "Sedang"
    return "Rendah"


def kategori_perilaku(skor):
    if skor >= 13:
        return "Baik"
    elif skor >= 10:
        return "Cukup"
    return "Kurang"


def hitung_perilaku(row):
    return (
        row["memperhatikan"]
        + row["mengerjakan_tugas"]
        + row["tepat_waktu"]
        + row["aktif_kelas"]
        + row["tertib"]
    )


def label_prestasi(row):
    nilai_akhir = (row["nilai_smt1"] + row["nilai_smt2"]) / 2
    return kategori_nilai(nilai_akhir)


def preprocessing(df):
    df = df.copy()

    numeric_columns = [
        "nilai_smt1", "nilai_smt2", "kehadiran", "jumlah_ekskul",
        "memperhatikan", "mengerjakan_tugas", "tepat_waktu",
        "aktif_kelas", "tertib",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["skor_perilaku"] = df.apply(hitung_perilaku, axis=1)
    df["kat_smt1"] = df["nilai_smt1"].apply(kategori_nilai)
    df["kat_smt2"] = df["nilai_smt2"].apply(kategori_nilai)
    df["kat_kehadiran"] = df["kehadiran"].apply(kategori_kehadiran)
    df["kat_ekskul"] = df["jumlah_ekskul"].apply(kategori_ekskul)
    df["kat_perilaku"] = df["skor_perilaku"].apply(kategori_perilaku)
    df["label_prestasi"] = df.apply(label_prestasi, axis=1)

    return df


def _records(df):
    clean = df.copy()
    clean = clean.where(pd.notna(clean), None)
    return clean.to_dict(orient="records")


def run_full_analysis(df):
    """Run the complete legacy ASPRI analysis flow in one call."""
    df_prep = preprocessing(df)

    # Label distribution
    label_counts = (
        df_prep["label_prestasi"]
        .value_counts()
        .reindex(LABELS, fill_value=0)
    )
    mayoritas_label = label_counts.idxmax()
    minoritas_label = label_counts.idxmin()

    mayoritas_jumlah = int(label_counts.max())
    minoritas_jumlah = int(label_counts.min())

    # Factor analysis on all data, matching the legacy cross-tab presentation.
    factor_analysis = {}
    for feature in FEATURES:
        cross = pd.crosstab(
            df_prep[feature],
            df_prep["label_prestasi"],
            normalize="index",
        ).reindex(columns=LABELS, fill_value=0)
        counts = pd.crosstab(
            df_prep[feature],
            df_prep["label_prestasi"],
        ).reindex(columns=LABELS, fill_value=0)
        dominant = cross.idxmax(axis=1).to_dict()
        factor_analysis[feature] = {
            "rows": [
                {"category": str(idx), **{k: float(v) for k, v in row.items()}}
                for idx, row in cross.iterrows()
            ],
            "counts": [
                {"category": str(idx), **{k: int(v) for k, v in row.items()}}
                for idx, row in counts.iterrows()
            ],
            "dominant": {str(k): str(v) for k, v in dominant.items()},
        }

    # Encoding
    X = df_prep[FEATURES]
    y = df_prep["label_prestasi"]

    encoder = OrdinalEncoder()
    X_enc = encoder.fit_transform(X)
    df_enc = pd.DataFrame(X_enc, columns=FEATURES, index=df_prep.index)

    mappings = {}
    for i, feature in enumerate(FEATURES):
        mappings[feature] = [
            {"category": str(category), "value": int(index)}
            for index, category in enumerate(encoder.categories_[i])
        ]

    # Split, exactly as legacy app: 70:30, random_state=42.
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.3, random_state=42
    )

    # Keep original categorical X for frequency tables.
    X_train_asli, X_test_asli, _, _ = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    train_dist = y_train.value_counts().reindex(LABELS, fill_value=0)
    test_dist = y_test.value_counts().reindex(LABELS, fill_value=0)
    split_dist = pd.DataFrame({"Training": train_dist, "Testing": test_dist})
    split_dist["Total"] = split_dist["Training"] + split_dist["Testing"]

    # Frequency tables used in the old app.
    frequency = {}
    for feature in FEATURES:
        freq = pd.crosstab(X_train_asli[feature], y_train)
        freq = freq.reindex(columns=LABELS, fill_value=0)
        freq["Total"] = freq.sum(axis=1)
        frequency[feature] = [
            {"category": str(idx), **{k: int(v) for k, v in row.items()}}
            for idx, row in freq.iterrows()
        ]

    # Train CategoricalNB with its default parameters, matching the legacy app.
    model = CategoricalNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    report = classification_report(
        y_test, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()

    report_rows = []
    for label in LABELS:
        report_rows.append({
            "label": label,
            "precision": float(report_df.loc[label, "precision"]),
            "recall": float(report_df.loc[label, "recall"]),
            "f1_score": float(report_df.loc[label, "f1-score"]),
            "support": int(report_df.loc[label, "support"]),
        })

    weighted = report_df.loc["weighted avg"]

    # Training/testing preview.
    encoded_preview = df_enc.head(10).reset_index(drop=True)
    preprocessing_preview = df_prep.head(10)

    # Full data for prediction page.
    prediction_df = df_prep.copy()
    prediction_df["hasil_prediksi"] = model.predict(encoder.transform(prediction_df[FEATURES]))

    return {
        "df_prep": df_prep,
        "preprocessing_preview": _records(preprocessing_preview),
        "preprocessing_columns": list(preprocessing_preview.columns),
        "total_data": int(len(df_prep)),
        "feature_count": int(len(df_prep.columns)),
        "label_counts": {k: int(v) for k, v in label_counts.items()},
        "factor_analysis": factor_analysis,
        "encoded_preview": _records(encoded_preview),
        "encoded_columns": FEATURES,
        "mappings": mappings,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "split_distribution": [
            {"label": str(idx), "training": int(row["Training"]),
             "testing": int(row["Testing"]), "total": int(row["Total"])}
            for idx, row in split_dist.iterrows()
        ],
        "frequency": frequency,
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_rows,
        "weighted_precision": float(weighted["precision"]),
        "weighted_recall": float(weighted["recall"]),
        "correct": int(cm.trace()),
        "test_total": int(cm.sum()),
        "prediction_df": prediction_df,
        "model": model,
        "encoder": encoder,
        "mayoritas_label": str(mayoritas_label),
        "mayoritas_jumlah": mayoritas_jumlah,
        "minoritas_label": str(minoritas_label),
        "minoritas_jumlah": minoritas_jumlah,
    }
