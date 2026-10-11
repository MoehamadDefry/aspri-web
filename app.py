from flask import Flask, render_template, request, send_file
from io import BytesIO

import pandas as pd

import data_store

from ml.preprocessing import preprocessing
from ml.analysis import run_full_analysis, FEATURES, LABELS

app = Flask(__name__)


# =========================
# HOME
# =========================

@app.route("/")
def home():

    return render_template("home.html")


# =========================
# UPLOAD DATA
# =========================
@app.route("/upload", methods=["GET", "POST"])
def upload_data():

    if request.method == "GET":
        return render_template("upload.html")

    file = request.files.get("dataset")

    if not file or file.filename == "":
        return render_template(
            "upload.html",
            error="Silakan pilih file dataset terlebih dahulu."
        )

    try:

        df = pd.read_excel(file)

    except Exception:

        return render_template(
            "upload.html",
            error="File tidak dapat dibaca. Pastikan file merupakan Excel yang valid."
        )

    # Simpan DataFrame hanya di memory
    data_store.current_data = df

    # Reset hasil analisis sebelumnya
    data_store.current_results = {}
    data_store.prediction_data = None

    return render_template(
        "upload.html",
        success=True,
        data=df.to_dict("records"),
        columns=df.columns.tolist(),
        total_data=len(df)
    )


# =========================
# PROSES & ANALISIS
# =========================

@app.route("/process")
def process_analysis():
    if data_store.current_data is None:
        return render_template(
            "process.html",
            error="Belum ada dataset. Silakan upload dataset terlebih dahulu."
        )

    try:
        result = run_full_analysis(data_store.current_data)
    except Exception as e:
        return render_template(
            "process.html",
            error=f"Proses analisis gagal: {e}"
        )

    data_store.processed_data = result["df_prep"]
    data_store.current_results = result

    return render_template(
        "process.html",

        # =========================
        # DATA AWAL
        # =========================
        original_data=(
            data_store.current_data
            .head(10)
            .where(data_store.current_data.head(10).notna(), None)
            .to_dict("records")
        ),

        original_columns=list(
            data_store.current_data.columns
        ),

        # =========================
        # PREPROCESSING
        # =========================
        total_data=result["total_data"],
        feature_count=result["feature_count"],
        preprocessing_preview=result["preprocessing_preview"],
        preprocessing_columns=result["preprocessing_columns"],

        # =========================
        # DISTRIBUSI LABEL
        # =========================
        label_counts=result["label_counts"],
        mayoritas_label=result["mayoritas_label"],
        mayoritas_jumlah=result["mayoritas_jumlah"],
        minoritas_label=result["minoritas_label"],
        minoritas_jumlah=result["minoritas_jumlah"],

        # =========================
        # ANALISIS FAKTOR
        # =========================
        features=FEATURES,
        factor_analysis=result["factor_analysis"],

        # =========================
        # ENCODING
        # =========================
        mappings=result["mappings"],
        encoded_preview=result["encoded_preview"],
        encoded_columns=result["encoded_columns"],

        # =========================
        # TRAIN TEST SPLIT
        # =========================
        train_size=result["train_size"],
        test_size=result["test_size"],
        split_distribution=result["split_distribution"],

        # =========================
        # FREKUENSI
        # =========================
        frequency=result["frequency"],

        # =========================
        # EVALUASI
        # =========================
        accuracy=result["accuracy"],
        confusion_matrix=result["confusion_matrix"],
        labels=LABELS,
        classification_report=result["classification_report"],
        weighted_precision=result["weighted_precision"],
        weighted_recall=result["weighted_recall"],
        correct=result["correct"],
        test_total=result["test_total"],
    )
# =========================
# HASIL PREDIKSI
# =========================

# =========================
# HASIL PREDIKSI
# =========================

@app.route("/prediction")
def prediction():

    # ---------------------------------
    # CEK DATA
    # ---------------------------------
    if data_store.current_data is None:
        return render_template(
            "prediction.html",
            prediction_ready=False,
            error="Belum ada dataset. Silakan upload dataset terlebih dahulu."
        )

    # ---------------------------------
    # CEK MODEL
    # ---------------------------------
    result = data_store.current_results

    if not result or "model" not in result or "encoder" not in result:
        return render_template(
            "prediction.html",
            prediction_ready=False,
            error="Model belum tersedia. Silakan jalankan Proses & Analisis terlebih dahulu."
        )

    try:

        # ---------------------------------
        # AMBIL DATA
        # ---------------------------------
        df = data_store.current_data.copy()

        model = result["model"]
        encoder = result["encoder"]

        # ---------------------------------
        # PREPROCESSING
        # ---------------------------------
        df_prep = preprocessing(df)

        # ---------------------------------
        # ENCODING
        # ---------------------------------
        X = df_prep[FEATURES]

        X_encoded = encoder.transform(X)

        # ---------------------------------
        # PREDIKSI
        # ---------------------------------
        predictions = model.predict(X_encoded)

        df_prep["hasil_prediksi"] = predictions

        # ---------------------------------
        # REKOMENDASI
        # ---------------------------------
        def buat_rekomendasi(row):

            rekomendasi = []

            # Kehadiran
            if row["kehadiran"] < 90:
                rekomendasi.append(
                    "Perlu meningkatkan kehadiran siswa."
                )
            elif row["kehadiran"] < 95:
                rekomendasi.append(
                    "Pertahankan dan tingkatkan kehadiran siswa."
                )

            # Akademik
            nilai_rata = (
                float(row["nilai_smt1"]) +
                float(row["nilai_smt2"])
            ) / 2

            if nilai_rata < 80:
                rekomendasi.append(
                    "Perlu pendampingan untuk meningkatkan hasil akademik."
                )
            elif nilai_rata < 90:
                rekomendasi.append(
                    "Perlu mempertahankan dan meningkatkan hasil akademik."
                )

            # Ekstrakurikuler
            if row["jumlah_ekskul"] == 0:
                rekomendasi.append(
                    "Disarankan mengikuti kegiatan ekstrakurikuler."
                )

            # Perilaku
            skor_perilaku = row["skor_perilaku"]

            if skor_perilaku < 10:
                rekomendasi.append(
                    "Perlu pembinaan perilaku dan kedisiplinan."
                )
            elif skor_perilaku < 13:
                rekomendasi.append(
                    "Perlu mempertahankan dan meningkatkan perilaku positif."
                )

            # Jika tidak ada rekomendasi khusus
            if not rekomendasi:
                if row["hasil_prediksi"] == "Tinggi":
                    return (
                        "Pertahankan prestasi akademik, kehadiran, "
                        "dan perilaku positif siswa."
                    )

                if row["hasil_prediksi"] == "Sedang":
                    return (
                        "Pertahankan pencapaian saat ini dan tingkatkan "
                        "aspek yang masih perlu diperbaiki."
                    )

                return (
                    "Perlu pemantauan dan pendampingan lebih lanjut."
                )

            return " ".join(rekomendasi)

        df_prep["rekomendasi"] = df_prep.apply(
            buat_rekomendasi,
            axis=1
        )

        # ---------------------------------
        # DISTRIBUSI PREDIKSI
        # ---------------------------------
        prediction_counts = (
            df_prep["hasil_prediksi"]
            .value_counts()
            .reindex(LABELS, fill_value=0)
        )

        prediction_counts = {
            label: int(prediction_counts.get(label, 0))
            for label in LABELS
        }

        total_prediction = len(df_prep)

        # ---------------------------------
        # DATA SISWA BERISIKO
        # ---------------------------------
        risk_df = df_prep[
            df_prep["hasil_prediksi"] == "Rendah"
        ].copy()

        risk_columns = [
            "nama",
            "kelas",
            "nilai_smt1",
            "nilai_smt2",
            "kehadiran",
            "hasil_prediksi",
            "rekomendasi"
        ]

        risk_columns = [
            c for c in risk_columns
            if c in risk_df.columns
        ]

        risk_students = (
            risk_df[risk_columns]
            .where(risk_df[risk_columns].notna(), None)
            .to_dict("records")
        )

        # ---------------------------------
        # HASIL PREDIKSI SEMUA SISWA
        # ---------------------------------
        prediction_columns = [
            "nama",
            "kelas",
            "nilai_smt1",
            "nilai_smt2",
            "kehadiran",
            "jumlah_ekskul",
            "skor_perilaku",
            "hasil_prediksi",
            "rekomendasi"
        ]

        prediction_columns = [
            c for c in prediction_columns
            if c in df_prep.columns
        ]

        predictions_data = (
            df_prep[prediction_columns]
            .where(df_prep[prediction_columns].notna(), None)
            .to_dict("records")
        )
        # ---------------------------------
        # PAGINATION
        # ---------------------------------

        # Jumlah data per halaman
        prediction_per_page = 20
        risk_per_page = 10

        # Halaman yang diminta dari URL
        try:
            prediction_page = int(request.args.get("page", 1))
        except (TypeError, ValueError):
            prediction_page = 1

        try:
            risk_page = int(request.args.get("risk_page", 1))
        except (TypeError, ValueError):
            risk_page = 1

        # Minimal halaman adalah 1
        prediction_page = max(prediction_page, 1)
        risk_page = max(risk_page, 1)

        # Total halaman
        total_prediction_pages = (
            (len(predictions_data) + prediction_per_page - 1)
            // prediction_per_page
        )

        total_risk_pages = (
            (len(risk_students) + risk_per_page - 1)
            // risk_per_page
        )

        # Jika halaman melebihi jumlah halaman,
        # arahkan ke halaman terakhir
        if total_prediction_pages > 0:
            prediction_page = min(
                prediction_page,
                total_prediction_pages
            )

        if total_risk_pages > 0:
            risk_page = min(
                risk_page,
                total_risk_pages
            )

        # Tentukan posisi data
        prediction_start = (
            prediction_page - 1
        ) * prediction_per_page

        prediction_end = (
            prediction_start + prediction_per_page
        )

        risk_start = (
            risk_page - 1
        ) * risk_per_page

        risk_end = (
            risk_start + risk_per_page
        )

        # Data yang benar-benar dikirim ke HTML
        paginated_predictions = predictions_data[
            prediction_start:prediction_end
        ]

        paginated_risk_students = risk_students[
            risk_start:risk_end
        ]

        # ---------------------------------
        # SIMPAN HASIL
        # ---------------------------------
        data_store.prediction_data = df_prep

        # ---------------------------------
        # RENDER
        # ---------------------------------
        return render_template(
        "prediction.html",

        prediction_ready=True,

        # =========================
        # RINGKASAN
        # =========================
        total_prediction=total_prediction,
        prediction_counts=prediction_counts,

        # =========================
        # SISWA PERLU PERHATIAN
        # =========================
        risk_students=paginated_risk_students,
        risk_columns=risk_columns,

        risk_page=risk_page,
        risk_per_page=risk_per_page,
        total_risk_pages=total_risk_pages,
        total_risk_students=len(risk_students),

        # =========================
        # SEMUA HASIL PREDIKSI
        # =========================
        predictions=paginated_predictions,
        prediction_columns=prediction_columns,

        prediction_page=prediction_page,
        prediction_per_page=prediction_per_page,
        total_prediction_pages=total_prediction_pages,

        # =========================
        # DOWNLOAD
        # =========================
        download_url="/prediction/download"
    )
    except Exception as e:

        return render_template(
            "prediction.html",
            prediction_ready=False,
            error=f"Proses prediksi gagal: {e}"
        )

@app.route("/prediction/download")
def prediction_download():

    if not hasattr(data_store, "prediction_data"):
        return "Hasil prediksi belum tersedia.", 404

    if data_store.prediction_data is None:
        return "Hasil prediksi belum tersedia.", 404

    try:

        df = data_store.prediction_data.copy()

        output = BytesIO()

        with pd.ExcelWriter(
            output,
            engine="openpyxl"
        ) as writer:

            df.to_excel(
                writer,
                index=False,
                sheet_name="Hasil Prediksi"
            )

        output.seek(0)

        return send_file(
            output,
            as_attachment=True,
            download_name="hasil_prediksi_aspri.xlsx",
            mimetype=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )
        )

    except Exception as e:

        return f"Gagal membuat file Excel: {e}", 500


# =========================
# DOWNLOAD TEMPLATE DATASET
# =========================
@app.route("/download-template")
def download_template():

    # Kolom yang dibutuhkan sistem
    columns = [
        "nama",
        "kelas",
        "nilai_smt1",
        "nilai_smt2",
        "kehadiran",
        "jumlah_ekskul",
        "skor_perilaku"
    ]

    # Buat DataFrame kosong
    df = pd.DataFrame(columns=columns)

    # Buat file Excel di memory
    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Dataset"
        )

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="template_dataset_aspri.xlsx",
        mimetype=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )


# =========================
# RUN
# =========================

if __name__ == "__main__":

    app.run(debug=True)