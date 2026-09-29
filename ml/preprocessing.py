import pandas as pd


def kategori_nilai(nilai):

    if nilai >= 90:
        return "Tinggi"

    elif nilai >= 80:
        return "Sedang"

    else:
        return "Rendah"


def kategori_kehadiran(persen):

    if persen >= 95:
        return "Sangat Baik"

    elif persen >= 90:
        return "Baik"

    else:
        return "Cukup"


def kategori_ekskul(jumlah):

    if jumlah >= 3:
        return "Tinggi"

    elif jumlah >= 1:
        return "Sedang"

    else:
        return "Rendah"


def kategori_perilaku(skor):

    if skor >= 13:
        return "Baik"

    elif skor >= 10:
        return "Cukup"

    else:
        return "Kurang"


def label_prestasi(row):

    nilai_akhir = (
        row["nilai_smt1"] +
        row["nilai_smt2"]
    ) / 2

    if nilai_akhir >= 90:
        return "Tinggi"

    elif nilai_akhir >= 80:
        return "Sedang"

    else:
        return "Rendah"


def hitung_perilaku(row):

    return (
        row["memperhatikan"] +
        row["mengerjakan_tugas"] +
        row["tepat_waktu"] +
        row["aktif_kelas"] +
        row["tertib"]
    )


def preprocessing(df):

    df = df.copy()

    # Konversi data numerik

    df["nilai_smt1"] = pd.to_numeric(
        df["nilai_smt1"],
        errors="coerce"
    )

    df["nilai_smt2"] = pd.to_numeric(
        df["nilai_smt2"],
        errors="coerce"
    )

    df["kehadiran"] = pd.to_numeric(
        df["kehadiran"],
        errors="coerce"
    )

    df["jumlah_ekskul"] = pd.to_numeric(
        df["jumlah_ekskul"],
        errors="coerce"
    )

    df["memperhatikan"] = pd.to_numeric(
        df["memperhatikan"],
        errors="coerce"
    )

    df["mengerjakan_tugas"] = pd.to_numeric(
        df["mengerjakan_tugas"],
        errors="coerce"
    )

    df["tepat_waktu"] = pd.to_numeric(
        df["tepat_waktu"],
        errors="coerce"
    )

    df["aktif_kelas"] = pd.to_numeric(
        df["aktif_kelas"],
        errors="coerce"
    )

    df["tertib"] = pd.to_numeric(
        df["tertib"],
        errors="coerce"
    )


    # Hitung skor perilaku

    df["skor_perilaku"] = df.apply(
        hitung_perilaku,
        axis=1
    )


    # Kategorisasi

    df["kat_smt1"] = df["nilai_smt1"].apply(
        kategori_nilai
    )

    df["kat_smt2"] = df["nilai_smt2"].apply(
        kategori_nilai
    )

    df["kat_kehadiran"] = df["kehadiran"].apply(
        kategori_kehadiran
    )

    df["kat_ekskul"] = df["jumlah_ekskul"].apply(
        kategori_ekskul
    )

    df["kat_perilaku"] = df["skor_perilaku"].apply(
        kategori_perilaku
    )


    # Label prestasi

    df["label_prestasi"] = df.apply(
        label_prestasi,
        axis=1
    )


    return df