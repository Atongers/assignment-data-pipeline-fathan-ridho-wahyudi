# Automobile Data Processing Pipeline (ETL)

Pipeline pengolahan data berbasis Python, Pandas, dan scikit-learn untuk membersihkan dan mentransformasi dataset otomotif (`data/raw/automobileEDA_dirty_training.csv`) tanpa mengubah data mentah. Hasil akhir disimpan ke `data/processed/automobileEDA_processed.csv`.

## Struktur Project

```text
data-pipeline-assignment/
├── data/
│   ├── raw/
│   │   └── automobileEDA_dirty_training.csv
│   └── processed/
│       ├── automobile_processed.csv
│       └── automobileEDA_processed.csv
├── src/
│   └── pipeline.py
├── README.md
└── requirements.txt
```

## Cara Menjalankan Pipeline

1. Pastikan virtual environment aktif dan dependency terpasang:

```bash
pip install -r requirements.txt
```

2. Jalankan pipeline ETL dengan satu perintah:

```bash
python src/pipeline.py
```

Pipeline menjalankan tahapan `Extract -> Inspect -> Clean -> Transform -> Load` secara berurutan melalui fungsi modular `extract_data`, `inspect_dataset`, `clean_dataset`, `transform_dataset`, dan `save_dataset`.

---

## 1. Temuan Awal Dataset (Inspection)

Berdasarkan hasil pemeriksaan awal pada `data/raw/automobileEDA_dirty_training.csv`:

- **Ukuran awal dataset**: `205` baris dan `30` kolom.
- **Jumlah baris terduplikasi**: `4` baris duplikat persis (baris identik pada seluruh 30 kolom).
- **Kolom dengan missing values** (total `17` sel kosong pada `7` kolom):
  - `stroke`: `4` missing values
  - `horsepower`: `3` missing values
  - `price`: `3` missing values
  - `transaction_date`: `2` missing values
  - `make`: `2` missing values (1 di antaranya berada pada baris duplikat)
  - `num-of-doors`: `2` missing values
  - `horsepower-binned`: `1` missing value
- **Kolom dengan tipe data belum sesuai**:
  - `transaction_date` masih bertipe `str` dengan 4 format tanggal campuran (`YYYY-MM-DD`, `DD/MM/YYYY`, `MM-DD-YYYY`, dan `DD-Mon-YYYY`, misalnya `2025-01-01`, `02/01/2025`, `01-03-2025`, `04-Jan-2025`).
  - `num-of-doors`, `num-of-cylinders`, dan `horsepower-binned` masih bertipe `str` meskipun merepresentasikan nilai numerik/ordinal bertingkat.
- **Penulisan kategori tidak konsisten**:
  - `make` (`27` nilai unik): campuran huruf kapital dan spasi berlebih di akhir (`'ALFA-ROMERO'`, `'Audi'`, `'BMW'`, `'dodge  '`, `'mercury  '`, `'porsche  '`), serta variasi ejaan `'peugot'`.
  - `body-style` (`7` nilai unik): `'sedan'`, `'SEDAN'`, `'Sedan'`.
  - `drive-wheels` (`5` nilai unik): `'rwd'`, `'RWD'`, `'4wd'`, `'AWD'`, `'fwd'`.
  - `fuel-system` (`10` nilai unik): `'mpfi'`, `'MPFI'`, `'Mpfi'`.
- **Permasalahan lain yang ditemukan**:
  - Skala fitur numerik belum seragam (misalnya `price` berada di rentang `5118.0–45400.0`, `peak-rpm` di `4150.0–6600.0`, `curb-weight` di `1488–4066`, sedangkan `length` dan `width` sudah berada di rentang `0.678–1.0` dan `0.838–1.0`).
  - Pada baris indeks 46 (`jaguar`), nilai `horsepower` adalah `262.0`, sedangkan `horsepower-binned` kosong karena berada di atas batas maksimum kategori `High` lainnya (`156.0–207.0`).

---

## 2. Proses Data Cleaning

Ringkasan perubahan jumlah data:

- **Jumlah data sebelum cleaning**: `205` baris (`17` missing values)
- **Jumlah data sesudah cleaning**: `198` baris (`0` missing values)
- **Duplicate rows dihapus**: `4` baris
- **Baris dihapus karena target kosong**: `3` baris
- **Kolom yang mengalami perubahan (10 kolom)**: `body-style`, `drive-wheels`, `fuel-system`, `horsepower`, `horsepower-binned`, `make`, `num-of-doors`, `price`, `stroke`, `transaction_date`

Detail metode cleaning dan alasannya:

| Kolom / Aspek | Permasalahan | Metode Cleaning | Alasan Pemilihan Metode |
|---|---|---|---|
| Seluruh baris | 4 baris duplikat persis | `drop_duplicates()` | Dilakukan sebelum imputasi agar data duplikat tidak mendistorsi statistik median/modus maupun frekuensi kategori. |
| `make`, `body-style`, `drive-wheels`, `fuel-system`, dsb. | Spasi berlebih & huruf kapital campuran | `str.strip()` + normalisasi huruf kecil (`title()` khusus `horsepower-binned`) | Menyatukan kategori identik (mis. `'dodge  '` dan `'dodge'`, `'SEDAN'` dan `'sedan'`, `'MPFI'` dan `'mpfi'`). |
| `transaction_date` | 4 format tanggal campuran & 2 missing values | Parsing eksplisit per format lalu interpolasi linear berbasis hari pada urutan kronologis | Menghindari kesalahan tukar hari/bulan (`DD/MM/YYYY` vs `MM-DD-YYYY`) dan mengisi 2 tanggal kosong secara akurat (`2025-02-10` dan `2025-05-21`). |
| `price` | 3 baris missing | **Dihapus** (`dropna(subset=["price"])`) | `price` adalah variabel target utama; mengimputasi nilai target berisiko menimbulkan bias label pada pemodelan. |
| `stroke`, `horsepower` | 4 dan 3 missing values | **Diisi** dengan median kolom | Mempertahankan baris fitur spesifikasi lain; median dipilih karena lebih tahan terhadap distribusi miring (*skewed*) dibanding mean. |
| `num-of-doors` | 2 missing values | **Diisi** dengan modus berdasarkan `body-style` | Jumlah pintu sangat bergantung pada tipe bodi (misalnya `sedan` umumnya memiliki `four` pintu). |
| `make` | 1 missing value tersisa setelah deduplikasi | **Diisi** dengan `'unknown'` | Merek tidak dapat disimpulkan secara pasti hanya dari fitur mekanis, sehingga label `'unknown'` mencegah atribusi merek yang keliru sekaligus mempertahankan baris data. |
| `horsepower-binned` | 1 missing value (`horsepower = 262.0`) | **Diturunkan** dari nilai `horsepower` (`<=101`: Low, `<=155`: Medium, `>155`: High) | Mengisi kategori berdasarkan ambang batas aktual dataset sehingga `262.0` hp diklasifikasikan secara tepat sebagai `High`. |

---

## 3. Proses Data Transformation

Setelah cleaning, pipeline mengubah dataset dari `30` kolom menjadi `75` kolom agar nilai asli tetap terdokumentasi dan dapat dibandingkan langsung dengan hasil transformasi:

1. **Min-Max Scaling (`sklearn.preprocessing.MinMaxScaler`)**:
   - **Kolom yang ditransformasi**: `normalized-losses`, `wheel-base`, `height`, `curb-weight`, `engine-size`, `bore`, `stroke`, `compression-ratio`, `horsepower`, `peak-rpm`, `city-mpg`, `highway-mpg`, `price`, dan `city-L/100km`.
   - **Alasan pemilihan**: Kolom-kolom tersebut memiliki skala dan satuan yang jauh berbeda (ribuan hingga puluhan), sementara `length` dan `width` sudah berada di rentang ternormalisasi `[0, 1]` dan `diesel`/`gas` sudah biner (`0`/`1`).
   - **Kolom baru**: `<nama_kolom>_scaled` dalam rentang `[0.0, 1.0]`.
2. **Encoding Kolom Kategorikal**:
   - **Ordinal Encoding** (`num-of-doors` -> `num-of-doors_encoded` `{two: 2, four: 4}`; `num-of-cylinders` -> `num-of-cylinders_encoded` `{two: 2, ..., twelve: 12}`; `horsepower-binned` -> `horsepower-binned_encoded` `{Low: 0, Medium: 1, High: 2}`): dipilih karena kategori memiliki urutan kuantitatif/tingkatan alami.
   - **Frequency Encoding** (`make` -> `make_freq`): dipilih karena `make` bersifat nominal dengan kardinalitas tinggi (23 kategori setelah normalisasi, termasuk `'unknown'`) sehingga mencegah ledakan dimensi kolom one-hot.
   - **One-Hot Encoding** (`aspiration`, `body-style`, `drive-wheels`, `engine-location`, `engine-type`, `fuel-system`): menghasilkan `27` kolom biner `0/1` baru (mis. `body-style_sedan`, `drive-wheels_fwd`) karena kolom-kolom tersebut bersifat nominal tanpa urutan dengan jumlah kategori kecil.
3. **Text & Date Normalization**:
   - Normalisasi spasi dan huruf pada seluruh kolom kategorikal teks serta penyeragaman `transaction_date` ke format ISO `YYYY-MM-DD`.

### Contoh Perbandingan Nilai Sebelum dan Sesudah Transformasi

Nilai berikut diambil langsung dari hasil eksekusi aktual pada `data/processed/automobileEDA_processed.csv`:

| Kolom | Transformasi | Nilai Sebelum | Nilai Sesudah |
|---|---|---:|---:|
| `price` -> `price_scaled` | Min-Max Scaling | `13495.0` | `0.207959` |
| `price` -> `price_scaled` | Min-Max Scaling | `36000.0` | `0.766645` |
| `horsepower` -> `horsepower_scaled` | Min-Max Scaling | `111.0` | `0.294393` |
| `horsepower` -> `horsepower_scaled` | Min-Max Scaling | `262.0` | `1.000000` |
| `curb-weight` -> `curb-weight_scaled` | Min-Max Scaling | `2548` | `0.411171` |
| `curb-weight` -> `curb-weight_scaled` | Min-Max Scaling | `3950` | `0.955004` |
| `num-of-doors` -> `num-of-doors_encoded` | Ordinal Encoding | `two` | `2` |
| `num-of-doors` -> `num-of-doors_encoded` | Ordinal Encoding | `four` | `4` |
| `horsepower-binned` -> `horsepower-binned_encoded` | Ordinal Encoding | `Medium` | `1` |
| `horsepower-binned` -> `horsepower-binned_encoded` | Ordinal Encoding | `High` | `2` |
| `make` -> `make_freq` | Frequency Encoding | `alfa-romero` | `0.015152` |
| `make` -> `make_freq` | Frequency Encoding | `audi` | `0.035354` |
| `body-style` -> `body-style_sedan` | One-Hot Encoding | `convertible` | `0` |
| `body-style` -> `body-style_sedan` | One-Hot Encoding | `sedan` | `1` |
