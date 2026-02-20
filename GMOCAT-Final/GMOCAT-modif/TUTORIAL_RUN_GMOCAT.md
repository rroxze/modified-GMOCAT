# Tutorial Menjalankan GMOCAT-Enhanced

Panduan ini menjelaskan langkah-langkah untuk menjalankan sistem GMOCAT yang telah dimodifikasi, mulai dari persiapan data hingga pengujian adaptif.

## Daftar Isi
1. [Persiapan Data](#1-persiapan-data)
2. [Konstruksi Graf](#2-konstruksi-graf)
3. [Pre-training Model CDM](#3-pre-training-model-cdm)
4. [Pelatihan Agen (Training)](#4-pelatihan-agen-training)
5. [Pengujian Adaptif (Testing)](#5-pengujian-adaptif-testing)

---

## 1. Persiapan Data

**File terkait:** `preprocessing.py`

Langkah ini mengubah data mentah (CSV/TXT) menjadi format JSON yang dibutuhkan oleh model (`train_task_*.json`, `concept_map_*.json`, dll).

**PENTING:** Untuk dataset `dbekt22`, file JSON biasanya sudah disediakan di folder `data/`. Anda dapat **MELEWATI** langkah ini dan langsung ke Langkah 2.

Jika Anda ingin menjalankan skrip, disarankan untuk masuk ke direktori kerja terlebih dahulu:

```bash
cd GMOCAT-Final/GMOCAT-modif
```

Jika Anda ingin menggunakan dataset publik seperti Assistments 2009, pastikan Anda telah mengunduh file CSV mentah (misal `assist09.csv`) dan meletakkannya di folder `raw_data/`.

```bash
# (Opsional) Hanya jika menggunakan data mentah baru
python preprocessing.py --dataset assist2009
```

Pastikan file-file berikut ada di folder `data/` sebelum lanjut ke tahap berikutnya:
- `concept_map_dbekt22.json`
- `train_task_dbekt22.json`
- `question_map_dbekt22.json` (Opsional, tapi disarankan)

---

## 2. Konstruksi Graf

**File terkait:** `construct_graphs.py`

Langkah ini membangun struktur graf pengetahuan (Knowledge Graph) berdasarkan hubungan antar konsep dan soal. Outputnya akan disimpan di folder `graph_data/`.

Jalankan perintah:

```bash
python construct_graphs.py --dataset dbekt22
```

**Output:** Folder `graph_data/dbekt22/` yang berisi file seperti `K_Directed.txt`, `k_from_e.txt`, dll.

---

## 3. Pre-training Model CDM

**File terkait:** `pretrain.py`

Sebelum melatih agen seleksi soal, kita perlu melatih model *Cognitive Diagnosis Model* (CDM) seperti NCD (Neural Cognitive Diagnosis). Model ini berfungsi sebagai simulator siswa dalam lingkungan pelatihan.

Jalankan perintah:

```bash
python pretrain.py --data_name dbekt22 --CDM NCD --training_epoch 20 --device cuda --gpu_no 0
```

*Gunakan `--device cpu` jika tidak menggunakan GPU.*

**Output:** File model `.pt` di dalam folder `models/dbekt22/`.

---

## 4. Pelatihan Agen (Training)

**File terkait:** `run_experiment.py`

Ini adalah skrip utama untuk melatih agen *Reinforcement Learning* (GMOCAT). Skrip ini menggunakan konfigurasi yang didefinisikan di dalam file (class `Args`).

Jalankan perintah:

```bash
python run_experiment.py
```

Jika Anda ingin mengubah parameter (seperti `training_epoch`, `batch_size`, dll), silakan edit bagian `class Args` di dalam file `run_experiment.py` sebelum menjalankannya.

**Output:**
- Log pelatihan di `baseline_log/dbekt22/`.
- Grafik *Concept Coverage* yang muncul setelah pelatihan selesai.

---

## 5. Pengujian Adaptif (Testing)

**File terkait:** `launch_adaptive_test.py`

Setelah agen dilatih, Anda dapat menjalankan simulasi tes adaptif atau tes interaktif.

Jalankan perintah:

```bash
python launch_adaptive_test.py --data_name dbekt22 --agent GCATAgent --student_ids "[0]" --device cuda
```

Skrip ini akan mensimulasikan sesi ujian untuk siswa dengan ID 0 menggunakan agen yang telah dilatih.

---

**Ringkasan Alur Eksekusi:**

`preprocessing.py` (jika perlu) -> `construct_graphs.py` -> `pretrain.py` -> `run_experiment.py` -> `launch_adaptive_test.py`