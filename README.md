# Modified GMOCAT

Repository ini berisi modifikasi untuk proyek GMOCAT (Graph-Enhanced Multi-Objective Method for Computerized Adaptive Testing).
Versi ini telah ditingkatkan dengan mekanisme *Coverage-Aware Reward*, *Uncertainty-Based Termination*, dan *Adaptive Diversity Weight*.

## Struktur Folder
- `GMOCAT-Final/`: Berisi kode sumber utama.
- `GMOCAT-Final/GMOCAT-modif/`: Berisi modifikasi spesifik.

## Skrip Utilitas

### 1. `fix_and_push.sh`
Skrip ini digunakan untuk memperbaiki masalah "nested repository" (repository bersarang) dan melakukan push ke GitHub.

**Cara penggunaan:**
```bash
bash fix_and_push.sh
```

### 2. `GMOCAT-Final/GMOCAT-modif/push.sh`
Skrip otomatisasi untuk menyimpan pekerjaan sehari-hari (auto-save).

**Cara penggunaan:**
```bash
bash GMOCAT-Final/GMOCAT-modif/push.sh
```