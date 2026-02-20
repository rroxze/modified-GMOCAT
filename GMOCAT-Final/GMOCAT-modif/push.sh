#!/bin/bash

# 0. Sinkronisasi: Ambil perubahan terbaru dari remote (rebase)
git pull --rebase origin main

# 1. Tambahkan semua perubahan (file baru, modifikasi, penghapusan)
git add .

# 2. Commit dengan pesan otomatis berisi tanggal dan waktu
if ! git diff --staged --quiet; then
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    git commit -m "Auto-save: Update semua file pada $TIMESTAMP"

    # 3. Push ke remote repository (branch saat ini)
    git push
    echo "Selesai! Semua perubahan telah di-push."
else
    echo "Tidak ada perubahan untuk di-commit."
fi