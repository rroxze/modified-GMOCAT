#!/bin/bash

echo "Memperbaiki status repository bersarang..."

# 0. Ambil perubahan terbaru dari remote
git pull --rebase origin main

# 1. Hapus direktori .git yang bersarang di dalam GMOCAT-Final
# Ini mengubah repository bersarang menjadi direktori biasa
if [ -d "GMOCAT-Final/.git" ]; then
    rm -rf GMOCAT-Final/.git
    echo "Direktori .git bersarang dihapus."
fi

# 2. Hapus referensi gitlink dari index jika ada
git rm --cached GMOCAT-Final 2>/dev/null

# 3. Tambahkan, Commit, dan Push
git add .
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
git commit -m "Fix nested repo and update content: $TIMESTAMP"
git push origin main

echo "Sukses! Perubahan berhasil di-push."