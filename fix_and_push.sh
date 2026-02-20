#!/bin/bash

echo "Fixing nested repository state..."

# 0. Ambil perubahan terbaru dari remote
git pull --rebase origin main

# 1. Remove the nested .git directory inside GMOCAT-Final
# This converts the nested repository into a regular directory
if [ -d "GMOCAT-Final/.git" ]; then
    rm -rf GMOCAT-Final/.git
    echo "Removed nested .git directory."
fi

# 2. Remove the gitlink reference from the index if it exists
git rm --cached GMOCAT-Final 2>/dev/null

# 3. Add, Commit, and Push
git add .
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
git commit -m "Fix nested repo and update content: $TIMESTAMP"
git push origin main

echo "Success! Changes pushed."