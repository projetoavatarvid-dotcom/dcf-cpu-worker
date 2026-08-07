# FIRST PUSH — Windows PowerShell

## 1. Create an empty GitHub repository

Recommended repository name:

dcf-cpu-worker

Do not add README, .gitignore or license in the GitHub web form because this package already contains them.

## 2. Extract this package

Open PowerShell in the extracted `dcf-cpu-worker` directory.

## 3. Configure Git identity if necessary

git config --global user.name "YOUR NAME"
git config --global user.email "YOUR GITHUB EMAIL"

## 4. Initialize

git init
git add .
git commit -m "DCF CPU Worker v1.0"

## 5. Connect the GitHub repository

Replace YOUR_GITHUB_USER:

git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USER/dcf-cpu-worker.git
git push -u origin main

## 6. Publish v1.0.0

git tag v1.0.0
git push origin v1.0.0

The tag triggers `.github/workflows/publish.yml`.

## 7. After GitHub Actions succeeds

Expected container image:

ghcr.io/YOUR_GITHUB_USER/dcf-cpu-worker:v1.0.0

Then enable it in DCF:

Enable-DCFCpuWorkerImage `
    -ImageName "ghcr.io/YOUR_GITHUB_USER/dcf-cpu-worker:v1.0.0"
