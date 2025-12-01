# Streamlit Cloud Deployment Guide

## Quick Deploy to Streamlit Cloud

1. **Create GitHub Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Configuration**
   - The app will automatically use `.streamlit/config.toml`
   - Max upload size: 500 MB
   - No additional settings needed

## Requirements
- GitHub account
- Repository must be public (or Streamlit Cloud Pro for private repos)
- Model file (best_model.pth) must be < 500 MB ✓

## App URL
After deployment, you'll get a URL like:
`https://<username>-<repo>-<branch>-<hash>.streamlit.app`

## Notes
- First run may take 2-3 minutes to install dependencies
- App will auto-sleep after inactivity (free tier)
- Wakes up automatically when accessed
