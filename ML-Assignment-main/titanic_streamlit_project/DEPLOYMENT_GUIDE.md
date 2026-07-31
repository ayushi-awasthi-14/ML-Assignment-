# DEPLOYMENT GUIDE FOR BITS WILP ASSIGNMENT

## STEP 1: Push to GitHub

### Option A: Using GitHub Desktop (Easiest)
1. Open GitHub Desktop
2. Click "File" → "Add Local Repository"
3. Select the folder: `c:\workspace\titanic_streamlit_project`
4. Click "Publish repository"
5. Name: `titanic-streamlit-classification`
6. Description: `BITS WILP ML Assignment 2 - Titanic Classification with Streamlit`
7. Click "Publish Repository"

### Option B: Using Git Command Line
```bash
cd c:\workspace\titanic_streamlit_project

# Add gitignore to clean up repo
git add .gitignore
git commit -m "Add .gitignore to exclude venv and cache files"

# Remove venv from git tracking (if already committed)
git rm -r --cached .venv
git commit -m "Remove venv from git tracking"

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/titanic-streamlit-classification.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

### GitHub Repository URL
After pushing, your repository will be at:
```
https://github.com/YOUR_USERNAME/titanic-streamlit-classification
```

---

## STEP 2: Deploy to Streamlit Community Cloud

### Prerequisites
- GitHub account (already have your repo there)
- Streamlit account (sign up at https://streamlit.io/cloud)

### Deployment Steps

1. **Go to Streamlit Cloud:** https://streamlit.io/cloud

2. **Sign In or Sign Up**
   - Click "Sign in with GitHub"
   - Authorize Streamlit to access your GitHub account

3. **Deploy New App**
   - Click "New app" button
   - Select your repository: `titanic-streamlit-classification`
   - Branch: `main`
   - Main file: `app.py`
   - Click "Deploy"

4. **Wait for Deployment**
   - Streamlit will install dependencies
   - Build and deploy the app
   - You'll get a public URL (typically: `https://[your-username]-titanic-streamlit-classification.streamlit.app`)

### Live Streamlit App URL
Once deployed, you can access it at:
```
https://[your-username]-titanic-streamlit-classification.streamlit.app
```

---

## STEP 3: Verify Everything Works

### Local Testing (Before Deployment)
```bash
# Train models (if not already done)
python train_models.py

# Run Streamlit app
streamlit run app.py
```

### Cloud Testing
1. Click the Streamlit app link
2. Upload test data (or use default)
3. Select a model from dropdown
4. Verify predictions, metrics, and confusion matrix display

---

## STEP 4: Prepare Final Submission PDF

Your submission PDF should contain (in order):

1. **GitHub Repository Link**
   ```
   https://github.com/YOUR_USERNAME/titanic-streamlit-classification
   ```

2. **Live Streamlit App Link**
   ```
   https://[your-username]-titanic-streamlit-classification.streamlit.app
   ```

3. **Screenshot from BITS Virtual Lab**
   - Run the training script on BITS Lab
   - Take a screenshot of the terminal output showing model metrics
   - Include in PDF

4. **Full README.md Content**
   - Copy all content from README.md file
   - Include in PDF

---

## STEP 5: Files Already Ready for Submission

✅ `README.md` - Complete with all required sections
✅ `requirements.txt` - All dependencies listed
✅ `app.py` - Streamlit application with all features
✅ `train_models.py` - Model training script
✅ `test_data.csv` - Test dataset for app
✅ `model_metadata.json` - Feature metadata
✅ `model_comparison.csv` - Metrics table export
✅ `models/` - All trained models (5 total)

---

## Troubleshooting

### App won't deploy on Streamlit Cloud
- Check `requirements.txt` has all dependencies
- Ensure `app.py` and all model files are in repo
- Models folder should have all 5 `.pkl` files
- Check GitHub status - ensure files are pushed

### Models not loading
- Verify all `.pkl` files are in `models/` folder
- Check `model_metadata.json` exists in root
- Check file permissions

### Missing dependencies error
- Run `pip install -r requirements.txt` locally
- Add any missing packages to `requirements.txt`
- Commit and push to GitHub

---

## Quick Command Reference

```bash
# Clone repo locally
git clone https://github.com/YOUR_USERNAME/titanic-streamlit-classification.git

# Navigate to project
cd titanic-streamlit-classification

# Install dependencies
pip install -r requirements.txt

# Train models
python train_models.py

# Run app locally
streamlit run app.py

# Push changes to GitHub
git add .
git commit -m "Your message"
git push origin main
```

---

## Important Dates & Links

- **Submission Deadline:** 18 August 2026, 23:59 PM
- **Streamlit Cloud:** https://streamlit.io/cloud
- **GitHub:** https://github.com
- **Support Email:** csislabsupport@wilp.bits-pilani.ac.in

---

**Good luck with your submission! 🚀**
