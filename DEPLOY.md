# Render Deployment Guide

## Quick Deploy to Render

1. **Push to GitHub** (already done ✅)
   ```bash
   git add .
   git commit -m "Add Render deployment files"
   git push origin main
   ```

2. **Connect to Render:**
   - Go to [render.com](https://render.com)
   - Sign up/Login with GitHub
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `emotion-detector-webapp`

3. **Configure Service:**
   - **Name:** `emotion-detector-webapp`
   - **Environment:** `Python 3`
   - **Build Command:** `./build.sh`
   - **Start Command:** `./start.sh`
   - **Plan:** Free (or paid for better performance)

4. **Environment Variables:**
   - `PYTHON_VERSION`: `3.11.6`
   - `MODEL_MODE`: `deepface`
   - `FLASK_ENV`: `production`

5. **Deploy:**
   - Click "Create Web Service"
   - Wait for build (5-10 minutes)
   - Your app will be live at: `https://your-app-name.onrender.com`

## Alternative: Auto-Deploy Method

If you have `render.yaml` in your repo root, Render can auto-configure:

1. Go to Render Dashboard
2. Click "New +" → "Blueprint"
3. Connect repository: `emotion-detector-webapp`
4. Render will use the `render.yaml` configuration

## Important Notes:

- **Free Tier Limitations:**
  - App sleeps after 15 minutes of inactivity
  - Cold start takes 30-60 seconds
  - 512MB RAM limit

- **For Production:**
  - Upgrade to paid plan for always-on service
  - Consider using external database for user data
  - Add Redis for session management

## Troubleshooting:

- **Build fails:** Check Python version compatibility
- **Timeout errors:** Increase timeout in start command
- **Memory issues:** Reduce model complexity or upgrade plan
- **Import errors:** Verify all dependencies in requirements.txt

## Post-Deployment:

Your emotion detection app will have:
- ✅ Real-time emotion detection
- ✅ File upload functionality  
- ✅ User history tracking
- ✅ Professional web interface
- ✅ HTTPS security (automatic on Render)