@echo off
cd /d "%~dp0"
py -3.11 -m streamlit run app.py --server.headless true --server.port 8501
