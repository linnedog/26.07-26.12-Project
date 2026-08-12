@echo off
chcp 65001 > nul
echo 계측 프로그램 웹 UI를 띄우는 중입니다. 잠시만 기다려주세요...
streamlit run app.py
cd C:\Users\KTR\Desktop\run
python -m streamlit run app.py