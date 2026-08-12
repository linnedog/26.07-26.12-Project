시험편 형상 정밀 계측 시스템 

본 프로젝트는 금속재료 인장시험편(KS B 0801 규격)의 형상을 사진만으로 자동 인식하고, 정밀하게 계측하여 규격을 판정하는 AI 기반 웹 어플리케이션입니다. 
로보플로우(Roboflow) 객체 탐지 모델과 OpenCV 컴퓨터 비전 기술을 결합하여, 사용자가 업로드한 이미지에서 시편의 종류(Round, Plate)를 파악하고 평행부의 너비와 길이를 자동으로 계산합니다.

주요 기능 (Features)
AI 자동 규격 판정: KS B 0801 기준에 따라 시편의 너비를 바탕으로 규격(Type 1, 4, 5, 10, 13, 14 등)을 자동 분류합니다.
스마트 각도 보정 계측 (Auto Tilt): 삐뚤어지게 촬영된 시편이라도 내부적으로 바르게 편 뒤, 양끝 물림부가 아닌 중앙 40% 평행부 구간만 정확히 스캔하여 오차를 최소화합니다.
API 크레딧 세이브 (캐싱 시스템): 파라미터나 측정 단위 변경 시 로보플로우 서버와 불필요한 재통신을 막아, 잔여 사용량(Credit)을 아끼고 처리 속도를 비약적으로 높였습니다.
세로 사진 자동 보정: 스마트폰으로 세로로 촬영한 이미지의 원근 및 인식 왜곡을 막기 위해 자동으로 90도 회전시켜 분석합니다.
NG / OK 자동 판정 (개발자 룸): 목표 규격과 허용 오차를 설정하면, 규격 미달/초과 시편을 표에 붉은색(NG)으로 경고 표시합니다.
원클릭 다운로드: 계측 결과 데이터(CSV)와 렌더링된 분석 이미지 전체를 한 번에 ZIP 파일로 다운로드할 수 있습니다.

요구 사항 및 설치 (Prerequisites)
파이썬 환경 구축 필요 (본 프로그램은 Python 3.10 이상 환경이 필요합니다)

프로그램을 실행하기 위해 아래의 파이썬 패키지들이 필요합니다.
```bash
pip install streamlit roboflow opencv-python numpy pandas requests

현재 설치된 패키지들에게 영향을 끼치지 않습니다. 구버전에서도 호환이 가능하도록 제작하였으나, 오류가 발생 가능합니다. 
발생한 오류의 경우 CMD창에 표시되니 CMD창 전체를 복사 후 저장하길 권고드립니다.




Precision Specimen Shape Measurement System

This project is an AI-based web application that automatically recognizes the shape of **metallic materials tensile test specimens (KS B 0801 standard)** from a single photo, precisely measures the dimensions, and determines the specifications. 
By combining a Roboflow object detection model with OpenCV computer vision technology, the system identifies the specimen type (Round or Plate) from user-uploaded images and automatically calculates the width and length of the parallel section.

Features
-AI-Automated Specification Classification: Automatically classifies the specimen type (Type 1, 4, 5, 10, 13, 14, etc.) based on the measured width according to the KS B 0801 standard.
-Smart Angle Correction (Auto Tilt): Even if the specimen is photographed at an angle, the system internally aligns it horizontally and precisely scans only the central 40% parallel section (ignoring the thicker grip ends) to minimize measurement errors.
-API Credit Saver (Caching System): Prevents unnecessary API calls to the Roboflow server when adjusting parameters or measurement units. This saves your remaining API credits and dramatically boosts processing speed.
-Auto-Correction for Portrait Photos: Automatically rotates vertically shot smartphone images by 90 degrees to prevent perspective distortion and improve recognition accuracy.
Automatic NG / OK Judgment (Developer Room):Set a target specification and tolerance limit. Any specimen falling outside the acceptable range will be visually highlighted with a red warning (NG) in the data table.
One-Click Download:Download the measurement result data (CSV) and all rendered analysis images at once in a single ZIP file.

Prerequisites & Installation
A Python environment setup is required. (This program requires Python 3.10 or higher).

To run the program, you need to install the following Python packages:
```bash
pip install streamlit roboflow opencv-python numpy pandas requests

This program does not affect your currently installed packages. While it is designed to be compatible with older versions, unexpected errors may still occur. If an error happens, the details will be displayed in the CMD (Command Prompt) window. We highly recommend copying and saving the entire console output for troubleshooting.