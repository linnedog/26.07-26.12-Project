# pip install streamlit roboflow opencv-python numpy pandas requests
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import tempfile
import zipfile
import io
import requests
import re
from roboflow import Roboflow

# ==========================================
# 1. API 기본 설정
# ==========================================
ROBOFLOW_API_KEY = "k"
WORKSPACE_ID     = "dogs-workspace-8moat"
PROJECT_ID       = "test2-6zyah"
VERSION_NUM      = 1

TARGET_NAMES = {"round", "plate"}
RULER_NAME   = "Ruler"

FONT_SIZE        = None
BASE_FONT_FACTOR = 0.025
BASE_LINE_W      = 2

# ==========================================
# 2. 로보플로우 API 남은 잔여량 조회
# ==========================================
def get_roboflow_usage(api_key):
    try:
        url = f"https://api.roboflow.com/?api_key={api_key}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            usage = data.get("user", {}).get("usage", {})
            return usage.get("image_count", "N/A"), usage.get("image_limit", "N/A")
    except Exception:
        pass
    return "확인 불가", "확인 불가"

# ==========================================
# 3. 정밀 계측 함수 (기본 & 스마트 보정)
# ==========================================
def measure_parallel_gap(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 90)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    edges = cv2.dilate(edges, k, iterations=1)

    h, w = edges.shape
    # 💡 [핵심] 시편의 중앙 40% (평행부)만 스캔하도록 범위 제한!
    x_start = int(w * 0.30)
    x_end   = int(w * 0.70)
    tops, bottoms = [], []

    for x in range(x_start, x_end):
        col = edges[:, x]
        ys = np.where(col > 0)[0]
        if ys.size < 2: continue
        tops.append(ys[0])
        bottoms.append(ys[-1])

    if not tops or not bottoms: return None

    top_y = int(np.median(tops))
    bottom_y = int(np.median(bottoms))
    gap_px = bottom_y - top_y

    if gap_px <= 0: return None
    return top_y, bottom_y, gap_px

def measure_smart_tilt(roi_bgr):
    """삐뚤어진 시편을 바르게 세운 뒤 계측하는 동적 파이프라인 함수"""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 90)
    
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: 
        return measure_parallel_gap(roi_bgr), False
        
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    w, h = rect[1]
    angle = rect[2]
    
    # 가로로 길게 눕히기 위한 각도 보정
    if w < h: angle += 90
        
    # 각도가 3도 이내면 굳이 연산하지 않고 기본 모드로 즉시 계측
    if abs(angle) % 180 < 3 or abs(angle) % 180 > 177:
        return measure_parallel_gap(roi_bgr), False
        
    # 3도 이상 비뚤어진 경우, 이미지를 수평으로 바르게 회전 (Un-rotate)
    roi_h, roi_w = roi_bgr.shape[:2]
    center = (roi_w // 2, roi_h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(roi_bgr, M, (roi_w, roi_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # 펴진 이미지의 중앙 40% 평행부만 계측
    return measure_parallel_gap(rotated), True

def classify_specimen(shape_name: str, gap_mm: float) -> str:
    s = shape_name.lower()
    if s == "round":
        if 13.2 <= gap_mm <= 15.5: return "Type 4 / 14A"
        elif 11.5 <= gap_mm <= 13.1: return "Type 10"
        elif 9.0 <= gap_mm <= 11.4: return "Type 14B"
        elif 7.0 <= gap_mm <= 8.9: return "Type 14C"
    elif s == "plate":
        if 37.0 <= gap_mm <= 43.0: return "Type 1 / 1A"
        elif 22.5 <= gap_mm <= 27.5: return "Type 5 / 1B"
        elif 18.0 <= gap_mm <= 22.4: return "Type 13A"
        elif 11.0 <= gap_mm <= 14.5: return "Type 13B"
    return "미분류"

# ==========================================
# 4. 어노테이션 및 렌더링
# ==========================================
def annotate_one(predictions, img, known_cm, display_unit="cm", use_smart_tilt=True, ng_ok_settings=None):
    out = img.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    base_fs = FONT_SIZE or max(14, int(min(h, w) * BASE_FONT_FACTOR))
    font_scale_small, font_scale_mid, top_font_scale = base_fs / 40.0, base_fs / 30.0, base_fs / 25.0
    lw, red_lw = max(2, int(BASE_LINE_W * min(h, w) / 800)), max(2, int(BASE_LINE_W * min(h, w) / 800))

    if not predictions: return out, []

    boxes = []
    for pred in predictions:
        cx, cy, bw, bh = pred['x'], pred['y'], pred['width'], pred['height']
        boxes.append({
            "name": pred['class'], "conf": pred['confidence'],
            "x1": int(cx - (bw / 2)), "y1": int(cy - (bh / 2)),
            "x2": int(cx + (bw / 2)), "y2": int(cy + (bh / 2))
        })

    cm_per_px = None
    ruler_bbox = None
    for b in boxes:
        if b["name"].lower() == RULER_NAME.lower():
            ruler_len_px = max(abs(b["x2"] - b["x1"]), abs(b["y2"] - b["y1"]))
            if ruler_len_px > 0: cm_per_px = known_cm / ruler_len_px
            ruler_bbox = (b["x1"], b["y1"], b["x2"], b["y2"])
            break

    unit_scale = 10 if display_unit == "mm" else 1
    measured_data = []

    for b in boxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        name, conf = b["name"], b["conf"]
        name_lower = name.lower()

        color = (0, 255, 0) if name_lower in {t.lower() for t in TARGET_NAMES} else (255, 0, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, lw)

        label_txt = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label_txt, font, font_scale_small, lw)
        cv2.rectangle(out, (x1, max(0, y1 - th - 4)), (x1 + tw, y1), (255, 255, 255), -1)
        text_color = (255, 0, 0) if name_lower == RULER_NAME.lower() or name_lower in {t.lower() for t in TARGET_NAMES} else (0, 0, 0)
        cv2.putText(out, label_txt, (x1, y1 - 2), font, font_scale_small, text_color, max(1, lw // 2), cv2.LINE_AA)

        x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
        roi = img[y1c:y2c, x1c:x2c].copy()
        
        if roi.size == 0 or name_lower not in {t.lower() for t in TARGET_NAMES}:
            continue

        gap_px = None
        is_tilted = False
        
        if use_smart_tilt:
            res, is_tilted = measure_smart_tilt(roi)
            if res: top_y, bottom_y, gap_px = res
        else:
            res = measure_parallel_gap(roi)
            if res: top_y, bottom_y, gap_px = res
            
        if gap_px is None: continue
        
        if is_tilted:
            cv2.rectangle(out, (x1c, y1c), (x2c, y2c), (255, 0, 255), red_lw)
        else:
            cv2.line(out, (x1c, y1c + top_y), (x2c, y1c + top_y), (0,0,255), red_lw)
            cv2.line(out, (x1c, y1c + bottom_y), (x2c, y1c + bottom_y), (0,0,255), red_lw)

        length_px = max(abs(x2 - x1), abs(y2 - y1))
        
        if cm_per_px is not None:
            gap_cm = max(0, (gap_px * cm_per_px) - 0.1)  
            length_cm = length_px * cm_per_px
            gap_val, length_val = gap_cm * unit_scale, length_cm * unit_scale
            cm_txt = f"W:{gap_val:.2f}{display_unit} / L:{length_val:.2f}{display_unit}"
        else:
            gap_cm, length_cm = None, None
            cm_txt = f"W:{gap_px}px"

        center_x, center_y = (x1c + x2c) // 2, y1c + (y2c - y1c) // 2
        (twm, thm), _ = cv2.getTextSize(cm_txt, font, font_scale_mid, lw)
        cv2.putText(out, cm_txt, (center_x - twm // 2, center_y + thm // 2), font, font_scale_mid, (0, 0, 0), max(1, lw), cv2.LINE_AA)
        cv2.putText(out, cm_txt, (center_x - twm // 2, center_y + thm // 2), font, font_scale_mid, (0, 0, 255), max(1, lw // 2), cv2.LINE_AA)

        spec_label = classify_specimen(name, gap_cm * 10) if gap_cm is not None else "미분류"
        
        judgment = ""
        if ng_ok_settings and ng_ok_settings['use'] and gap_cm is not None:
            match = re.search(r'\((\d+\.\d+)mm\)', ng_ok_settings['target'])
            if match:
                target_mm = float(match.group(1))
                if abs((gap_cm * 10) - target_mm) <= ng_ok_settings['tolerance']:
                    judgment = "OK"
                else:
                    judgment = "NG"

        data_row = {
            "FileName": "", "Shape": name, "Type": spec_label,
            f"Width({display_unit})": round(gap_val, 2) if gap_cm else None,
            f"Length({display_unit})": round(length_val, 2) if length_cm else None
        }
        if ng_ok_settings and ng_ok_settings['use']:
            data_row["Judgment"] = judgment
            
        measured_data.append(data_row)

    return out, measured_data

# ==========================================
# 5. Streamlit UI 및 세션
# ==========================================
st.set_page_config(page_title="시험편 형상 계측 시스템(Beta.ver3.0)", layout="wide")

if 'api_predictions_cache' not in st.session_state:
    st.session_state.api_predictions_cache = {}

with st.sidebar:
    st.header("⚙️ 분석 파라미터 설정")
    
    display_unit = st.radio("📏 측정 단위 변환", ["cm", "mm"], horizontal=True, help="결과물에 표시될 단위를 선택합니다.")
    conf_thres = st.slider("신뢰도 컷오프 (Confidence %)", 1, 100, 25, 1)
    known_cm = st.number_input("기준 자 길이 (Ruler cm)", 1.0, 100.0, 15.0, 0.5)
    
    auto_rotate = st.checkbox("📱 세로 사진 자동 90도 회전", value=True, help="스마트폰으로 세로로 찍은 사진을 인식률 향상을 위해 자동으로 가로로 돌립니다.")
    
    with st.expander("🛠️ 개발자 룸 (고급/실험 기능)", expanded=False):
        use_smart_tilt = st.checkbox("📐 스마트 각도 보정 계측", value=True, help="삐뚤어진 시편을 내부적으로 바르게 펴서 가운데 물림부만 정확히 스캔합니다.")
        
        st.markdown("---")
        use_ng_ok = st.checkbox("✅ NG / OK 판정 기능 활성화", value=False)
        target_type, tolerance_mm = None, 0.5
        if use_ng_ok:
            targets = ["Type 4 / 14A (14.0mm)", "Type 10 (12.5mm)", "Type 14B (10.0mm)", "Type 14C (8.0mm)", 
                       "Type 1 / 1A (40.0mm)", "Type 5 / 1B (25.0mm)", "Type 13A (20.0mm)", "Type 13B (12.5mm)"]
            target_type = st.selectbox("🎯 목표 규격 지정", targets)
            tolerance_mm = st.number_input("허용 오차 (±mm)", 0.1, 5.0, 0.5, 0.1)

    st.markdown("---")
    st.subheader("💳 API 크레딧 상태(미완성, 구현 중)")
    used_count, limit_count = get_roboflow_usage(ROBOFLOW_API_KEY)
    try:
        used_int, limit_int = int(used_count), int(limit_count)
        st.metric(label="이번 달 사용량 / 한도", value=f"{used_int:,} / {limit_int:,} 장")
        st.info(f"💡 남은 분석 가능 횟수: 약 **{limit_int - used_int:,}회**")
    except:
        st.caption("현재 로보플로우 사용량을 불러올 수 없습니다.")

st.title("📏 시험편 정밀 계측 시스템 (V3.0)")
st.write("이미지를 업로드하면 규격을 자동 판정합니다. 우측 옵션을 변경해도 API 잔여량이 소모되지 않습니다.")

uploaded_files = st.file_uploader("이미지를 여러 장 선택하세요 (jpg, png 등)", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if st.button("🚀 분석 시작") and uploaded_files:
    if not ROBOFLOW_API_KEY or ROBOFLOW_API_KEY == "여기에_API_키를_입력하세요":
        st.error("⚠️ API 키가 설정되지 않았습니다.")
        st.stop()

    with st.spinner("이미지 계측 중... (옵션만 변경 시 서버 통신 생략)"):
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
        model = project.version(VERSION_NUM).model
            
        all_results = []
        processed_images = {} 
        progress_bar = st.progress(0)
        
        ng_ok_settings = {"use": use_ng_ok, "target": target_type, "tolerance": tolerance_mm}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for idx, file in enumerate(uploaded_files):
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                
                if auto_rotate and img.shape[0] > img.shape[1]:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                
                api_cache_key = f"{file.name}_{conf_thres}_{auto_rotate}"
                
                if api_cache_key in st.session_state.api_predictions_cache:
                    predictions = st.session_state.api_predictions_cache[api_cache_key]
                else:
                    temp_img_path = os.path.join(temp_dir, file.name)
                    cv2.imwrite(temp_img_path, img)
                    try:
                        api_result = model.predict(temp_img_path, confidence=conf_thres, overlap=30).json()
                        predictions = api_result.get('predictions', [])
                        st.session_state.api_predictions_cache[api_cache_key] = predictions
                    except Exception as e:
                        st.warning(f"{file.name} 분석 실패: {e}")
                        continue
                
                out_img, img_data_list = annotate_one(predictions, img, known_cm, display_unit, use_smart_tilt, ng_ok_settings)
                
                processed_images[file.name] = out_img
                for data in img_data_list:
                    data['FileName'] = file.name
                    all_results.append(data)
                
                progress_bar.progress((idx + 1) / len(uploaded_files))

        st.success("✅ 이미지 분석 완료!")

        st.markdown("---")
        st.subheader("📊 분석 요약 대시보드")
        total_specs = len(all_results)
        unclassified_count = sum(1 for d in all_results if d['Type'] == '미분류')
        classified_count = total_specs - unclassified_count
        
        col1, col2, col3 = st.columns(3)
        col1.metric(label="총 계측된 시편", value=f"{total_specs} 개")
        if unclassified_count > 0: col2.metric(label="판정 성공 / 미분류", value=f"{classified_count} / {unclassified_count}", delta=f"-{unclassified_count} 미분류", delta_color="inverse")
        else: col2.metric(label="판정 성공 / 미분류", value=f"{classified_count} / 0", delta="All Clear!", delta_color="normal")
        
        if use_ng_ok:
            ng_count = sum(1 for d in all_results if d.get('Judgment') == 'NG')
            col3.metric(label="NG (불량) 발견", value=f"{ng_count} 개", delta=f"{'경고!' if ng_count > 0 else '우수'}", delta_color="inverse")

        st.markdown("---")
        tab1, tab2 = st.tabs(["🖼️ 이미지 뷰어", "📝 데이터 편집 및 다운로드"])
        
        with tab1:
            cols = st.columns(2)
            for i, (fname, out_img) in enumerate(processed_images.items()):
                with cols[i % 2]:
                    # 💡 패치 1: use_container_width 대신 구버전 문법인 use_column_width 사용
                    st.image(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), caption=fname, use_column_width=True)
                    is_success, img_buf = cv2.imencode(".jpg", out_img)
                    if is_success: st.download_button(label=f"💾 {fname} 다운로드", data=img_buf.tobytes(), file_name=f"measured_{fname}", mime="image/jpeg", key=f"dl_btn_{i}")

        with tab2:
            if all_results:
                df = pd.DataFrame(all_results)
                
                if use_ng_ok:
                    def highlight_ng(val):
                        return 'color: red; font-weight: bold' if val == 'NG' else 'color: green; font-weight: bold' if val == 'OK' else ''
                    # 💡 패치 2: 구버전 호환을 위해 에러를 유발하는 use_container_width 파라미터 삭제
                    st.dataframe(df.style.applymap(highlight_ng, subset=['Judgment']))
                else:
                    # 💡 패치 3: 정말 옛날 버전이라 data_editor 기능이 아예 없는 PC를 위한 방어 코드 추가
                    try:
                        st.data_editor(df, num_rows="dynamic")
                    except AttributeError:
                        st.dataframe(df)

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for fname, out_img in processed_images.items():
                        is_success, buffer = cv2.imencode(".jpg", out_img)
                        if is_success: zip_file.writestr(fname, buffer.tobytes())
                    zip_file.writestr("measurement_results.csv", df.to_csv(index=False, encoding='utf-8-sig'))
                zip_buffer.seek(0)
                csv_bytes = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                
                st.markdown("<br>", unsafe_allow_html=True)
                dl_col1, dl_col2 = st.columns(2)
                # 💡 패치 4: 다운로드 버튼에서도 구버전 에러를 유발하는 use_container_width 파라미터 삭제
                with dl_col1: st.download_button(label="📄 CSV 표만 다운로드", data=csv_bytes, file_name="measurement_results.csv", mime="text/csv")
                with dl_col2: st.download_button(label="📦 전체 다운로드 (ZIP)", data=zip_buffer, file_name="Total_Measurement_Results.zip", mime="application/zip")
            else:
                st.info("계측된 시편 데이터가 없습니다.")