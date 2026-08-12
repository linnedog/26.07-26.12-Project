import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import os

# 윈도우 한글/마이너스 깨짐 방지
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

base_dir = os.path.dirname(os.path.abspath(__file__))
parquet_path = os.path.join(base_dir, 'AE_Master_Data.parquet')

# =====================================================================
# [STEP 0] 독립적인 스마트 로더 (CSV 파일 자동 병합 엔진 내장)
# =====================================================================
df = None

if os.path.exists(parquet_path):
    print("📦 기존 통합 데이터(Parquet)를 1초 만에 불러옵니다!")
    df = pd.read_parquet(parquet_path)
else:
    print("🔍 통합 파일이 없습니다. CSV 파일 병합을 시작합니다 (독립 실행)...")
    
    def analyze_file_structure(file_path):
        encodings = ['utf-8', 'cp949', 'utf-16', 'cp1252']
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    for i, line in enumerate(f):
                        if 'Hit Id' in line or 'Hit\t' in line or 'Hit,' in line:
                            sep = ',' if ',' in line else '\t'
                            return i, sep, enc
            except UnicodeDecodeError:
                continue
        return 0, '\t', 'utf-8'

    target_pattern = os.path.join(base_dir, '**', '*.csv')
    file_list = sorted(glob.glob(target_pattern, recursive=True))

    if len(file_list) == 0:
        print("❌ 에러: CSV 파일을 찾을 수 없습니다. 데이터가 있는 폴더에서 실행해주세요.")
        exit()

    print(f"🚀 총 {len(file_list)}개의 CSV 파일을 합칩니다...")
    processed_dfs = []
    last_toh = -1
    current_session_start = None

    for i, file_path in enumerate(file_list):
        header_idx, sep_char, enc = analyze_file_structure(file_path)
        try:
            df_chunk = pd.read_csv(file_path, sep=sep_char, skiprows=header_idx, encoding=enc, on_bad_lines='skip', low_memory=False)
        except Exception:
            continue
            
        df_chunk = df_chunk.dropna(axis=1, how='all')
        df_chunk.columns = df_chunk.columns.str.strip()
        if df_chunk.empty or 'Channel' not in df_chunk.columns:
            continue

        first_toh_in_chunk = df_chunk['TOH [us]'].iloc[0]
        if first_toh_in_chunk < last_toh or current_session_start is None:
            try:
                df_chunk['Date Time'] = pd.to_datetime(df_chunk['Date Time'])
                time_delta = pd.to_timedelta(first_toh_in_chunk, unit='us')
                current_session_start = df_chunk['Date Time'].iloc[0] - time_delta
            except Exception:
                continue

        df_chunk['Abs_Time'] = current_session_start + pd.to_timedelta(df_chunk['TOH [us]'], unit='us')
        last_toh = df_chunk['TOH [us]'].iloc[-1]
        
        # 노이즈 필터링
        df_filtered = df_chunk[df_chunk['Channel'] != 1].copy()
        if 'Count [#]' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['Count [#]'] >= 3]
            
        if df_filtered.empty:
            continue
            
        # 필수 컬럼만 추출
        cols_to_keep = ['Abs_Time', 'Hit Id', 'Channel', 'Peak Amplitude [mV]', 'Duration [us]', 'Rise-time [us]', 'Signal Energy [EU]']
        processed_dfs.append(df_filtered[[c for c in cols_to_keep if c in df_filtered.columns]])

        if i % 500 == 0 and i > 0:
            print(f"  └ {i}개 파일 처리 완료...")

    df = pd.concat(processed_dfs, ignore_index=True)
    df = df.sort_values('Abs_Time').reset_index(drop=True)
    df.to_parquet(parquet_path, engine='pyarrow')
    print(f"✅ CSV 병합 및 저장 완료! (총 {len(df):,}개)")

df = df.set_index('Abs_Time')

# =====================================================================
# [STEP 1] 이벤트 자동 검출 (초기 노이즈 무시, 7/22 이후 Top 10 추출)
# =====================================================================
print("\n🔍 [이벤트 자동 검출] 7/22 이후 에너지가 폭발한 Top 10 구간을 찾습니다...")

target_start_time = pd.to_datetime('2026-07-22 00:00:00')
df_target = df[df.index >= target_start_time].copy()

window_size = '30min'
energy_bins = df_target['Signal Energy [EU]'].resample(window_size).sum().fillna(0)

top_events = []
temp_energy = energy_bins.copy()

for i in range(10):
    if temp_energy.empty or temp_energy.max() == 0: 
        break
    peak_time = temp_energy.idxmax()
    top_events.append(peak_time)
    
    # 겹침 방지 (전후 1시간 제외)
    mask = (temp_energy.index <= peak_time - pd.Timedelta(hours=1)) | (temp_energy.index >= peak_time + pd.Timedelta(hours=1))
    temp_energy = temp_energy[mask]

top_events = sorted(top_events) 

# =====================================================================
# [STEP 2] 구간별 파라미터 텍스트 추출 및 [Hit Id] 출력
# =====================================================================
event_results = []
print("\n" + "="*75)
print(" 🎯 [Top 10 AE Event 파라미터 정밀 추출 리포트]")
print("="*75)

for i, ev_time in enumerate(top_events):
    start = ev_time
    end = ev_time + pd.Timedelta(minutes=30)
    
    mask = (df.index >= start) & (df.index < end)
    ev_df = df.loc[mask].copy()
    
    hit_count = len(ev_df)
    if hit_count == 0: continue
        
    max_amp = ev_df['Peak Amplitude [mV]'].max()
    avg_energy = ev_df['Signal Energy [EU]'].mean()
    avg_duration = ev_df['Duration [us]'].mean()
    avg_risetime = ev_df['Rise-time [us]'].mean()
    
    # 🎯 [핵심] 해당 이벤트 구간에서 에너지가 가장 강력했던 대장 Hit의 번호 찾기
    max_hit_row = ev_df.loc[ev_df['Signal Energy [EU]'].idxmax()]
    target_hit_id = max_hit_row['Hit Id']
    target_energy = max_hit_row['Signal Energy [EU]']
    
    event_name = f"Event {i+1}"
    event_results.append({
        'name': event_name,
        'start': start,
        'end': end,
        'data': ev_df
    })
    
    print(f"▶ {event_name} ({start.strftime('%m-%d %H:%M')} ~ {end.strftime('%H:%M')})")
    print(f"   - 총 Hit 수     : {hit_count:,} 개")
    print(f"   - 최대 진폭     : {max_amp:.2f} mV")
    print(f"   - 평균 에너지   : {avg_energy:.2f} EU")
    print(f"   - 평균 Duration : {avg_duration:.2f} us")
    print(f"   - 평균 Rise-time: {avg_risetime:.2f} us")
    print(f"   🎯 [Target] 파형 추출용 번호 -> Hit Id: {int(target_hit_id)} (에너지: {target_energy:.2f} EU)\n")

# =====================================================================
# [STEP 3] 추출된 10개 이벤트의 패턴 분류 시각화
# =====================================================================
print("🎨 이벤트 패턴 분류(Clustering) 시각화를 진행합니다...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [1, 1.5]})
colors = plt.cm.tab10.colors

# [상단] 타임라인
ax1.plot(energy_bins.index, energy_bins.values, color='#bdc3c7', linewidth=1.5, label='Energy Trend (30min)')
for i, ev in enumerate(event_results):
    c = colors[i % 10]
    ax1.axvspan(ev['start'], ev['end'], color=c, alpha=0.4)
    y_pos = energy_bins.max() * (0.9 - (i % 2) * 0.15)
    ax1.text(ev['start'], y_pos, f"E{i+1}", color=c, fontweight='bold', fontsize=12)

ax1.set_title("Timeline of Top 10 Events (7/22 이후 Signal Energy 기준)", fontsize=16, fontweight='bold', pad=10)
ax1.set_ylabel("Total Energy [EU]", fontsize=12)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
ax1.grid(True, linestyle='--', alpha=0.5)

# [하단] 산점도
for i, ev in enumerate(event_results):
    c = colors[i % 10]
    ev_data = ev['data']
    ax2.scatter(ev_data['Duration [us]'], ev_data['Peak Amplitude [mV]'], 
                color=c, label=f"{ev['name']} ({ev['start'].strftime('%m-%d %H:%M')})", 
                alpha=0.6, s=20, edgecolors='white', linewidth=0.5)

ax2.set_title("Pattern Classification by Event (Duration vs Peak Amplitude)", fontsize=16, fontweight='bold', pad=10)
ax2.set_xlabel("Duration [us] (Log Scale)", fontsize=12)
ax2.set_ylabel("Peak Amplitude [mV] (Log Scale)", fontsize=12)
ax2.set_xscale('log') 
ax2.set_yscale('log')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11, title="Top 10 Events", title_fontsize=13)

plt.tight_layout()
save_path = os.path.join(base_dir, 'AE_Top10_Event_Analysis.png')
plt.savefig(save_path, dpi=600, bbox_inches='tight')
print(f"✅ 이벤트 정밀 분석 그래프 자동 저장 완료: {save_path}")