import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

base_dir = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# [입력 1] 파라미터 통합 데이터 경로
# =====================================================================
parquet_path = os.path.join(base_dir, 'AE_Master_Data.parquet')

# =====================================================================
# [입력 2] 58GB 원본 파형(Waveform) 데이터 최상위 폴더 경로
# =====================================================================
waveform_dir = r'F:\AE' 

desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'AE_DeepDive_Reports')
os.makedirs(desktop_path, exist_ok=True)

if not os.path.exists(parquet_path):
    print("❌ 에러: Parquet 파일이 없습니다. 경로를 확인해 주세요.")
    exit()

print("🔍 [STEP 1] 데이터를 분석하여 타겟 11개(비교군 1개 + Top 10)를 선정합니다...")
df = pd.read_parquet(parquet_path)
df = df.set_index('Abs_Time')

target_hits_info = {}

# ---------------------------------------------------------
# 🎯 [추가] 비교군 (Event 0: Baseline Noise) 추출 로직
# 7월 21일의 평온한 시간대 중, 진폭이 매우 낮고(5mV 이하) 일반적인 잡음 찾기
# ---------------------------------------------------------
baseline_mask = (df.index >= '2026-07-21 00:00:00') & (df.index < '2026-07-22 00:00:00')
df_base = df[baseline_mask]

if not df_base.empty:
    typical_noise = df_base[(df_base['Peak Amplitude [mV]'] < 5) & (df_base['Peak Amplitude [mV]'] > 0)]
    if not typical_noise.empty:
        base_hit_row = typical_noise.iloc[len(typical_noise)//2] # 중간에 있는 무난한 노이즈 선택
        base_hit_id = int(base_hit_row['Hit Id'])
        
        target_hits_info[str(base_hit_id)] = {
            'rank': 0,
            'name': 'Event 0 (Baseline / Gas Noise)',
            'time': typical_noise.index[len(typical_noise)//2],
            'amp': base_hit_row['Peak Amplitude [mV]'],
            'duration': base_hit_row['Duration [us]'],
            'risetime': base_hit_row['Rise-time [us]'],
            'energy': base_hit_row['Signal Energy [EU]']
        }
        print(f"   └ ✅ [비교군] Event 0 (Hit Id: {base_hit_id}) 선정 완료")

# ---------------------------------------------------------
# 🎯 Top 10 이벤트 추출 로직 (7/22 이후)
# ---------------------------------------------------------
target_start_time = pd.to_datetime('2026-07-22 00:00:00')
df_target = df[df.index >= target_start_time].copy()

window_size = '30min'
energy_bins = df_target['Signal Energy [EU]'].resample(window_size).sum().fillna(0)

top_events = []
temp_energy = energy_bins.copy()

for i in range(10):
    if temp_energy.empty or temp_energy.max() == 0: break
    peak_time = temp_energy.idxmax()
    top_events.append(peak_time)
    mask = (temp_energy.index <= peak_time - pd.Timedelta(hours=1)) | (temp_energy.index >= peak_time + pd.Timedelta(hours=1))
    temp_energy = temp_energy[mask]

top_events = sorted(top_events)

for i, ev_time in enumerate(top_events):
    start = ev_time
    end = ev_time + pd.Timedelta(minutes=30)
    
    mask = (df.index >= start) & (df.index < end)
    ev_df = df.loc[mask]
    
    if len(ev_df) == 0: continue
        
    max_hit_row = ev_df.loc[ev_df['Signal Energy [EU]'].idxmax()]
    hit_id = int(max_hit_row['Hit Id'])
    
    target_hits_info[str(hit_id)] = {
        'rank': i + 1,
        'name': f'Event {i+1}',
        'time': ev_df['Signal Energy [EU]'].idxmax(),
        'amp': max_hit_row['Peak Amplitude [mV]'],
        'duration': max_hit_row['Duration [us]'],
        'risetime': max_hit_row['Rise-time [us]'],
        'energy': max_hit_row['Signal Energy [EU]']
    }

print(f"\n🚀 [STEP 2] 총 {len(target_hits_info)}개의 타겟 파형을 58GB 폴더에서 스나이핑합니다...")

remaining_targets = set(target_hits_info.keys())

for root, dirs, files in os.walk(waveform_dir):
    if not remaining_targets: break

    for file in files:
        file_base_name = os.path.splitext(file)[0]
        
        for target in list(remaining_targets):
            if file_base_name.endswith(target):
                info = target_hits_info[target]
                print(f"   └ 🎯 [{info['name']}] Hit Id {target} 렌더링 중...")
                file_path = os.path.join(root, file)
                
                # 스마트 로딩 엔진
                wf_df = None
                success = False
                for enc in ['utf-8', 'utf-16', 'cp949']:
                    if success: break
                    for sep in ['\t', ',']:
                        try:
                            temp_df = pd.read_csv(file_path, sep=sep, encoding=enc, engine='python')
                            if len(temp_df.columns) >= 2:
                                wf_df = temp_df
                                success = True
                                break
                        except Exception:
                            continue
                
                if wf_df is None or len(wf_df.columns) < 2:
                    remaining_targets.remove(target)
                    continue

                wf_df.columns = ['Time', 'Voltage'] + list(wf_df.columns[2:])
                wf_df['Time'] = pd.to_numeric(wf_df['Time'], errors='coerce')
                wf_df['Voltage'] = pd.to_numeric(wf_df['Voltage'], errors='coerce')
                wf_df = wf_df.dropna(subset=['Time', 'Voltage'])

                # ---------------------------------------------------------
                # 🎨 2단 콤보 뷰 시각화 (Waveform + Hit Count Context)
                # ---------------------------------------------------------
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
                fig.subplots_adjust(hspace=0.3)
                
                # [상단 패널] 리얼 파형 (Waveform)
                event_color = '#7f8c8d' if info['rank'] == 0 else '#e74c3c'
                ax1.plot(wf_df['Time'], wf_df['Voltage'], color=event_color, linewidth=1.2)
                
                time_str = info['time'].strftime('%Y-%m-%d %H:%M:%S')
                ax1.set_title(f"{info['name']} Waveform (Hit: {target} / {time_str})", fontsize=15, fontweight='bold', pad=15)
                ax1.set_ylabel('Voltage [mV]', fontsize=11)
                ax1.grid(True, linestyle='--', alpha=0.5)

                # 수치 데이터 표(Text Box)
                text_content = (
                    f"Amplitude : {info['amp']:.2f} mV\n"
                    f"Energy    : {info['energy']:,.0f} EU\n"
                    f"Duration  : {info['duration']:.1f} \u03bcs\n"
                    f"Rise-time : {info['risetime']:.1f} \u03bcs"
                )
                props = dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', alpha=0.9, edgecolor='#bdc3c7')
                ax1.text(0.98, 0.95, text_content, transform=ax1.transAxes, fontsize=11, fontweight='500',
                         verticalalignment='top', horizontalalignment='right', bbox=props, fontfamily='monospace')

                # [하단 패널] 해당 파형 전후 30분(총 1시간)의 Hit 발생 히스토그램
                context_start = info['time'] - pd.Timedelta(minutes=30)
                context_end = info['time'] + pd.Timedelta(minutes=30)
                
                # 해당 시간대 데이터만 자르기
                context_df = df[(df.index >= context_start) & (df.index <= context_end)]
                
                # 1분 단위로 Hit 개수 세기
                hit_rate = context_df.resample('1min').size()
                
                ax2.fill_between(hit_rate.index, 0, hit_rate.values, color='#34495e', alpha=0.6, step='mid')
                
                # 정확히 파형이 터진 순간에 빨간색 점선 스나이퍼 조준!
                ax2.axvline(info['time'], color='red', linestyle='--', linewidth=2, label='Moment of this Waveform')
                
                ax2.set_title("AE Hit Count Context (\u00b130 Minutes)", fontsize=12, fontweight='bold')
                ax2.set_ylabel('Hits / min', fontsize=11)
                ax2.set_xlabel('Experiment Time', fontsize=11)
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax2.grid(True, linestyle='--', alpha=0.5)
                ax2.legend(loc='upper right')

                plt.tight_layout()
                save_path = os.path.join(desktop_path, f"DeepDive_{info['rank']:02d}_Hit_{target}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                remaining_targets.remove(target)
                break

if remaining_targets:
    print(f"\n⚠️ 끝내 찾지 못한 타겟: {remaining_targets}")
else:
    print(f"\n🎉 분석 완료! 바탕화면의 [AE_DeepDive_Reports] 폴더를 확인하세요.")