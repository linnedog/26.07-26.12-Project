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
parquet_path = os.path.join(
    base_dir,
    'AE_Total_Cleaned_Data_AbsTime.parquet'
)


# =========================================================
# Y축 상단 압축 관련 함수
# =========================================================

# 선을 끊지 않고 Y축 윗부분만 압축하기 위한 매핑 파라미터 계산
def get_scale_params(data_max, T, top_ratio=0.25):
    mapped_max = T / (1 - top_ratio)

    if pd.isna(data_max) or data_max <= T:
        return 1.0, mapped_max

    scale_factor = (mapped_max - T) / (data_max - T)

    return scale_factor, mapped_max


# 임계값 T보다 큰 데이터만 압축
def apply_mapped_data(data, T, scale_factor):
    mapped = data.copy()

    mask = mapped > T
    mapped[mask] = T + (mapped[mask] - T) * scale_factor

    return mapped


# 압축된 Y좌표에 실제 Hit Count 눈금값 표시
def set_mapped_ticks(
    ax,
    T,
    data_max,
    scale_factor,
    mapped_max
):
    # 하단 상세 구간 눈금 간격
    if T <= 20:
        step_lower = 5
    elif T <= 50:
        step_lower = 10
    elif T <= 100:
        step_lower = 20
    elif T <= 500:
        step_lower = 100
    else:
        step_lower = 500

    ticks_lower = np.arange(
        0,
        T + step_lower,
        step_lower
    )

    ticks_lower = ticks_lower[
        ticks_lower <= T
    ]

    # 상단 압축 구간 눈금 계산
    if data_max > T:
        magnitude = 10 ** np.floor(
            np.log10(data_max)
        )

        if (data_max / magnitude) < 3:
            step_upper = magnitude / 2
        else:
            step_upper = magnitude

        start_upper = (
            np.ceil(T / step_upper)
            * step_upper
        )

        if start_upper <= T:
            start_upper += step_upper

        ticks_upper = np.arange(
            start_upper,
            data_max + step_upper,
            step_upper
        )

    else:
        ticks_upper = np.array([])

    real_ticks = np.unique(
        np.concatenate([
            ticks_lower,
            ticks_upper
        ])
    )

    mapped_ticks = np.where(
        real_ticks <= T,
        real_ticks,
        T + (real_ticks - T) * scale_factor
    )

    ax.set_yticks(mapped_ticks)

    ax.set_yticklabels([
        f"{int(x)}"
        for x in real_ticks
    ])

    # 그래프 상단에 5% 여백
    ax.set_ylim(
        0,
        mapped_max * 1.05
    )


# =========================================================
# Parquet 또는 CSV 데이터 불러오기
# =========================================================

final_df = None

if os.path.exists(parquet_path):

    print(
        "📦 기존에 병합된 Parquet 데이터를 "
        "발견했습니다. 빠르게 불러옵니다!"
    )

    final_df = pd.read_parquet(
        parquet_path
    )

else:

    print(
        "🔍 Parquet 파일이 없습니다. "
        "폴더를 스캔하여 CSV 파일 병합을 시작합니다..."
    )

    def analyze_file_structure(file_path):

        encodings = [
            'utf-8',
            'cp949',
            'utf-16',
            'cp1252'
        ]

        for enc in encodings:

            try:

                with open(
                    file_path,
                    'r',
                    encoding=enc
                ) as f:

                    for i, line in enumerate(f):

                        if (
                            'Hit Id' in line
                            or 'Hit\t' in line
                            or 'Hit,' in line
                        ):

                            if ',' in line:
                                sep = ','
                            else:
                                sep = '\t'

                            return i, sep, enc

            except UnicodeDecodeError:
                continue

        return 0, '\t', 'utf-8'


    target_pattern = os.path.join(
        base_dir,
        '**',
        '*.csv'
    )

    file_list = sorted(
        glob.glob(
            target_pattern,
            recursive=True
        )
    )

    if len(file_list) == 0:

        print(
            "❌ 에러: CSV 파일을 찾을 수 없습니다!"
        )

        raise SystemExit


    print(
        f"🚀 총 {len(file_list)}개의 "
        "파일을 통합합니다...\n"
    )

    processed_dfs = []

    last_toh = -1
    current_session_start = None

    for i, file_path in enumerate(file_list):

        header_idx, sep_char, enc = (
            analyze_file_structure(file_path)
        )

        try:

            df_chunk = pd.read_csv(
                file_path,
                sep=sep_char,
                skiprows=header_idx,
                encoding=enc,
                on_bad_lines='skip',
                low_memory=False
            )

        except Exception as e:

            print(
                f"⚠️ 읽기 실패: "
                f"{os.path.basename(file_path)} "
                f"({e})"
            )

            continue


        df_chunk = df_chunk.dropna(
            axis=1,
            how='all'
        )

        df_chunk.columns = (
            df_chunk.columns
            .str.strip()
        )


        required_columns = [
            'Channel',
            'TOH [us]',
            'Date Time'
        ]

        if (
            df_chunk.empty
            or not all(
                col in df_chunk.columns
                for col in required_columns
            )
        ):
            continue


        # 숫자형 변환
        df_chunk['TOH [us]'] = pd.to_numeric(
            df_chunk['TOH [us]'],
            errors='coerce'
        )

        df_chunk['Channel'] = pd.to_numeric(
            df_chunk['Channel'],
            errors='coerce'
        )

        df_chunk = df_chunk.dropna(
            subset=[
                'TOH [us]',
                'Channel',
                'Date Time'
            ]
        )

        if df_chunk.empty:
            continue


        first_toh_in_chunk = (
            df_chunk['TOH [us]']
            .iloc[0]
        )


        # TOH가 이전 파일보다 작아졌다면 새 세션으로 판단
        if (
            first_toh_in_chunk < last_toh
            or current_session_start is None
        ):

            try:

                df_chunk['Date Time'] = (
                    pd.to_datetime(
                        df_chunk['Date Time'],
                        errors='coerce'
                    )
                )

                df_chunk = df_chunk.dropna(
                    subset=['Date Time']
                )

                if df_chunk.empty:
                    continue

                time_delta = pd.to_timedelta(
                    first_toh_in_chunk,
                    unit='us'
                )

                current_session_start = (
                    df_chunk['Date Time'].iloc[0]
                    - time_delta
                )

            except Exception:
                continue


        df_chunk['Abs_Time'] = (
            current_session_start
            + pd.to_timedelta(
                df_chunk['TOH [us]'],
                unit='us'
            )
        )

        last_toh = (
            df_chunk['TOH [us]']
            .iloc[-1]
        )


        # 채널 1 제외
        df_filtered = df_chunk[
            df_chunk['Channel'] != 1
        ].copy()


        # Count 3 미만 Hit 제외
        if 'Count [#]' in df_filtered.columns:

            df_filtered['Count [#]'] = pd.to_numeric(
                df_filtered['Count [#]'],
                errors='coerce'
            )

            df_filtered = df_filtered[
                df_filtered['Count [#]'] >= 3
            ]


        if df_filtered.empty:
            continue


        cols_to_keep = [
            'Abs_Time',
            'Hit Id',
            'Channel'
        ]

        available_columns = [
            col
            for col in cols_to_keep
            if col in df_filtered.columns
        ]

        processed_dfs.append(
            df_filtered[available_columns]
        )


        if i % 500 == 0 and i > 0:

            print(
                f"  └ {i}개 파일 처리 완료..."
            )


    if not processed_dfs:

        print(
            "❌ 에러: 합칠 데이터가 없습니다."
        )

        raise SystemExit


    final_df = pd.concat(
        processed_dfs,
        ignore_index=True
    )

    final_df = (
        final_df
        .sort_values('Abs_Time')
        .reset_index(drop=True)
    )


    final_df.to_parquet(
        parquet_path,
        engine='pyarrow',
        index=False
    )

    print(
        "✅ 데이터 병합 및 Parquet 저장 완료! "
        f"(총 Hit 수: {len(final_df):,}개)"
    )


# =========================================================
# 1분 및 10분 데이터 생성
# =========================================================

print(
    "\n🎨 데이터 시계열 동기화를 시작합니다..."
)

df_viz = final_df.set_index(
    'Abs_Time'
)


# 전체 1분 시간축 생성
full_idx = pd.date_range(
    start=df_viz.index.min().floor('1min'),
    end=df_viz.index.max().ceil('1min'),
    freq='1min'
)


df_1m = pd.DataFrame(
    index=full_idx
)


# 채널별 1분 Hit Count
df_1m['CH2'] = (
    df_viz[
        df_viz['Channel'] == 2
    ]
    .resample('1min')
    .size()
)


df_1m['CH3'] = (
    df_viz[
        df_viz['Channel'] == 3
    ]
    .resample('1min')
    .size()
)


# 데이터가 없는 구간을 0으로 처리
df_1m = df_1m.fillna(0)


# 10분 단위 Hit Count
df_10m = (
    df_1m
    .resample('10min')
    .sum()
)


print(
    "🎨 선이 끊기지 않는 단일 축 "
    "Piecewise 그래프를 렌더링합니다..."
)


# =========================================================
# 그래프 생성 함수
# =========================================================

def plot_continuous_broken_graph(
    df_binned,
    title,
    y_label_unit,
    save_filename,
    lw_ch2,
    lw_ch3,
    alpha_ch2,
    alpha_ch3,
    png_dpi=600
):

    fig, ax = plt.subplots(
        figsize=(16, 6)
    )

    ax_r = ax.twinx()


    # 시험 시작 후 1일 이후 데이터를 후반부 안정화 구간으로 판단
    mask_post = (
        df_binned.index
        > (
            df_binned.index.min()
            + pd.Timedelta(days=1)
        )
    )


    if mask_post.any():

        ch2_post = df_binned.loc[
            mask_post,
            'CH2'
        ]

        ch3_post = df_binned.loc[
            mask_post,
            'CH3'
        ]

    else:

        ch2_post = df_binned['CH2']
        ch3_post = df_binned['CH3']


    # 안정화 구간 99.9 분위수의 1.5배를 압축 시작점으로 설정
    T_ch2 = max(
        ch2_post.quantile(0.999) * 1.5,
        10.0
    )

    T_ch3 = max(
        ch3_post.quantile(0.999) * 1.5,
        10.0
    )


    ch2_max = df_binned['CH2'].max()
    ch3_max = df_binned['CH3'].max()


    # 상위 25% 영역에 큰 값을 압축
    sf_ch2, m_max_ch2 = get_scale_params(
        ch2_max,
        T_ch2,
        top_ratio=0.25
    )

    sf_ch3, m_max_ch3 = get_scale_params(
        ch3_max,
        T_ch3,
        top_ratio=0.25
    )


    # 실제 데이터를 Piecewise 좌표로 변환
    y_ch2 = apply_mapped_data(
        df_binned['CH2'],
        T_ch2,
        sf_ch2
    )

    y_ch3 = apply_mapped_data(
        df_binned['CH3'],
        T_ch3,
        sf_ch3
    )


    # CH2 그래프
    ax.plot(
        df_binned.index,
        y_ch2,
        color='#3498db',
        lw=lw_ch2,
        alpha=alpha_ch2,
        label='CH2'
    )


    # CH3 그래프
    ax_r.plot(
        df_binned.index,
        y_ch3,
        color='#e67e22',
        lw=lw_ch3,
        alpha=alpha_ch3,
        label='CH3'
    )


    # 실제 Hit Count 값으로 Y축 눈금 표시
    set_mapped_ticks(
        ax,
        T_ch2,
        ch2_max,
        sf_ch2,
        m_max_ch2
    )

    set_mapped_ticks(
        ax_r,
        T_ch3,
        ch3_max,
        sf_ch3,
        m_max_ch3
    )


    # =====================================================
    # 축 압축 표시
    # =====================================================

    # 전체 Y축 범위의 약 71.4% 지점
    h_frac = 0.75 / 1.05
    d = 0.015

    break_kwargs = dict(
        transform=ax.transAxes,
        color='gray',
        clip_on=False,
        lw=1.5
    )


    if (
        ch2_max > T_ch2
        or ch3_max > T_ch3
    ):

        # 왼쪽 Y축 절단 표시
        ax.plot(
            (-d, +d),
            (h_frac - d, h_frac + d),
            **break_kwargs
        )

        ax.plot(
            (-d, +d),
            (
                h_frac - d - 0.02,
                h_frac + d - 0.02
            ),
            **break_kwargs
        )


        # 오른쪽 Y축 절단 표시
        ax.plot(
            (1 - d, 1 + d),
            (h_frac - d, h_frac + d),
            **break_kwargs
        )

        ax.plot(
            (1 - d, 1 + d),
            (
                h_frac - d - 0.02,
                h_frac + d - 0.02
            ),
            **break_kwargs
        )


        # CH2 압축 기준선
        ax.axhline(
            T_ch2,
            color='#bdc3c7',
            linestyle='--',
            lw=1,
            alpha=0.5,
            zorder=0
        )


    # =====================================================
    # 제목 및 축 디자인
    # =====================================================

    ax.set_title(
        title,
        fontsize=16,
        fontweight='bold',
        pad=15
    )


    ax.set_ylabel(
        f'CH2 Hit Count (hits/{y_label_unit})',
        color='#2980b9',
        fontsize=12,
        fontweight='bold'
    )


    ax_r.set_ylabel(
        f'CH3 Hit Count (hits/{y_label_unit})',
        color='#d35400',
        fontsize=12,
        fontweight='bold'
    )


    ax.tick_params(
        axis='y',
        colors='#2980b9'
    )

    ax_r.tick_params(
        axis='y',
        colors='#d35400'
    )


    # Y축 테두리도 채널 색상과 일치
    ax.spines['left'].set_color(
        '#2980b9'
    )

    ax_r.spines['right'].set_color(
        '#d35400'
    )


    ax.grid(
        True,
        linestyle='--',
        alpha=0.4,
        color='#bdc3c7'
    )


    # 날짜 및 시간 형식
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            '%m-%d %H:%M'
        )
    )


    plt.setp(
        ax.xaxis.get_majorticklabels(),
        rotation=20,
        ha='right',
        fontsize=11
    )


    ax.set_xlabel(
        'Experiment Time (Date & Time)',
        fontsize=13,
        fontweight='bold',
        labelpad=10
    )


    # =====================================================
    # 범례
    # =====================================================

    lines, labels = (
        ax.get_legend_handles_labels()
    )

    lines_r, labels_r = (
        ax_r.get_legend_handles_labels()
    )


    ax.legend(
        lines + lines_r,
        labels + labels_r,
        loc='upper right',
        fontsize=11,
        framealpha=1.0
    )


    # =====================================================
    # 그래프 설명문
    # =====================================================

    fig.text(
        0.5,
        0.015,
        (
            "Note: CH2 and CH3 use different "
            "Y-axis scales due to the difference "
            "in hit occurrence magnitude."
        ),
        ha='center',
        va='bottom',
        fontsize=11,
        fontweight='bold',
        color='#7f8c8d'
    )


    # 아래쪽 설명문 공간 확보
    fig.tight_layout(
        rect=[0, 0.06, 1, 1]
    )


    # =====================================================
    # 파일 저장
    # =====================================================

    # 전달받은 파일명에서 확장자를 제외한 기본 경로 생성
    base_save_path = os.path.splitext(
        save_filename
    )[0]


    png_filename = (
        base_save_path
        + '.png'
    )

    pdf_filename = (
        base_save_path
        + '.pdf'
    )


    # 1. PPT, 한글, Word 삽입용 고해상도 PNG
    fig.savefig(
        png_filename,
        dpi=png_dpi,
        bbox_inches='tight',
        facecolor='white'
    )


    # 2. 확대 및 인쇄용 벡터 PDF
    fig.savefig(
        pdf_filename,
        bbox_inches='tight',
        facecolor='white'
    )


    print(
        f"✅ PNG 저장 완료: {png_filename}"
    )

    print(
        f"✅ PDF 저장 완료: {pdf_filename}"
    )


# =========================================================
# 1분 단위 그래프
# =========================================================

path_1m = os.path.join(
    base_dir,
    'AE_1min_Report.png'
)

plot_continuous_broken_graph(
    df_binned=df_1m,
    title='1-min AE Hit Count',
    y_label_unit='min',
    save_filename=path_1m,
    lw_ch2=0.3,
    lw_ch3=0.9,
    alpha_ch2=0.6,
    alpha_ch3=0.9,
    png_dpi=600
)


# =========================================================
# 10분 단위 그래프
# =========================================================

path_10m = os.path.join(
    base_dir,
    'AE_10min_Report.png'
)

plot_continuous_broken_graph(
    df_binned=df_10m,
    title='10-min AE Hit Count',
    y_label_unit='10 min',
    save_filename=path_10m,
    lw_ch2=0.8,
    lw_ch3=1.5,
    alpha_ch2=0.8,
    alpha_ch3=1.0,
    png_dpi=600
)


plt.show()