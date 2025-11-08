import yfinance as yf
import numpy as np
import pandas as pd

# --- [1. '최강 전략' 파라미터 설정] ---
# (2004~2024년 백테스트 기준 최적 비중)
BASE_WEIGHTS = {
    'QQQ': 0.40,
    'GLD': 0.20,
    'Tactical_Bond': 0.40
}

# MA 전략 설정
MA_WINDOWS = [20, 120, 200]
SCALAR_MAP = {3: 1.0, 2: 0.75, 1: 0.50, 0: 0.0} # 시나리오 A

# 채권 스위칭 설정
RATE_MA_WINDOW = 200
BOND_RISING_RATE = 'IEF'
BOND_FALLING_RATE = 'TLT'

# 현금 대체 설정
CASH_ASSET_TICKER = '^IRX' # SGOV 대용 (3개월 T-Bill)

# 분석할 티커 목록
tickers_to_download = ['QQQ', 'GLD', 'TLT', 'IEF', '^TNX', '^IRX']

# --- [2. 일일 신호 계산 함수] ---
def get_daily_signals_and_report():
    
    print("... 최신 시장 데이터 다운로드 중 ...")
    # MA 계산을 위해 최소 200일 + 100일(버퍼) 데이터 다운로드
    # (yfinance는 가끔 '^' 티커의 최근 데이터를 누락하므로 400d로 더 넉넉하게 받음)
    data_full = yf.download(tickers_to_download, period="400d", progress=False)
    
    if data_full.empty:
        raise ValueError("데이터 다운로드에 실패했습니다.")
    
    all_prices_df = data_full['Close']
    
    # --- Tactical_Bond (IEF/TLT) 생성 ---
    # 데이터가 누락될 수 있으므로 ffill()로 채움
    rate_prices = all_prices_df['^TNX'].ffill()
    rate_ma = rate_prices.rolling(window=RATE_MA_WINDOW).mean()
    # 금리 상승기(True) / 하락기(False)
    is_rising_rates = (rate_prices > rate_ma)
    
    # Tactical_Bond의 가격 데이터 생성
    bond_prices = pd.Series(
        np.where(
            is_rising_rates, 
            all_prices_df[BOND_RISING_RATE].ffill(),
            all_prices_df[BOND_FALLING_RATE].ffill()
        ), 
        index=all_prices_df.index
    )
    bond_prices.name = 'Tactical_Bond'
    
    # --- SGOV_Synth (현금) 수익률 생성 ---
    irx_yield = all_prices_df[CASH_ASSET_TICKER].ffill() / 100
    sgov_daily_return = (1 + irx_yield) ** (1/252) - 1
    sgov_daily_return.name = 'SGOV_Synth'

    # --- 최종 분석 데이터 준비 ---
    # MA 신호 계산용 가격 데이터 (QQQ, GLD, Tactical_Bond)
    prices_for_signal = pd.concat([all_prices_df[['QQQ', 'GLD']].ffill(), bond_prices.ffill()], axis=1)
    
    # --- [3. 오늘 비중 계산] ---
    
    # 1. MA 신호(0~3점) 계산 (어제 종가 기준)
    ma_scores = pd.Series(0, index=['QQQ', 'GLD', 'Tactical_Bond'])
    
    # 어제 날짜 (가장 마지막 데이터)
    yesterday = prices_for_signal.index[-1]
    
    for ticker in ma_scores.index:
        score = 0
        for window in MA_WINDOWS:
            ma_value = prices_for_signal[ticker].rolling(window=window).mean().loc[yesterday]
            current_price = prices_for_signal[ticker].loc[yesterday]
            
            # MA 값이 NaN이면 (데이터가 부족하면) 신호를 0점으로 처리 (하락으로 간주)
            if pd.isna(ma_value) or current_price < ma_value:
                score += 0
            else:
                score += 1
        ma_scores[ticker] = score

    # 2. 시나리오 A 스케일러(Scalar) 적용
    scalars = ma_scores.map(SCALAR_MAP) # 예: QQQ 0.75, GLD 0.50, Bond 1.0

    # 3. 최종 비중 계산
    invested_qqq = BASE_WEIGHTS['QQQ'] * scalars['QQQ']
    invested_gld = BASE_WEIGHTS['GLD'] * scalars['GLD']
    invested_bond = BASE_WEIGHTS['Tactical_Bond'] * scalars['Tactical_Bond']

    # 4. 현금(SGOV) 비중 계산
    cash_qqq = BASE_WEIGHTS['QQQ'] * (1 - scalars['QQQ'])
    cash_gld = BASE_WEIGHTS['GLD'] * (1 - scalars['GLD'])
    cash_bond = BASE_WEIGHTS['Tactical_Bond'] * (1 - scalars['Tactical_Bond'])
    total_sgov = cash_qqq + cash_gld + cash_bond
    
    # 5. 전일 대비 수익률 계산 (가격 데이터 기준)
    price_change = prices_for_signal.pct_change().iloc[-1]
    
    # --- [4. 알림 메시지 생성] ---
    
    # Tactical_Bond가 현재 IEF인지 TLT인지 확인
    current_bond_ticker = BOND_RISING_RATE if is_rising_rates.iloc[-1] else BOND_FALLING_RATE
    
    # SGOV(현금) 수익률
    sgov_yield = irx_yield.iloc[-1]
    
    report = []
    report.append(f"🔔 '최강 전략 (SGOV 1.28)' 일일 리포트")
    report.append(f"   ({yesterday.strftime('%Y-%m-%d')} 마감 기준)")
    report.append("="*30)
    report.append("📈 [1] 전일 시장 현황")
    report.append(f"  - QQQ: {price_change['QQQ']:.2%}")
    report.append(f"  - GLD: {price_change['GLD']:.2%}")
    report.append(f"  - 채권({current_bond_ticker}): {price_change['Tactical_Bond']:.2%}")
    report.append(f"  - 현금({CASH_ASSET_TICKER}): 연 {sgov_yield:.2%}")

    report.append("\n" + "="*30)
    report.append("📊 [2] MA 신호 (20/120/200일)")
    report.append(f"  - QQQ: {ma_scores['QQQ']}/3개 ON  (→ {scalars['QQQ']:.0%} 투자)")
    report.append(f"  - GLD: {ma_scores['GLD']}/3개 ON  (→ {scalars['GLD']:.0%} 투자)")
    report.append(f"  - Bond: {ma_scores['Tactical_Bond']}/3개 ON (→ {scalars['Tactical_Bond']:.0%} 투자)")

    report.append("\n" + "="*30)
    report.append("💰 [3] 오늘 목표 비중 (리밸런싱)")
    report.append(f"  - QQQ: {invested_qqq:.2%}")
    report.append(f"  - GLD: {invested_gld:.2%}")
    
    if current_bond_ticker == 'IEF':
        report.append(f"  - IEF (채권): {invested_bond:.2%}")
        report.append(f"  - TLT (채권): 0.00%")
    else:
        report.append(f"  - IEF (채권): 0.00%")
        report.append(f"  - TLT (채권): {invested_bond:.2%}")
        
    report.append(f"  - SGOV (현금): {total_sgov:.2%}")
    report.append("-" * 30)
    report.append(f"  * 총합: {invested_qqq + invested_gld + invested_bond + total_sgov:.2%}")
    
    return "\n".join(report)

# --- [5. 메인 실행] ---
if __name__ == "__main__":
    try:
        daily_report = get_daily_signals_and_report()
        print(daily_report)
        
        # --- [텔레그램 전송 (추가 작업)] ---
        # 이 아래에 텔레그램 봇 API 코드를 추가하여
        # 'daily_report' 변수에 담긴 텍스트를 전송할 수 있습니다.
        
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
