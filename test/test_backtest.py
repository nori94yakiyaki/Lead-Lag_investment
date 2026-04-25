"""
Step 2: バックテスト再現テスト
==============================
論文 表2（PCA SUB）の数値を再現し、実装の正確性を確認する。

論文の数値（表2 PCA SUB）:
  AR   = 23.79%
  RISK = 10.70%
  R/R  = 2.22
  MDD  = 9.58%

サンプル期間: 2010-01-01 〜 2025-12-31
ロング/ショート: 上位・下位 q=0.3（3分位）

実行方法:
  python -X utf8 test/test_backtest.py

注意: yfinance のデータと論文のデータソースには差異がある場合があります。
      ±5pp 程度の乖離は許容範囲とします。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from generate_portfolio import (
    build_prior_subspace, build_prior_exposure,
    compute_reg_pca, compute_signal,
    fetch_close_prices,
    US_TICKERS, JP_TICKERS,
    L, K, LAMBDA,
    TRAIN_START, TRAIN_END,
)

# ---------------------------------------------------------------------------
# 論文 表2 の参照値（PCA SUB）
# ---------------------------------------------------------------------------
PAPER = {
    "AR":   23.79,   # 年率リターン (%)
    "RISK": 10.70,   # 年率リスク (%)
    "RR":    2.22,   # R/R = AR/RISK
    "MDD":   9.58,   # 最大ドローダウン (%)
}
TOLERANCE = 5.0      # 許容誤差 (pp)

# ---------------------------------------------------------------------------
# バックテスト用 Open データ取得
# ---------------------------------------------------------------------------

def fetch_open_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)["Open"]
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    available = [t for t in tickers if t in raw.columns]
    return raw[available].sort_index()


# ---------------------------------------------------------------------------
# バックテスト本体
# ---------------------------------------------------------------------------

def run_pca_sub_backtest(avail_us, avail_jp, C0,
                          bt_start="2014-07-01",
                          bt_end="2025-12-31",
                          eval_start="2015-01-01") -> pd.Series:
    """
    PCA SUB 戦略のバックテストを実行し、日次戦略リターン系列を返す。

    論文 2.2節 ロング/ショートポートフォリオの構築に準拠:
      - シグナル上位 q=0.3 → ロング（等ウェイト +1/|L|）
      - シグナル下位 q=0.3 → ショート（等ウェイト -1/|S|）
      - 戦略リターン Rt+1 = sum_j w_j * r^oc_j,t+1  （式(7)）
    """
    all_av = avail_us + avail_jp
    Q      = 0.3

    print("  終値（Close）データ取得中...")
    close_us = fetch_close_prices(avail_us, bt_start, bt_end)
    close_jp = fetch_close_prices(avail_jp, bt_start, bt_end)

    print("  始値（Open）データ取得中...")
    open_jp = fetch_open_prices(avail_jp, bt_start, bt_end)

    # Close-to-Close リターン（PCA の推定に使用, 論文 式(1)）
    close_all = pd.concat([close_us[avail_us], close_jp[avail_jp]], axis=1)
    close_all = close_all.dropna(thresh=int(len(all_av) * 0.7)).ffill().dropna()
    cc_ret    = close_all.pct_change().dropna()

    # Open-to-Close リターン（戦略評価用, 論文 式(2)）
    oc_jp = (close_jp[avail_jp] / open_jp[avail_jp] - 1).dropna(how="all")

    eval_start_ts = pd.Timestamp(eval_start)
    eval_dates    = cc_ret.index[cc_ret.index >= eval_start_ts]

    strategy_returns = []
    signal_dates_out = []

    print(f"  ローリングバックテスト実行中（{len(eval_dates)}日分）...")
    for signal_date in eval_dates:
        loc = cc_ret.index.get_loc(signal_date)
        if loc < L:
            continue

        window = cc_ret.iloc[loc - L: loc][all_av]

        # 翌営業日（日本市場のOpen-to-Close を評価する日）
        next_days = oc_jp.index[oc_jp.index > signal_date]
        if len(next_days) == 0:
            continue
        next_date = next_days[0]
        if next_date not in oc_jp.index:
            continue

        # 正則化 PCA（論文 3.2節 式(13)(14)(15)(16)）
        V_K, mu_w, sigma_w = compute_reg_pca(window.values, C0)

        # 米国当日標準化リターン（論文 3.3節 式(17)）
        us_cc  = cc_ret.loc[signal_date, avail_us].values
        mu_us  = mu_w[:len(avail_us)]
        sig_us = np.where(sigma_w[:len(avail_us)] < 1e-10, 1.0, sigma_w[:len(avail_us)])
        z_us   = (us_cc - mu_us) / sig_us

        # リードラグシグナル（論文 3.3節 式(18)(19)）
        sig = compute_signal(z_us, V_K, len(avail_us))

        # ロング/ショート構築（論文 2.2節 式(3)(4)(5)）
        n   = len(sig)
        n_q = max(1, int(n * Q))
        idx = np.argsort(sig)
        long_idx  = idx[-n_q:]    # 上位 q
        short_idx = idx[:n_q]     # 下位 q

        oc_next = oc_jp.loc[next_date, avail_jp].values

        r_long  = np.nanmean(oc_next[long_idx])
        r_short = np.nanmean(oc_next[short_idx])
        r_strat = r_long - r_short          # 式(7) に対応

        if np.isfinite(r_strat):
            strategy_returns.append(r_strat)
            signal_dates_out.append(next_date)

    return pd.Series(strategy_returns, index=pd.DatetimeIndex(signal_dates_out))


# ---------------------------------------------------------------------------
# パフォーマンス指標の計算（論文 4.2節 式(27)(28)(29)(30)）
# ---------------------------------------------------------------------------

def calc_metrics(ret: pd.Series) -> dict:
    """論文 4.2節 の評価指標を計算する。"""
    # 年率リターン（式(27)）
    AR = ret.mean() * 252 * 100

    # 年率リスク（式(28)）
    RISK = ret.std() * np.sqrt(252) * 100

    # R/R（式(29)）
    RR = AR / RISK if RISK > 0 else 0.0

    # 最大ドローダウン（式(30)）
    cum         = (1 + ret).cumprod()
    rolling_max = cum.cummax()
    MDD         = abs((cum / rolling_max - 1).min()) * 100

    return {"AR": AR, "RISK": RISK, "RR": RR, "MDD": MDD}


# ---------------------------------------------------------------------------
# 結果比較・判定
# ---------------------------------------------------------------------------

def compare_results(ours: dict):
    print("\n  ┌──────────────────────────────────────────────────────┐")
    print("  │      バックテスト結果 vs 論文 表2（PCA SUB）          │")
    print("  ├──────────────┬──────────────┬──────────────┬─────────┤")
    print("  │ 指標         │ 本コード     │ 論文値       │ 差      │")
    print("  ├──────────────┼──────────────┼──────────────┼─────────┤")

    labels = {
        "AR":   "年率リターン(%)",
        "RISK": "年率リスク(%)",
        "RR":   "R/R",
        "MDD":  "最大DD(%)",
    }
    all_pass = True
    for key, label in labels.items():
        diff = ours[key] - PAPER[key]
        tol  = TOLERANCE if key != "RR" else TOLERANCE * 0.5
        ok   = abs(diff) < tol
        mark = "✅" if ok else "❌"
        print(f"  │ {label:<14}│ {ours[key]:>12.2f} │ {PAPER[key]:>12.2f} │{diff:>+8.2f} │ {mark}")
        if not ok:
            all_pass = False

    print("  └──────────────┴──────────────┴──────────────┴─────────┘")
    print(f"\n  許容誤差: AR/RISK/MDD ±{TOLERANCE}pp, R/R ±{TOLERANCE*0.5:.1f}")

    if all_pass:
        print("\n  ✅ バックテスト結果は論文と概ね一致しています。")
    else:
        print("\n  ⚠️  乖離があります。考えられる原因:")
        print("     1. データソース（yfinance vs 論文の Bloomberg/Refinitiv 等）の違い")
        print("     2. 日米祝日・共通営業日の取り扱いの差異")
        print("     3. 株式分割・配当調整方法の違い")
        print("     4. 評価開始年の差異（論文は 2010年〜）")

    return all_pass


# ---------------------------------------------------------------------------
# 追加検証: ベースライン（MOM, PCA PLAIN）との大小関係
# ---------------------------------------------------------------------------

def test_relative_performance(ret_pca_sub, avail_us, avail_jp, cc_ret, oc_jp):
    """
    論文の主張: PCA SUB > DOUBLE > MOM ≈ PCA PLAIN
    R/R の大小関係が正しいかを確認する。
    """
    print("\n── 相対パフォーマンスの大小関係確認 ───────────────────")
    print("  論文の主張: R/R は PCA SUB > DOUBLE > MOM ≈ PCA PLAIN")

    all_av = avail_us + avail_jp
    Q = 0.3
    eval_start_ts = pd.Timestamp("2015-01-01")
    eval_dates = cc_ret.index[cc_ret.index >= eval_start_ts]

    mom_ret = []
    mom_dates = []

    for signal_date in eval_dates:
        loc = cc_ret.index.get_loc(signal_date)
        if loc < L:
            continue

        window_jp = cc_ret.iloc[loc - L: loc][avail_jp]

        next_days = oc_jp.index[oc_jp.index > signal_date]
        if len(next_days) == 0:
            continue
        next_date = next_days[0]
        if next_date not in oc_jp.index:
            continue

        # MOM: 日本側のウィンドウ内平均（論文 4.3節 式(31)）
        mom_sig = window_jp.mean(axis=0).values
        n   = len(mom_sig)
        n_q = max(1, int(n * Q))
        idx = np.argsort(mom_sig)
        long_idx  = idx[-n_q:]
        short_idx = idx[:n_q]

        oc_next = oc_jp.loc[next_date, avail_jp].values
        r = np.nanmean(oc_next[long_idx]) - np.nanmean(oc_next[short_idx])
        if np.isfinite(r):
            mom_ret.append(r)
            mom_dates.append(next_date)

    mom_series  = pd.Series(mom_ret, index=pd.DatetimeIndex(mom_dates))
    mom_metrics = calc_metrics(mom_series)
    sub_metrics = calc_metrics(ret_pca_sub)

    print(f"\n  MOM    R/R = {mom_metrics['RR']:.2f}  (論文: 0.53)")
    print(f"  PCA SUB R/R = {sub_metrics['RR']:.2f}  (論文: 2.22)")

    rr_ok = sub_metrics["RR"] > mom_metrics["RR"]
    mark  = "✅" if rr_ok else "❌"
    print(f"\n  {mark} PCA SUB の R/R > MOM の R/R : {rr_ok}")
    print(f"     （論文の主要な主張: 正則化により R/R が大幅改善）")

    return rr_ok


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 62)
    print("  Step 2: バックテスト再現（論文 表2 との比較）")
    print("=" * 62)
    print(f"  論文参照値 (PCA SUB): AR={PAPER['AR']}%, RISK={PAPER['RISK']}%, "
          f"R/R={PAPER['RR']}, MDD={PAPER['MDD']}%")

    # データ・事前情報の準備
    print("\n訓練データ取得中...")
    train_us = fetch_close_prices(US_TICKERS, TRAIN_START, TRAIN_END)
    train_jp = fetch_close_prices(JP_TICKERS, TRAIN_START, TRAIN_END)

    avail_us = [t for t in US_TICKERS if t in train_us]
    avail_jp = [t for t in JP_TICKERS if t in train_jp]
    all_av   = avail_us + avail_jp

    combined  = pd.concat([train_us[avail_us], train_jp[avail_jp]], axis=1)
    combined  = combined.dropna(thresh=int(len(all_av) * 0.8)).ffill().dropna()
    train_ret = combined.pct_change().dropna()

    V0 = build_prior_subspace(avail_us, avail_jp)
    C0 = build_prior_exposure(V0, train_ret)

    # バックテスト実行
    print("\nバックテスト実行中（数分かかります）...")
    ret_series = run_pca_sub_backtest(avail_us, avail_jp, C0)
    print(f"  有効な戦略リターン日数: {len(ret_series)}日")

    if len(ret_series) < 100:
        print("❌ データ不足のためバックテストを完了できませんでした。")
        sys.exit(1)

    # パフォーマンス計算
    metrics = calc_metrics(ret_series)
    print(f"\n  年率リターン: {metrics['AR']:.2f}%")
    print(f"  年率リスク  : {metrics['RISK']:.2f}%")
    print(f"  R/R         : {metrics['RR']:.2f}")
    print(f"  最大DD      : {metrics['MDD']:.2f}%")

    # 論文との比較
    compare_results(metrics)

    # MOM との相対比較
    close_all = pd.concat(
        [fetch_close_prices(avail_us, "2014-07-01", "2025-12-31"),
         fetch_close_prices(avail_jp, "2014-07-01", "2025-12-31")], axis=1
    ).dropna(thresh=int(len(all_av) * 0.7)).ffill().dropna()
    cc_ret = close_all.pct_change().dropna()

    from generate_portfolio import fetch_close_prices as fcp
    close_jp_bt = fcp(avail_jp, "2014-07-01", "2025-12-31")
    open_jp_bt  = fetch_open_prices(avail_jp, "2014-07-01", "2025-12-31")
    oc_jp_bt    = (close_jp_bt[avail_jp] / open_jp_bt[avail_jp] - 1).dropna(how="all")

    test_relative_performance(ret_series, avail_us, avail_jp, cc_ret, oc_jp_bt)

    print("\n" + "=" * 62)
    print("  Step 2 完了")
    print("=" * 62)
