"""
budgetモードの検証：銘柄数が減ってもパフォーマンスは維持されるか
=================================================================
「上位から1口ずつ・予算が尽きたら打ち切り」方式の性能を
フルN銘柄（等ウェイト）と比較する。

検証内容:
  - 上位3銘柄ロング vs 上位5銘柄ロング の成績比較
  - 必要最低資金の目安算出（各期間の銘柄価格から）

実行:
  python -X utf8 test/test_budget_mode.py
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
    US_TICKERS, JP_TICKERS, JP_NAMES,
    L, K, LAMBDA, TRAIN_START, TRAIN_END,
)
from test_backtest import fetch_open_prices, calc_metrics

# ---------------------------------------------------------------------------
# バックテスト（上位N銘柄ロングのみ・等ウェイト）
# ---------------------------------------------------------------------------

def backtest_top_n(avail_us, avail_jp, C0, cc_ret, oc_jp,
                   n_long: int, eval_start="2015-01-01") -> pd.Series:
    """
    シグナル上位 n_long 銘柄を等ウェイトでロングする戦略のリターン系列を返す。
    budgetモード（上から順に買う）と等価：買える銘柄数 = n_long と固定した場合。
    """
    all_av      = avail_us + avail_jp
    eval_start_ts = pd.Timestamp(eval_start)
    eval_dates  = cc_ret.index[cc_ret.index >= eval_start_ts]

    rets, dates = [], []
    for signal_date in eval_dates:
        loc = cc_ret.index.get_loc(signal_date)
        if loc < L:
            continue
        window    = cc_ret.iloc[loc - L: loc][all_av]
        next_days = oc_jp.index[oc_jp.index > signal_date]
        if len(next_days) == 0:
            continue
        next_date = next_days[0]
        if next_date not in oc_jp.index:
            continue

        V_K, mu_w, sigma_w = compute_reg_pca(window.values, C0)
        us_cc  = cc_ret.loc[signal_date, avail_us].values
        mu_us  = mu_w[:len(avail_us)]
        sig_us = np.where(sigma_w[:len(avail_us)] < 1e-10, 1.0, sigma_w[:len(avail_us)])
        z_us   = (us_cc - mu_us) / sig_us
        sig    = compute_signal(z_us, V_K, len(avail_us))

        # 上位 n_long 銘柄のインデックス
        top_idx   = np.argsort(sig)[-n_long:]
        oc_next   = oc_jp.loc[next_date, avail_jp].values
        r_long    = np.nanmean(oc_next[top_idx])

        if np.isfinite(r_long):
            rets.append(r_long)
            dates.append(next_date)

    return pd.Series(rets, index=pd.DatetimeIndex(dates))


# ---------------------------------------------------------------------------
# 必要最低資金の試算
# ---------------------------------------------------------------------------

def estimate_min_budget(avail_jp, close_jp, n_long: int,
                        eval_start="2015-01-01") -> pd.DataFrame:
    """
    各営業日に「上位シグナルn_long銘柄を1口ずつ買うのに必要な最低資金」を算出する。
    実際には翌日Openで買うため、前日Closeを代理として使用。
    """
    eval_start_ts = pd.Timestamp(eval_start)
    prices = close_jp[avail_jp][close_jp.index >= eval_start_ts].dropna(how="all")

    # 各日、全銘柄の価格上位n_long個の合計（最悪ケース：最高値銘柄を買う場合）
    records = []
    for date, row in prices.iterrows():
        valid = row.dropna().sort_values(ascending=False)
        if len(valid) >= n_long:
            worst_case  = int(valid.iloc[:n_long].sum())   # 高値n銘柄の合計
            best_case   = int(valid.iloc[-n_long:].sum())  # 安値n銘柄の合計
            median_case = int(valid.sample(n_long, random_state=0).sum())
        else:
            worst_case = best_case = median_case = None
        records.append({"date": date, "最高値ケース": worst_case,
                        "中央値ケース": median_case, "最安値ケース": best_case})

    df = pd.DataFrame(records).set_index("date").dropna()
    return df


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 62)
    print("  budgetモード検証：上位N銘柄ロングのパフォーマンス比較")
    print("=" * 62)

    # データ準備
    print("\nデータ取得中...")
    train_us = fetch_close_prices(US_TICKERS, TRAIN_START, TRAIN_END)
    train_jp = fetch_close_prices(JP_TICKERS, TRAIN_START, TRAIN_END)

    min_cov  = int(len(train_us) * 0.5)
    avail_us = [t for t in US_TICKERS
                if t in train_us.columns and train_us[t].notna().sum() >= min_cov]
    avail_jp = [t for t in JP_TICKERS if t in train_jp]
    all_av   = avail_us + avail_jp

    combined  = pd.concat([train_us[avail_us], train_jp[avail_jp]], axis=1)
    combined  = combined.dropna(thresh=int(len(all_av)*0.8)).ffill().dropna()
    train_ret = combined.pct_change().dropna()
    V0 = build_prior_subspace(avail_us, avail_jp)
    C0 = build_prior_exposure(V0, train_ret)

    close_us = fetch_close_prices(avail_us, "2014-07-01", "2025-12-31")
    close_jp = fetch_close_prices(avail_jp, "2014-07-01", "2025-12-31")
    open_jp  = fetch_open_prices(avail_jp,  "2014-07-01", "2025-12-31")
    close_all = pd.concat([close_us, close_jp[avail_jp]], axis=1)
    close_all = close_all.dropna(thresh=int(len(all_av)*0.7)).ffill().dropna()
    cc_ret    = close_all.pct_change().dropna()
    oc_jp     = (close_jp[avail_jp] / open_jp[avail_jp] - 1).dropna(how="all")

    # ── 上位N銘柄ロング成績比較 ──────────────────────────────
    print("\nバックテスト実行中...")
    print("\n  ┌──────────────────────────────────────────────────────┐")
    print("  │   上位N銘柄ロング 成績比較（ロングのみ・等ウェイト）   │")
    print("  ├────────┬──────────┬──────────┬──────┬────────────────┤")
    print("  │ 銘柄数 │ AR(%)    │ RISK(%)  │  R/R │ MDD(%)         │")
    print("  ├────────┼──────────┼──────────┼──────┼────────────────┤")

    results = {}
    for n in [1, 2, 3, 4, 5]:
        ret = backtest_top_n(avail_us, avail_jp, C0, cc_ret, oc_jp, n_long=n)
        m   = calc_metrics(ret)
        results[n] = m
        marker = " ◀ 5銘柄（論文）" if n == 5 else ""
        print(f"  │  上位{n}銘柄 │ {m['AR']:>8.2f} │ {m['RISK']:>8.2f} │ "
              f"{m['RR']:>4.2f} │ {m['MDD']:>8.2f}%      │{marker}")

    print("  └────────┴──────────┴──────────┴──────┴────────────────┘")

    # ── 上位3銘柄 vs 5銘柄 の詳細比較 ──────────────────────
    r3 = results[3]["RR"]
    r5 = results[5]["RR"]
    diff_pct = (r3 - r5) / r5 * 100

    print(f"""
  【判定】上位3銘柄ロング の R/R は上位5銘柄の {r3/r5*100:.0f}% 水準
  　→ 銘柄数を5→3に減らしても R/R の低下は {abs(diff_pct):.0f}% 程度
  　→ {'許容範囲内' if abs(diff_pct) < 30 else '影響が大きいため注意が必要'}
    """)

    # ── 必要最低資金の試算 ──────────────────────────────────
    print("─" * 62)
    print("  必要最低資金の試算（1口ずつ・2015〜2025年の実績価格ベース）")
    print("─" * 62)

    for n in [3, 4, 5]:
        budget_df = estimate_min_budget(avail_jp, close_jp, n)
        print(f"\n  上位{n}銘柄を1口ずつ買う場合:")
        print(f"    最低でも必要な資金（期間中の最大値）:")
        print(f"      最高値ケース（最も高い{n}銘柄を買う）: "
              f"{budget_df['最高値ケース'].max():>10,} 円")
        print(f"      中央値ケース（ランダム{n}銘柄）    : "
              f"{budget_df['中央値ケース'].median():>10,.0f} 円（中央値）")
        print(f"      最安値ケース（最も安い{n}銘柄を買う）: "
              f"{budget_df['最安値ケース'].max():>10,} 円")
        print(f"    → 余裕を持った推奨資金（最高値ケース+20%）: "
              f"{int(budget_df['最高値ケース'].max() * 1.2):,} 円")

    print("\n" + "=" * 62)
    print("  検証完了")
    print("=" * 62)
