"""
Step 3: シグナル品質の診断テスト
=================================
バックテスト乖離の原因を切り分け、実売買への影響を判断するための分析。

確認項目:
  Q1. シグナルと翌日日本リターンの相関は正か？（方向性の検証）
  Q2. Open価格の品質は問題ないか？（データ品質）
  Q3. 評価期間を合わせると数値は改善するか？（期間効果）
  Q4. 取引コストを考慮してもプラスか？（実取引への影響）

実行方法:
  python -X utf8 test/test_signal_quality.py
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
from test_backtest import fetch_open_prices, run_pca_sub_backtest, calc_metrics

PASS_MARK = "✅"
FAIL_MARK = "❌"
WARN_MARK = "⚠️ "

def check(label, cond, detail=""):
    mark = PASS_MARK if cond else FAIL_MARK
    print(f"  {mark} {label}")
    if detail:
        print(f"     → {detail}")
    return cond


# ---------------------------------------------------------------------------
# データ準備
# ---------------------------------------------------------------------------

def prepare(bt_start="2014-07-01", bt_end="2025-12-31"):
    print("データ準備中...")
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

    close_us = fetch_close_prices(avail_us, bt_start, bt_end)
    close_jp = fetch_close_prices(avail_jp, bt_start, bt_end)
    open_jp  = fetch_open_prices(avail_jp, bt_start, bt_end)

    close_all = pd.concat([close_us, close_jp[avail_jp]], axis=1)
    close_all = close_all.dropna(thresh=int(len(all_av)*0.7)).ffill().dropna()
    cc_ret    = close_all.pct_change().dropna()
    oc_jp     = (close_jp[avail_jp] / open_jp[avail_jp] - 1).dropna(how="all")

    return avail_us, avail_jp, C0, cc_ret, oc_jp, close_jp, open_jp


# ---------------------------------------------------------------------------
# Q1. シグナルと翌日リターンの予測相関（IC分析）
# ---------------------------------------------------------------------------

def q1_information_coefficient(avail_us, avail_jp, C0, cc_ret, oc_jp):
    """
    IC (Information Coefficient) = シグナルと翌日実現リターンのSpearman相関。
    IC > 0 ならシグナルの方向性が正しい。
    論文が主張するリードラグが存在するなら IC > 0 のはず。
    """
    print("\n── Q1. シグナルの予測力（IC分析）─────────────────────")
    print("  IC = シグナルと翌日 Open-to-Close リターンの Spearman 相関")
    print("  IC > 0 → 方向性が正しい  |  IC > 0.05 → 実用的に有意")

    all_av    = avail_us + avail_jp
    eval_start = pd.Timestamp("2015-01-01")
    eval_dates = cc_ret.index[cc_ret.index >= eval_start]

    ics = []
    for signal_date in eval_dates:
        loc = cc_ret.index.get_loc(signal_date)
        if loc < L:
            continue
        window = cc_ret.iloc[loc - L: loc][all_av]

        next_days = oc_jp.index[oc_jp.index > signal_date]
        if len(next_days) == 0:
            continue
        next_date = next_days[0]

        V_K, mu_w, sigma_w = compute_reg_pca(window.values, C0)
        us_cc  = cc_ret.loc[signal_date, avail_us].values
        mu_us  = mu_w[:len(avail_us)]
        sig_us = np.where(sigma_w[:len(avail_us)] < 1e-10, 1.0, sigma_w[:len(avail_us)])
        z_us   = (us_cc - mu_us) / sig_us
        sig    = compute_signal(z_us, V_K, len(avail_us))

        oc_next = oc_jp.loc[next_date, avail_jp].values
        if np.isfinite(sig).all() and np.isfinite(oc_next).all():
            from scipy.stats import spearmanr
            ic, _ = spearmanr(sig, oc_next)
            if np.isfinite(ic):
                ics.append(ic)

    ics = np.array(ics)
    mean_ic = ics.mean()
    std_ic  = ics.std()
    icir    = mean_ic / std_ic if std_ic > 0 else 0  # IC Information Ratio
    pct_pos = (ics > 0).mean() * 100

    print(f"\n  平均IC    : {mean_ic:+.4f}  (論文が正しければ > 0)")
    print(f"  IC標準偏差: {std_ic:.4f}")
    print(f"  ICIR      : {icir:.3f}  (> 0.3 で実用的に有意とされる)")
    print(f"  IC > 0 の日: {pct_pos:.1f}%  (> 50% で方向性が正しい)")
    print(f"  サンプル数: {len(ics)}日")

    check("平均IC > 0（シグナルが翌日リターンと正の相関）",
          mean_ic > 0,
          f"IC = {mean_ic:.4f}")
    check("IC > 0 の日が 50% 超（方向性が半数以上で正しい）",
          pct_pos > 50,
          f"{pct_pos:.1f}%")
    check("ICIR > 0.3（実用的な予測力の目安）",
          icir > 0.3,
          f"ICIR = {icir:.3f}")

    return mean_ic, icir


# ---------------------------------------------------------------------------
# Q2. Open 価格の品質確認
# ---------------------------------------------------------------------------

def q2_open_price_quality(avail_jp, close_jp, open_jp):
    """
    yfinance の Open 価格が異常でないかを確認する。
    Open = 0 や Open/Close 比率が極端な日は Open 価格が信頼できない。
    """
    print("\n── Q2. Open 価格の品質確認 ──────────────────────────")
    print("  yfinance の日本ETF Open 価格の信頼性を検証する")

    issues = {}
    for t in avail_jp:
        if t not in open_jp.columns or t not in close_jp.columns:
            continue
        o = open_jp[t].dropna()
        c = close_jp[t].reindex(o.index).dropna()
        common = o.index.intersection(c.index)
        o, c = o[common], c[common]

        # Open = 0 または負の値
        zero_open = (o <= 0).sum()
        # Open と Close の乖離が 10% 超（異常値の疑い）
        ratio = (o / c - 1).abs()
        extreme = (ratio > 0.10).sum()
        # Open が前日 Close と同じ（実は Open が Close で代替されている疑い）
        prev_close = c.shift(1).reindex(o.index)
        same_as_prev_close = (np.abs(o - prev_close) < 1e-6).sum()
        pct_same = same_as_prev_close / len(o) * 100

        if zero_open > 0 or extreme > 5 or pct_same > 20:
            issues[t] = {
                "zero_open": zero_open,
                "extreme_ratio": extreme,
                "same_as_prev_close(%)": round(pct_same, 1),
            }

    if not issues:
        print(f"  {PASS_MARK} 全 {len(avail_jp)} 銘柄の Open 価格に異常なし")
    else:
        print(f"  {WARN_MARK} Open 価格に疑わしい銘柄が {len(issues)} 件あります:")
        for t, info in issues.items():
            name = JP_NAMES.get(t, t)
            print(f"     {t}({name}): {info}")
        print()
        print("  ⚠️  「前日Close = 当日Open」の割合が高い場合、")
        print("     Open価格がClose価格で代替されており、")
        print("     Open-to-Close リターンが過小評価されている可能性があります。")

    # Open-to-Close リターンの統計
    oc = (close_jp[avail_jp] / open_jp[avail_jp] - 1).dropna(how="all")
    mean_oc = oc.mean().mean() * 252 * 100
    print(f"\n  日本ETF の年率換算 平均 Open-to-Close リターン: {mean_oc:.2f}%")
    check("Open-to-Close リターンが正（市場全体が上昇トレンド）",
          mean_oc > 0,
          f"{mean_oc:.2f}% / 年")

    return issues


# ---------------------------------------------------------------------------
# Q3. 評価期間別パフォーマンス（期間効果の確認）
# ---------------------------------------------------------------------------

def q3_period_performance(avail_us, avail_jp, C0, cc_ret, oc_jp):
    """
    論文は 2010-2025 年全体だが、本コードは 2015 年以降。
    期間ごとの成績を分解して、どの期間でアルファが出ているかを確認する。
    """
    print("\n── Q3. 期間別パフォーマンス ────────────────────────")
    print("  論文のサンプル期間（2010-2025）との差異を確認する")

    all_av     = avail_us + avail_jp
    Q          = 0.3
    periods    = [
        ("2015-2018", "2015-01-01", "2018-12-31"),
        ("2019-2021", "2019-01-01", "2021-12-31"),
        ("2022-2025", "2022-01-01", "2025-12-31"),
    ]

    print(f"\n  {'期間':<12} {'AR(%)':>8} {'RISK(%)':>8} {'R/R':>6} {'MDD(%)':>8}")
    print("  " + "-" * 48)

    for label, start, end in periods:
        start_ts = pd.Timestamp(start)
        end_ts   = pd.Timestamp(end)
        period_dates = cc_ret.index[
            (cc_ret.index >= start_ts) & (cc_ret.index <= end_ts)
        ]

        rets = []
        for signal_date in period_dates:
            loc = cc_ret.index.get_loc(signal_date)
            if loc < L:
                continue
            window = cc_ret.iloc[loc - L: loc][all_av]
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

            n   = len(sig)
            n_q = max(1, int(n * Q))
            idx = np.argsort(sig)
            long_idx  = idx[-n_q:]
            short_idx = idx[:n_q]

            oc_next = oc_jp.loc[next_date, avail_jp].values
            r = np.nanmean(oc_next[long_idx]) - np.nanmean(oc_next[short_idx])
            if np.isfinite(r):
                rets.append(r)

        if len(rets) < 10:
            print(f"  {label:<12} データ不足")
            continue

        s = pd.Series(rets)
        m = calc_metrics(s)
        print(f"  {label:<12} {m['AR']:>8.2f} {m['RISK']:>8.2f} {m['RR']:>6.2f} {m['MDD']:>8.2f}")


# ---------------------------------------------------------------------------
# Q4. 取引コスト考慮後のパフォーマンス
# ---------------------------------------------------------------------------

def q4_after_cost(ret_series: pd.Series):
    """
    実売買では取引コストが発生する。
    日本ETFの現実的なコスト水準を考慮したパフォーマンスを計算する。
    """
    print("\n── Q4. 取引コスト考慮後のパフォーマンス ────────────")
    print("  日本ETFの現実的な取引コスト（片道）を考慮する")
    print("  ロング5銘柄・ショート5銘柄を毎日入れ替える想定")

    # 現実的なコスト見積もり
    # TOPIX-17 ETFの流動性は低め、スプレッド+手数料で片道0.05～0.1%
    cost_scenarios = {
        "低コスト (片道0.03%)": 0.0003,
        "中コスト (片道0.05%)": 0.0005,
        "高コスト (片道0.10%)": 0.0010,
    }

    # 毎日全銘柄入れ替え: ロング+ショート両側で 2 × 片道コスト
    base = calc_metrics(ret_series)
    print(f"\n  コストなし: AR={base['AR']:.2f}%, R/R={base['RR']:.2f}")
    print(f"\n  {'シナリオ':<28} {'AR(%)':>8} {'R/R':>6} {'判定'}")
    print("  " + "-" * 50)

    for name, cost_one_way in cost_scenarios.items():
        daily_cost = cost_one_way * 2   # ロング+ショートの両側
        ret_after  = ret_series - daily_cost
        m          = calc_metrics(ret_after)
        viable     = m["RR"] > 0.5 and m["AR"] > 0
        mark       = PASS_MARK if viable else FAIL_MARK
        print(f"  {name:<28} {m['AR']:>8.2f} {m['RR']:>6.2f} {mark}")

    print("\n  ※ショートが難しい場合（ロングのみ）:")
    # ロングのみの場合: 毎日5銘柄入れ替え、実際は部分入れ替えになる
    # ターンオーバーを50%と仮定
    for name, cost_one_way in cost_scenarios.items():
        # ロングのみ、ターンオーバー50%/日
        daily_cost = cost_one_way * 0.5
        ret_long_only = ret_series / 2 + 0  # ロングのみ（ショート除く近似）
        ret_after = ret_long_only - daily_cost
        m = calc_metrics(ret_after)
        viable = m["RR"] > 0.3 and m["AR"] > 0
        mark   = PASS_MARK if viable else FAIL_MARK
        print(f"  {name:<28} {m['AR']:>8.2f} {m['RR']:>6.2f} {mark}")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 62)
    print("  Step 3: シグナル品質の診断（実売買への影響評価）")
    print("=" * 62)

    avail_us, avail_jp, C0, cc_ret, oc_jp, close_jp, open_jp = prepare()

    # Q1: IC 分析
    mean_ic, icir = q1_information_coefficient(
        avail_us, avail_jp, C0, cc_ret, oc_jp
    )

    # Q2: Open 価格品質
    issues = q2_open_price_quality(avail_jp, close_jp, open_jp)

    # Q3: 期間別パフォーマンス
    q3_period_performance(avail_us, avail_jp, C0, cc_ret, oc_jp)

    # Q4: 取引コスト考慮
    print("\n取引コスト分析のためバックテスト再実行中...")
    ret_series = run_pca_sub_backtest(avail_us, avail_jp, C0)
    q4_after_cost(ret_series)

    # 総合判断
    print("\n" + "=" * 62)
    print("  総合判断")
    print("=" * 62)
    print(f"  IC = {mean_ic:+.4f}, ICIR = {icir:.3f}")

    if mean_ic > 0.02 and icir > 0.3:
        print("""
  ✅ シグナルの方向性は統計的に有意です。
     バックテスト数値の乖離はデータソースの差によるもので、
     シグナル自体の品質には問題ない可能性が高いです。
     ただし実売買では取引コストと流動性リスクを要確認。
        """)
    elif mean_ic > 0:
        print("""
  ⚠️  シグナルは正の方向性を持ちますが、統計的有意性は限定的です。
     論文と同等のパフォーマンスが得られるかは不確かです。
     少額での実運用検証（ペーパートレード）を先に行うことを推奨します。
        """)
    else:
        print("""
  ❌ シグナルの方向性が確認できませんでした。
     論文の実装か入力データに問題がある可能性があります。
     実売買への適用は推奨しません。
        """)

    if issues:
        print(f"""
  ⚠️  Open 価格に {len(issues)} 件の品質問題があります。
     これがバックテスト乖離の主因である可能性があります。
     より正確なデータソース（SBI/楽天証券APIなど）の利用を検討してください。
        """)
