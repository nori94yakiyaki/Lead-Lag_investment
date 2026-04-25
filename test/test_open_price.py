"""
Open 価格の異常値診断
====================
yfinance の日本ETF Open 価格の具体的な異常日・異常銘柄を特定する。

確認内容:
  1. 調整済み vs 未調整 Open/Close の比較
  2. Open-to-Close リターンが極端に大きい/小さい日の一覧
  3. 「当日Open = 前日Close」になっている日の一覧（代替疑い）
  4. 決算・配当落ち日の影響確認

実行:
  python -X utf8 test/test_open_price.py
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

from generate_portfolio import JP_TICKERS, JP_NAMES

DIAG_START = "2020-01-01"
DIAG_END   = "2025-12-31"

# ── データ取得 ──────────────────────────────────────────────

def download_both(tickers, start, end):
    """調整済み・未調整の両方を取得して返す。"""
    adj = yf.download(tickers, start=start, end=end,
                      auto_adjust=True,  progress=False)
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=False, progress=False)

    def flatten(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(c).strip() for c in df.columns]
        return df

    return flatten(adj), flatten(raw)


# ── 診断1: 調整 Open と未調整 Open の乖離 ───────────────────

def diag_adj_vs_raw(adj, raw, ticker):
    """
    auto_adjust=True の Open と auto_adjust=False の Open を比較する。
    乖離が大きい場合、調整の掛け方に問題がある可能性。
    """
    t = ticker
    adj_o_col = f"Open_{t}" if f"Open_{t}" in adj.columns else None
    adj_c_col = f"Close_{t}" if f"Close_{t}" in adj.columns else None
    raw_o_col = f"Open_{t}" if f"Open_{t}" in raw.columns else None
    raw_c_col = f"Close_{t}" if f"Close_{t}" in raw.columns else None

    if not all([adj_o_col, adj_c_col, raw_o_col, raw_c_col]):
        return None

    df = pd.DataFrame({
        "adj_open":  adj[adj_o_col],
        "adj_close": adj[adj_c_col],
        "raw_open":  raw[raw_o_col],
        "raw_close": raw[raw_c_col],
    }).dropna()

    # 調整係数 = adj / raw
    df["ratio_open"]  = df["adj_open"]  / df["raw_open"]
    df["ratio_close"] = df["adj_close"] / df["raw_close"]
    # 調整係数の Open/Close 乖離（これが大きいほど Open の調整がおかしい）
    df["ratio_diff"]  = (df["ratio_open"] - df["ratio_close"]).abs()

    # OC リターン（調整済み vs 未調整）
    df["oc_adj"] = df["adj_close"] / df["adj_open"] - 1
    df["oc_raw"] = df["raw_close"] / df["raw_open"] - 1

    return df


# ── 診断2: Open-to-Close リターンの異常日特定 ───────────────

def diag_extreme_oc(df, ticker, threshold=0.03):
    """
    |OC リターン| > threshold の日を列挙する。
    """
    name   = JP_NAMES.get(ticker, ticker)
    oc_adj = df["oc_adj"].dropna()

    extreme = oc_adj[oc_adj.abs() > threshold].sort_values()
    return extreme


# ── 診断3: 「当日Open ≒ 前日Close」の日を検出 ───────────────

def diag_open_equals_prev_close(df, ticker, tol=1e-4):
    """
    当日の adj_open が前日の adj_close と一致している日を検出する。
    yfinance がデータ欠損を前日 Close で埋めている疑い。
    """
    prev_close = df["adj_close"].shift(1)
    mask = (df["adj_open"] - prev_close).abs() / prev_close.abs() < tol
    return df[mask].index


# ── 診断4: OC リターン分布と年率換算 ────────────────────────

def diag_oc_distribution(df, ticker):
    oc  = df["oc_adj"].dropna()
    ann = oc.mean() * 252 * 100
    med = oc.median() * 100
    pos = (oc > 0).mean() * 100
    return {
        "年率平均OC(%)": round(ann, 2),
        "中央値OC(%)":   round(med * 252, 2),
        "正の日(%)":     round(pos, 1),
        "標準偏差":      round(oc.std() * 100, 3),
    }


# ── メイン ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  Open 価格の異常値診断")
    print(f"  期間: {DIAG_START} 〜 {DIAG_END}")
    print("=" * 70)

    print("\nデータ取得中（調整済み・未調整の両方）...")
    adj, raw = download_both(JP_TICKERS, DIAG_START, DIAG_END)

    # ─── サマリテーブル ──────────────────────────────────────
    print("\n" + "─" * 70)
    print("【1】 銘柄別 Open-to-Close リターン サマリ")
    print("─" * 70)
    print(f"  {'ティッカー':<10} {'業種名':<22} {'年率OC%':>8} "
          f"{'中央値OC%':>10} {'正の日%':>8} {'異常疑い'}")
    print("  " + "-" * 68)

    summary_rows = []
    for t in JP_TICKERS:
        df = diag_adj_vs_raw(adj, raw, t)
        if df is None or len(df) < 10:
            continue
        stats = diag_oc_distribution(df, t)
        ann   = stats["年率平均OC(%)"]
        med   = stats["中央値OC(%)"]
        pos   = stats["正の日(%)"]

        # 異常フラグ: 年率OC が -5% 未満 または 中央値が異常
        flag = "⚠️  要確認" if ann < -5 or pos < 40 else ""
        name = JP_NAMES.get(t, t)
        print(f"  {t:<10} {name:<22} {ann:>8.2f} {med:>10.2f} {pos:>8.1f}  {flag}")
        summary_rows.append((t, name, ann, med, pos, df))

    # ─── 調整係数の乖離（Open vs Close）────────────────────────
    print("\n" + "─" * 70)
    print("【2】 調整係数の Open/Close 乖離（大きいほど Open 調整がおかしい）")
    print("─" * 70)
    print(f"  {'ティッカー':<10} {'業種名':<22} {'平均乖離':>10} "
          f"{'最大乖離日':<14} {'最大乖離値':>10}")
    print("  " + "-" * 68)

    for t, name, ann, med, pos, df in summary_rows:
        worst_date  = df["ratio_diff"].idxmax()
        worst_val   = df["ratio_diff"].max()
        mean_diff   = df["ratio_diff"].mean()
        flag        = "⚠️ " if mean_diff > 0.001 else ""
        print(f"  {t:<10} {name:<22} {mean_diff:>10.6f} "
              f"  {str(worst_date.date()):<14} {worst_val:>10.6f}  {flag}")

    # ─── 「Open = 前日Close」の日数 ────────────────────────────
    print("\n" + "─" * 70)
    print("【3】 「当日Open ≈ 前日Close」の日（Open が前日 Close で代替疑い）")
    print("─" * 70)
    print(f"  {'ティッカー':<10} {'業種名':<22} {'該当日数':>8} "
          f"{'該当率%':>8} {'最近の該当日'}")
    print("  " + "-" * 68)

    for t, name, ann, med, pos, df in summary_rows:
        suspect = diag_open_equals_prev_close(df, t)
        pct     = len(suspect) / len(df) * 100
        latest  = str(suspect[-1].date()) if len(suspect) > 0 else "なし"
        flag    = "⚠️ " if pct > 5 else ""
        print(f"  {t:<10} {name:<22} {len(suspect):>8d} "
              f"{pct:>8.1f}  {latest}  {flag}")

    # ─── 極端な OC リターン日の具体的なリスト ──────────────────
    print("\n" + "─" * 70)
    print("【4】 銘柄ごとの極端な OC リターン日（|OC| > 3%）")
    print("─" * 70)

    all_extreme = []
    for t, name, ann, med, pos, df in summary_rows:
        extreme = diag_extreme_oc(df, t, threshold=0.03)
        for date, oc_val in extreme.items():
            all_extreme.append({
                "日付":     date.date(),
                "ティッカー": t,
                "業種名":   name,
                "adj_Open":  round(df.loc[date, "adj_open"], 2),
                "adj_Close": round(df.loc[date, "adj_close"], 2),
                "raw_Open":  round(df.loc[date, "raw_open"], 2),
                "raw_Close": round(df.loc[date, "raw_close"], 2),
                "OC_adj(%)": round(oc_val * 100, 3),
                "OC_raw(%)": round(df.loc[date, "oc_raw"] * 100, 3),
            })

    if all_extreme:
        extreme_df = pd.DataFrame(all_extreme).sort_values("OC_adj(%)")
        print(f"\n  最もマイナス側 20 件（Open が Close より大幅に高い日）:")
        print(extreme_df.head(20).to_string(index=False))
        print(f"\n  最もプラス側 10 件（参考）:")
        print(extreme_df.tail(10).to_string(index=False))

        # CSV 保存
        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "extreme_oc_dates.csv"
        )
        extreme_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n  → 全 {len(extreme_df)} 件を {out_path} に保存しました（Excel で開けます）")
    else:
        print("  極端な OC リターン日は検出されませんでした。")

    # ─── 結論 ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  診断結論")
    print("=" * 70)

    neg_oc_tickers = [t for t, name, ann, *_ in summary_rows if ann < -5]
    if neg_oc_tickers:
        print(f"\n  ⚠️  年率 Open-to-Close < -5% の銘柄: {neg_oc_tickers}")
        print("""
  これは以下のいずれかを意味します:
  (A) yfinance の Open 価格が配当落ち調整されていない
      → Close は調整済みだが Open は未調整 → OC が系統的にマイナス
  (B) 流動性が低く、寄付きで前日比大幅高から始まり日中に戻る
  (C) yfinance が Open データ欠損を前日 Close（高値）で補完している

  【目視確認すべき日】
  上記【4】の「OC_adj が最もマイナスな日」と「raw_Open ≠ adj_Open な日」を
  証券会社のチャートや日経データなどで実際の寄付き値と比較してください。
        """)
    else:
        print("\n  Open 価格に系統的な異常は見られません。")
