"""
日米業種リードラグ投資戦略 - セクターシグナル生成（6時〜9時実行用）
==========================================================================
論文: 中川慧 et al.「部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略」
      SIG-FIN-036-13, 人工知能学会金融情報学研究会

【動作概要】
  1. 前営業日の米国業種ETF終値リターン（Close-to-Close）を取得
  2. 部分空間正則化付きPCA（PCA SUB）でシグナルを計算
  3. 日本業種ETF 全17銘柄のシグナル強度を出力
     - 買いシグナル上位5銘柄（◆ BUY）：当日の追い風セクター
     - 売り・回避シグナル下位5銘柄（▼ SELL）：当日の向かい風セクター

【実行タイミング】
  日本市場寄付き前（6:00〜9:00 JST）に実行する
  → 前日確定した米国終値リターンを使用して当日の日本市場を予測
"""

import sys
import io
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Windows コンソールで UTF-8 出力を強制（スクリプトとして直接実行した場合のみ）
def _force_utf8():
    try:
        if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ===========================================================================
# パラメータ設定（論文に準拠）
# ===========================================================================

# 米国業種ETF（S&P500 Select Sector SPDR、11業種）
US_TICKERS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
US_NAMES = {
    "XLB": "素材",       "XLC": "通信サービス", "XLE": "エネルギー",
    "XLF": "金融",       "XLI": "資本財",       "XLK": "情報技術",
    "XLP": "生活必需品", "XLRE": "不動産",      "XLU": "公益事業",
    "XLV": "ヘルスケア", "XLY": "一般消費財"
}

# 日本業種ETF（NEXT FUNDS TOPIX-17、17業種）
JP_TICKERS = [
    "1617.T", "1618.T", "1619.T", "1620.T", "1621.T", "1622.T",
    "1623.T", "1624.T", "1625.T", "1626.T", "1627.T", "1628.T",
    "1629.T", "1630.T", "1631.T", "1632.T", "1633.T"
]
JP_NAMES = {
    "1617.T": "食品",               "1618.T": "エネルギー資源",
    "1619.T": "建設・資材",         "1620.T": "素材・化学",
    "1621.T": "医薬品",             "1622.T": "自動車・輸送機",
    "1623.T": "鉄鋼・非鉄",         "1624.T": "機械",
    "1625.T": "電機・精密",         "1626.T": "情報通信・サービス",
    "1627.T": "電力・ガス",         "1628.T": "運輸・物流",
    "1629.T": "商社・卸売",         "1630.T": "小売",
    "1631.T": "銀行",               "1632.T": "金融（除く銀行）",
    "1633.T": "不動産"
}

# シクリカル / ディフェンシブ ラベル（論文4.1節に準拠）
US_CYCLICAL  = ["XLB", "XLE", "XLF", "XLRE"]
US_DEFENSIVE = ["XLK", "XLP", "XLU", "XLV"]
JP_CYCLICAL  = ["1618.T", "1625.T", "1629.T", "1631.T"]
JP_DEFENSIVE = ["1617.T", "1621.T", "1627.T", "1630.T"]

# ハイパーパラメータ（論文に準拠）
L      = 60   # ローリングウィンドウ長（営業日）
K      = 3    # 抽出する主成分数
K0     = 3    # 事前部分空間の次元
LAMBDA = 0.9  # 正則化強度（λ）
N_LONG  = 5   # 買いポートフォリオ銘柄数
N_SHORT = 5   # 売り（回避）シグナル銘柄数

# 事前情報推定期間（Cfull の推定に使用）
TRAIN_START = "2010-01-01"
TRAIN_END   = "2014-12-31"


# ===========================================================================
# 事前部分空間の構築（論文 3.1節）
# ===========================================================================

def build_prior_subspace(us_tickers: list, jp_tickers: list) -> np.ndarray:
    """
    事前固有ベクトル V0 ∈ R^{N × K0} を構築する。

    3つの直交ベクトルを用いる:
      v1: グローバルファクター（全銘柄等ウェイト）
      v2: 国スプレッドファクター（米国+, 日本-）、v1に直交化
      v3: シクリカル・ディフェンシブファクター、v1, v2 に直交化
    """
    N_US  = len(us_tickers)
    N_JP  = len(jp_tickers)
    N     = N_US + N_JP
    all_t = us_tickers + jp_tickers

    # v1: グローバルファクター
    v1 = np.ones(N) / np.sqrt(N)

    # v2: 国スプレッドファクター
    v2_raw = np.concatenate([
        np.ones(N_US) / np.sqrt(N_US),
        -np.ones(N_JP) / np.sqrt(N_JP)
    ])
    v2_raw -= np.dot(v2_raw, v1) * v1
    v2 = v2_raw / np.linalg.norm(v2_raw)

    # v3: シクリカル・ディフェンシブファクター
    v3_raw = np.zeros(N)
    for i, t in enumerate(all_t):
        if t in US_CYCLICAL or t in JP_CYCLICAL:
            v3_raw[i] = +1.0
        elif t in US_DEFENSIVE or t in JP_DEFENSIVE:
            v3_raw[i] = -1.0
    v3_raw -= np.dot(v3_raw, v1) * v1
    v3_raw -= np.dot(v3_raw, v2) * v2
    norm = np.linalg.norm(v3_raw)
    v3 = v3_raw / norm if norm > 1e-10 else v3_raw

    return np.column_stack([v1, v2, v3])  # (N, K0)


def build_prior_exposure(V0: np.ndarray, returns_df: pd.DataFrame) -> np.ndarray:
    """
    事前エクスポージャー行列 C0 を構築する（論文 3.1節）。

      D0     = diag(V0^T * Cfull * V0)
      C0_raw = V0 * D0 * V0^T
      C0     = 対角正規化して相関行列に変換し、対角を1に調整
    """
    z = returns_df.dropna()
    z = (z - z.mean()) / z.std()
    Cfull = (z.values.T @ z.values) / max(len(z) - 1, 1)

    D0     = np.diag(np.diag(V0.T @ Cfull @ V0))
    C0_raw = V0 @ D0 @ V0.T

    diag_sqrt = np.sqrt(np.maximum(np.diag(C0_raw), 1e-20))
    C0 = C0_raw / np.outer(diag_sqrt, diag_sqrt)
    np.fill_diagonal(C0, 1.0)

    return C0


# ===========================================================================
# 部分空間正則化 PCA（論文 3.2節）
# ===========================================================================

def compute_reg_pca(
    window_returns: np.ndarray,
    C0: np.ndarray,
    lambda_: float = LAMBDA,
    k: int = K,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    正則化相関行列を固有分解して上位 K 固有ベクトルを返す。

      C_reg = (1 - λ) * Ct + λ * C0
      固有分解後、降順で上位 K 列を返す。

    Returns:
      V_K    : (N, K) 上位K固有ベクトル
      mu_w   : (N,)  ウィンドウ内平均（標準化用）
      sigma_w: (N,)  ウィンドウ内標準偏差（標準化用）
    """
    mu_w    = window_returns.mean(axis=0)
    sigma_w = window_returns.std(axis=0)
    sigma_w = np.where(sigma_w < 1e-10, 1.0, sigma_w)

    z  = (window_returns - mu_w) / sigma_w
    Ct = (z.T @ z) / max(len(z) - 1, 1)

    C_reg = (1 - lambda_) * Ct + lambda_ * C0

    eigenvalues, eigenvectors = np.linalg.eigh(C_reg)
    idx      = np.argsort(eigenvalues)[::-1]
    V_K      = eigenvectors[:, idx[:k]]

    return V_K, mu_w, sigma_w


# ===========================================================================
# リードラグ・シグナル計算（論文 3.3節）
# ===========================================================================

def compute_signal(
    z_us: np.ndarray,
    V_K: np.ndarray,
    n_us: int,
) -> np.ndarray:
    """
    日本業種の翌営業日シグナルを計算する（論文 式(18)(19)）。

      ft         = V_U^T * z_US          （共通ファクタースコア）
      z_J_hat    = V_J * ft              （日本業種への写像）
      B_t^(K)    = V_J * V_U^T          （低ランク伝播行列）

    Args:
      z_us  : (N_US,) 標準化済み米国業種リターン
      V_K   : (N, K)  上位K固有ベクトル
      n_us  : 米国銘柄数
    Returns:
      signal: (N_JP,) 日本業種シグナル
    """
    V_U = V_K[:n_us, :]   # 米国ブロック (N_US, K)
    V_J = V_K[n_us:, :]   # 日本ブロック (N_JP, K)

    ft     = V_U.T @ z_us  # (K,)
    signal = V_J @ ft      # (N_JP,)

    return signal


# ===========================================================================
# 発注口数の計算
# ===========================================================================

def calc_order_units(
    long_portfolio: pd.DataFrame,
    budget: int = 1_000_000,
    mode: str = "equal",
) -> tuple:
    """
    買いポートフォリオの発注口数を計算する。

    mode:
      "equal"   論文 式(5) 等ウェイト。各ポジション = budget / N_LONG で口数を算出。
                予算が足りなくても全銘柄に1口以上割り当てようとする。
      "budget"  予算優先・最低口数モード。
                シグナル上位から順に1口ずつ購入を試み、
                予算が尽きたらそこで打ち切る。
                資金が少ないときに自然に銘柄数を絞れる。

    Returns:
      (order_df, remaining_cash)
    """
    tickers = long_portfolio["Ticker"].tolist()
    raw = yf.download(tickers, period="3d", auto_adjust=True, progress=False)["Close"]
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    latest_price = {}
    for t in tickers:
        if t in raw.columns:
            s = raw[t].dropna()
            if len(s) > 0:
                latest_price[t] = float(s.iloc[-1])

    rows      = []
    remaining = budget

    for rank, (_, row) in enumerate(long_portfolio.iterrows(), start=1):
        t     = row["Ticker"]
        price = latest_price.get(t)

        if price is None or price <= 0:
            rows.append(_order_row(row, rank, "取得失敗", 0, 0, 0, "価格取得失敗"))
            continue

        if mode == "budget":
            # シグナル上位から1口ずつ。残金が足りなければ購入しない。
            if remaining >= price:
                units    = 1
                invested = int(price)
                note     = ""
            else:
                units    = 0
                invested = 0
                note     = "残金不足 → 購入見送り"
        else:
            # equal: 各ポジション = budget / N_LONG、最低1口
            alloc    = budget / len(long_portfolio)
            units    = max(1, int(alloc / price))
            invested = int(units * price)
            note     = ""

        remaining -= invested
        rows.append(_order_row(row, rank, int(price), units, invested,
                               budget / len(long_portfolio), note))

    result        = pd.DataFrame(rows)
    result.index  = range(1, len(result) + 1)
    return result, remaining


def _order_row(row, rank, price, units, invested, alloc, note):
    return {
        "Ticker":    row["Ticker"],
        "Name":      row["Name"],
        "Signal":    row["Signal"],
        "前日終値(円)": price,
        "目標配分(円)": int(alloc) if isinstance(alloc, float) else alloc,
        "口数":       units,
        "投資額(円)":  invested,
        "備考":       note,
    }


def print_order_table(order_df: pd.DataFrame, remaining: int, budget: int,
                      mode: str = "equal") -> None:
    mode_label = {
        "equal":  "等ウェイト（論文 式(5)）",
        "budget": "予算優先・最低口数（上位から順に1口）",
    }.get(mode, mode)
    bought = order_df[order_df["口数"] > 0]
    skipped = order_df[order_df["口数"] == 0]

    print(f"\n  【発注口数】  運用資金: {budget:,}円  配分方式: {mode_label}")
    print(f"  {'順位':<4} {'ティッカー':<10} {'業種名':<20} "
          f"{'前日終値':>8} {'目標配分':>8} {'口数':>4} {'投資額':>8}  備考")
    print("  " + "-" * 78)
    for rank, row in order_df.iterrows():
        note = f"  {row['備考']}" if row["備考"] else ""
        print(f"  {rank:<4} {row['Ticker']:<10} {row['Name']:<20} "
              f"{str(row['前日終値(円)']):>8} {row['目標配分(円)']:>8,} "
              f"{row['口数']:>4} {row['投資額(円)']:>8,}{note}")

    total_invested = order_df["投資額(円)"].sum()
    print("  " + "-" * 78)
    print(f"  購入銘柄数: {len(bought)}/{len(order_df)}銘柄  "
          f"合計投資額: {total_invested:,}円  "
          f"残金: {remaining:,}円  "
          f"投資比率: {total_invested/budget*100:.1f}%")
    if len(skipped) > 0:
        skipped_names = ", ".join(
            f"{r['Ticker']}({r['Name']})" for _, r in skipped.iterrows()
        )
        print(f"  ※ 購入見送り: {skipped_names}")


# ===========================================================================
# Open 価格の信頼性チェック
# ===========================================================================

# 「当日Open ≈ 前日Close」と判定する許容誤差（相対）
_OPEN_EQUAL_TOL = 1e-4
# 過去何営業日で発生率を計算するか
_OPEN_CHECK_WINDOW = 60


def check_open_reliability(tickers: list, close_df: pd.DataFrame) -> dict:
    """
    直近 _OPEN_CHECK_WINDOW 日の Open 価格を取得し、
    「当日Open ≈ 前日Close」の発生率を銘柄ごとに返す。

    Returns:
      {ticker: {"rate": float, "dates": [date, ...]}}
        rate  : 発生率 (0.0 〜 1.0)
        dates : 直近の該当日リスト（最大5件）
    """
    if close_df.empty:
        return {}

    start = close_df.index[-_OPEN_CHECK_WINDOW].strftime("%Y-%m-%d")
    end   = (close_df.index[-1] + pd.DateOffset(days=3)).strftime("%Y-%m-%d")

    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)["Open"]
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    result = {}
    for t in tickers:
        if t not in raw.columns or t not in close_df.columns:
            continue
        open_s  = raw[t].dropna()
        close_s = close_df[t].reindex(open_s.index).dropna()
        common  = open_s.index.intersection(close_s.index)
        if len(common) < 5:
            continue

        open_s, close_s = open_s[common], close_s[common]
        prev_close = close_s.shift(1)
        mask = ((open_s - prev_close).abs() / prev_close.abs() < _OPEN_EQUAL_TOL) \
               & prev_close.notna()
        suspect_dates = common[mask]
        result[t] = {
            "rate":  len(suspect_dates) / len(common),
            "dates": [d.date() for d in suspect_dates[-5:]],  # 直近5件
        }
    return result


def print_open_warnings(long_portfolio: pd.DataFrame,
                        reliability: dict,
                        warn_threshold: float = 0.08) -> None:
    """
    買いポートフォリオの銘柄について Open 価格の信頼性警告を表示する。

    warn_threshold: この発生率を超えたら警告（デフォルト 8%）
    """
    warned = False
    for _, row in long_portfolio.iterrows():
        t    = row["Ticker"]
        info = reliability.get(t)
        if info is None:
            continue
        rate = info["rate"]
        if rate >= warn_threshold:
            if not warned:
                print("\n  ━━━ Open 価格 信頼性チェック ━━━━━━━━━━━━━━━━━━━━━━")
                print(f"  「当日Open ≈ 前日Close」発生率が {warn_threshold*100:.0f}% 超の銘柄:")
                print(f"  （yfinance がデータ欠損を前日Closeで補完している疑い）")
                warned = True
            dates_str = ", ".join(str(d) for d in info["dates"][-3:])
            print(f"\n  ⚠️  {t} ({row['Name']})")
            print(f"     発生率: {rate*100:.1f}%  直近該当日: {dates_str}")
            print(f"     → 本日の寄付き執行前に実際のOpen値を確認してください")

    if not warned:
        print("\n  ✅ Open 価格チェック: 全銘柄で異常なし")


# ===========================================================================
# データ取得
# ===========================================================================

def fetch_close_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """yfinance で終値（調整済み）を取得する。"""
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    available = [t for t in tickers if t in raw.columns]
    return raw[available].sort_index()


# ===========================================================================
# メイン：ポートフォリオ生成
# ===========================================================================

def generate_portfolio(
    target_date: datetime.date = None,
    write_summary: bool = False,
    budget: int = 1_000_000,
    order_mode: str = "equal",
) -> pd.DataFrame:
    """
    target_date 朝の日本業種セクターシグナルを生成する。

    Args:
      target_date   : 予測対象日（None なら本日 JST）
      write_summary : True のとき GitHub Step Summary / portfolio_result.md を出力
    Returns:
      long_portfolio: 買いシグナル上位 N_LONG 銘柄の DataFrame [Ticker, Name, Signal]
                      （売りシグナル下位 N_SHORT 銘柄は内部で生成し Summary に出力）
    """
    if target_date is None:
        # GitHub Actions は UTC で動作するため JST(+9) に変換して日付を取得
        import datetime as _dt
        JST = _dt.timezone(_dt.timedelta(hours=9))
        target_date = datetime.now(JST).date()
    target_ts       = pd.Timestamp(target_date)
    args_budget     = budget
    args_order_mode = order_mode

    print("=" * 62)
    print("  日米業種リードラグ投資戦略（部分空間正則化付きPCA）")
    print("=" * 62)
    print(f"  予測対象日  : {target_date}")
    print(f"  ウィンドウ  : L={L}日, λ={LAMBDA}, K={K}")

    # ---- 1. データ取得 ----
    # 事前情報用（Cfull の推定）
    print(f"\n[1/4] 訓練データ取得 ({TRAIN_START} 〜 {TRAIN_END}) ...")
    train_us = fetch_close_prices(US_TICKERS, TRAIN_START, TRAIN_END)
    train_jp = fetch_close_prices(JP_TICKERS, TRAIN_START, TRAIN_END)

    # 直近データ（ローリングウィンドウ用）
    recent_start = (target_ts - pd.DateOffset(months=5)).strftime("%Y-%m-%d")
    recent_end   = (target_ts + pd.DateOffset(days=3)).strftime("%Y-%m-%d")
    print(f"[1/4] 直近データ取得  ({recent_start} 〜 {target_date}) ...")
    recent_us = fetch_close_prices(US_TICKERS, recent_start, recent_end)
    recent_jp = fetch_close_prices(JP_TICKERS, recent_start, recent_end)

    # ---- 2. 利用可能銘柄の確認・整合 ----
    # 訓練期間に十分なデータがあるもののみ使用（全期間の50%以上が非NaN）
    min_cov = int(len(train_us) * 0.5)
    avail_us = [t for t in US_TICKERS
                if t in train_us.columns
                and train_us[t].notna().sum() >= min_cov
                and t in recent_us.columns]
    avail_jp = [t for t in JP_TICKERS if t in train_jp and t in recent_jp]
    N_US = len(avail_us)
    N_JP = len(avail_jp)
    print(f"  米国ETF: {N_US}銘柄  |  日本ETF: {N_JP}銘柄")

    if N_US < 3 or N_JP < N_LONG:
        raise RuntimeError(
            f"利用可能銘柄が不足しています（米国:{N_US}, 日本:{N_JP}）。"
        )

    # ---- 3. 訓練データで共通の取引日を揃えてリターン計算 ----
    print("\n[2/4] 事前部分空間を構築中 ...")

    # 日米共通の取引日で結合（欠損が多い行は除外）
    train_combined = pd.concat(
        [train_us[avail_us], train_jp[avail_jp]], axis=1
    ).dropna(how="all")
    # 列欠損の多い日を除外
    thresh = int((N_US + N_JP) * 0.8)
    train_combined = train_combined.dropna(thresh=thresh).ffill().dropna()

    train_ret = train_combined.pct_change().dropna()

    # ---- 4. 事前部分空間 V0、C0 の構築 ----
    V0 = build_prior_subspace(avail_us, avail_jp)
    C0 = build_prior_exposure(V0, train_ret)

    # ---- 5. 直近ローリングウィンドウデータの準備 ----
    print("[3/4] シグナル計算中 ...")

    recent_combined = pd.concat(
        [recent_us[avail_us], recent_jp[avail_jp]], axis=1
    ).dropna(how="all")
    thresh_r = int((N_US + N_JP) * 0.7)
    recent_combined = recent_combined.dropna(thresh=thresh_r).ffill().dropna()
    recent_ret = recent_combined.pct_change().dropna()

    # 米国シグナル確定日 = target_date の直前営業日
    # （target_date 朝6〜9時時点では前日の米国終値が確定済み）
    us_dates = recent_ret.index.normalize()
    past_dates = us_dates[us_dates < target_ts]
    if len(past_dates) == 0:
        raise RuntimeError(
            f"{target_date} 以前のデータが見つかりません。"
            "市場の休日や取得範囲を確認してください。"
        )
    signal_date = past_dates[-1]
    print(f"  米国終値確定日: {signal_date.date()}")

    # ローリングウィンドウの切り出し
    loc = recent_ret.index.get_loc(signal_date)
    if isinstance(loc, slice):
        loc = loc.stop - 1
    win_start = max(0, loc - L + 1)
    window_data = recent_ret.iloc[win_start: loc + 1]

    if len(window_data) < max(K * 3, 10):
        raise RuntimeError(
            f"ウィンドウデータが不足しています（{len(window_data)}行）。"
        )

    # ---- 6. 正則化 PCA 実行 ----
    all_avail = avail_us + avail_jp
    V_K, mu_w, sigma_w = compute_reg_pca(window_data[all_avail].values, C0)

    # ---- 7. 当日米国標準化リターンの計算 ----
    us_ret_today = recent_ret.loc[signal_date, avail_us].values
    mu_us    = mu_w[:N_US]
    sig_us   = np.where(sigma_w[:N_US] < 1e-10, 1.0, sigma_w[:N_US])
    z_us     = (us_ret_today - mu_us) / sig_us

    # ---- 8. シグナル計算・ポートフォリオ構築 ----
    signal = compute_signal(z_us, V_K, N_US)

    signal_df = pd.DataFrame({
        "Ticker": avail_jp,
        "Name":   [JP_NAMES.get(t, t) for t in avail_jp],
        "Signal": signal,
    }).sort_values("Signal", ascending=False).reset_index(drop=True)
    signal_df.index += 1

    long_portfolio  = signal_df.head(N_LONG).copy()
    short_portfolio = signal_df.tail(N_SHORT).copy()

    # ---- 9. 結果表示 ----
    print("\n[4/4] ポートフォリオ生成完了")

    print("\n" + "=" * 62)
    print(f"  【セクターシグナル】  {target_date} 参考")
    print(f"  シグナル源  : {signal_date.date()} 米国業種ETF終値リターン")
    print(f"  買い上位{N_LONG}銘柄（◆ BUY）/ 回避下位{N_SHORT}銘柄（▼ SELL）")
    print("=" * 62)
    print(f"  {'順位':<4} {'ティッカー':<12} {'業種名':<22} {'シグナル':>8}")
    print("  " + "-" * 52)
    for rank, row in long_portfolio.iterrows():
        print(f"  {rank:<4} {row['Ticker']:<12} {row['Name']:<22} {row['Signal']:>+8.4f}")
    print("  " + "-" * 52)

    print("\n  【参考】全日本業種ETFシグナル順位")
    print(f"  {'順位':<4} {'ティッカー':<12} {'業種名':<22} {'シグナル':>8}  {'判定'}")
    print("  " + "-" * 60)
    n_total = len(signal_df)
    for i, row in signal_df.iterrows():
        if i <= N_LONG:
            tag = "◆ BUY "
        elif i > n_total - N_SHORT:
            tag = "▼ SELL"
        else:
            tag = ""
        print(f"  {i:<4} {row['Ticker']:<12} {row['Name']:<22} {row['Signal']:>+8.4f}  {tag}")

    print(f"\n  【参考】{signal_date.date()} 米国業種ETFリターン")
    us_ret_df = pd.DataFrame({
        "Ticker": avail_us,
        "Name":   [US_NAMES.get(t, t) for t in avail_us],
        "標準化リターン (z)": z_us.round(4),
        "リターン (%)":     (us_ret_today * 100).round(2),
    }).sort_values("リターン (%)", ascending=False)
    print(us_ret_df.to_string(index=False))

    # ---- 10. 発注口数の計算 ----
    order_df, remaining = calc_order_units(long_portfolio, budget=args_budget,
                                           mode=args_order_mode)
    print_order_table(order_df, remaining, budget=args_budget, mode=args_order_mode)

    # ---- 11. Open 価格の信頼性チェック ----
    print("\n[Open 価格チェック中...]")
    reliability = check_open_reliability(avail_jp, recent_jp[avail_jp])
    print_open_warnings(long_portfolio, reliability)

    # GitHub Actions / portfolio_result.md への書き出し
    if write_summary:
        write_github_summary(
            long_portfolio, short_portfolio, signal_df, us_ret_df,
            target_date, signal_date, reliability,
        )

    return long_portfolio


# ===========================================================================
# GitHub Actions 用：Step Summary への Markdown 書き出し
# ===========================================================================

def write_github_summary(
    long_portfolio: pd.DataFrame,
    short_portfolio: pd.DataFrame,
    signal_df: pd.DataFrame,
    us_ret_df: pd.DataFrame,
    target_date,
    signal_date,
    reliability: dict = None,
) -> None:
    """$GITHUB_STEP_SUMMARY に Markdown を書き込む（Actions 上でのみ有効）。"""
    import os
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    output_path = "portfolio_result.md"

    WARN_THRESHOLD = 0.08

    lines = []
    lines.append(f"# 日米リードラグ セクターシグナル  {target_date}\n")
    lines.append(
        f"> シグナル源：{signal_date.date()} 米国業種ETF終値リターン  "
        f"｜ 等ウェイト各 {100/N_LONG:.0f}%\n"
    )

    # --- Open 価格の警告ブロック ---
    if reliability:
        warned_tickers = [
            (row["Ticker"], row["Name"], reliability[row["Ticker"]])
            for _, row in long_portfolio.iterrows()
            if row["Ticker"] in reliability
            and reliability[row["Ticker"]]["rate"] >= WARN_THRESHOLD
        ]
        if warned_tickers:
            lines.append("## ⚠️ Open 価格 要確認\n")
            lines.append(
                "> yfinance で「当日Open ≈ 前日Close」が頻発している銘柄があります。  \n"
                "> 寄付き執行前に証券会社の画面で実際のOpen値を確認してください。\n"
            )
            lines.append("| ティッカー | 業種名 | 発生率 | 直近の該当日 |")
            lines.append("|:---:|:---|---:|:---|")
            for t, name, info in warned_tickers:
                dates_str = "、".join(str(d) for d in info["dates"][-3:])
                lines.append(
                    f"| **{t}** | {name} | `{info['rate']*100:.1f}%` | {dates_str} |"
                )
            lines.append("")
        else:
            lines.append("## ✅ Open 価格チェック：異常なし\n")

    lines.append("## ◆ 買いシグナル（上位5銘柄）\n")
    lines.append("| 順位 | ティッカー | 業種名 | シグナル値 |")
    lines.append("|:---:|:---:|:---|---:|")
    for rank, row in long_portfolio.iterrows():
        lines.append(
            f"| {rank} | **{row['Ticker']}** | {row['Name']} | `{row['Signal']:+.4f}` |"
        )

    lines.append("\n## ▼ 売り・回避シグナル（下位5銘柄）\n")
    lines.append("| 順位 | ティッカー | 業種名 | シグナル値 |")
    lines.append("|:---:|:---:|:---|---:|")
    for rank, row in short_portfolio.iterrows():
        lines.append(
            f"| {rank} | **{row['Ticker']}** | {row['Name']} | `{row['Signal']:+.4f}` |"
        )

    lines.append("\n## 全日本業種ETF シグナル順位\n")
    lines.append("| 順位 | ティッカー | 業種名 | シグナル値 | 判定 |")
    lines.append("|:---:|:---:|:---|---:|:---:|")
    n_total = len(signal_df)
    for i, row in signal_df.iterrows():
        if i <= N_LONG:
            tag = "**◆ BUY**"
        elif i > n_total - N_SHORT:
            tag = "**▼ SELL**"
        else:
            tag = ""
        lines.append(
            f"| {i} | {row['Ticker']} | {row['Name']} | `{row['Signal']:+.4f}` | {tag} |"
        )

    lines.append(f"\n## 参考：{signal_date.date()} 米国業種ETFリターン\n")
    lines.append("| ティッカー | 業種名 | リターン(%) | 標準化(z) |")
    lines.append("|:---:|:---|---:|---:|")
    for _, row in us_ret_df.iterrows():
        lines.append(
            f"| {row['Ticker']} | {row['Name']} "
            f"| `{row['リターン (%)']:+.2f}` | `{row['標準化リターン (z)']:+.4f}` |"
        )

    md_text = "\n".join(lines)

    # アーティファクト用ファイルに書き込む
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    # GitHub Actions Step Summary に書き込む
    if summary_path:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"\n→ Step Summary に書き込みました: {summary_path}")


# ===========================================================================
# エントリーポイント
# ===========================================================================

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="日米リードラグ セクターシグナル生成")
    parser.add_argument(
        "--date", "-d",
        type=str,
        default=None,
        help="予測対象日 (YYYY-MM-DD)。省略時は本日。",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="GitHub Actions の Step Summary / portfolio_result.md に結果を出力する。",
    )
    parser.add_argument(
        "--budget", "-b",
        type=int,
        default=1_000_000,
        help="運用資金（円）。デフォルト 1000000（100万円）。",
    )
    parser.add_argument(
        "--order-mode",
        choices=["equal", "budget"],
        default="equal",
        help=(
            "equal : 等ウェイト（論文準拠、デフォルト）。"
            "budget: 予算優先・最低口数。上位から1口ずつ、予算が尽きたら打ち切り。"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    _force_utf8()
    args = parse_args()

    target_date = None
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    portfolio = generate_portfolio(
        target_date,
        write_summary=args.summary,
        budget=args.budget,
        order_mode=args.order_mode,
    )
