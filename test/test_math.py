"""
Step 1: 数学的性質テスト
========================
論文の各数式・命題が Python コードで正しく実装されているかを検証する。

対応箇所:
  3.1節  事前部分空間 V0、事前エクスポージャー C0
  3.2節  正則化 PCA（C_reg、固有分解）
  3.3節  リードラグシグナル、命題1（低ランク線形予測器）

実行方法:
  python -X utf8 test/test_math.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
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
# テストユーティリティ
# ---------------------------------------------------------------------------

results = []

def check(label: str, cond: bool, detail: str = "") -> bool:
    status = "PASS" if cond else "FAIL"
    symbol = "✅" if cond else "❌"
    line = f"  {symbol} [{status}] {label}"
    if detail:
        line += f"\n           → {detail}"
    print(line)
    results.append((label, cond))
    return cond


# ---------------------------------------------------------------------------
# データ準備（全テスト共通）
# ---------------------------------------------------------------------------

def load_data():
    print("  訓練データ取得中...")
    train_us = fetch_close_prices(US_TICKERS, TRAIN_START, TRAIN_END)
    train_jp = fetch_close_prices(JP_TICKERS, TRAIN_START, TRAIN_END)

    # 訓練期間に十分なデータがある銘柄のみ（50%以上）
    min_cov  = int(len(train_us) * 0.5)
    avail_us = [t for t in US_TICKERS
                if t in train_us.columns and train_us[t].notna().sum() >= min_cov]
    avail_jp = [t for t in JP_TICKERS if t in train_jp]
    all_av   = avail_us + avail_jp

    combined = pd.concat([train_us[avail_us], train_jp[avail_jp]], axis=1)
    combined = combined.dropna(thresh=int(len(all_av) * 0.8)).ffill().dropna()
    train_ret = combined.pct_change().dropna()

    return avail_us, avail_jp, train_ret


# ---------------------------------------------------------------------------
# Test A: 事前部分空間 V0（論文 3.1節）
# ---------------------------------------------------------------------------

def test_A_prior_subspace(avail_us, avail_jp):
    print("\n── A. 事前部分空間 V0（論文 3.1節）──────────────────")

    V0 = build_prior_subspace(avail_us, avail_jp)
    N  = len(avail_us) + len(avail_jp)

    # A-1: 形状
    check("A-1  形状が (N, K0=3)",
          V0.shape == (N, 3),
          f"実際: {V0.shape}")

    # A-2: 列直交性  V0^T V0 = I_3（論文 3.1節「列直交」）
    VtV = V0.T @ V0
    err = np.abs(VtV - np.eye(3)).max()
    check("A-2  列直交性  V0^T V0 ≈ I_3",
          err < 1e-10,
          f"最大誤差: {err:.2e}")

    # A-3: 各列の単位ノルム
    norms = np.linalg.norm(V0, axis=0)
    check("A-3  各列ベクトルが単位ノルム  ||vi|| = 1",
          np.allclose(norms, 1.0, atol=1e-10),
          f"ノルム: {np.round(norms, 12)}")

    # A-4: v1 = 全銘柄等ウェイト（論文「v1 ∝ 1」）
    v1  = V0[:, 0]
    exp = np.ones(N) / np.sqrt(N)
    check("A-4  v1 がグローバルファクター（全銘柄等ウェイト, v1 ∝ 1）",
          np.allclose(np.abs(v1), np.abs(exp), atol=1e-10),
          f"最大偏差: {np.abs(np.abs(v1) - np.abs(exp)).max():.2e}")

    # A-5: v2 の符号 = 米国+, 日本-（論文「v2 ∝ (1_Nu, -1_Nj)」）
    v2   = V0[:, 1]
    n_us = len(avail_us)
    us_pos = (v2[:n_us] > 0).all()
    jp_neg = (v2[n_us:] < 0).all()
    check("A-5  v2 が国スプレッドファクター（米国+, 日本-）",
          us_pos and jp_neg,
          f"米国が正: {us_pos}, 日本が負: {jp_neg}")

    # A-6: v2 が v1 に直交（グラム・シュミット）
    dot_12 = abs(V0[:, 0] @ V0[:, 1])
    check("A-6  v2 ⊥ v1（グラム・シュミット直交化）",
          dot_12 < 1e-10,
          f"|v1·v2| = {dot_12:.2e}")

    # A-7: v3 が v1, v2 に直交
    dot_13 = abs(V0[:, 0] @ V0[:, 2])
    dot_23 = abs(V0[:, 1] @ V0[:, 2])
    check("A-7  v3 ⊥ v1 かつ v3 ⊥ v2",
          dot_13 < 1e-10 and dot_23 < 1e-10,
          f"|v1·v3|={dot_13:.2e}, |v2·v3|={dot_23:.2e}")

    return V0


# ---------------------------------------------------------------------------
# Test B: 事前エクスポージャー C0（論文 3.1節）
# ---------------------------------------------------------------------------

def test_B_prior_exposure(V0, train_ret):
    print("\n── B. 事前エクスポージャー C0（論文 3.1節）──────────")

    C0 = build_prior_exposure(V0, train_ret)

    # B-1: 対称性
    err_sym = np.abs(C0 - C0.T).max()
    check("B-1  対称行列  C0 = C0^T",
          err_sym < 1e-10,
          f"最大非対称誤差: {err_sym:.2e}")

    # B-2: 対角が 1（「diag(C0)=1 となるよう調整」）
    diag_err = np.abs(np.diag(C0) - 1.0).max()
    check("B-2  対角要素がすべて 1.0  diag(C0) = 1",
          diag_err < 1e-10,
          f"最大誤差: {diag_err:.2e}")

    # B-3: 正半定値（相関行列の必要条件）
    eigs = np.linalg.eigvalsh(C0)
    check("B-3  正半定値  固有値 ≥ -1e-10",
          eigs.min() >= -1e-10,
          f"最小固有値: {eigs.min():.4e}")

    # B-4: ランク ≤ K0=3（C0_raw = V0 D0 V0^T なので rank ≤ 3）
    rank = np.linalg.matrix_rank(C0, tol=1e-8)
    check("B-4  ランク ≤ K0=3（論文 式(11): C0_raw = V0 D0 V0^T）",
          rank <= 3,
          f"実際のランク: {rank}")

    # B-5: 事前方向の固有値 D0 が非負（対角行列の非負性）
    z   = train_ret.dropna()
    z   = (z - z.mean()) / z.std()
    Cf  = (z.values.T @ z.values) / max(len(z) - 1, 1)
    D0  = np.diag(V0.T @ Cf @ V0)
    check("B-5  D0 = diag(V0^T Cfull V0) の対角が非負",
          (D0 >= 0).all(),
          f"D0 対角: {np.round(D0, 4)}")

    return C0


# ---------------------------------------------------------------------------
# Test C: 正則化 PCA（論文 3.2節）
# ---------------------------------------------------------------------------

def test_C_reg_pca(C0, window_data):
    print("\n── C. 正則化 PCA（論文 3.2節）───────────────────────")

    V_K, mu_w, sigma_w = compute_reg_pca(window_data.values, C0)
    N = window_data.shape[1]

    # C-1: 形状
    check(f"C-1  V_K の形状が (N={N}, K={K})",
          V_K.shape == (N, K),
          f"実際: {V_K.shape}")

    # C-2: 列直交性
    VtV = V_K.T @ V_K
    err = np.abs(VtV - np.eye(K)).max()
    check("C-2  V_K が列直交  V_K^T V_K ≈ I_K",
          err < 1e-10,
          f"最大誤差: {err:.2e}")

    # C-3: 正則化行列が正定値
    z  = (window_data.values - mu_w) / np.where(sigma_w < 1e-10, 1.0, sigma_w)
    Ct = (z.T @ z) / max(len(z) - 1, 1)
    C_reg = (1 - LAMBDA) * Ct + LAMBDA * C0
    eigs  = np.linalg.eigvalsh(C_reg)
    check(f"C-3  C_reg = (1-λ)Ct + λC0 が正定値（λ={LAMBDA}）",
          eigs.min() > 0,
          f"最小固有値: {eigs.min():.4e}")

    # C-4: V_K が C_reg の真の上位固有ベクトルと一致
    true_eigs, true_vecs = np.linalg.eigh(C_reg)
    idx_desc = np.argsort(true_eigs)[::-1]
    V_ref = true_vecs[:, idx_desc[:K]]
    # 符号不定なので部分空間距離で比較
    P_est = V_K @ V_K.T
    P_ref = V_ref @ V_ref.T
    subspace_err = np.linalg.norm(P_est - P_ref, ord="fro")
    check("C-4  抽出された固有空間が正しい（部分空間距離 < 1e-8）",
          subspace_err < 1e-8,
          f"部分空間距離: {subspace_err:.2e}")

    # C-5: 固有値が降順
    evals_K = [V_K[:, k] @ C_reg @ V_K[:, k] for k in range(K)]
    check("C-5  抽出された固有値が降順",
          all(evals_K[i] >= evals_K[i+1] - 1e-10 for i in range(K-1)),
          f"Rayleigh 商: {[round(e, 4) for e in evals_K]}")

    return V_K, mu_w, sigma_w


# ---------------------------------------------------------------------------
# Test D: リードラグシグナル / 命題1（論文 3.3節）
# ---------------------------------------------------------------------------

def test_D_signal(V_K, n_us, n_jp):
    print("\n── D. リードラグシグナル / 命題1（論文 3.3節）────────")

    np.random.seed(42)
    z_us = np.random.randn(n_us)
    signal = compute_signal(z_us, V_K, n_us)

    V_U = V_K[:n_us, :]
    V_J = V_K[n_us:, :]

    # D-1: 出力形状
    check(f"D-1  シグナルの形状が (N_JP={n_jp},)",
          signal.shape == (n_jp,),
          f"実際: {signal.shape}")

    # D-2: 命題1 式(20)  ẑ_J = V_J (V_U^T z_U)
    expected = V_J @ (V_U.T @ z_us)
    err = np.abs(signal - expected).max()
    check("D-2  ẑ_J = V_J (V_U^T z_U)  が成立（命題1 式(20)）",
          err < 1e-12,
          f"最大誤差: {err:.2e}")

    # D-3: 命題1 式(21)  B_t^(K) = V_J V_U^T  で一致
    B   = V_J @ V_U.T
    err2 = np.abs(signal - B @ z_us).max()
    check("D-3  ẑ_J = B_t^(K) z_U（伝播行列による線形写像）が成立",
          err2 < 1e-12,
          f"最大誤差: {err2:.2e}")

    # D-4: 命題1 式(22)  rank(B_t^(K)) ≤ K
    rank_B = np.linalg.matrix_rank(B, tol=1e-8)
    check(f"D-4  rank(B_t^(K)) ≤ K={K}（命題1 式(22)）",
          rank_B <= K,
          f"実際のランク: {rank_B}")

    # D-5: 命題1「二段階構造」—— ft = V_U^T z_U → ẑ_J = V_J ft
    ft  = V_U.T @ z_us
    ẑ_J = V_J @ ft
    err3 = np.abs(signal - ẑ_J).max()
    check("D-5  二段階構造（射影→復元）と直接計算が一致",
          err3 < 1e-12,
          f"最大誤差: {err3:.2e}")

    # D-6: 線形性確認  f(az+bw) = a*f(z) + b*f(w)
    a, b = 1.5, -0.7
    z2   = np.random.randn(n_us)
    s1   = compute_signal(a * z_us + b * z2, V_K, n_us)
    s2   = a * compute_signal(z_us, V_K, n_us) + b * compute_signal(z2, V_K, n_us)
    err4 = np.abs(s1 - s2).max()
    check("D-6  シグナルが z_U に対して線形（重ね合わせの原理）",
          err4 < 1e-12,
          f"最大誤差: {err4:.2e}")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 62)
    print("  Step 1: 数学的性質テスト（論文の数式との整合性）")
    print("=" * 62)

    avail_us, avail_jp, train_ret = load_data()
    all_avail = avail_us + avail_jp

    V0 = test_A_prior_subspace(avail_us, avail_jp)
    C0 = test_B_prior_exposure(V0, train_ret)

    window_data = train_ret[all_avail].iloc[-L:]
    V_K, mu_w, sigma_w = test_C_reg_pca(C0, window_data)

    test_D_signal(V_K, len(avail_us), len(avail_jp))

    # 結果サマリ
    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    print("\n" + "=" * 62)
    print(f"  結果: {passed}/{total} テスト通過")
    if passed == total:
        print("  ✅ 全テスト PASS — 論文の数式と一致しています")
    else:
        failed = [label for label, ok in results if not ok]
        print(f"  ❌ FAIL: {failed}")
    print("=" * 62)
