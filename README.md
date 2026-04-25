# Lead-Lag Investment — 日米業種リードラグ投資戦略

## 概要

前日の**米国業種ETF**の値動きを信号として、翌朝の**日本業種ETF**の有望セクターを予測するアルゴリズムです。

中川他「部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略」(SIG-FIN-036-13) の手法を実装しています。

### 戦略の仕組み

```
米国市場（前日終値）→ 部分空間正則化PCA → 日本セクター予測シグナル
```

- **買いシグナル上位5セクター**：当日の寄り付きで追い風が期待されるセクター
- **売り・回避シグナル下位5セクター**：当日の向かい風が予想されるセクター

個別株の銘柄選定や売買タイミングの補助情報として活用できます。

### 主なパラメータ

| パラメータ | 値 | 意味 |
|-----------|-----|------|
| ローリングウィンドウ | L = 60日 | PCA推定に使う直近データ期間 |
| 正則化強度 | λ = 0.9 | 事前部分空間への引き戻し強度 |
| 抽出因子数 | K = 3 | グローバル・国別・景気循環 |
| 対象ETF | 米国9銘柄 / 日本17銘柄 | 業種別ETF |

---

## 使い方

### GitHub Actions で毎朝自動実行（推奨）

リポジトリをForkまたはCloneし、**Actions タブ**を有効化するだけで毎朝6:05 JST（平日）に自動実行されます。

結果は Actions → 該当ワークフロー → **Summary** タブで確認できます（スマートフォン対応）。

手動実行：`Actions` → `Generate Portfolio` → `Run workflow`

### ローカルで実行

```bash
pip install -r requirements.txt

# 当日のポートフォリオ生成
python generate_portfolio.py

# 日付を指定して実行
python generate_portfolio.py --date 2026-04-07

# 予算・発注口数も計算（等ウェイト、100万円）
python generate_portfolio.py --budget 1000000 --order-mode equal
```

---

## 依存ライブラリ

```
numpy
pandas
scipy
yfinance
```

---

## ファイル構成

```
.
├── generate_portfolio.py   # メインスクリプト
├── app.py                  # Streamlit Webアプリ（オプション）
├── requirements.txt
├── .github/
│   └── workflows/
│       └── portfolio.yml   # GitHub Actions ワークフロー
└── test/                   # 検証コード
    ├── test_math.py        # 数学的実装の検証（23テスト）
    ├── test_backtest.py    # バックテスト再現検証
    ├── test_signal_quality.py
    ├── test_open_price.py
    └── test_budget_mode.py
```

---

## 参考文献

中川慧、他「部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略」  
人工知能学会 金融情報学研究会 SIG-FIN-036-13

---

## 免責事項

> **本ソフトウェアおよび出力される情報は、投資の勧誘や売買推奨を目的としたものではありません。**
>
> 本ツールの使用によって生じた**いかなる損失・損害についても、作者は一切の責任を負いません**。
> 投資はご自身の判断と責任において行ってください。
>
> 過去のバックテスト結果は将来の利益を保証するものではありません。
> 市場環境の変化により、戦略の有効性が失われる可能性があります。

---

## ライセンス

[MIT License](LICENSE)
