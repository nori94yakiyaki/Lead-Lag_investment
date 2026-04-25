"""
Streamlit Web アプリ - 日米リードラグ 買いポートフォリオ
スマホ・タブレットのブラウザからアクセスして実行できる。

デプロイ先: https://streamlit.io/cloud  (無料)
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta

# generate_portfolio.py の関数を再利用
from generate_portfolio import generate_portfolio

st.set_page_config(
    page_title="日米リードラグ ポートフォリオ",
    page_icon="📈",
    layout="centered",
)

st.title("📈 日米リードラグ 買いポートフォリオ")
st.caption("部分空間正則化付きPCA（PCA SUB）｜ 論文: 中川慧 et al. SIG-FIN-036-13")

# --- 日付入力 ---
st.divider()
target_date = st.date_input(
    "予測対象日（日本市場の寄付き日）",
    value=date.today(),
    min_value=date(2015, 1, 1),
    max_value=date.today() + timedelta(days=1),
)

run_btn = st.button("🚀 ポートフォリオを生成", type="primary", use_container_width=True)

if run_btn:
    with st.spinner("データ取得・シグナル計算中... (約60秒)"):
        try:
            portfolio = generate_portfolio(target_date)

            st.success(f"✅ {target_date} 寄付き 買いポートフォリオ（等ウェイト各20%）")

            # 買いポートフォリオ表示
            st.subheader("◆ 買いポートフォリオ 上位5銘柄")
            display_df = portfolio.copy()
            display_df.columns = ["ティッカー", "業種名", "シグナル値"]
            display_df["シグナル値"] = display_df["シグナル値"].map("{:+.4f}".format)
            st.dataframe(display_df, use_container_width=True, hide_index=False)

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
            st.info("市場休日や祝日の場合、前営業日のデータが使用されます。")
