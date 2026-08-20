"""
制造型产品成本分析平台 v2
========================
更新：4项成本结构（直接材料/直接人工/变动制造费用/固定制造费用）
新增：盈亏平衡分析 · 敏感度分析 · 预测模拟

数据源：standard_cost_final.csv + actual_cost_final.csv + variance_analysis_final.csv
应用场景：博世等德企成本控制岗位面试作品集
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")

# ─── 页面配置 ─────────────────────────────────────────────
st.set_page_config(
    page_title="成本分析平台 | Cost Control",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ─── 调色板 ──────────────────────────────────────────────
C = {
    "primary": "#1A3A5C",
    "secondary": "#2E86AB",
    "accent": "#F18F01",
    "positive": "#2E8B57",
    "negative": "#C1292E",
    "bg": "#F5F7FA",
    "grid": "#E1E5EB",
    "text": "#2C3E50",
    "muted": "#7F8C9B",
    "material": "#5B8DEF",
    "labor": "#E67E22",
    "var_oh": "#27AE60",
    "fix_oh": "#E74C3C",
    "revenue": "#2E86AB",
    "bep": "#C1292E",
    "capacity": "#F18F01",
}

# ─── 固定参数 ────────────────────────────────────────────
FIXED_COST_MONTHLY = 17_290_000       # 产线级11,700,000 + 工厂级5,590,000
DIRECT_LABOR_STD = 20.0               # 元/件
VAR_OH_STD = 29.615384615384617       # 元/件（59.2元/h × 0.5h）
FIX_OH_STD = 307.8703703703704        # 元/件（标准分摊）
STD_HOURS_PER_UNIT = 0.5              # 标准工时/件
TOTAL_AVAILABLE_HOURS = 28080         # 总可用工时/月
MAX_CAPACITY_UNITS = TOTAL_AVAILABLE_HOURS / STD_HOURS_PER_UNIT  # 56,160件

# 产线信息
LINE_INFO = {
    "A1": {"devices": 13, "hours": 9360, "std_hours_pct": 9360/28080},
    "A2": {"devices": 12, "hours": 8640, "std_hours_pct": 8640/28080},
    "A3": {"devices": 14, "hours": 10080, "std_hours_pct": 10080/28080},
}

# ─── 数据加载 ─────────────────────────────────────────────
@st.cache_data
def load_data():
    dfs = {}
    # 标准成本（4项拆分）
    dfs["standard"] = pd.read_csv(
        os.path.join(DATA_DIR, "standard_cost_final.csv"), encoding="utf-8-sig"
    )
    # 实际成本（2个月）
    dfs["actual"] = pd.read_csv(
        os.path.join(DATA_DIR, "actual_cost_final.csv"), encoding="utf-8-sig"
    )
    # 差异分析
    dfs["variance"] = pd.read_csv(
        os.path.join(DATA_DIR, "variance_analysis_final.csv"), encoding="utf-8-sig"
    )
    # 制造费用拆分
    dfs["mfg_cost"] = pd.read_csv(
        os.path.join(DATA_DIR, "manufacturing_cost_corrected.csv"), encoding="utf-8-sig"
    )
    # BOM结构
    dfs["bom"] = pd.read_csv(
        os.path.join(DATA_DIR, "bom_structure_v2.csv"), encoding="utf-8-sig"
    )
    return dfs

dfs = load_data()
std_df = dfs["standard"]
act_df = dfs["actual"]
var_df = dfs["variance"]
mfg_df = dfs["mfg_cost"]
bom_df = dfs["bom"]

# 标准成本去重（文件有重复行）
std_df = std_df.drop_duplicates(subset=["产品名称", "产线"]).reset_index(drop=True)

# 数据预处理
act_df["月份"] = pd.to_datetime(act_df["月份"].astype(str) + "01", format="%Y年%m月%d")
var_df["月份"] = pd.to_datetime(var_df["月份"].astype(str) + "01", format="%Y年%m月%d")

# 产品列表
products = std_df[["产品名称", "产线"]].drop_duplicates().values.tolist()
product_names = sorted(set(p[0] for p in products))
product_line_map = {p[0]: p[1] for p in products}

# 标准成本索引
std_map = {}
for _, row in std_df.iterrows():
    pname = row["产品名称"]
    std_map[pname] = {
        "material": row["直接材料标准"],
        "labor": row["直接人工标准"],
        "var_oh": row["变动制造费用标准"],
        "fix_oh": row["固定制造费用标准"],
        "total": row["标准单位成本"],
        "std_vol": row["标准产量"],
        "line": row["产线"],
    }

# 产线标准产量汇总
line_std_vol = std_df.groupby("产线")["标准产量"].sum().to_dict()
# 产品标准产量占比（用于加权计算）
total_std_vol = std_df["标准产量"].sum()
std_df["mix_ratio"] = std_df["标准产量"] / total_std_vol


# ─── 工具函数 ─────────────────────────────────────────────
def fmt_money(val):
    if abs(val) >= 1_0000_0000:
        return f"¥{val/1e8:.2f}亿"
    if abs(val) >= 1_0000:
        return f"¥{val:,.0f}"
    return f"¥{val:,.2f}"

def fmt_pct(val):
    return f"{val:+.1f}%" if val != 0 else f"{val:.1f}%"


# ─── BOM树解析与半成品成本计算 ─────────────────────────────
@st.cache_data
def calc_semi_finished_costs(bom_df, product_name, std_map):
    """
    基于BOM结构计算半成品节点的材料成本完工率。
    返回各工序节点数据，含材料消耗占比和完工率估计。
    """
    # 找到该产品的根物料号
    prod_rows = bom_df[bom_df["产品名称"] == product_name]
    if len(prod_rows) == 0:
        return [], None
    
    root_mat_no = prod_rows.iloc[0]["物料号(母件)"]
    
    # 构建父→子映射
    parent_map = {}
    for _, row in bom_df.iterrows():
        p = row["物料号(母件)"]
        if p not in parent_map:
            parent_map[p] = []
        parent_map[p].append({
            "child": row["组件号(子件)"],
            "name": row["材料名称"],
            "qty": row["材料用量"],
            "parent_name": row["产品名称"],
        })
    
    # 收集所有叶子节点（原材料）和半成品节点
    raw_materials = {}
    semi_nodes = []
    
    def collect_tree(parent, visited):
        if parent in visited:
            return
        visited.add(parent)
        children = parent_map.get(parent, [])
        for child in children:
            if child["child"] not in parent_map:
                # 叶子节点→原材料
                raw_materials[child["child"]] = {
                    "name": child["name"], "qty": child["qty"]
                }
            else:
                if "(半)" in child["name"]:
                    semi_nodes.append({
                        "node": child["child"],
                        "name": child["name"],
                        "parent_name": child["parent_name"],
                    })
                collect_tree(child["child"], visited)
    
    collect_tree(root_mat_no, set())
    
    if not raw_materials:
        return [], None
    
    total_raw_qty = sum(r["qty"] for r in raw_materials.values())
    
    # 判断节点是否在到目标的路径上
    def is_on_path_to_target(node, target, visited):
        if node in visited:
            return False
        visited.add(node)
        if node == target:
            return True
        for child in parent_map.get(node, []):
            if is_on_path_to_target(child["child"], target, visited):
                return True
        return False
    
    # 计算半成品节点前的上游材料消耗
    stage_data = []
    for sn in semi_nodes:
        upstream_qty = 0
        def calc_upstream(parent, target, visited):
            qty = 0
            if parent in visited:
                return 0
            visited.add(parent)
            children = parent_map.get(parent, [])
            for child in children:
                if child["child"] == target:
                    continue  # 找到目标，不计入该节点材料
                if child["child"] not in parent_map:
                    # 叶子节点 → 加入
                    qty += child["qty"]
                else:
                    # 检查子节点是否在通往目标的路径上
                    v2 = set()
                    if is_on_path_to_target(child["child"], target, v2):
                        qty += calc_upstream(child["child"], target, visited)
                    else:
                        qty += child["qty"]
            return qty
        
        upstream_qty = calc_upstream(root_mat_no, sn["node"], set())
        mat_pct = upstream_qty / total_raw_qty * 100 if total_raw_qty > 0 else 0
        
        stage_data.append({
            "name": sn["name"],
            "parent_name": sn["parent_name"],
            "upstream_qty": upstream_qty,
            "total_qty": total_raw_qty,
            "mat_completion_pct": mat_pct,
        })
    
    return stage_data, {"total_raw_qty": total_raw_qty, "raw_count": len(raw_materials)}


# ─── 侧边栏 ──────────────────────────────────────────────
st.sidebar.markdown(
    f"""
    <div style="padding: 12px 0; border-bottom: 2px solid {C['primary']}; margin-bottom: 16px;">
        <h2 style="color: {C['primary']}; margin: 0; font-size: 1.3rem;">📊 成本分析平台</h2>
        <p style="color: {C['muted']}; font-size: 0.8rem; margin: 4px 0 0;">汽车零部件制造 | Cost Control v2</p>
    </div>
    """,
    unsafe_allow_html=True,
)

selected_product = st.sidebar.selectbox("选择产品", product_names, index=0)

# 日期范围（用字符串避免前端sprintf兼容问题）
months = sorted(act_df["月份"].unique())
month_labels = [m.strftime("%Y-%m") for m in months]
date_range_label = st.sidebar.select_slider(
    "分析期间",
    options=month_labels,
    value=(month_labels[0], month_labels[-1]),
)
# 转为Timestamp用于筛选
start_date = pd.Timestamp(date_range_label[0] + "-01")
end_date = pd.Timestamp(date_range_label[1] + "-01")

# BOM视图
bom_view = st.sidebar.radio("BOM视图", ["层级树", "成本结构"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style="font-size:0.8rem; color:{C['muted']};">
    <b>固定参数</b><br>
    固定成本: {fmt_money(FIXED_COST_MONTHLY)}/月<br>
    直接人工: ¥{DIRECT_LABOR_STD}/件<br>
    变动制造费: ¥{VAR_OH_STD:.2f}/件<br>
    固定制造费: ¥{FIX_OH_STD:.2f}/件<br>
    总产能: {MAX_CAPACITY_UNITS:,.0f} 件/月
    </div>
    """,
    unsafe_allow_html=True,
)

# 筛选
prod_act = act_df[
    (act_df["产品名称"] == selected_product) &
    (act_df["月份"] >= start_date) &
    (act_df["月份"] <= end_date)
].sort_values("月份")

prod_var = var_df[
    (var_df["产品名称"] == selected_product) &
    (var_df["月份"] >= start_date) &
    (var_df["月份"] <= end_date)
].sort_values("月份")


# ════════════════════════════════════════════════════════════
# TAB 1：成本预测与差异分析
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📈 成本分析", "🔧 BOM成本拆解", "🎯 盈亏预测与模拟"])

with tab1:
    s = std_map[selected_product]
    line = product_line_map[selected_product]

    # ── 摘要卡片 ──
    st.markdown(f"### {selected_product} — 成本概览")
    st.caption(f"产线: {line} | 标准产能: {s['std_vol']:,.0f} 件/月 | 标准工时: {STD_HOURS_PER_UNIT}h/件")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("标准单位成本", fmt_money(s["total"]),
                  help=f"材料 {fmt_money(s['material'])} + 人工 {fmt_money(s['labor'])} + 变动制造费 {fmt_money(s['var_oh'])} + 固定制造费 {fmt_money(s['fix_oh'])}")

    with col2:
        if len(prod_act) > 0:
            latest = prod_act.iloc[-1]
            delta = latest["实际单位成本"] - s["total"]
            st.metric("最新实际成本", fmt_money(latest["实际单位成本"]),
                      delta=f"{'⬆' if delta > 0 else '⬇'} {fmt_money(abs(delta))}",
                      delta_color="inverse" if delta > 0 else "normal")

    with col3:
        if len(prod_act) > 0:
            latest = prod_act.iloc[-1]
            total_diff = latest["月实际总成本"] - s["total"] * latest["实际产量"]
            st.metric("月总差异", fmt_money(total_diff),
                      delta=f"产量 {latest['实际产量']:,.0f} 件",
                      delta_color="inverse" if total_diff > 0 else "normal")

    with col4:
        if len(prod_act) > 0:
            latest = prod_act.iloc[-1]
            util = latest["实际产量"] / s["std_vol"] * 100
            st.metric("产能利用率", f"{util:.1f}%",
                      delta=f"标准 {s['std_vol']:,.0f} 件/月",
                      delta_color="inverse" if util < 80 else "normal")

    # ── 成本结构对比 ──
    st.markdown("### 成本结构对比（标准 vs 实际）")
    if len(prod_act) > 0:
        latest = prod_act.iloc[-1]
        cost_items = [
            ("直接材料", s["material"], latest["直接材料实际"]),
            ("直接人工", s["labor"], latest["直接人工实际"]),
            ("变动制造费用", s["var_oh"], latest["变动制造费用实际"]),
            ("固定制造费用", s["fix_oh"], latest["固定制造费用实际"]),
        ]

        fig_struct = go.Figure()
        cats = [x[0] for x in cost_items]
        fig_struct.add_trace(go.Bar(
            name="标准", x=cats, y=[x[1] for x in cost_items],
            marker_color=C["secondary"],
            text=[fmt_money(x[1]) for x in cost_items],
            textposition="outside",
        ))
        fig_struct.add_trace(go.Bar(
            name=f"实际 ({latest['月份'].strftime('%Y-%m')})",
            x=cats, y=[x[2] for x in cost_items],
            marker_color=C["primary"],
            text=[fmt_money(x[2]) for x in cost_items],
            textposition="outside",
            opacity=0.85,
        ))

        fig_struct.update_layout(
            height=350, barmode="group",
            hovermode="x unified",
            xaxis=dict(title="", gridcolor=C["grid"]),
            yaxis=dict(title="单件成本 (元)", gridcolor=C["grid"]),
            plot_bgcolor="white", margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_struct, use_container_width=True)

    # ── 月度成本趋势 ──
    st.markdown("### 月度成本趋势")
    if len(prod_act) > 0:
        fig_trend = go.Figure()

        fig_trend.add_trace(go.Scatter(
            x=prod_act["月份"], y=prod_act["实际单位成本"],
            mode="lines+markers", name="实际单件成本",
            line=dict(color=C["primary"], width=2.5),
            marker=dict(size=10, color=C["primary"]),
        ))

        fig_trend.add_trace(go.Scatter(
            x=prod_act["月份"],
            y=[s["total"]] * len(prod_act),
            mode="lines", name=f"标准成本 ({fmt_money(s['total'])})",
            line=dict(color=C["muted"], width=2, dash="dash"),
        ))

        # 场景标注
        for _, row in prod_act.iterrows():
            desc = row.get("场景说明", "")
            if desc and pd.notna(desc):
                fig_trend.add_annotation(
                    x=row["月份"], y=row["实际单位成本"],
                    text=desc[:10] + "…" if len(desc) > 10 else desc,
                    showarrow=True, arrowhead=2, arrowsize=1,
                    ax=0, ay=-30, font=dict(size=10, color=C["accent"]),
                )

        fig_trend.update_layout(
            height=380, hovermode="x unified",
            xaxis=dict(title="", gridcolor=C["grid"]),
            yaxis=dict(title="单件成本 (元)", gridcolor=C["grid"]),
            plot_bgcolor="white", margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # ── 差异分析 ──
    if len(prod_var) > 0:
        st.markdown("### 差异分析明细")
        st.caption("8项差异：材料价格差 · 材料用量差 · 人工工资率差 · 人工效率差 · 变动制造费耗费差 · 变动制造费效率差 · 固定制造费耗费差 · 固定制造费产能差")

        diff_cols = [
            "材料价格差异", "材料用量差异", "人工工资率差异", "人工效率差异",
            "变动制造费用耗费差异", "变动制造费用效率差异",
            "固定制造费用耗费差异", "固定制造费用产能差异",
        ]
        diff_display = prod_var[["月份"] + diff_cols + ["总差异合计"]].copy()
        diff_display["月份"] = diff_display["月份"].dt.strftime("%Y-%m")
        for c in diff_cols + ["总差异合计"]:
            diff_display[c] = diff_display[c].apply(lambda x: f"{x:+.2f}")

        st.dataframe(diff_display, use_container_width=True, hide_index=True)

        # 差异瀑布图（最近月份）
        st.markdown("#### 差异瀑布图（最近月份）")
        latest_v = prod_var.iloc[-1]
        waterfall_items = [
            ("标准成本", s["total"]),
            ("材料价格差", latest_v["材料价格差异"]),
            ("材料用量差", latest_v["材料用量差异"]),
            ("工资率差", latest_v["人工工资率差异"]),
            ("效率差", latest_v["人工效率差异"]),
            ("变动制造费耗费差", latest_v["变动制造费用耗费差异"]),
            ("变动制造费效率差", latest_v["变动制造费用效率差异"]),
            ("固定制造费产能差", latest_v["固定制造费用产能差异"]),
            ("实际成本", prod_act.iloc[-1]["实际单位成本"]),
        ]

        measures = []
        texts = []
        vals = []
        for name, val in waterfall_items:
            if name == "标准成本":
                measures.append("absolute")
                vals.append(val)
            elif name == "实际成本":
                measures.append("total")
                vals.append(val)
            else:
                measures.append("relative")
                vals.append(val)
            texts.append(name)

        fig_wf = go.Figure(go.Waterfall(
            orientation="v", measure=measures, x=texts, y=vals,
            text=[fmt_money(v) for v in vals],
            textposition="outside",
            connector=dict(line=dict(color=C["grid"], width=2)),
            decreasing=dict(marker=dict(color=C["positive"])),
            increasing=dict(marker=dict(color=C["negative"])),
            totals=dict(marker=dict(color=C["primary"])),
        ))
        fig_wf.update_layout(
            height=350, showlegend=False,
            margin=dict(l=40, r=20, t=20, b=40),
            plot_bgcolor="white",
            xaxis=dict(gridcolor=C["grid"]),
            yaxis=dict(title="单件成本 (元)", gridcolor=C["grid"]),
        )
        st.plotly_chart(fig_wf, use_container_width=True)

    # ── 成本预测（趋势外推） ──
    st.markdown("### 成本预测（基于趋势外推）")
    st.caption("基于过去月份趋势推测未来，仅供参考")

    if len(prod_act) >= 2:
        x = np.arange(len(prod_act))
        y = prod_act["实际单位成本"].values

        coeffs = np.polyfit(x, y, 1)
        trend = np.poly1d(coeffs)

        # 预测未来1个月
        future_x = np.arange(len(prod_act), len(prod_act) + 1)
        pred_y = trend(future_x)

        residuals = y - trend(x)
        std_err = np.std(residuals) if len(residuals) > 1 else 0

        last_month = prod_act["月份"].iloc[-1]
        next_month = last_month + pd.DateOffset(months=1)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=prod_act["月份"], y=y,
            mode="lines+markers", name="历史实际成本",
            line=dict(color=C["primary"], width=2.5),
            marker=dict(size=8),
        ))
        fig_pred.add_trace(go.Scatter(
            x=[next_month], y=[pred_y[0]],
            mode="markers", name="预测成本",
            marker=dict(size=12, symbol="diamond", color=C["accent"]),
        ))
        fig_pred.add_trace(go.Scatter(
            x=prod_act["月份"].tolist() + [next_month],
            y=[s["total"]] * (len(prod_act) + 1),
            mode="lines", name=f"标准成本 ({fmt_money(s['total'])})",
            line=dict(color=C["muted"], width=1.5, dash="dash"),
        ))

        fig_pred.update_layout(
            height=300, hovermode="x unified",
            xaxis=dict(title="", gridcolor=C["grid"]),
            yaxis=dict(title="单件成本 (元)", gridcolor=C["grid"]),
            plot_bgcolor="white", margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        pred_data = [{
            "预测月份": next_month.strftime("%Y-%m"),
            "预测单件成本": fmt_money(pred_y[0]),
            "vs 标准成本": fmt_pct((pred_y[0] - s["total"]) / s["total"] * 100),
            "vs 最近实际": fmt_pct((pred_y[0] - y[-1]) / y[-1] * 100),
        }]
        if std_err > 0:
            pred_data[0]["预测区间"] = f"{fmt_money(pred_y[0] - 1.96*std_err)} ~ {fmt_money(pred_y[0] + 1.96*std_err)}"
        st.dataframe(pd.DataFrame(pred_data), use_container_width=True, hide_index=True)
    else:
        st.info("数据量不足，无法进行成本预测")


# ════════════════════════════════════════════════════════════
# TAB 2：BOM成本拆解
# ════════════════════════════════════════════════════════════
with tab2:
    s = std_map[selected_product]

    # 成本结构饼图
    st.markdown(f"### {selected_product} — 成本结构")

    cost_breakdown = {
        "直接材料": s["material"],
        "直接人工": s["labor"],
        "变动制造费用": s["var_oh"],
        "固定制造费用": s["fix_oh"],
    }

    colors_pie = [C["material"], C["labor"], C["var_oh"], C["fix_oh"]]

    fig_pie = go.Figure(data=[go.Pie(
        labels=list(cost_breakdown.keys()),
        values=list(cost_breakdown.values()),
        marker=dict(colors=colors_pie),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value:,.2f} 元 (%{percent})<extra></extra>",
    )])
    fig_pie.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # 成本结构明细表
    st.markdown("#### 成本要素明细")
    detail_rows = []
    for k, v in cost_breakdown.items():
        detail_rows.append({
            "成本项目": k,
            "单件金额(元)": f"{v:,.2f}",
            "占比": f"{v/s['total']*100:.1f}%",
        })
    detail_rows.append({
        "成本项目": "合计",
        "单件金额(元)": f"{s['total']:,.2f}",
        "占比": "100%",
    })
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

    # 成本结构说明
    st.markdown("#### 成本构成说明")
    st.markdown(f"""
    ```
    标准单位成本 = 直接材料 + 直接人工 + 变动制造费用 + 固定制造费用
    
    直接材料    : {fmt_money(s['material'])}/件（按BOM材料清单×标准单价）
    直接人工    : ¥{DIRECT_LABOR_STD:.2f}/件（标准工时{STD_HOURS_PER_UNIT}h × 标准工资率¥40/h）
    变动制造费用 : ¥{VAR_OH_STD:.2f}/件（能耗+维修+物料消耗，¥59.2/h × {STD_HOURS_PER_UNIT}h）
    固定制造费用 : ¥{FIX_OH_STD:.2f}/件（设备折旧分摊，¥{416.7+199.1:.1f}/h × {STD_HOURS_PER_UNIT}h）
    
    固定成本总额: {fmt_money(FIXED_COST_MONTHLY)}/月
    ├─ 产线级固定: {fmt_money(11_700_000)}/月（设备折旧 ¥416.7/h）
    └─ 工厂级固定: {fmt_money(5_590_000)}/月（管理+厂房 ¥199.1/h）
    ```
    """)

    # ── 敏感度模拟（材料价格影响） ──
    st.markdown("### 敏感度模拟 — 材料价格变动影响")
    st.caption("模拟石油衍生材料价格波动对成本的影响")

    mat_change = st.slider(
        "石油衍生材料价格变动幅度",
        min_value=-30, max_value=50, value=0, step=5,
        help="石油衍生材料（PP/ABS/PC/PA/PU等）统一调整",
    )

    # 材料成本中石油衍生材料占比约70%
    OIL_MATERIAL_RATIO = 0.70
    impact = s["material"] * OIL_MATERIAL_RATIO * (mat_change / 100)
    new_material = s["material"] + impact
    new_total = s["total"] + impact

    col1, col2, col3 = st.columns(3)
    col1.metric("标准材料成本", fmt_money(s["material"]))
    col2.metric("模拟后材料成本", fmt_money(new_material),
                delta=f"{'⬆' if impact > 0 else '⬇'} {fmt_money(abs(impact))}")
    col3.metric("模拟后总成本", fmt_money(new_total),
                delta=f"{'⬆' if impact > 0 else '⬇'} {fmt_money(abs(impact))}")


    # ════════════════════════════════════════════════════════════
    # 新增：半成品成本与WIP在制品模拟
    # ════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📦 半成品成本与WIP在制品模拟")
    st.caption("基于BOM结构识别各工序半成品节点，计算材料成本完工率，并模拟在制品(WIP)对单位成本的影响")

    # 计算半成品数据
    sf_data, bom_info = calc_semi_finished_costs(bom_df, selected_product, std_map)

    if sf_data and bom_info:
        s = std_map[selected_product]

        # ── 工序结构图 ──
        st.markdown("#### 工序划分与BOM成本积累")
        st.caption(f"该产品BOM含 {bom_info['raw_count']} 种原材料，总材料用量 {bom_info['total_raw_qty']:.2f} 单位")

        # 工序流程说明
        process_stages = [
            ("① 基材成型", "骨架/基体注塑成型", "材料消耗约 30-50%"),
            ("② 覆面层加工", "表皮/包覆/隔音层贴合", "材料消耗约 15-25%"),
            ("③ 半成品组装", "紧固件/嵌件/面料装配", "半成品节点完工"),
            ("④ 总成检验", "功能检测+包装入库", "成品完工"),
        ]

        stage_cols = st.columns(4)
        for i, (stage_name, stage_desc, stage_mat) in enumerate(process_stages):
            with stage_cols[i]:
                if i == 2:  # 半成品节点高亮
                    pct = sf_data[0]["mat_completion_pct"]
                    st.markdown(
                        f"""<div style="background:#E8F4FD; border:2px solid {C['secondary']};
                        border-radius:8px; padding:10px; text-align:center;">
                        <b style="color:{C['primary']}">{stage_name}</b><br>
                        <span style="font-size:0.8rem;color:{C['muted']}">{stage_desc}</span><br>
                        <span style="color:{C['secondary']};font-weight:bold;font-size:1.1rem;">
                        材料完工 {pct:.1f}%</span>
                        </div>""", unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""<div style="background:#F5F7FA; border:1px solid {C['grid']};
                        border-radius:8px; padding:10px; text-align:center;">
                        <b style="color:{C['muted']}">{stage_name}</b><br>
                        <span style="font-size:0.8rem;color:{C['muted']}">{stage_desc}</span><br>
                        <span style="font-size:0.8rem;color:gray;">{stage_mat}</span>
                        </div>""", unsafe_allow_html=True,
                    )

        # ── 半成品成本明细表 ──
        st.markdown("#### 半成品节点成本明细")

        # 计算各工序的成本
        mat_completion = sf_data[0]["mat_completion_pct"] / 100
        # 制造过程按4个阶段均分转换成本（人工+制造费用）
        conv_to_stage3 = 0.75  # 3/4的转换工序已完成

        # 半成品节点成本
        semi_material = s["material"] * mat_completion
        semi_labor = s["labor"] * conv_to_stage3
        semi_var_oh = s["var_oh"] * conv_to_stage3
        semi_fix_oh = s["fix_oh"] * conv_to_stage3
        semi_total = semi_material + semi_labor + semi_var_oh + semi_fix_oh

        cost_rows = [
            {
                "成本项目": "直接材料",
                "成品成本(元)": f"{s['material']:,.2f}",
                "半成品(元)": f"{semi_material:,.2f}",
                "完工率": f"{mat_completion*100:.1f}%",
                "说明": f"半成品节点前已消耗 {sf_data[0]['upstream_qty']:.0f}/{sf_data[0]['total_qty']:.0f} 材料用量单位",
            },
            {
                "成本项目": "直接人工",
                "成品成本(元)": f"{s['labor']:,.2f}",
                "半成品(元)": f"{semi_labor:,.2f}",
                "完工率": "75.0%",
                "说明": "4道工序中已完成3道（基材→覆面→组装）",
            },
            {
                "成本项目": "变动制造费用",
                "成品成本(元)": f"{s['var_oh']:,.2f}",
                "半成品(元)": f"{semi_var_oh:,.2f}",
                "完工率": "75.0%",
                "说明": "按工序进度同比例分摊",
            },
            {
                "成本项目": "固定制造费用",
                "成品成本(元)": f"{s['fix_oh']:,.2f}",
                "半成品(元)": f"{semi_fix_oh:,.2f}",
                "完工率": "75.0%",
                "说明": "按工序进度同比例分摊",
            },
            {
                "成本项目": "合计",
                "成品成本(元)": f"{s['total']:,.2f}",
                "半成品(元)": f"{semi_total:,.2f}",
                "完工率": f"{semi_total/s['total']*100:.1f}%",
                "说明": "半成品单位成本 vs 成品单位成本",
            },
        ]
        st.dataframe(pd.DataFrame(cost_rows), use_container_width=True, hide_index=True)

        # 半成品成本结构图
        fig_semi = go.Figure()
        fig_semi.add_trace(go.Bar(
            name="成品", x=["直接材料", "直接人工", "变动制造费", "固定制造费"],
            y=[s["material"], s["labor"], s["var_oh"], s["fix_oh"]],
            marker_color=[C["material"], C["labor"], C["var_oh"], C["fix_oh"]],
            text=[fmt_money(v) for v in [s["material"], s["labor"], s["var_oh"], s["fix_oh"]]],
            textposition="outside", opacity=0.6,
        ))
        fig_semi.add_trace(go.Bar(
            name="半成品", x=["直接材料", "直接人工", "变动制造费", "固定制造费"],
            y=[semi_material, semi_labor, semi_var_oh, semi_fix_oh],
            marker_color=[C["material"], C["labor"], C["var_oh"], C["fix_oh"]],
            text=[fmt_money(v) for v in [semi_material, semi_labor, semi_var_oh, semi_fix_oh]],
            textposition="outside",
        ))
        fig_semi.update_layout(
            height=300, barmode="group",
            hovermode="x unified",
            xaxis=dict(title="", gridcolor=C["grid"]),
            yaxis=dict(title="单件成本 (元)", gridcolor=C["grid"]),
            plot_bgcolor="white", margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_semi, use_container_width=True)

        # ── WIP在制品模拟 ──
        st.markdown("#### WIP在制品模拟")
        st.caption("设定在制品数量和完工进度，模拟对单位成本的影响（参考半成品成本结构）")

        wip_col1, wip_col2 = st.columns(2)
        with wip_col1:
            wip_qty = st.number_input(
                "在制品数量（件）", min_value=0, max_value=50000,
                value=int(s["std_vol"] * 0.3), step=100,
                help="月末盘点在制品数量",
            )
        with wip_col2:
            # 默认完工率 = 半成品成本完工率，用户可调
            default_completion = round(semi_total / s["total"] * 100)
            wip_completion = st.slider(
                "在制品平均完工率（%）", 5, 95, default_completion, 5,
                help=f"BOM推算参考值: 材料 {mat_completion*100:.0f}% / 转换成本 75%",
            )

        wip_pct = wip_completion / 100

        # 计算约当产量
        if len(prod_act) > 0:
            latest = prod_act.iloc[-1]
            actual_vol = latest["实际产量"]
            equiv_units = wip_qty * wip_pct
            total_equiv = actual_vol + equiv_units

            if total_equiv > 0:
                total_cost = latest["月实际总成本"]
                cost_per_unit_no_wip = s["total"]
                cost_per_unit_with_wip = total_cost / total_equiv

                wip_c1, wip_c2, wip_c3, wip_c4 = st.columns(4)
                with wip_c1:
                    st.metric("完工产量", f"{actual_vol:,.0f}")
                with wip_c2:
                    st.metric("在制品数量", f"{wip_qty:,.0f}",
                              delta=f"完工率 {wip_completion}% → 约当 {equiv_units:,.0f} 件")
                with wip_c3:
                    st.metric("总约当产量", f"{total_equiv:,.0f}",
                              delta=f"+ {equiv_units:,.0f} WIP约当")
                with wip_c4:
                    delta_amt = cost_per_unit_with_wip - cost_per_unit_no_wip
                    st.metric("调整后单位成本", fmt_money(cost_per_unit_with_wip),
                              delta=f"{'⬇' if delta_amt < 0 else '⬆'} {fmt_money(abs(delta_amt))} vs 标准",
                              delta_color="inverse" if delta_amt > 0 else "normal")

                # WIP影响说明
                wip_impact = (cost_per_unit_with_wip - cost_per_unit_no_wip) / cost_per_unit_no_wip * 100
                st.info(
                    f"**WIP影响分析**: 考虑在制品后，单位成本从 {fmt_money(cost_per_unit_no_wip)} "
                    f"{'下降' if wip_impact < 0 else '上升'} {abs(wip_impact):.1f}% 至 "
                    f"{fmt_money(cost_per_unit_with_wip)}。"
                    f"WIP约当 {equiv_units:,.0f} 件占总约当产量 {total_equiv:,.0f} 件的 "
                    f"{equiv_units/total_equiv*100:.1f}%。"
                    if wip_impact != 0 else
                    f"**WIP影响分析**: WIP对单位成本无显著影响"
                )

                # 完工率敏感度曲线
                if wip_qty > 0:
                    st.markdown("##### 不同完工率下的单位成本")
                    completion_range = list(range(5, 100, 5))
                    costs = []
                    for pct in completion_range:
                        eq = wip_qty * pct / 100
                        costs.append(total_cost / (actual_vol + eq) if (actual_vol + eq) > 0 else 0)

                    fig_wip = go.Figure()
                    fig_wip.add_trace(go.Scatter(
                        x=completion_range, y=costs,
                        mode="lines+markers",
                        name="单位成本 vs 完工率",
                        line=dict(color=C["secondary"], width=2.5),
                        marker=dict(size=6),
                    ))
                    fig_wip.add_hline(
                        y=cost_per_unit_no_wip,
                        line=dict(color=C["muted"], width=1.5, dash="dash"),
                        annotation_text=f"不含WIP: {fmt_money(cost_per_unit_no_wip)}",
                        annotation_position="top right",
                    )
                    fig_wip.add_vline(
                        x=wip_completion,
                        line=dict(color=C["accent"], width=2, dash="dot"),
                        annotation_text=f"当前: {wip_completion}%",
                        annotation_position="bottom",
                    )
                    fig_wip.update_layout(
                        height=280,
                        hovermode="x unified",
                        xaxis=dict(title="在制品完工率 (%)", gridcolor=C["grid"]),
                        yaxis=dict(title="单位成本 (元)", gridcolor=C["grid"]),
                        plot_bgcolor="white", margin=dict(l=40, r=20, t=20, b=40),
                    )
                    st.plotly_chart(fig_wip, use_container_width=True)
            else:
                st.info("当前产品无实际产量数据，无法进行WIP模拟")
        else:
            st.info("当前产品无实际产量数据，无法进行WIP模拟")
    else:
        st.info("当前产品在BOM中未找到半成品结构数据")


with tab3:
    # ─── 全局参数滑块 ─────────────────────────────────────────
    st.markdown("### 经营参数设定")
    st.caption("以下分析基于工厂整体汇总数据，统一毛利率倒推售价")

    col_margin, col_vol, col_mat = st.columns(3)
    with col_margin:
        margin_rate = st.slider("毛利率目标", 10, 50, 25, 1,
                                help="目标毛利率，用于倒推统一售价")
    with col_vol:
        vol_factor = st.slider("产量系数", 50, 150, 100, 5,
                               help="实际产量占标准产能的比例")
    with col_mat:
        mat_factor = st.slider("材料价格调整系数", 80, 120, 100, 2,
                               help="材料价格相对标准的变化")

    margin_pct = margin_rate / 100
    vol_pct = vol_factor / 100
    mat_pct = mat_factor / 100

    # ─── 计算工厂整体数据 ─────────────────────────────────────
    def calc_factory_level(margin_pct, vol_pct, mat_pct):
        """计算工厂整体盈亏数据"""
        total_fixed = FIXED_COST_MONTHLY

        products_data = []
        total_units = 0
        total_revenue = 0
        total_variable_cost = 0
        total_contribution = 0

        for _, row in std_df.iterrows():
            pname = row["产品名称"]
            std_vol = row["标准产量"]
            mat_std = row["直接材料标准"]
            lab_std = row["直接人工标准"]
            var_oh_std = row["变动制造费用标准"]
            fix_oh_std = row["固定制造费用标准"]
            total_std = row["标准单位成本"]

            # 实际产量
            actual_vol = std_vol * vol_pct
            # 实际材料成本（考虑材料价格调整）
            actual_mat = mat_std * mat_pct
            # 变动成本 = 调整后材料 + 人工 + 变动制造费
            var_cost_per_unit = actual_mat + lab_std + var_oh_std
            # 售价 = 标准成本 ÷ (1 - 毛利率)
            selling_price = total_std / (1 - margin_pct)
            # 边际贡献
            cm_per_unit = selling_price - var_cost_per_unit

            units = actual_vol
            revenue = selling_price * units
            var_cost_total = var_cost_per_unit * units
            contribution = cm_per_unit * units

            total_units += units
            total_revenue += revenue
            total_variable_cost += var_cost_total
            total_contribution += contribution

            products_data.append({
                "产品": pname,
                "产线": row["产线"],
                "标准产量": int(std_vol),
                "实际产量": int(round(units)),
                "售价": selling_price,
                "变动成本": var_cost_per_unit,
                "边际贡献": cm_per_unit,
                "收入": revenue,
                "变动成本总额": var_cost_total,
                "边际贡献总额": contribution,
            })

        profit = total_contribution - total_fixed
        # 盈亏平衡计算
        if total_units > 0:
            avg_cm = total_contribution / total_units
            bep_units = total_fixed / avg_cm if avg_cm > 0 else float("inf")
            bep_revenue = bep_units * (total_revenue / total_units) if total_units > 0 else 0
        else:
            avg_cm = 0
            bep_units = float("inf")
            bep_revenue = 0

        # 产能利用率
        capacity_util = total_units / MAX_CAPACITY_UNITS * 100

        return {
            "products": products_data,
            "total_units": total_units,
            "total_revenue": total_revenue,
            "total_variable_cost": total_variable_cost,
            "total_contribution": total_contribution,
            "total_fixed": total_fixed,
            "profit": profit,
            "avg_cm": avg_cm,
            "bep_units": bep_units,
            "bep_revenue": bep_revenue,
            "capacity_util": capacity_util,
        }

    factory = calc_factory_level(margin_pct, vol_pct, mat_pct)


    # ════════════════════════════════════════════════════════════
    # 卡片区 1：盈亏平衡分析
    # ════════════════════════════════════════════════════════════
    st.markdown("### 📉 盈亏平衡分析")

    # 盈亏平衡核心指标
    bep_col1, bep_col2, bep_col3, bep_col4 = st.columns(4)

    with bep_col1:
        st.metric(
            "盈亏平衡点（产量）",
            f"{factory['bep_units']:,.0f}" if factory['bep_units'] != float("inf") else "∞",
            delta=f"产能 {factory['bep_units']/MAX_CAPACITY_UNITS*100:.1f}%" if factory['bep_units'] != float("inf") else None,
            help="达到此产量时盈亏平衡",
        )

    with bep_col2:
        st.metric(
            "盈亏平衡点（收入）",
            fmt_money(factory['bep_revenue']) if factory['bep_revenue'] != float("inf") else "∞",
        )

    with bep_col3:
        profit_color = "normal" if factory["profit"] >= 0 else "inverse"
        st.metric(
            "当月利润",
            fmt_money(factory["profit"]),
            delta=f"产量 {factory['total_units']:,.0f} 件",
            delta_color=profit_color,
        )

    with bep_col4:
        util_color = "normal" if factory["capacity_util"] <= 100 else "inverse"
        util_warning = ""
        if factory["capacity_util"] > 110:
            util_warning = "🔴 超红线"
        elif factory["capacity_util"] > 100:
            util_warning = "🟠 超产能"
        st.metric(
            "产能利用率",
            f"{factory['capacity_util']:.1f}%",
            delta=util_warning if util_warning else f"上限 {MAX_CAPACITY_UNITS:,.0f} 件",
            delta_color=util_color,
        )

    # 盈亏平衡图
    st.markdown("#### 盈亏平衡图")
    st.caption("总收入线 vs 总成本线，交点即为盈亏平衡点。红色竖线 = 产能上限")

    # 生成产量范围（0 ~ 产能上限×1.2）
    max_x = MAX_CAPACITY_UNITS * 1.2
    x_units = np.linspace(0, max_x, 200)

    # 总成本 = 固定成本 + 变动成本 × 产量
    var_cost_per_unit_avg = factory["total_variable_cost"] / factory["total_units"] if factory["total_units"] > 0 else 0
    total_cost_line = FIXED_COST_MONTHLY + var_cost_per_unit_avg * x_units

    # 总收入 = 售价 × 产量
    avg_price = factory["total_revenue"] / factory["total_units"] if factory["total_units"] > 0 else 0
    total_revenue_line = avg_price * x_units

    fig_bep = go.Figure()

    # 总成本线
    fig_bep.add_trace(go.Scatter(
        x=x_units, y=total_cost_line,
        mode="lines", name="总成本",
        line=dict(color=C["negative"], width=2.5),
    ))

    # 总收入线
    fig_bep.add_trace(go.Scatter(
        x=x_units, y=total_revenue_line,
        mode="lines", name="总收入",
        line=dict(color=C["revenue"], width=2.5),
    ))

    # 固定成本线
    fig_bep.add_trace(go.Scatter(
        x=x_units, y=[FIXED_COST_MONTHLY] * len(x_units),
        mode="lines", name=f"固定成本 ({fmt_money(FIXED_COST_MONTHLY)})",
        line=dict(color=C["muted"], width=1.5, dash="dot"),
    ))

    # 盈亏平衡点
    if factory["bep_units"] != float("inf") and factory["bep_units"] <= max_x:
        bep_y = FIXED_COST_MONTHLY + var_cost_per_unit_avg * factory["bep_units"]
        fig_bep.add_trace(go.Scatter(
            x=[factory["bep_units"]], y=[bep_y],
            mode="markers+text",
            name=f"BEP ({factory['bep_units']:,.0f} 件)",
            marker=dict(symbol="star", size=16, color=C["bep"],
                        line=dict(width=2, color="white")),
            text=["BEP"],
            textposition="top center",
            textfont=dict(size=14, color=C["bep"]),
        ))

    # 当前产量点
    if factory["total_units"] > 0:
        current_cost = FIXED_COST_MONTHLY + var_cost_per_unit_avg * factory["total_units"]
        current_rev = avg_price * factory["total_units"]
        fig_bep.add_trace(go.Scatter(
            x=[factory["total_units"]], y=[current_cost],
            mode="markers", name="当前成本",
            marker=dict(symbol="circle", size=12, color=C["primary"]),
        ))
        fig_bep.add_trace(go.Scatter(
            x=[factory["total_units"]], y=[current_rev],
            mode="markers", name="当前收入",
            marker=dict(symbol="circle", size=12, color=C["revenue"]),
        ))

    # 产能红线
    fig_bep.add_vline(
        x=MAX_CAPACITY_UNITS,
        line=dict(color=C["capacity"], width=2.5, dash="dash"),
        annotation_text=f"产能上限 {MAX_CAPACITY_UNITS:,.0f} 件",
        annotation_position="top right",
        annotation_font=dict(size=12, color=C["capacity"]),
    )

    # 超产能警告区域
    if factory["capacity_util"] > 100:
        fig_bep.add_vrect(
            x0=MAX_CAPACITY_UNITS * 0.95,
            x1=MAX_CAPACITY_UNITS * 1.05,
            fillcolor="rgba(241, 143, 1, 0.15)",
            layer="below",
            line_width=0,
        )

    fig_bep.update_layout(
        height=450,
        hovermode="x unified",
        xaxis=dict(title="产量 (件)", gridcolor=C["grid"],
                   range=[0, max_x]),
        yaxis=dict(title="金额 (元)", gridcolor=C["grid"]),
        plot_bgcolor="white",
        margin=dict(l=40, r=30, t=20, b=40),
        legend=dict(orientation="h", y=1.1),
    )

    st.plotly_chart(fig_bep, use_container_width=True)

    # 盈亏平衡说明
    bep_status = "🟢 盈利" if factory["profit"] > 0 else ("🔴 亏损" if factory["profit"] < 0 else "🟡 盈亏平衡")
    st.info(
        f"**当前状态**: {bep_status} | "
        f"实际产量 {factory['total_units']:,.0f} 件 vs BEP {factory['bep_units']:,.0f} 件 | "
        f"安全边际 {(factory['total_units'] - factory['bep_units'])/factory['total_units']*100:.1f}%"
        if factory['total_units'] > 0 and factory['bep_units'] != float("inf") and factory['total_units'] > factory['bep_units']
        else f"**当前状态**: {bep_status} | 未达到盈亏平衡点"
    )


    # ════════════════════════════════════════════════════════════
    # 卡片区 2：敏感度分析（Tornado Chart）
    # ════════════════════════════════════════════════════════════
    st.markdown("### 🔄 成本因素敏感度分析（±10%变动对利润影响）")
    st.caption("以当前参数为基准，各成本因素单独变动±10%对月利润的影响程度")

    # 基准利润
    base_profit = factory["profit"]

    # 分析5个因素
    factors = {
        "直接材料": {
            "impact": lambda pct: calc_factory_level(margin_pct, vol_pct, mat_pct * (1 + pct/100))["profit"],
            "pct_range": [-10, 10],
        },
        "直接人工": {
            "impact": lambda pct: calc_factory_level(margin_pct, vol_pct, mat_pct)["profit"]
            - (factory["total_units"] * DIRECT_LABOR_STD * (pct/100)),
            "pct_range": [-10, 10],
        },
        "变动制造费用": {
            "impact": lambda pct: calc_factory_level(margin_pct, vol_pct, mat_pct)["profit"]
            - (factory["total_units"] * VAR_OH_STD * (pct/100)),
            "pct_range": [-10, 10],
        },
        "固定成本": {
            "impact": lambda pct: base_profit - (FIXED_COST_MONTHLY * (pct/100)),
            "pct_range": [-10, 10],
        },
        "售价（毛利率）": {
            "impact": lambda pct: calc_factory_level((margin_pct * (1 + pct/100)), vol_pct, mat_pct)["profit"],
            "pct_range": [-10, 10],
        },
    }

    tornado_data = []
    for name, config in factors.items():
        profit_low = config["impact"](config["pct_range"][0])
        profit_high = config["impact"](config["pct_range"][1])
        impact_low = profit_low - base_profit
        impact_high = profit_high - base_profit
        # tornado chart 显示范围：从最小值到最大值
        tornado_data.append({
            "因素": name,
            "下限": min(impact_low, impact_high),
            "上限": max(impact_low, impact_high),
            "影响幅度": max(abs(impact_low), abs(impact_high)),
            "low_label": f"{fmt_money(profit_low)}",
            "high_label": f"{fmt_money(profit_high)}",
        })

    tornado_df = pd.DataFrame(tornado_data)
    tornado_df = tornado_df.sort_values("影响幅度", ascending=True)

    fig_tornado = go.Figure()

    # Tornado chart: 水平条形图，以基准为中线
    y_labels = tornado_df["因素"].tolist()

    fig_tornado.add_trace(go.Bar(
        y=y_labels,
        x=tornado_df["下限"],
        orientation="h",
        name="-10%（不利）",
        marker_color=C["negative"],
        text=tornado_df["low_label"],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>-10%: %{x:+,.0f} 元<extra></extra>",
    ))

    fig_tornado.add_trace(go.Bar(
        y=y_labels,
        x=tornado_df["上限"],
        orientation="h",
        name="+10%（有利）",
        marker_color=C["positive"],
        text=tornado_df["high_label"],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>+10%: %{x:+,.0f} 元<extra></extra>",
    ))

    # 基准线
    fig_tornado.add_vline(
        x=0,
        line=dict(color=C["primary"], width=1.5, dash="dash"),
        annotation_text=f"基准利润 {fmt_money(base_profit)}",
        annotation_position="top",
        annotation_font=dict(size=11, color=C["primary"]),
    )

    fig_tornado.update_layout(
        height=350,
        barmode="relative",
        hovermode="y unified",
        xaxis=dict(title="利润变动 (元)", gridcolor=C["grid"]),
        yaxis=dict(title="", gridcolor=C["grid"]),
        plot_bgcolor="white",
        margin=dict(l=40, r=30, t=20, b=40),
        legend=dict(orientation="h", y=1.1),
    )

    st.plotly_chart(fig_tornado, use_container_width=True)

    # 敏感度排序表
    st.markdown("#### 敏感度排序（按影响幅度）")
    tornado_sort = tornado_df.sort_values("影响幅度", ascending=False)
    tornado_sort["影响幅度"] = tornado_sort["影响幅度"].apply(fmt_money)
    tornado_sort["下限"] = tornado_sort["low_label"]
    tornado_sort["上限"] = tornado_sort["high_label"]
    st.dataframe(
        tornado_sort[["因素", "下限", "上限", "影响幅度"]].rename(
            columns={"下限": "-10% 利润", "上限": "+10% 利润", "影响幅度": "影响幅度"}
        ),
        use_container_width=True, hide_index=True,
    )


    # ════════════════════════════════════════════════════════════
    # 卡片区 3：预测模拟
    # ════════════════════════════════════════════════════════════
    st.markdown("### 🔮 预测模拟")
    st.caption("基于三个输入参数，动态模拟工厂整体经营结果")

    # 模拟结果指标卡
    sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)

    with sim_col1:
        rev_per_unit = factory["total_revenue"] / factory["total_units"] if factory["total_units"] > 0 else 0
        cost_per_unit = (factory["total_variable_cost"] + factory["total_fixed"]) / factory["total_units"] if factory["total_units"] > 0 else 0
        actual_margin = (rev_per_unit - cost_per_unit) / rev_per_unit * 100 if rev_per_unit > 0 else 0
        st.metric("实际毛利率", f"{actual_margin:.1f}%",
                  delta=f"目标 {margin_rate}%",
                  delta_color="inverse" if actual_margin < margin_pct * 100 else "normal")

    with sim_col2:
        st.metric("总收入", fmt_money(factory["total_revenue"]),
                  help=f"产量 {factory['total_units']:,.0f} 件 × 均价 {fmt_money(avg_price)}")

    with sim_col3:
        total_cost = factory["total_variable_cost"] + factory["total_fixed"]
        st.metric("总成本", fmt_money(total_cost),
                  help=f"变动 {fmt_money(factory['total_variable_cost'])} + 固定 {fmt_money(factory['total_fixed'])}")

    with sim_col4:
        profit_color = "normal" if factory["profit"] >= 0 else "inverse"
        st.metric("净利润", fmt_money(factory["profit"]),
                  delta_color=profit_color)

    # 产线级详情
    st.markdown("#### 产线经营明细")
    line_summary = []
    for line_name in ["A1", "A2", "A3"]:
        line_products = [p for p in factory["products"] if p["产线"] == line_name]
        if line_products:
            line_units = sum(p["实际产量"] for p in line_products)
            line_rev = sum(p["收入"] for p in line_products)
            line_var = sum(p["变动成本总额"] for p in line_products)
            line_cm = sum(p["边际贡献总额"] for p in line_products)
            line_std_vol = sum(p["标准产量"] for p in line_products)
            line_util = line_units / (LINE_INFO[line_name]["hours"] / STD_HOURS_PER_UNIT) * 100
            line_fix = line_units * FIX_OH_STD  # 分摊的固定制造费用
            line_factory_fix = line_units * (FIXED_COST_MONTHLY / MAX_CAPACITY_UNITS)  # 工厂级分摊
            line_profit = line_cm - line_factory_fix * line_units

            line_summary.append({
                "产线": line_name,
                "设备数": LINE_INFO[line_name]["devices"],
                "实际产量": f"{line_units:,.0f}",
                "产能利用率": f"{line_util:.1f}%",
                "收入": fmt_money(line_rev),
                "变动成本": fmt_money(line_var),
                "边际贡献": fmt_money(line_cm),
                "分摊固定成本": fmt_money(line_fix),
            })

    if line_summary:
        st.dataframe(pd.DataFrame(line_summary), use_container_width=True, hide_index=True)

    # 产品级明细
    st.markdown("#### 产品经营明细")
    prod_summary = pd.DataFrame(factory["products"])
    prod_summary["售价"] = prod_summary["售价"].apply(fmt_money)
    prod_summary["变动成本"] = prod_summary["变动成本"].apply(lambda x: f"{x:,.2f}")
    prod_summary["边际贡献"] = prod_summary["边际贡献"].apply(lambda x: f"{x:,.2f}")
    prod_summary["收入"] = prod_summary["收入"].apply(fmt_money)
    prod_summary["边际贡献总额"] = prod_summary["边际贡献总额"].apply(fmt_money)

    st.dataframe(
        prod_summary[["产品", "产线", "标准产量", "实际产量", "售价", "变动成本", "边际贡献", "收入", "边际贡献总额"]],
        use_container_width=True, hide_index=True,
    )

    # 产能预警
    if factory["capacity_util"] > 100:
        over_pct = factory["capacity_util"] - 100
        if factory["capacity_util"] > 110:
            st.error(f"🚨 **产能严重超限**: 当前产能利用率 {factory['capacity_util']:.1f}%，超出产能上限 {over_pct:.1f}%。"
                     f"需关注设备过载风险和加班成本。")
        else:
            st.warning(f"⚠️ **产能超限警告**: 当前产能利用率 {factory['capacity_util']:.1f}%，超出产能上限 {over_pct:.1f}%。"
                       f"建议关注设备负荷。")
    elif factory["capacity_util"] > 85:
        st.info(f"ℹ️ **产能利用率偏高**: {factory['capacity_util']:.1f}%，接近满产状态。")
    else:
        st.info(f"ℹ️ **产能利用率**: {factory['capacity_util']:.1f}%，尚有 {100-factory['capacity_util']:.1f}% 产能空间。")

    # 预测建议
    st.markdown("#### 情景建议")
    bep_ratio = factory["bep_units"] / MAX_CAPACITY_UNITS * 100 if factory["bep_units"] != float("inf") else 0
    suggestions = []

    if factory["profit"] < 0:
        suggestions.append("🔴 **当前亏损**：建议提高毛利率目标或降低材料成本")
        suggestions.append(f"🟡 **BEP产能占比过高**：盈亏平衡点需占用 {bep_ratio:.1f}% 产能，经营杠杆风险较高")
    elif factory["profit"] > 0 and factory["profit"] < FIXED_COST_MONTHLY * 0.1:
        suggestions.append("🟡 **利润微薄**：利润率不足固定成本的10%，建议优化成本结构")
        suggestions.append(f"🟢 **已过BEP**：安全边际 {(factory['total_units'] - factory['bep_units'])/factory['total_units']*100:.1f}%")

    if factory["capacity_util"] > 100:
        suggestions.append("🟠 **产能瓶颈**：考虑增加班次或外包部分产能")
    else:
        suggestions.append(f"🟢 **产能充裕**：可承接 {100-factory['capacity_util']:.1f}% 的增量订单")

    if mat_pct < 1.0:
        suggestions.append("🟢 **材料成本下降**：当前材料价格低于标准，有利于利润提升")
    elif mat_pct > 1.0:
        suggestions.append(f"🔴 **材料成本上升**：材料价格较标准上涨 {(mat_pct-1)*100:.0f}%，建议关注供应链风险")

    for s in suggestions:
        st.markdown(s)


# ─── 页脚 ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: {C['muted']}; font-size: 0.8rem;">
        <b>制造型产品成本分析平台 v2</b> · 数据基于汽车零部件行业公开数据规格化模拟<br>
        Python + Streamlit + Pandas + Plotly · 吸收成本法 · 4项成本结构
    </div>
    """,
    unsafe_allow_html=True,
)