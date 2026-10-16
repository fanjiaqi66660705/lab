# -*- coding: utf-8 -*-
"""
=================================================================================
《“饺”点时刻，谁在买单？——Z世代大学生正大速冻食品消费意愿影响机制
  与市场细分研究》  数据分析全流程脚本（可复现 / One-click Reproducible）
---------------------------------------------------------------------------------
正大杯全国大学生市场调查与分析大赛 · 参赛数据分析流水线 v1.0

流程：
  1. 按真实问卷结构生成模拟作答（合成数据 Synthetic Data，随机种子可复现）
  2. 问卷质控与清洗（注意力检验题 / 直线作答 / 缺失率 / 反向计分 / 异常值）
  3. 信度分析（Cronbach's α）与效度检验（KMO + Bartlett 球形检验）
  4. 描述性统计与样本结构
  5. 交叉分析（卡方独立性检验）
  6. 二元 Logistic 回归（消费高意愿影响因素：β / OR / 95%CI / p / VIF / AUC）
  7. K-Means 聚类市场细分（轮廓系数选 K + 四类用户画像雷达）
  8. NPS 净推荐值分析
  9. 结果导出：results.json（供可视化报告使用）+ 静态图表 charts/*.png + 清洗后数据

真实参赛时：把 STEP 1 替换为 pd.read_csv("问卷网导出原始数据.csv") 即可，
STEP 2-9 全部无需改动 —— 这是一条“数据进来、报告出去”的标准化流水线。

依赖：pandas numpy matplotlib scipy scikit-learn statsmodels
作者：[入队申请人]    日期：2026-10
=================================================================================
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ 全局配置
SEED = 20261016
rng = np.random.default_rng(SEED)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHART_DIR = ROOT / "charts"
DATA_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)

# Windows 中文字体
for f in ["Microsoft YaHei", "SimHei", "DengXian"]:
    if any(f in ff.name for ff in font_manager.fontManager.ttflist):
        matplotlib.rcParams["font.sans-serif"] = [f]
        break
matplotlib.rcParams["axes.unicode_minus"] = False
PALETTE = ["#E4572E", "#F3A953", "#3AA17E", "#3E7CB1", "#7B5EA7", "#D7B446"]
plt.rcParams["figure.dpi"] = 130

# ------------------------------------------------------------------ 问卷元数据
# 量表题：Likert 5 级（1=非常不同意 … 5=非常同意），BC3 为反向陈述题
CONSTRUCTS = {
    "PQ": {"name": "产品品质感知", "items": [
        "PQ1 正大速冻食品的口味符合我的预期",
        "PQ2 我认为正大速冻食品安全、食材有保障",
        "PQ3 正大速冻食品的馅料/用料比同类更实在"]},
    "PV": {"name": "价格价值感知", "items": [
        "PV1 考虑到品质，正大速冻食品定价是合理的",
        "PV2 促销活动时正大速冻食品的性价比很高"]},
    "BC": {"name": "品牌认知信任", "items": [
        "BC1 我能识别正大品牌并与竞品区分开",
        "BC2 购买速冻食品时我会优先想到正大",
        "BC3r 速冻食品都差不多，买哪个品牌无所谓"]},   # 反向计分
    "CV": {"name": "渠道便利感知", "items": [
        "CV1 我很容易在常逛的渠道买到正大速冻食品",
        "CV2 从下单到吃上，正大速冻食品足够省时省力"]},
    "SN": {"name": "主观规范", "items": [
        "SN1 家人/室友的推荐会影响我购买正大速冻食品",
        "SN2 美食博主/同学“种草”会勾起我的尝试欲",
        "SN3 身边越来越多人吃速冻食品，我也愿意尝试"]},
    "HA": {"name": "健康意识", "items": [
        "HA1 我关注配料表、热量与营养成分",
        "HA2 我偏好低脂/高蛋白的“轻负担”速冻产品"]},
    "PI": {"name": "购买意愿（结果变量）", "items": [
        "PI1 下个月我愿意购买正大速冻食品",
        "PI2 我愿意尝试正大推出的速冻新品",
        "PI3 我愿意把正大速冻食品推荐给朋友"]},
}
SCALE_ITEMS = [it for c in CONSTRUCTS.values() for it in c["items"]]
SCALE_CODES = [it.split(" ")[0] for it in SCALE_ITEMS]
REVERSE_ITEMS = ["BC3r"]
ATTENTION_ITEM = "AT1 本题为质量检验题，请选择“比较同意”(4)"

QUESTIONNAIRE = {
    "sections": [
        {"part": "A 甄别与背景", "questions": ["性别", "年级", "月生活费(元)", "校园所在城市线级"]},
        {"part": "B 消费行为", "questions": ["购买频次", "单次客单价(元)", "购买渠道(多选)", "消费场景(多选)"]},
        {"part": "C 感知量表", "questions": [f"7 个构念共 18 题 Likert 5 级量表 + 1 道反向计分题 + 1 道注意力检验题"]},
        {"part": "D 推荐意愿", "questions": ["NPS：您向同学推荐正大速冻食品的可能性(0-10 分)"]},
    ],
    "sample_size_plan": {
        "formula": "n0 = Z²·p(1-p)/e² = 1.96²×0.5×0.5/0.05² ≈ 385",
        "planned_distribution": 580,
        "expected_valid": 526,
        "note": "按无限总体 95% 置信度、5% 误差计算最小样本量 385；上浮约 50% 覆盖质控剔除",
    },
}

# ==================================================================== STEP 1
def generate_raw_data(n_valid=526, n_bad=54):
    """按预设的四类潜在消费群体生成模拟问卷原始作答（含真实感的脏数据）。"""
    # 四类潜在群体的构念基线（1-5）：品质忠诚 / 价格敏感 / 便利速食 / 健康尝鲜
    seg_means = {
        "品质忠诚型": dict(PQ=4.35, PV=3.75, BC=4.25, CV=3.65, SN=3.70, HA=3.95, off=0.45, p=0.22),
        "价格敏感型": dict(PQ=3.25, PV=2.25, BC=2.85, CV=3.30, SN=2.80, HA=3.40, off=-0.55, p=0.28),
        "便利速食型": dict(PQ=3.50, PV=3.05, BC=3.30, CV=4.40, SN=3.40, HA=2.75, off=0.15, p=0.27),
        "健康尝鲜型": dict(PQ=3.55, PV=3.10, BC=3.15, CV=2.90, SN=4.25, HA=4.35, off=0.10, p=0.23),
    }
    seg_names = list(seg_means)
    seg_p = [seg_means[s]["p"] for s in seg_names]

    rows, true_seg = [], []
    for i in range(n_valid):
        seg = rng.choice(seg_names, p=seg_p)
        m = seg_means[seg]
        # —— A 背景信息 ——
        gender = rng.choice(["女", "男"], p=[0.58, 0.42])
        grade = rng.choice(["大一", "大二", "大三", "大四"], p=[0.34, 0.31, 0.24, 0.11])
        expense = rng.choice(["1500以下", "1500-2000", "2000-2500", "2500以上"],
                             p=[0.18, 0.37, 0.28, 0.17])
        city = rng.choice(["一线/新一线", "二线", "三线及以下"], p=[0.46, 0.34, 0.20])
        # —— B 行为（与群体、生活费相关） ——
        freq_opt = ["从不/几乎不", "每月少于1次", "每月1-2次", "每月3-5次", "每周1次及以上"]
        freq_by_seg = {
            "品质忠诚型": [0.01, 0.04, 0.15, 0.40, 0.40],
            "价格敏感型": [0.20, 0.38, 0.30, 0.10, 0.02],
            "便利速食型": [0.02, 0.08, 0.30, 0.42, 0.18],
            "健康尝鲜型": [0.05, 0.27, 0.40, 0.23, 0.05]}
        freq = rng.choice(freq_opt, p=freq_by_seg[seg])
        # 客单价受生活费影响
        price_p_base = {"1500以下": [0.42, 0.40, 0.14, 0.04], "1500-2000": [0.24, 0.46, 0.23, 0.07],
                        "2000-2500": [0.13, 0.38, 0.36, 0.13], "2500以上": [0.07, 0.26, 0.40, 0.27]}
        price = rng.choice(["10元以下", "10-20元", "20-30元", "30元以上"], p=price_p_base[expense])
        channel_p = {
            "品质忠诚型": [0.82, 0.60, 0.35, 0.30, 0.22], "价格敏感型": [0.55, 0.30, 0.58, 0.45, 0.18],
            "便利速食型": [0.40, 0.52, 0.30, 0.70, 0.66], "健康尝鲜型": [0.60, 0.66, 0.30, 0.38, 0.35]}
        channels = "┋".join([c for c, pp in zip(
            ["大型商超", "电商平台", "社区团购", "校园便利店", "外卖平台"], channel_p[seg]) if rng.random() < pp]) or "大型商超"
        scene_p = {
            "品质忠诚型": [0.35, 0.30, 0.55, 0.62, 0.25], "价格敏感型": [0.45, 0.30, 0.35, 0.20, 0.10],
            "便利速食型": [0.85, 0.78, 0.40, 0.25, 0.20], "健康尝鲜型": [0.50, 0.45, 0.50, 0.40, 0.35]}
        scenes = "┋".join([s for s, pp in zip(
            ["宿舍速食", "夜宵加餐", "快手早餐", "改善伙食", "朋友聚会"], scene_p[seg]) if rng.random() < pp]) or "宿舍速食"
        # —— C 量表：个体构念特质(σ=0.40) + 题目特有误差(σ=0.32)，
        #     使同一构念题项间相关≈0.6（保证分量表 α≈0.75-0.85 的真实信度结构）——
        ans, trait = {}, {}
        for con in ["PQ", "PV", "BC", "CV", "SN", "HA"]:
            trait[con] = m[con] + rng.normal(0, 0.40)
        for code in SCALE_CODES:
            if code == "BC3r" or code.startswith("PI"):
                continue
            con = code[:2]
            ans[code] = int(np.clip(round(trait[con] + rng.normal(0, 0.32)), 1, 5))
        # 反向陈述题：真实品牌信任越高 → 越不认同“品牌无所谓” → 6 分量表反转
        ans["BC3r"] = int(np.clip(round(6 - trait["BC"] + rng.normal(0, 0.32)), 1, 5))
        # 购买意愿由个体层面各感知构念共同驱动（含群体偏移与随机扰动，避免完全分离）
        pi_star = (0.45 + 0.22 * trait["PQ"] + 0.18 * trait["BC"] + 0.18 * trait["PV"]
                   + 0.12 * trait["CV"] + 0.14 * trait["SN"] + 0.10 * trait["HA"]
                   + m["off"] + rng.normal(0, 0.30))
        for code in ["PI1", "PI2", "PI3"]:
            ans[code] = int(np.clip(round(pi_star + rng.normal(0, 0.28)), 1, 5))
        # 注意力检验题：认真作答的目标样本全部通过（失败样本由“脏问卷”模拟）
        ans["AT1"] = 4
        # —— D NPS（与群体挂钩，取 0-10 整数）——
        nps_center = {"品质忠诚型": 9.3, "价格敏感型": 6.0, "便利速食型": 8.0, "健康尝鲜型": 7.9}[seg]
        nps_sd = {"品质忠诚型": 1.0, "价格敏感型": 1.2, "便利速食型": 1.0, "健康尝鲜型": 1.0}[seg]
        nps = int(np.clip(round(nps_center + rng.normal(0, nps_sd)), 0, 10))

        row = dict(样本编号=f"S{i+1:04d}", 性别=gender, 年级=grade, 月生活费=expense,
                   城市线级=city, 购买频次=freq, 客单价=price, 购买渠道=channels,
                   消费场景=scenes, NPS=nps, **ans)
        # 2% 随机缺失（MCAR），模拟真实作答漏填
        for col in rng.choice(SCALE_CODES + ["NPS", "客单价"], size=max(1, int(23 * 0.02)), replace=False):
            if rng.random() < 0.5:
                row[col] = np.nan
        rows.append(row)
        true_seg.append(seg)

    # —— 54 份“脏问卷”：25 注意力题失败 / 15 直线作答 / 14 缺失过多（三类互斥，
    #    分别对应三条质控规则，模拟问卷网常见的敷衍/机器人作答）——
    for j in range(n_bad):
        if j < 25:
            # ① 注意力检验失败（其余作答正常）
            ans = {c: int(rng.integers(1, 6)) for c in SCALE_CODES}
            at1 = rng.choice([1, 2, 3, 5])
        elif j < 40:
            # ② 直线作答（全部 18 道量表题同一选项）
            flat = int(rng.choice([2, 3, 4]))
            ans = {c: flat for c in SCALE_CODES}
            at1 = 4
        else:
            # ③ 关键字段缺失率 > 20%（其余作答正常，避免与直线规则重叠）
            ans = {c: int(rng.integers(1, 6)) for c in SCALE_CODES}
            for col in rng.choice(SCALE_CODES, size=8, replace=False):
                ans[col] = np.nan
            at1 = 4
        row = dict(样本编号=f"B{j+1:04d}",
                   性别=rng.choice(["女", "男"]), 年级=rng.choice(["大一", "大二", "大三", "大四"]),
                   月生活费=rng.choice(["1500以下", "1500-2000", "2000-2500", "2500以上"]),
                   城市线级=rng.choice(["一线/新一线", "二线", "三线及以下"]),
                   购买频次=rng.choice(freq_opt), 客单价=rng.choice(["10元以下", "10-20元", "20-30元", "30元以上"]),
                   购买渠道="电商平台", 消费场景="宿舍速食", NPS=int(rng.integers(0, 11)),
                   AT1=at1, **ans)
        rows.append(row)

    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.to_csv(DATA_DIR / "raw_data.csv", index=False, encoding="utf-8-sig")
    print(f"[STEP1] 原始问卷已生成：{len(df)} 份 -> data/raw_data.csv")
    return df

# ==================================================================== STEP 2
def clean_data(df):
    """问卷质控：注意力检验 → 直线作答 → 缺失率 → 反向计分 → 缺失插补。"""
    n0 = len(df)
    report = {}
    # ① 注意力检验题
    bad_att = df["AT1"] != 4
    report["注意力检验题未通过"] = int(bad_att.sum())
    df = df[~bad_att]
    # ② 直线作答：18 题量表标准差 < 0.3
    flat = df[SCALE_CODES].std(axis=1) < 0.3
    report["直线作答(量表SD<0.3)"] = int(flat.sum())
    df = df[~flat]
    # ③ 关键字段缺失率 > 20%
    miss_rate = df[SCALE_CODES + ["NPS"]].isna().mean(axis=1)
    bad_miss = miss_rate > 0.20
    report["缺失率超20%"] = int(bad_miss.sum())
    df = df[~bad_miss]
    # ④ 反向题重新计分
    df = df.copy()
    df["BC3r_raw"] = df["BC3r"]
    df["BC3r"] = 6 - df["BC3r"]
    # ⑤ 少量缺失值：量表题用中位数插补，保留样本
    n_miss = int(df[SCALE_CODES + ["NPS"]].isna().sum().sum())
    for col in SCALE_CODES + ["NPS"]:
        df[col] = df[col].fillna(df[col].median())
    df = df.reset_index(drop=True)
    df.to_csv(DATA_DIR / "clean_data.csv", index=False, encoding="utf-8-sig")
    report["插补缺失值个数"] = n_miss
    report["发放份数"] = n0
    report["有效份数"] = len(df)
    report["有效回收率"] = round(len(df) / n0 * 100, 1)
    print(f"[STEP2] 质控完成：{n0} -> {len(df)} 份，有效回收率 {report['有效回收率']}%")
    return df, report

# ==================================================================== STEP 3
def cronbach_alpha(item_df):
    item_df = item_df.dropna()
    k = item_df.shape[1]
    var_sum = item_df.var(axis=0, ddof=1).sum()
    total_var = item_df.sum(axis=1).var(ddof=1)
    return k / (k - 1) * (1 - var_sum / total_var)

def kmo_bartlett(item_df):
    """KMO 取样适切性与 Bartlett 球形检验（相关矩阵法手写实现）。"""
    X = item_df.dropna().astype(float)
    R = np.corrcoef(X.values, rowvar=False)
    R_inv = np.linalg.pinv(R)
    d = np.sqrt(np.diag(R_inv))
    partial = -R_inv / np.outer(d, d)          # 偏相关矩阵
    np.fill_diagonal(partial, 0)
    np.fill_diagonal(R, 0)
    kmo = (R ** 2).sum() / ((R ** 2).sum() + (partial ** 2).sum())
    n, p = X.shape
    det = max(np.linalg.det(np.corrcoef(X.values, rowvar=False)), 1e-12)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(det)
    dof = p * (p - 1) / 2
    pval = 1 - stats.chi2.cdf(chi2, dof)
    return kmo, chi2, dof, pval

def reliability_validity(df):
    alpha = {}
    construct_scores = pd.DataFrame(index=df.index)
    for code, meta in CONSTRUCTS.items():
        codes = [it.split(" ")[0] for it in meta["items"]]
        alpha[meta["name"]] = round(cronbach_alpha(df[codes]), 3)
        construct_scores[code] = df[codes].mean(axis=1)
    alpha["总量表"] = round(cronbach_alpha(df[SCALE_CODES]), 3)
    kmo, chi2, dof, pval = kmo_bartlett(df[SCALE_CODES])
    out = {"cronbach_alpha": alpha,
           "kmo": round(kmo, 3), "bartlett_chi2": round(chi2, 1),
           "bartlett_df": int(dof), "bartlett_p": float(f"{pval:.2e}")}
    print(f"[STEP3] 总量表 α={alpha['总量表']}，KMO={kmo:.3f}，Bartlett χ²={chi2:.1f}(p<0.001)")
    return construct_scores, out

# ==================================================================== STEP 4
def descriptives(df):
    def dist(col, order=None):
        vc = df[col].value_counts(normalize=True)
        if order:
            return {k: round(vc.get(k, 0) * 100, 1) for k in order}
        return {k: round(v * 100, 1) for k, v in vc.items()}

    out = {
        "n": len(df),
        "性别": dist("性别"),
        "年级": dist("年级", ["大一", "大二", "大三", "大四"]),
        "月生活费": dist("月生活费", ["1500以下", "1500-2000", "2000-2500", "2500以上"]),
        "城市线级": dist("城市线级"),
        "购买频次": dist("购买频次", ["从不/几乎不", "每月少于1次", "每月1-2次", "每月3-5次", "每周1次及以上"]),
        "客单价": dist("客单价", ["10元以下", "10-20元", "20-30元", "30元以上"]),
    }
    # 多选拆分
    for col, label in [("购买渠道", "渠道"), ("消费场景", "场景")]:
        all_items = sorted({x for v in df[col] for x in str(v).split("┋")})
        out[label] = {k: round((df[col].str.contains(k, regex=False)).mean() * 100, 1) for k in all_items}
    # 构念均值（消费者总体态度画像）
    out["构念均值"] = {CONSTRUCTS[c]["name"]: round(construct_scores_global[c].mean(), 2)
                    for c in ["PQ", "PV", "BC", "CV", "SN", "HA", "PI"]}
    print(f"[STEP4] 描述统计完成：{len(df)} 个有效样本")
    return out

# ==================================================================== STEP 5
def crosstab_analysis(df):
    results = []
    for row_var, col_var, row_order, col_order in [
        ("月生活费", "客单价", ["1500以下", "1500-2000", "2000-2500", "2500以上"],
         ["10元以下", "10-20元", "20-30元", "30元以上"]),
        ("年级", "购买频次", ["大一", "大二", "大三", "大四"],
         ["从不/几乎不", "每月少于1次", "每月1-2次", "每月3-5次", "每周1次及以上"])]:
        ct = pd.crosstab(df[row_var], df[col_var]).reindex(index=row_order, columns=col_order, fill_value=0)
        chi2, p, dof, _ = chi2_contingency(ct)
        results.append({"交叉表": f"{row_var} × {col_var}", "卡方": round(chi2, 2),
                        "自由度": int(dof), "p值": round(p, 4),
                        "列联表": ct.to_dict(),
                        "结论": "存在显著关联(p<0.05)" if p < 0.05 else "关联不显著"})
    print(f"[STEP5] 卡方交叉分析完成：{len(results)} 组交叉表")
    return results

# ==================================================================== STEP 6
def logistic_model(df, scores):
    """y=购买意愿高(PI均值≥4)，自变量为 6 个前置感知构念。"""
    model_df = scores.copy()
    model_df["y"] = (model_df["PI"] >= 4).astype(int)
    X_cols = ["PQ", "PV", "BC", "CV", "SN", "HA"]
    # 自变量均值中心化：截距变为“平均感知水平个体”的 logit，具有实际意义并缓解完全分离
    Xc = model_df[X_cols] - model_df[X_cols].mean()
    X = sm.add_constant(Xc)
    y = model_df["y"]
    fit = sm.Logit(y, X).fit(disp=0, maxiter=200)
    coef = pd.DataFrame({
        "变量": ["截距"] + [CONSTRUCTS[c]["name"] for c in X_cols],
        "β": fit.params.round(3).values,
        "SE": fit.bse.round(3).values,
        "p值": fit.pvalues.round(4).values,
        "OR": np.exp(fit.params).round(3).values,
        "CI_low": np.exp(fit.conf_int()[0]).round(3).values,
        "CI_high": np.exp(fit.conf_int()[1]).round(3).values,
    })
    vif = pd.DataFrame({"变量": [CONSTRUCTS[c]["name"] for c in X_cols],
                        "VIF": [round(variance_inflation_factor(X.values, i + 1), 2)
                                for i in range(len(X_cols))]})
    pred = fit.predict(X)
    # ROC 曲线数据（供交互式报告绘制）
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, pred)
    out = {
        "高意愿样本占比": round(y.mean() * 100, 1),
        "McFadden_R2": round(fit.prsquared, 3),
        "AUC": round(roc_auc_score(y, pred), 3),
        "roc": [{"fpr": round(a, 3), "tpr": round(b, 3)} for a, b in zip(fpr, tpr)],
        "LLR_p": float(f"{fit.llr_pvalue:.2e}"),
        "coef": coef.to_dict("records"),
        "vif": vif.to_dict("records"),
        "方程": "logit(P_高意愿) = " + " + ".join(
            [f"{fit.params['const']:.2f}"] + [f"{fit.params[c]:.2f}·{CONSTRUCTS[c]['name']}" for c in X_cols]),
    }
    print(f"[STEP6] Logistic 模型：AUC={out['AUC']}，McFadden R²={out['McFadden_R2']}，"
          f"高意愿占比 {out['高意愿样本占比']}%")
    return out, model_df

# ==================================================================== STEP 7
def kmeans_segment(df, scores):
    X = StandardScaler().fit_transform(scores[["PQ", "PV", "BC", "CV", "SN", "HA"]])
    eval_rows = []
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
        eval_rows.append({"k": k, "轮廓系数": round(silhouette_score(X, km.labels_), 3),
                          "SSE": round(km.inertia_, 1)})
    best_k = max(eval_rows[:4], key=lambda r: r["轮廓系数"])["k"]
    km = KMeans(n_clusters=best_k, n_init=30, random_state=SEED).fit(X)
    scores = scores.copy()
    scores["segment"] = km.labels_
    # 簇画像（构念均值 + 行为占比）
    prof = scores.groupby("segment")[["PQ", "PV", "BC", "CV", "SN", "HA", "PI"]].mean().round(2)
    size = scores["segment"].value_counts(normalize=True).round(3).sort_index()
    # 依据簇画像做互斥式命名：按各客群的典型特征得分贪心指派簇编号
    label_rules = [("懒人便利速食型", lambda r: r["CV"]),
                   ("品质品牌忠诚型", lambda r: r["PQ"] + r["BC"]),
                   ("价格敏感观望型", lambda r: -r["PV"]),
                   ("健康尝鲜潜力型", lambda r: r["HA"] + r["SN"])]
    radar_names, used = {}, set()
    for label, score_fn in label_rules:
        cand = max((c for c in prof.index if c not in used), key=lambda c: score_fn(prof.loc[c]))
        radar_names[cand] = label
        used.add(cand)
    out = {"k_evaluation": eval_rows, "best_k": int(best_k),
           "segments": [{"簇": int(c), "命名": radar_names[c], "占比": round(size[c] * 100, 1),
                          **{CONSTRUCTS[k]["name"] if k in CONSTRUCTS else k: float(v)
                             for k, v in prof.loc[c].items()}} for c in prof.index]}
    print(f"[STEP7] K-Means 聚类：最优 K={best_k}，"
          " / ".join(f"{r['命名']}{r['占比']}%" for r in out["segments"]))
    return out, scores

# ==================================================================== STEP 8
def nps_analysis(df):
    promoters = (df["NPS"] >= 9).mean() * 100
    passives = df["NPS"].between(7, 8).mean() * 100
    detractors = (df["NPS"] <= 6).mean() * 100
    out = {"推荐者(9-10)占比": round(promoters, 1), "被动者(7-8)占比": round(passives, 1),
           "贬损者(0-6)占比": round(detractors, 1), "NPS": round(promoters - detractors, 1),
           "均分": round(df["NPS"].mean(), 2),
           "分布": {str(i): int((df["NPS"] == i).sum()) for i in range(11)}}
    print(f"[STEP8] NPS = {out['NPS']}（推荐者 {out['推荐者(9-10)占比']}% / "
          f"贬损者 {out['贬损者(0-6)占比']}%）")
    return out

# ==================================================================== 出图
def plot_charts(desc, logit, seg, nps, scores):
    # 图1 样本结构
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, key, title in zip(axes, ["性别", "年级", "月生活费"], ["性别结构", "年级分布", "月生活费(元)"]):
        d = desc[key]
        ax.bar(list(d.keys()), list(d.values()), color=PALETTE[:len(d)])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("占比(%)")
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(d.values()):
            ax.text(i, v + 1, f"{v}%", ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(CHART_DIR / "fig1_sample_profile.png", bbox_inches="tight"); plt.close()

    # 图2 行为：频次 + 渠道 + 场景
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    d = desc["购买频次"]
    axes[0].barh(list(d.keys()), list(d.values()), color=PALETTE[3])
    axes[0].set_title("购买频次", fontweight="bold"); axes[0].set_xlabel("占比(%)")
    for i, v in enumerate(d.values()):
        axes[0].text(v + 0.5, i, f"{v}%", va="center", fontsize=9)
    for ax, key, title, c in [(axes[1], "渠道", "购买渠道(多选)", PALETTE[0]),
                              (axes[2], "场景", "消费场景(多选)", PALETTE[2])]:
        dd = dict(sorted(desc[key].items(), key=lambda x: x[1]))
        ax.barh(list(dd.keys()), list(dd.values()), color=c)
        ax.set_title(title, fontweight="bold"); ax.set_xlabel("选择率(%)")
        for i, v in enumerate(dd.values()):
            ax.text(v + 0.5, i, f"{v}%", va="center", fontsize=9)
    plt.tight_layout(); plt.savefig(CHART_DIR / "fig2_behavior.png", bbox_inches="tight"); plt.close()

    # 图3 Logistic OR 森林图
    fig, ax = plt.subplots(figsize=(9, 4.2))
    rows = [r for r in logit["coef"] if r["变量"] != "截距"]
    y = np.arange(len(rows))
    colors = ["#C0392B" if r["OR"] < 1 else "#2E7D5B" for r in rows]
    ax.errorbar([r["OR"] for r in rows], y,
                xerr=([r["OR"] - r["CI_low"] for r in rows], [r["CI_high"] - r["OR"] for r in rows]),
                fmt="o", color="none", ecolor="#888", capsize=4, zorder=1)
    ax.scatter([r["OR"] for r in rows], y, c=colors, s=90, zorder=2)
    ax.axvline(1, color="#999", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([r["变量"] for r in rows])
    ax.set_xlabel("发生比 OR（95% 置信区间）"); ax.set_title("高购买意愿影响因素：Logistic 回归森林图", fontweight="bold")
    for i, r in enumerate(rows):
        p_txt = "<0.001" if r["p值"] < 0.001 else r["p值"]
        ax.text(r["CI_high"] + 0.03, i, f"OR={r['OR']}  p={p_txt}", va="center", fontsize=9)
    ax.set_xlim(0.5, max(r["CI_high"] for r in rows) * 1.35)
    plt.tight_layout(); plt.savefig(CHART_DIR / "fig3_logistic_forest.png", bbox_inches="tight"); plt.close()

    # 图4 聚类雷达
    labels = ["品质", "价值", "品牌", "便利", "规范", "健康", "意愿"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist(); angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.2, 6.2), subplot_kw=dict(polar=True))
    for i, s in enumerate(seg["segments"]):
        vals = [s["产品品质感知"], s["价格价值感知"], s["品牌认知信任"], s["渠道便利感知"],
                s["主观规范"], s["健康意识"], s["购买意愿（结果变量）"]]
        vals += vals[:1]
        ax.plot(angles, vals, color=PALETTE[i], lw=2, label=f"{s['命名']}({s['占比']}%)")
        ax.fill(angles, vals, color=PALETTE[i], alpha=0.08)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels); ax.set_ylim(2, 5)
    ax.set_title("四类客群画像雷达图", fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=9)
    plt.tight_layout(); plt.savefig(CHART_DIR / "fig4_segment_radar.png", bbox_inches="tight"); plt.close()

    # 图5 NPS
    fig, ax = plt.subplots(figsize=(9, 3.2))
    dist = nps["分布"]
    colors = ["#C0392B"] * 7 + ["#F3A953"] * 2 + ["#2E7D5B"] * 2
    ax.bar(list(dist.keys()), list(dist.values()), color=colors)
    ax.set_title(f"NPS 净推荐值 = {nps['NPS']}（红=贬损者 0-6｜黄=被动者 7-8｜绿=推荐者 9-10）",
                 fontweight="bold", fontsize=11)
    ax.set_ylabel("人数"); ax.set_xlabel("推荐评分")
    plt.tight_layout(); plt.savefig(CHART_DIR / "fig5_nps.png", bbox_inches="tight"); plt.close()

    # 图6 ROC 曲线
    roc = logit["roc"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([p["fpr"] for p in roc], [p["tpr"] for p in roc], color=PALETTE[0], lw=2.2,
            label=f"Logistic 模型  AUC = {logit['AUC']}")
    ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1, label="随机分类参考线")
    ax.set_xlabel("假阳性率 FPR"); ax.set_ylabel("真阳性率 TPR")
    ax.set_title("模型区分能力：ROC 曲线", fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    plt.tight_layout(); plt.savefig(CHART_DIR / "fig6_roc.png", bbox_inches="tight"); plt.close()
    print("[CHARTS] 6 张静态图表已导出至 charts/")

# ==================================================================== MAIN
if __name__ == "__main__":
    raw = generate_raw_data()
    clean, qc_report = clean_data(raw)

    construct_scores_global, rv_report = reliability_validity(clean)
    desc = descriptives(clean)
    cross = crosstab_analysis(clean)
    logit, model_df = logistic_model(clean, construct_scores_global)
    seg, scored = kmeans_segment(clean, construct_scores_global)
    nps = nps_analysis(clean)
    plot_charts(desc, logit, seg, nps, scored)

    results = {
        "title": "“饺”点时刻，谁在买单？——Z世代大学生正大速冻食品消费意愿影响机制与市场细分研究",
        "data_disclaimer": "本作品数据为按真实问卷结构生成的合成数据(Synthetic Data, seed=20261016)，仅用于展示可复现的分析流程；正式参赛将以问卷网真实采集数据替换，STEP2-9 代码无需改动。",
        "questionnaire": QUESTIONNAIRE,
        "quality_control": qc_report,
        "reliability_validity": rv_report,
        "descriptives": desc,
        "crosstab": cross,
        "logistic": logit,
        "segmentation": seg,
        "nps": nps,
    }
    with open(DATA_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("[DONE] 全部结果已写入 data/results.json，可直接供可视化报告调用。")
