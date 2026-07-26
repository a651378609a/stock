#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一代策略基因进化：仅场内ETF → 检查器 → 突变 → 盲测回放（含交易明细）→ 晋级/保持 → 右移。"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVOL = ROOT / "evolution"
DATA = EVOL / "data"
LOG = EVOL / "log"
TRADES = EVOL / "trades"
STATE = ROOT / "state"
GENOME_PATH = ROOT / "strategy_genome.md"
SKILL_PATH = ROOT / "skill.md"
WINDOW_PATH = EVOL / "window.json"
SCORECARD_PATH = STATE / "live_scorecard.json"
ETF_MAP_PATH = DATA / "etf_role_map.json"

# 冻结评估成本（与 skill.md 一致）
COMMISSION = 0.0003
STAMP_SELL = 0.001
SLIPPAGE = 0.001
EPS_DD = 0.10
W_LIVE = 0.70
W_HIST = 0.30
MIN_LIVE_DAYS = 20
INIT_CASH = 1_000_000.0
STRESS_MONTHS = ("2026-04", "2026-07")
STRESS_MDD_CAP = 0.25
TRAIN_END = date(2025, 12, 31)
PRED_YEAR = 2026
PRED_START = date(2026, 1, 1)
PRED_END_CAL = date(2026, 12, 31)

# 角色→数据文件（指数序列仅作 ETF 报价代理；日志按 ETF 记账）
PROXY = {
    "进攻": "sz399006",
    "次线": "sh000905",
    "防御": "sh000012",
    "工具": "sh000300",
    "基准": "sh000300",
}

FORBIDDEN_NAME_PATTERNS = [
    r"\b(?:茅台|宁德|比亚迪|中芯|寒武纪|赛力斯)\b",
    r"\b[036]\d{5}\b",  # 正股代码（ETF 的 51/15/56 等不在此列）
]


def load_etf_map() -> dict[str, Any]:
    if ETF_MAP_PATH.exists():
        return json.loads(ETF_MAP_PATH.read_text(encoding="utf-8"))
    return {}


def etf_label(role: str, emap: dict[str, Any]) -> tuple[str, str]:
    meta = emap.get(role) or {}
    return str(meta.get("ETF角色名") or f"{role}ETF"), str(meta.get("示例代码") or "ETF")


@dataclass
class GenomeParams:
    version: str
    cash_lo: float = 0.05
    offense_hi: float = 0.70
    secondary_hi: float = 0.30
    defense_hi: float = 0.25
    cycle_hi: float = 0.30
    tool_hi: float = 0.40
    trend_pos: tuple[float, float] = (0.80, 0.95)
    range_pos: tuple[float, float] = (0.60, 0.75)
    chaos_pos: tuple[float, float] = (0.00, 0.50)
    open_step: float = 0.08
    stop_from_peak: float = 0.20
    warn_from_peak: float = 0.15
    two_day_crash: float = 0.20
    outflow_days: int = 3
    abandon_weeks: int = 4
    chase_breakout: bool = False
    min_rr: float = 3.0
    catalyst_hits: int = 2
    max_sells_day: int = 2
    prefer_divergence: bool = True
    chaos_low_activity: bool = True

    def clamp(self) -> "GenomeParams":
        self.offense_hi = min(self.offense_hi, 0.95)
        self.secondary_hi = min(self.secondary_hi, 0.95)
        self.defense_hi = min(self.defense_hi, 0.95)
        self.tool_hi = min(self.tool_hi, 0.95)
        self.open_step = float(np.clip(self.open_step, 0.05, 0.10))
        self.stop_from_peak = float(np.clip(self.stop_from_peak, 0.08, 0.35))
        self.warn_from_peak = float(np.clip(self.warn_from_peak, 0.05, self.stop_from_peak))
        self.outflow_days = int(np.clip(self.outflow_days, 1, 10))
        self.catalyst_hits = int(np.clip(self.catalyst_hits, 1, 5))
        return self


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_v0_genome() -> GenomeParams:
    return GenomeParams(version="v0").clamp()


def load_panel(*, for_training: bool = False) -> pd.DataFrame:
    """for_training=True 时硬截断到训练截止日，训练开始不可见预测年。"""
    frames = []
    for role, sym in PROXY.items():
        fp = DATA / f"{sym}.csv"
        df = pd.read_csv(fp, parse_dates=["date"])
        df = df.sort_values("date").drop_duplicates("date")
        df = df.rename(
            columns={
                "open": f"{role}_open",
                "high": f"{role}_high",
                "low": f"{role}_low",
                "close": f"{role}_close",
                "volume": f"{role}_volume",
            }
        )
        frames.append(df.set_index("date")[[c for c in df.columns if c.startswith(role)]])
    panel = pd.concat(frames, axis=1, join="inner").sort_index()
    today = datetime.now(timezone.utc).date()
    panel = panel[panel.index.date < today]
    if for_training:
        panel = panel[panel.index.date <= TRAIN_END]
    return panel


def last_closed_month_end(last_data_day: date) -> date:
    """最近已封闭自然月终点（该月全部交易日均已落在数据内）。"""
    y, m = last_data_day.year, last_data_day.month
    # 若 last_data_day 不是该月最后日历日，则上一自然月才算封闭
    if last_data_day.month == 12:
        next_month = date(y + 1, 1, 1)
    else:
        next_month = date(y, m + 1, 1)
    month_calendar_end = next_month - timedelta(days=1)
    if last_data_day < month_calendar_end:
        # 回退一个月
        if m == 1:
            return date(y - 1, 12, 31)
        first = date(y, m, 1)
        return first - timedelta(days=1)
    return month_calendar_end


def add_months(d: date, months: int) -> date:
    m0 = d.month - 1 + months
    y = d.year + m0 // 12
    m = m0 % 12 + 1
    if m == 12:
        nxt = date(y + 1, 1, 1)
    else:
        nxt = date(y, m + 1, 1)
    return nxt - timedelta(days=1)


def month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def window_from_evol_start(e0: date) -> tuple[date, date, date, date]:
    """进化段约 24 个自然月，随后盲测 6 个自然月。"""
    e1 = date(e0.year + 2, e0.month, 1) - timedelta(days=1)
    b0 = date(e0.year + 2, e0.month, 1)
    bm = b0.month - 1 + 5
    by = b0.year + bm // 12
    bmonth = bm % 12 + 1
    if bmonth == 12:
        b1 = date(by + 1, 1, 1) - timedelta(days=1)
    else:
        b1 = date(by, bmonth + 1, 1) - timedelta(days=1)
    return e0, e1, b0, b1


def shift_evol_start(e0: date, months: int = 6) -> date:
    """按自然月平移月初日期（支持负偏移）。"""
    idx = e0.year * 12 + (e0.month - 1) + months
    year, month0 = divmod(idx, 12)
    return date(year, month0 + 1, 1)


def pred_segment_end(full_panel: pd.DataFrame) -> date:
    last = full_panel.index.max().date()
    closed_end = last_closed_month_end(last)
    return min(PRED_END_CAL, closed_end, last)


def init_or_load_window(train_panel: pd.DataFrame, full_panel: pd.DataFrame) -> dict[str, Any]:
    """训练窗≤2025-12-31；预测年2026固定为样本外。"""
    win = json.loads(WINDOW_PATH.read_text(encoding="utf-8"))
    first = train_panel.index.min().date()
    p1 = pred_segment_end(full_panel)
    p0 = PRED_START

    # 进化段：训练截止日前约 24 个月
    e1 = TRAIN_END
    e0 = date(TRAIN_END.year - 2, TRAIN_END.month, 1)
    if e0 < month_start(first):
        e0 = month_start(first)

    # 强制纠正旧窗口（若越界训练截止或缺少预测年字段）
    need_reset = (
        not win.get("进化段起")
        or not win.get("预测段起")
        or date.fromisoformat(str(win.get("进化段止") or "1900-01-01")) > TRAIN_END
        or str(win.get("训练截止日")) != str(TRAIN_END)
    )
    if need_reset or True:
        # 始终对齐预测年协议（可保留基因版本）
        ver = win.get("现任基因版本") or "v0"
        win.update(
            {
                "训练截止日": str(TRAIN_END),
                "进化段起": str(e0),
                "进化段止": str(e1),
                "预测年": PRED_YEAR,
                "预测段起": str(p0),
                "预测段止": str(p1),
                # 兼容旧字段名：盲测=预测段
                "盲测段起": str(p0),
                "盲测段止": str(p1),
                "现任基因版本": ver,
                "现任实盘交易日数": int(json.loads(SCORECARD_PATH.read_text()).get("交易日数") or 0),
                "状态": "评估中",
                "数据最早日": str(first),
                "数据最末日": str(full_panel.index.max().date()),
                "已封闭月终点": str(last_closed_month_end(full_panel.index.max().date())),
                "压力月": list(STRESS_MONTHS),
                "交易宇宙": "仅场内ETF",
                "初始化说明": "训练硬截断≤2025-12-31；预测年2026仅样本外评测（全年+压力月）。",
                "更新时间": utc_now(),
            }
        )
    return win


def shift_window(win: dict[str, Any], train_panel: pd.DataFrame, full_panel: pd.DataFrame) -> dict[str, Any]:
    """仅在训练期内右移进化窗；预测年锚点不变。"""
    e0 = date.fromisoformat(win["进化段起"])
    ne0 = shift_evol_start(e0, 6)
    ne1 = TRAIN_END
    # 保持进化段终点锚定训练截止；起点右移但不得使段长失控：仍要求 ne0 < ne1
    if ne0 >= TRAIN_END:
        win["状态"] = "训练窗已锚定_预测年复评"
        win["备注"] = "进化段已贴训练截止日；继续在固定训练段上突变，并用预测年2026复评。"
        win["预测段止"] = str(pred_segment_end(full_panel))
        win["盲测段止"] = win["预测段止"]
        win["更新时间"] = utc_now()
        return win

    if len(train_panel.loc[str(ne0) : str(ne1)]) < 20:
        win["状态"] = "训练窗已锚定_预测年复评"
        win["备注"] = "训练段交易日不足右移；锚定复评预测年。"
        win["更新时间"] = utc_now()
        return win

    p1 = pred_segment_end(full_panel)
    win.update(
        {
            "进化段起": str(ne0),
            "进化段止": str(ne1),
            "预测段起": str(PRED_START),
            "预测段止": str(p1),
            "盲测段起": str(PRED_START),
            "盲测段止": str(p1),
            "状态": "突变中",
            "更新时间": utc_now(),
            "备注": "训练期内右移进化起点；预测年保持2026样本外。",
        }
    )
    return win


def checker() -> dict[str, Any]:
    skill = SKILL_PATH.read_text(encoding="utf-8")
    genome = GENOME_PATH.read_text(encoding="utf-8")
    issues = []
    if "冻结执行壳" not in skill and "自进化多头技能" not in skill:
        issues.append("skill.md 壳标识异常")
    # 壳哈希存档比对（若存在）
    hash_path = EVOL / "skill_shell.sha256"
    h = hashlib.sha256(skill.encode("utf-8")).hexdigest()
    if hash_path.exists():
        old = hash_path.read_text().strip()
        if old != h:
            issues.append("skill.md 壳被改动（哈希不一致）")
    else:
        hash_path.write_text(h + "\n", encoding="utf-8")

    for pat in FORBIDDEN_NAME_PATTERNS:
        if re.search(pat, genome):
            issues.append(f"策略基因含违禁点名模式: {pat}")

    # 细分行业/概念固定偏好粗检
    bad_phrases = ["永远优先", "固定超配", "只买", "永不满仓某", "专做"]
    for p in bad_phrases:
        if p in genome:
            issues.append(f"疑似非法偏好措辞: {p}")

    # 禁止条款/非法列表示例会提到「盲测段…涨跌」，仅在非约束语境下报警
    in_illegal_section = False
    for line in genome.splitlines():
        if "非法突变" in line or "检查器拒收" in line:
            in_illegal_section = True
            continue
        if in_illegal_section and line.startswith("## "):
            in_illegal_section = False
        if in_illegal_section:
            continue
        if any(k in line for k in ("禁止", "非法", "不得", "拒收", "不可读", "不可用")):
            continue
        if re.search(r"因盲测段.*(?:涨|跌|收益)|根据盲测.*(?:涨|跌)", line):
            issues.append("策略基因疑似引用盲测未来涨跌")
            break

    return {"通过": not issues, "问题": issues, "壳哈希": h}


def latest_live_files() -> dict[str, list[str]]:
    refs = sorted([p.name for p in (EVOL / "live_reflections").glob("*.md")])
    props = sorted([p.name for p in (EVOL / "live_proposals").glob("*.md")])
    # 忽略 gitkeep
    refs = [x for x in refs if not x.startswith(".")]
    props = [x for x in props if not x.startswith(".")]
    return {"反思": refs[-5:], "提案": props[-5:]}


def annualize(total_return: float, n_days: int) -> float:
    if n_days <= 0:
        return 0.0
    return (1.0 + total_return) ** (252.0 / n_days) - 1.0


def max_drawdown(equity: list[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    mdd = 0.0
    for x in equity:
        peak = max(peak, x)
        if peak > 0:
            mdd = max(mdd, (peak - x) / peak)
    return mdd


def month_bounds(ym: str) -> tuple[str, str]:
    y, m = map(int, ym.split("-"))
    start = date(y, m, 1)
    end = date(y + 1, 1, 1) - timedelta(days=1) if m == 12 else date(y, m + 1, 1) - timedelta(days=1)
    return str(start), str(end)


def backtest(
    panel: pd.DataFrame,
    g: GenomeParams,
    start: str,
    end: str,
    *,
    collect_trades: bool = False,
    tag: str = "",
) -> dict[str, Any]:
    emap = load_etf_map()
    seg = panel.loc[start:end].copy()
    if len(seg) < 5:
        return {
            "总收益": 0.0,
            "年化": 0.0,
            "最大回撤": 1.0,
            "交易日数": float(len(seg)),
            "期末权益": INIT_CASH,
            "成交次数": 0.0,
            "交易明细": [],
        }

    idx = panel.index
    start_ts = pd.Timestamp(start)
    warm_idx = idx[idx <= start_ts]
    warm_start = warm_idx[-120] if len(warm_idx) >= 120 else warm_idx[0]
    full = panel.loc[warm_start:end].copy()

    base_close = full["基准_close"]
    ma20 = base_close.rolling(20).mean()
    ma60 = base_close.rolling(60).mean()
    vol20 = base_close.pct_change().rolling(20).std()

    weights = {"进攻": 0.0, "次线": 0.0, "防御": 0.0, "工具": 0.0}
    peak_px = {k: np.nan for k in weights}
    equity = INIT_CASH
    pending: dict[str, Any] | None = None
    dates = list(full.index)
    in_blind = False
    blind_eq: list[float] = []
    trade_count = 0
    trades: list[dict[str, Any]] = []

    for i, dt in enumerate(dates):
        row = full.loc[dt]
        dstr = str(dt.date())
        if dstr >= start:
            in_blind = True

        if pending is not None and in_blind and i > 0:
            target = pending["weights"]
            signal_day = pending["signal_day"]
            rationale = pending["rationale"]
            old_w = dict(weights)
            turnover = sum(abs(target.get(k, 0.0) - weights[k]) for k in weights)
            if turnover / 2 > 0.30:
                scale = 0.30 / (turnover / 2)
                for k in weights:
                    weights[k] = weights[k] + (target[k] - weights[k]) * scale
            else:
                weights = {k: float(target[k]) for k in weights}
            cost = equity * (turnover * (COMMISSION + SLIPPAGE) + 0.5 * turnover * STAMP_SELL)
            equity = max(1.0, equity - cost)
            if turnover > 1e-6:
                trade_count += 1
                if collect_trades:
                    for k in weights:
                        dw = weights[k] - old_w[k]
                        if abs(dw) < 1e-4:
                            continue
                        name, code = etf_label(k, emap)
                        op = "加仓" if dw > 0 else "减仓"
                        if old_w[k] < 1e-6 and dw > 0:
                            op = "开仓"
                        if weights[k] < 1e-6 and dw < 0:
                            op = "清仓"
                        trades.append(
                            {
                                "标签": tag,
                                "信号日": signal_day,
                                "成交日": dstr,
                                "ETF名称": name,
                                "ETF代码": code,
                                "角色": k,
                                "操作": op,
                                "仓位变化": round(dw, 4),
                                "成交后仓位": round(weights[k], 4),
                                "成交价": float(row[f"{k}_open"]),
                                "盘面依据": rationale["盘面"],
                                "动机形态": rationale["动机"],
                            }
                        )
            pending = None
            for k in weights:
                if weights[k] > 0:
                    peak_px[k] = row[f"{k}_open"] if np.isnan(peak_px[k]) else max(peak_px[k], row[f"{k}_open"])

        if in_blind and i > 0:
            prev = full.loc[dates[i - 1]]
            day_ret = 0.0
            for k, w in weights.items():
                px0 = prev[f"{k}_close"]
                px1 = row[f"{k}_close"]
                if px0 > 0:
                    day_ret += w * (px1 / px0 - 1.0)
            equity *= 1.0 + day_ret
            for k, w in weights.items():
                if w > 0:
                    peak_px[k] = row[f"{k}_close"] if np.isnan(peak_px[k]) else max(peak_px[k], row[f"{k}_close"])

        if in_blind:
            blind_eq.append(equity)

        if not in_blind or i >= len(dates) - 1:
            continue
        if pd.isna(ma20.loc[dt]) or pd.isna(ma60.loc[dt]):
            continue

        c = base_close.loc[dt]
        regime = "震荡"
        if c > ma60.loc[dt] and ma20.loc[dt] > ma60.loc[dt]:
            regime = "趋势"
        if vol20.loc[dt] > vol20.loc[:dt].dropna().quantile(0.9):
            regime = "混沌"

        if regime == "趋势":
            pos_lo, pos_hi = g.trend_pos
        elif regime == "混沌":
            pos_lo, pos_hi = g.chaos_pos
        else:
            pos_lo, pos_hi = g.range_pos

        target_gross = (pos_lo + pos_hi) / 2
        if g.chaos_low_activity and regime == "混沌":
            target_gross = min(target_gross, g.chaos_pos[1])

        rets = base_close.pct_change()
        recent = rets.loc[:dt].tail(g.outflow_days)
        weak = bool((recent < 0).sum() >= max(1, g.outflow_days - 1))
        recent_ret = float(recent.sum()) if len(recent) else 0.0

        tgt = {"进攻": 0.0, "次线": 0.0, "防御": 0.0, "工具": 0.0}
        motive = "格局持有"
        if weak and regime != "趋势":
            tgt["防御"] = min(g.defense_hi, target_gross * 0.5)
            tgt["工具"] = min(g.tool_hi, max(0.0, target_gross - tgt["防御"]) * 0.5)
            motive = "主题资金偏弱切防御/工具ETF"
        elif regime == "趋势":
            off = min(g.offense_hi, target_gross * 0.55)
            sec = min(g.secondary_hi, target_gross * 0.20)
            tool = min(g.tool_hi, max(0.0, target_gross - off - sec))
            if g.chase_breakout:
                off = min(g.offense_hi, off + g.open_step * 0.5)
            tgt.update({"进攻": off, "次线": sec, "工具": tool})
            motive = "趋势体制超配成长/卫星/宽基ETF"
        else:
            sec = min(g.secondary_hi, target_gross * 0.35)
            tool = min(g.tool_hi, target_gross * 0.35)
            dfn = min(g.defense_hi, max(0.0, target_gross - sec - tool))
            tgt.update({"次线": sec, "工具": tool, "防御": dfn})
            motive = "震荡体制均衡配置ETF角色"

        risk_hit = []
        for k in list(tgt):
            px = row[f"{k}_close"]
            pk = peak_px[k]
            if weights.get(k, 0) > 0 and not np.isnan(pk) and pk > 0:
                dd = (pk - px) / pk
                if dd >= g.stop_from_peak:
                    tgt[k] = 0.0
                    risk_hit.append(f"{k}触及止损")
                elif dd >= g.warn_from_peak:
                    tgt[k] = min(tgt[k], g.open_step)
                    risk_hit.append(f"{k}预警减压")

        if i >= 2:
            for k in tgt:
                p0 = full.loc[dates[i - 2], f"{k}_close"]
                p1 = row[f"{k}_close"]
                if p0 > 0 and (p0 - p1) / p0 > g.two_day_crash:
                    tgt[k] = 0.0
                    risk_hit.append(f"{k}两日暴跌清仓")

        for k in tgt:
            tgt[k] = min(tgt[k], 0.20)
        s = sum(tgt.values())
        if s > 0.95:
            for k in tgt:
                tgt[k] *= 0.95 / s

        cur = weights
        delta_sum = sum(abs(tgt[k] - cur[k]) for k in tgt)
        max_step = max(g.open_step * 3, 0.15)
        if delta_sum > max_step:
            for k in tgt:
                tgt[k] = cur[k] + (tgt[k] - cur[k]) * (max_step / delta_sum)

        gross = sum(tgt.values())
        if 1.0 - gross < g.cash_lo:
            scale = max(0.0, 1.0 - g.cash_lo) / max(gross, 1e-9)
            for k in tgt:
                tgt[k] *= scale

        if risk_hit:
            motive = motive + "；" + "、".join(risk_hit)

        board = (
            f"截至{dstr}收盘前可得：宽基体制={regime}；MA20={float(ma20.loc[dt]):.2f}；"
            f"MA60={float(ma60.loc[dt]):.2f}；近{g.outflow_days}日基准累计涨跌={recent_ret:.2%}；"
            f"资金连续偏弱={weak}"
        )
        pending = {
            "weights": tgt,
            "signal_day": dstr,
            "rationale": {"盘面": board, "动机": motive},
        }

    if not blind_eq:
        return {
            "总收益": 0.0,
            "年化": 0.0,
            "最大回撤": 1.0,
            "交易日数": 0.0,
            "期末权益": INIT_CASH,
            "成交次数": 0.0,
            "交易明细": [],
        }

    total_ret = blind_eq[-1] / INIT_CASH - 1.0
    n = len(blind_eq)
    return {
        "总收益": float(total_ret),
        "年化": float(annualize(total_ret, n)),
        "最大回撤": float(max_drawdown(blind_eq)),
        "交易日数": float(n),
        "期末权益": float(blind_eq[-1]),
        "成交次数": float(trade_count),
        "交易明细": trades,
    }


def stress_slices(panel: pd.DataFrame, g: GenomeParams) -> dict[str, Any]:
    out: dict[str, Any] = {}
    last = panel.index.max().date()
    for ym in STRESS_MONTHS:
        s, e = month_bounds(ym)
        if date.fromisoformat(s) > last:
            out[ym] = {"可得": False, "说明": "数据尚未覆盖"}
            continue
        e_eff = min(date.fromisoformat(e), last)
        bt = backtest(panel, g, s, str(e_eff), collect_trades=False, tag=f"压力月{ym}")
        out[ym] = {
            "可得": True,
            "区间": f"{s}~{e_eff}",
            "扣成本收益": bt["总收益"],
            "最大回撤": bt["最大回撤"],
            "成交次数": bt["成交次数"],
            "交易日数": bt["交易日数"],
        }
    return out


def stress_gate(cand: dict[str, Any], base: dict[str, Any]) -> tuple[bool, str]:
    notes = []
    ok = True
    for ym in STRESS_MONTHS:
        c, b = cand.get(ym) or {}, base.get(ym) or {}
        if not c.get("可得"):
            continue
        cdd, bdd = float(c["最大回撤"]), float(b.get("最大回撤") or 0)
        if b.get("可得") and b.get("交易日数", 0) > 0:
            if cdd > bdd * (1 + EPS_DD) + 1e-12:
                ok = False
                notes.append(f"{ym}回撤{cdd:.2%}>现任{bdd:.2%}×1.1")
        elif cdd > STRESS_MDD_CAP + 1e-12:
            ok = False
            notes.append(f"{ym}回撤{cdd:.2%}>上限{STRESS_MDD_CAP:.0%}")
    return ok, ("；".join(notes) if notes else "压力月通过")


def year_return_gate(cand_ret: float, base_ret: float) -> tuple[bool, str]:
    """预测年全年/迄今收益不得显著差于现任。"""
    floor = base_ret - abs(base_ret) * EPS_DD - 0.02  # 相对ε再加2pct绝对缓冲
    if base_ret >= 0:
        floor = min(floor, base_ret - 0.02)
    ok = cand_ret + 1e-12 >= floor
    return ok, (f"全年收益通过({cand_ret:.2%}≥{floor:.2%})" if ok else f"全年收益不足({cand_ret:.2%}<{floor:.2%})")


def live_score(scorecard: dict[str, Any]) -> tuple[float | None, str, bool]:
    """返回 (分数或None, 说明, 是否就绪)。未就绪时不得阻塞纯历史晋级。"""
    days = int(scorecard.get("交易日数") or 0)
    if days < MIN_LIVE_DAYS or scorecard.get("扣成本年化") is None:
        return None, f"实盘未就绪（交易日={days} < {MIN_LIVE_DAYS}）", False
    ann = float(scorecard["扣成本年化"])
    viol = int(scorecard.get("违规次数") or 0)
    score = ann - 0.05 * viol
    return score, f"任期内年化={ann:.4f}，违规={viol}", True


def mutate_candidates(base: GenomeParams, live_meta: dict[str, list[str]], live_ready: bool) -> list[dict[str, Any]]:
    """实盘就绪且有文档：至少3/5实盘；否则允许5/5历史（含压力月抽象）。"""
    has_live_docs = bool(live_meta["反思"] or live_meta["提案"])
    live_theme = (
        "响应近期实盘反思/提案"
        if has_live_docs
        else "冷启动/无实盘文档：围绕压力月降回撤与ETF角色风控做历史向突变"
    )

    cands: list[dict[str, Any]] = []

    def pack(name: str, src: str, g: GenomeParams, diff: str, rationale: str) -> dict[str, Any]:
        g = g.clamp()
        return {
            "名称": name,
            "来源": src,
            "差异": diff,
            "理由": rationale,
            "参数": g,
        }

    src_primary = "实盘" if (live_ready and has_live_docs) else "历史"

    g1 = deepcopy(base)
    g1.stop_from_peak = 0.16
    g1.warn_from_peak = 0.12
    cands.append(
        pack(
            "C1_收紧止损",
            src_primary,
            g1,
            "G4.自高点强制止损 0.20→0.16；预警 0.15→0.12",
            live_theme + "；降低ETF组合回撤（针对压力月）",
        )
    )

    g2 = deepcopy(base)
    g2.chaos_pos = (0.00, 0.35)
    g2.cash_lo = 0.10
    cands.append(
        pack(
            "C2_混沌降仓",
            src_primary,
            g2,
            "G1.混沌期总仓上限 0.50→0.35；现金下限 0.05→0.10",
            live_theme + "；回落市更大现金",
        )
    )

    g3 = deepcopy(base)
    g3.open_step = 0.06
    g3.outflow_days = 2
    cands.append(
        pack(
            "C3_小步更快流出反应",
            src_primary,
            g3,
            "G2.新开底仓 0.08→0.06；G5.主题流出退出天数 3→2",
            live_theme + "；ETF试错更小、流出反应更快",
        )
    )

    g4 = deepcopy(base)
    g4.trend_pos = (0.85, 0.95)
    g4.offense_hi = 0.70
    cands.append(
        pack(
            "C4_历史_趋势抬仓",
            "历史",
            g4,
            "G1.趋势上行总仓下限 0.80→0.85",
            "历史窗对照：趋势段ETF参与度略升",
        )
    )

    g5 = deepcopy(base)
    g5.stop_from_peak = 0.18
    g5.warn_from_peak = 0.13
    g5.chaos_pos = (0.00, 0.40)
    cands.append(
        pack(
            "C5_历史_压力月防回撤",
            "历史",
            g5,
            "G4.止损0.20→0.18；G1.混沌上限0.50→0.40",
            "历史/压力月抽象归因：加快减仓，禁正股只做ETF角色轮换",
        )
    )

    if live_ready and has_live_docs:
        assert sum(1 for c in cands if c["来源"] == "实盘") >= 3
        assert sum(1 for c in cands if c["来源"] == "历史") <= 2
    return cands


def genome_md_from_params(g: GenomeParams, parent: str, generation: int, note: str) -> str:
    # 仅在晋级时调用；本轮因实盘不足预期不晋级
    return f"""---
name: strategy-genome
version: {g.version}
generation: {generation}
description: 可进化战术基因。禁止个股名与细分行业/概念名。
---

# 策略基因 {g.version}

> 进化代理只能改本文件。任何点名股票/细分行业/概念的条款视为非法，检查器拒收。

## G0. 元数据

- 版本：{g.version}
- 父版本：{parent}
- 说明：{note}

## G1. 角色预算目标（在硬顶内）

| 角色 | 目标下限 | 目标上限 | 说明 |
|------|----------|----------|------|
| 现金 | {g.cash_lo:.2f} | 1.00 | 不确定时抬升 |
| 主线进攻 | 0.00 | {g.offense_hi:.2f} | 当日动态主线，非预设名单 |
| 次线配置 | 0.00 | {g.secondary_hi:.2f} | 低位配置、降低换手 |
| 防御 | 0.00 | {g.defense_hi:.2f} | 降组合波动时启用 |
| 周期 | 0.00 | {g.cycle_hi:.2f} | 与进攻波动互补 |
| 工具（宽基/货币类 ETF） | 0.00 | {g.tool_hi:.2f} | 角色级，不点名代码 |

**总仓软目标（受硬顶约束）**：

- 趋势上行总仓：{g.trend_pos[0]:.2f}–{g.trend_pos[1]:.2f}
- 震荡轮动总仓：{g.range_pos[0]:.2f}–{g.range_pos[1]:.2f}
- 混沌期总仓：{g.chaos_pos[0]:.2f}–{g.chaos_pos[1]:.2f}
- 事件前风险总仓：0.60–0.75

## G2. 单笔步长

| 动作 | 默认步长 | 允许区间 |
|------|----------|----------|
| 新开底仓 | {g.open_step:.2f} | 0.05–0.10 |
| 试错加仓 | 0.02 | 0.01–0.03 |
| 趋势加仓 | 0.02–0.03 | 单票加后仍 ≤ 壳硬顶与 G3 |
| 减压至底仓 | 目标回到 0.08–0.09 | — |
| 平换目标仓位 | 0.08–0.10 | 可略高于被替换仓位 |
| 日内 T 加仓层 | 0.02 | 隔日不强则去掉加仓层 |

## G3. 单票软顶（硬顶仍为 20%）

- 减压带：0.12–0.15
- 强制下调线：0.15
- 底仓带：0.08–0.09

## G4. 风控触发（抽象）

- 自高点强制止损：{g.stop_from_peak:.2f}
- 自高点预警：{g.warn_from_peak:.2f}
- 两日暴跌止损：连续 2 日累计跌幅 > {g.two_day_crash:.2f} → 强制退出优先
- 中线补仓需回撤达到：0.10
- 弱势禁止补仓：是
- 短线角色禁止补仓：是

## G5. 资金连续性（主题级，非点名）

- 主题流出退出天数：{g.outflow_days}
- 主题放弃周数：{g.abandon_weeks}
- 单日流出不足以为据：是

## G6. 买入触发权重（形态开关）

- 新主题建底仓：开
- 主线扩散第二梯队：开
- 降波动开防御：开
- 相对滞后补涨：开
- 同主题换更高壁垒：开
- 同主题换业绩可验证：开
- 同主题换容量：开
- 公司行为后容量介入：开
- 分歧试错：开
- 均线回踩加仓：开
- 早盘弱势介入：开
- 追涨延伸突破：{"开" if g.chase_breakout else "关"}

## G7. 卖出触发权重

- 目标到达分批减：开
- 催化失望减仓：开
- 超配减压带减仓：开
- 日内 T 失败去掉加仓层：开
- 基本面不及预期退出：开
- 估值压制退出：开
- 主题资金退出：开
- 弱于同主题退出：开
- 风控止损退出：开

## G8. 时机偏好（抽象）

- 偏好分歧尾盘试错：{"是" if g.prefer_divergence else "否"}
- 偏好早盘弱势加仓：是
- 禁止高开冲高追涨：是
- 反抽减而非下杀减：是
- 混沌期尽量少动：{"是" if g.chaos_low_activity else "否"}
- 试错最低盈亏比：{g.min_rr:.1f}

## G9. 催化密度（多笔同步动作门槛）

- 催化最少命中数：{g.catalyst_hits}
- 催化信号（抽象，运行时实例化）：
  - 动态主线出现广度高潮
  - 动态主线出现可验证超预期催化
  - 持仓中存在与当前主线背离的防御/弱仓需腾位

## G10. 换手与交易频率软约束

- 无独立动机时单日最多清/减只数：{g.max_sells_day}
- 逻辑未坏优先持有：是
- 允许日内 T：是

## G11. 进化约束（写给突变器）

合法突变示例：调整 G1–G3 端点；开关 G6/G7；微调 G4/G5/G8/G9；响应实盘提案。
非法：点名；改壳；放宽成本/ε/权重；用盲测事后涨跌倒推。

## G12. 评分权重提示（只读）

- 实盘权重：0.70
- 历史权重：0.30
- 最少实盘交易日：20

## G13. 版本记录

| 版本 | 代数 | 变更 |
|------|------|------|
| {g.version} | {generation} | {note} |
"""


def write_trades_md(gen_id: str, trades: list[dict[str, Any]], title: str) -> Path:
    TRADES.mkdir(parents=True, exist_ok=True)
    path = TRADES / f"{gen_id}_trades.md"
    lines = [
        f"# 交易明细 {gen_id}",
        "",
        f"- 宇宙：**仅场内 ETF**（禁止正股）",
        f"- 说明：{title}",
        f"- 笔数：{len(trades)}",
        "",
    ]
    if not trades:
        lines.append("（本窗无调仓成交）")
    for i, t in enumerate(trades, 1):
        lines += [
            f"## 第{i}笔",
            f"- 信号日：{t['信号日']} → 成交日：{t['成交日']}",
            f"- ETF：{t['ETF名称']}（{t['ETF代码']}）角色={t['角色']}",
            f"- 操作：{t['操作']}；仓位变化：{t['仓位变化']:.2%} → 成交后 {t['成交后仓位']:.2%}",
            f"- 成交价：{t['成交价']:.4f}",
            f"- 动机形态：{t['动机形态']}",
            f"- 盘面依据：{t['盘面依据']}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
    (TRADES / f"{gen_id}_trades.json").write_text(
        json.dumps(trades, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def main() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    TRADES.mkdir(parents=True, exist_ok=True)
    train_panel = load_panel(for_training=True)
    full_panel = load_panel(for_training=False)
    # 断言：训练面板不得含预测年
    if len(train_panel) and train_panel.index.max().date() > TRAIN_END:
        raise RuntimeError("训练面板泄漏：含训练截止日之后数据")

    win = init_or_load_window(train_panel, full_panel)
    scorecard = json.loads(SCORECARD_PATH.read_text(encoding="utf-8"))
    live_days = int(scorecard.get("交易日数") or 0)
    win["现任实盘交易日数"] = live_days

    check = checker()
    # 额外：训练截止检查
    if date.fromisoformat(win["进化段止"]) > TRAIN_END:
        check["通过"] = False
        check["问题"] = list(check.get("问题") or []) + ["进化段越过训练截止日"]

    live_meta = latest_live_files()
    base = parse_v0_genome()
    base.version = win.get("现任基因版本") or "v0"

    e0, e1 = win["进化段起"], win["进化段止"]
    p0, p1 = win["预测段起"], win["预测段止"]

    live_s, live_note, live_ready = live_score(scorecard)
    mode = "实盘加权" if live_ready else "纯历史"
    win["晋级模式"] = mode

    # 预测年评分（全年/迄今）— 仅评测可用 full_panel
    base_pred = backtest(full_panel, base, p0, p1, collect_trades=True, tag="现任预测年")
    base_stress = stress_slices(full_panel, base)
    if live_ready:
        assert live_s is not None
        base_composite = W_LIVE * live_s + W_HIST * base_pred["年化"]
    else:
        base_composite = base_pred["年化"]

    paper_mdd = float(scorecard.get("最大回撤") or 0.0)
    cands = mutate_candidates(base, live_meta, live_ready)
    rows = []
    best = None
    all_trades: list[dict[str, Any]] = list(base_pred.get("交易明细") or [])

    for c in cands:
        if any(re.search(p, c["差异"] + c["理由"]) for p in FORBIDDEN_NAME_PATTERNS):
            rows.append({**c, "拒收": True, "原因": "差异含违禁点名"})
            continue
        pred = backtest(full_panel, c["参数"], p0, p1, collect_trades=True, tag=c["名称"])
        all_trades.extend(pred.get("交易明细") or [])
        cstress = stress_slices(full_panel, c["参数"])
        s_ok, s_note = stress_gate(cstress, base_stress)
        y_ok, y_note = year_return_gate(float(pred["总收益"]), float(base_pred["总收益"]))
        if live_ready:
            assert live_s is not None
            comp = W_LIVE * live_s + W_HIST * pred["年化"]
            dd_live_ok = True
            if scorecard.get("最大回撤") is not None:
                dd_live_ok = paper_mdd <= float(scorecard.get("最大回撤") or 0) * (1 + EPS_DD) + 1e-12
        else:
            comp = pred["年化"]
            dd_live_ok = True
        dd_pred_ok = pred["最大回撤"] <= base_pred["最大回撤"] * (1 + EPS_DD) + 1e-12
        pass_gate = (
            check["通过"]
            and dd_pred_ok
            and dd_live_ok
            and s_ok
            and y_ok
            and (comp > base_composite + 1e-12)
        )
        rec = {
            "名称": c["名称"],
            "来源": c["来源"],
            "差异": c["差异"],
            "理由": c["理由"],
            "实盘分": live_s if live_ready else "未就绪",
            "预测年分": pred["年化"],
            "预测年收益": pred["总收益"],
            "综合分": comp,
            "预测年最大回撤": pred["最大回撤"],
            "成交次数": pred["成交次数"],
            "压力月": cstress,
            "压力月门槛": s_note,
            "全年收益门槛": y_note,
            "回撤门槛_预测年通过": dd_pred_ok,
            "回撤门槛_纸交易通过": dd_live_ok,
            "压力月通过": s_ok,
            "全年收益通过": y_ok,
            "可晋级": pass_gate,
            "拒收": False,
        }
        rows.append(rec)
        if pass_gate and (best is None or rec["综合分"] > best["综合分"]):
            best = {**rec, "参数": c["参数"]}

    decision = "保持现任"
    new_version = base.version
    pending_audit = False
    if best:
        decision = f"候选晋级:{best['名称']}（待监督审计确认后生效）"
        pending_audit = True
        m = re.search(r"v(\d+)", base.version)
        new_version = f"v{int(m.group(1))+1}" if m else "v1"
        best["参数"].version = new_version
        GENOME_PATH.write_text(
            genome_md_from_params(
                best["参数"],
                parent=base.version,
                generation=int(m.group(1)) + 1 if m else 1,
                note=best["差异"],
            ),
            encoding="utf-8",
        )
        win["现任基因版本"] = new_version
        win["待监督确认"] = True
    else:
        decision = "保持现任（无候选过预测年全年/压力月/回撤门槛）"
        win["待监督确认"] = False

    win["状态"] = "右移中"
    win = shift_window(win, train_panel, full_panel)

    win["最近决策"] = decision
    win["最近一代时间"] = utc_now()
    win["现任实盘交易日数"] = live_days
    win["更新时间"] = utc_now()
    WINDOW_PATH.write_text(json.dumps(win, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    gen_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trades_path = write_trades_md(gen_id, all_trades, "预测年2026内ETF调仓明细（仅盘面依据，无消息面）")

    log = {
        "代数时间": gen_id,
        "晋级模式": mode,
        "训练截止日": str(TRAIN_END),
        "预测年": PRED_YEAR,
        "交易宇宙": "仅场内ETF",
        "待监督确认": pending_audit,
        "窗口": {
            "进化段": f"{e0}~{e1}",
            "预测段": f"{p0}~{p1}",
            "右移后状态": win.get("状态"),
        },
        "检查器": check,
        "消费的实盘文件": live_meta,
        "现任": {
            "版本": base.version,
            "实盘交易日": live_days,
            "实盘分": live_s if live_ready else "未就绪",
            "实盘注": live_note,
            "预测年分": base_pred["年化"],
            "预测年收益": base_pred["总收益"],
            "预测年最大回撤": base_pred["最大回撤"],
            "综合分": base_composite,
            "压力月": base_stress,
            "预测年成交次数": base_pred["成交次数"],
        },
        "权重": {"实盘": W_LIVE if live_ready else 0.0, "预测年": W_HIST if live_ready else 1.0},
        "候选": [{k: v for k, v in r.items() if k != "参数"} for r in rows],
        "决策": decision,
        "新版本": new_version,
        "交易明细文件": str(trades_path.relative_to(ROOT)),
        "回测说明": "训练≤2025-12-31不可见2026；预测年2026样本外评测全年+压力月；仅盘面依据。",
    }

    lines = [
        f"# 进化日志 {gen_id}",
        "",
        f"- 决策：**{decision}**",
        f"- 晋级模式：**{mode}**",
        f"- 训练截止：**{TRAIN_END}**（训练不可见其后数据）",
        f"- 预测年：**{PRED_YEAR}** 段 {p0} ~ {p1}",
        f"- 进化段（仅训练期）：{e0} ~ {e1}",
        f"- 交易宇宙：仅场内 ETF；依据：仅盘面（无消息面）",
        f"- 现任版本：{base.version} → {new_version}",
        f"- 检查器：{'通过' if check['通过'] else '失败'} {check['问题']}",
        f"- 实盘：{live_note}",
        f"- 现任预测年收益={base_pred['总收益']:.2%}；年化={base_pred['年化']:.4f}；最大回撤={base_pred['最大回撤']:.2%}",
        f"- 综合分={base_composite:.4f}",
        f"- 交易明细：`{trades_path.relative_to(ROOT)}`（{len(all_trades)} 笔）",
        f"- 待监督确认：{pending_audit}",
        "",
        "## 预测年全年 / 迄今（必填）",
        "",
        f"- 区间：{p0} ~ {p1}",
        f"- 扣成本收益：{base_pred['总收益']:.2%}",
        f"- 年化：{base_pred['年化']:.4f}",
        f"- 最大回撤：{base_pred['最大回撤']:.2%}",
        f"- 成交次数：{int(base_pred['成交次数'])}",
        "",
        "## 压力月专项（必填）",
        "",
    ]
    for ym in STRESS_MONTHS:
        s = base_stress.get(ym) or {}
        if not s.get("可得"):
            lines.append(f"- **{ym}**：数据未覆盖")
        else:
            lines.append(
                f"- **{ym}**（{s['区间']}）：扣成本收益={s['扣成本收益']:.2%}；最大回撤={s['最大回撤']:.2%}；成交={int(s['成交次数'])}次"
            )
    lines += [
        "",
        "## 候选",
        "",
        "| 名称 | 来源 | 实盘分 | 预测年分 | 全年收益 | 回撤 | 全年门槛 | 压力月 | 可晋级 | 差异 |",
        "|---|---|---|---:|---:|---:|---|---|---|---|",
    ]
    for r in rows:
        if r.get("拒收"):
            lines.append(f"| {r['名称']} | {r['来源']} | - | - | - | - | 拒收 | - | 否 | {r.get('原因')} |")
            continue
        live_cell = f"{r['实盘分']:.4f}" if isinstance(r["实盘分"], float) else r["实盘分"]
        lines.append(
            f"| {r['名称']} | {r['来源']} | {live_cell} | {r['预测年分']:.4f} | {r['预测年收益']:.2%} | "
            f"{r['预测年最大回撤']:.2%} | {r['全年收益门槛']} | {r['压力月门槛']} | "
            f"{'是' if r['可晋级'] else '否'} | {r['差异']} |"
        )
    lines += [
        "",
        "## 说明",
        "",
        "- 训练硬截断≤2025-12-31；2026仅样本外。",
        "- 纯历史：综合分=预测年年化；实盘加权：0.70×实盘+0.30×预测年。",
        "- 须同时过：预测年回撤、全年收益、压力月（04/07）。",
        "- 交易明细仅盘面依据；晋级须监督通过。",
        "",
        "```json",
        json.dumps(log, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    log_path = LOG / f"{gen_id}_gen.md"
    log_path.write_text("\n".join(lines), encoding="utf-8")
    (LOG / f"{gen_id}_gen.json").write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "决策": decision,
        "晋级模式": mode,
        "训练截止": str(TRAIN_END),
        "预测段": f"{p0}~{p1}",
        "现任预测年收益": base_pred["总收益"],
        "交易明细": str(trades_path.relative_to(ROOT)),
        "交易笔数": len(all_trades),
        "日志": str(log_path.relative_to(ROOT)),
        "待监督确认": pending_audit,
        "压力月现任": base_stress,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))



if __name__ == "__main__":
    main()

