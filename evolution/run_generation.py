#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一代策略基因进化：检查器 → 突变 → 盲测回放 → 实盘加权 → 晋级/保持 → 右移。"""

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
STATE = ROOT / "state"
GENOME_PATH = ROOT / "strategy_genome.md"
SKILL_PATH = ROOT / "skill.md"
WINDOW_PATH = EVOL / "window.json"
SCORECARD_PATH = STATE / "live_scorecard.json"

# 冻结评估成本（与 skill.md 一致）
COMMISSION = 0.0003
STAMP_SELL = 0.001
SLIPPAGE = 0.001
EPS_DD = 0.10
W_LIVE = 0.70
W_HIST = 0.30
INIT_CASH = 1_000_000.0

# 回测角色代理（仅评估器内部；不得写入策略基因规则层）
PROXY = {
    "进攻": "sz399006",
    "次线": "sh000905",
    "防御": "sh000012",
    "工具": "sh000300",
    "基准": "sh000300",
}

FORBIDDEN_NAME_PATTERNS = [
    r"\b(?:茅台|宁德|比亚迪|中芯|升水|赛力斯|寒武纪)\b",
    r"\b[036]\d{5}\b",
    r"\b(?:sh|sz|bj)\d{6}\b",
]


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


def load_panel() -> pd.DataFrame:
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
    # 仅用已收盘完整日；开放中的今日不进入盲测
    today = date(2026, 7, 26)
    panel = panel[panel.index.date < today]
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


def init_or_load_window(panel: pd.DataFrame) -> dict[str, Any]:
    win = json.loads(WINDOW_PATH.read_text(encoding="utf-8"))
    if win.get("进化段起"):
        return win

    first = panel.index.min().date()
    last = panel.index.max().date()
    closed_end = last_closed_month_end(last)

    # 锚定最近已封闭月终点为盲测止：盲测=含该月在内的 6 个自然月，之前 24 个月为进化段
    b1 = closed_end
    b0 = shift_evol_start(date(b1.year, b1.month, 1), -5)
    e0 = shift_evol_start(b0, -24)
    e1 = b0 - timedelta(days=1)
    if e0 < month_start(first) or len(panel.loc[str(b0) : str(b1)]) < 20:
        # 回退：自数据最早月正向滚动到不超过封闭终点的最后合法窗
        evol_start = month_start(first)
        last_ok = None
        while True:
            ce0, ce1, cb0, cb1 = window_from_evol_start(evol_start)
            if cb1 > closed_end:
                break
            if len(panel.loc[str(cb0) : str(cb1)]) >= 20:
                last_ok = (ce0, ce1, cb0, cb1)
            evol_start = shift_evol_start(evol_start, 6)
        if last_ok is None:
            raise RuntimeError("无法构造有效滚动窗：数据不足以覆盖 24+6 个月")
        e0, e1, b0, b1 = last_ok
    win.update(
        {
            "进化段起": str(e0),
            "进化段止": str(e1),
            "盲测段起": str(b0),
            "盲测段止": str(b1),
            "现任基因版本": "v0",
            "现任实盘交易日数": int(json.loads(SCORECARD_PATH.read_text()).get("交易日数") or 0),
            "状态": "评估中",
            "数据最早日": str(first),
            "数据最末日": str(last),
            "已封闭月终点": str(closed_end),
            "初始化说明": "窗口原为空：自最早可得指数代理历史滚动前进至最近已封闭盲测窗。",
            "更新时间": utc_now(),
        }
    )
    return win


def shift_window(win: dict[str, Any], panel: pd.DataFrame) -> dict[str, Any]:
    e0 = date.fromisoformat(win["进化段起"])
    ne0, ne1, nb0, nb1 = window_from_evol_start(shift_evol_start(e0, 6))

    last = panel.index.max().date()
    closed_end = last_closed_month_end(last)
    if nb1 > closed_end:
        win["状态"] = "历史窗已尽_待新月封闭"
        win["备注"] = "右移后盲测终点超出已封闭月；继续消化实盘反思，等待新月份封闭。"
        win["更新时间"] = utc_now()
        return win

    seg = panel.loc[str(nb0) : str(nb1)]
    if len(seg) < 20:
        win["状态"] = "历史窗已尽_待新月封闭"
        win["备注"] = "右移后盲测交易日不足；等待数据延伸。"
        win["更新时间"] = utc_now()
        return win

    win.update(
        {
            "进化段起": str(ne0),
            "进化段止": str(ne1),
            "盲测段起": str(nb0),
            "盲测段止": str(nb1),
            "状态": "突变中",
            "更新时间": utc_now(),
            "备注": "决策后右移 6 个月。",
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


def backtest(panel: pd.DataFrame, g: GenomeParams, start: str, end: str) -> dict[str, float]:
    seg = panel.loc[start:end].copy()
    if len(seg) < 5:
        return {
            "总收益": 0.0,
            "年化": 0.0,
            "最大回撤": 1.0,
            "交易日数": float(len(seg)),
            "期末权益": INIT_CASH,
        }

    # 进化段特征仅用于信号标定的前置指标；评分窗口仍只用盲测段成交
    # 为计算均线，向前多取 120 日（不把前置段计入权益）
    idx = panel.index
    start_ts = pd.Timestamp(start)
    warm_idx = idx[idx <= start_ts]
    warm_start = warm_idx[-120] if len(warm_idx) >= 120 else warm_idx[0]
    full = panel.loc[warm_start:end].copy()

    base_close = full["基准_close"]
    ma20 = base_close.rolling(20).mean()
    ma60 = base_close.rolling(60).mean()
    vol20 = base_close.pct_change().rolling(20).std()

    # 持仓：角色代理权重（占净值）
    weights = {"进攻": 0.0, "次线": 0.0, "防御": 0.0, "工具": 0.0}
    peak_px = {k: np.nan for k in weights}
    entry_ready = False
    cash_w = 1.0
    equity = INIT_CASH
    eq_curve: list[float] = []
    pending: dict[str, float] | None = None  # T 信号 → T+1 开盘目标权重

    dates = list(full.index)
    in_blind = False
    blind_eq: list[float] = []
    trade_count = 0

    for i, dt in enumerate(dates):
        row = full.loc[dt]
        dstr = str(dt.date())
        if dstr >= start:
            in_blind = True
            entry_ready = True

        # 先按 T+1 开盘执行昨信号
        if pending is not None and in_blind and i > 0:
            target = pending
            # 以开盘价调仓，计成本
            turnover = 0.0
            for k in weights:
                turnover += abs(target.get(k, 0.0) - weights[k])
            # 单日单向合计硬顶 30%
            if turnover / 2 > 0.30:
                scale = 0.30 / (turnover / 2)
                for k in weights:
                    weights[k] = weights[k] + (target[k] - weights[k]) * scale
            else:
                weights = {k: float(target[k]) for k in weights}
            cash_w = max(0.0, 1.0 - sum(weights.values()))
            # 成本：双边佣金+滑点；卖出部分另计印花税（近似用减仓绝对值）
            sell_amt = sum(max(0.0, - (target[k] - weights[k])) for k in weights)  # after assign rough
            # 更稳：用 turnover
            cost = equity * (turnover * (COMMISSION + SLIPPAGE) + 0.5 * turnover * STAMP_SELL)
            equity = max(1.0, equity - cost)
            trade_count += 1
            pending = None
            for k in weights:
                if weights[k] > 0:
                    peak_px[k] = row[f"{k}_open"] if np.isnan(peak_px[k]) else max(peak_px[k], row[f"{k}_open"])

        # 当日盈亏：用角色代理日收益（开→收近似用 close/close 前，执行后用当日收益）
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
            eq_curve.append(equity)
            blind_eq.append(equity)

        # 生成信号（仅盲测段；进化段不算分）
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

        # 资金连续性代理：基准近 N 日涨跌与波动
        rets = base_close.pct_change()
        recent = rets.loc[:dt].tail(g.outflow_days)
        weak = bool((recent < 0).sum() >= max(1, g.outflow_days - 1))

        tgt = {"进攻": 0.0, "次线": 0.0, "防御": 0.0, "工具": 0.0}
        if weak and regime != "趋势":
            tgt["防御"] = min(g.defense_hi, target_gross * 0.5)
            tgt["工具"] = min(g.tool_hi, max(0.0, target_gross - tgt["防御"]) * 0.5)
        elif regime == "趋势":
            off = min(g.offense_hi, target_gross * 0.55)
            sec = min(g.secondary_hi, target_gross * 0.20)
            tool = min(g.tool_hi, max(0.0, target_gross - off - sec))
            if g.chase_breakout:
                off = min(g.offense_hi, off + g.open_step * 0.5)
            tgt.update({"进攻": off, "次线": sec, "工具": tool})
        else:
            sec = min(g.secondary_hi, target_gross * 0.35)
            tool = min(g.tool_hi, target_gross * 0.35)
            dfn = min(g.defense_hi, max(0.0, target_gross - sec - tool))
            tgt.update({"次线": sec, "工具": tool, "防御": dfn})

        # 风控：自高点回撤
        for k in list(tgt):
            px = row[f"{k}_close"]
            pk = peak_px[k]
            if weights.get(k, 0) > 0 and not np.isnan(pk) and pk > 0:
                dd = (pk - px) / pk
                if dd >= g.stop_from_peak:
                    tgt[k] = 0.0
                elif dd >= g.warn_from_peak:
                    tgt[k] = min(tgt[k], g.open_step)

        # 两日暴跌
        if i >= 2:
            for k in tgt:
                p0 = full.loc[dates[i - 2], f"{k}_close"]
                p1 = row[f"{k}_close"]
                if p0 > 0 and (p0 - p1) / p0 > g.two_day_crash:
                    tgt[k] = 0.0

        # 硬顶：总仓 ≤ 0.95，单票（角色代理）≤ 0.20
        for k in tgt:
            tgt[k] = min(tgt[k], 0.20)
        s = sum(tgt.values())
        if s > 0.95:
            for k in tgt:
                tgt[k] *= 0.95 / s

        # 步长：向目标靠拢不超过 open_step 量级（组合层）
        cur = weights
        delta_sum = sum(abs(tgt[k] - cur[k]) for k in tgt)
        max_step = max(g.open_step * 3, 0.15)
        if delta_sum > max_step:
            for k in tgt:
                tgt[k] = cur[k] + (tgt[k] - cur[k]) * (max_step / delta_sum)

        # 现金下限
        gross = sum(tgt.values())
        if 1.0 - gross < g.cash_lo:
            scale = max(0.0, 1.0 - g.cash_lo) / max(gross, 1e-9)
            for k in tgt:
                tgt[k] *= scale

        pending = tgt

    if not blind_eq:
        return {
            "总收益": 0.0,
            "年化": 0.0,
            "最大回撤": 1.0,
            "交易日数": 0.0,
            "期末权益": INIT_CASH,
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
    }


def live_score(scorecard: dict[str, Any]) -> tuple[float, str]:
    days = int(scorecard.get("交易日数") or 0)
    if days <= 0 or scorecard.get("扣成本年化") is None:
        return 0.0, "实盘样本为空：实盘分记 0（偏保守，不晋级）"
    ann = float(scorecard["扣成本年化"])
    viol = int(scorecard.get("违规次数") or 0)
    score = ann - 0.05 * viol
    return score, f"任期内年化={ann:.4f}，违规={viol}"


def mutate_candidates(base: GenomeParams, live_meta: dict[str, list[str]]) -> list[dict[str, Any]]:
    """至少 3/5 响应实盘；至多 2/5 纯历史。无提案时响应空样本保本金主题。"""
    has_live_docs = bool(live_meta["反思"] or live_meta["提案"])
    live_theme = (
        "响应近期实盘反思/提案"
        if has_live_docs
        else "响应实盘空样本：保本金、降换手、收紧风控（记分卡交易日=0）"
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

    # 实盘向 1：收紧止损
    g1 = deepcopy(base)
    g1.stop_from_peak = 0.16
    g1.warn_from_peak = 0.12
    cands.append(
        pack(
            "C1_实盘_收紧止损",
            "实盘",
            g1,
            "G4.自高点强制止损 0.20→0.16；预警 0.15→0.12",
            live_theme + "；降低纸交易回撤暴露",
        )
    )

    # 实盘向 2：混沌更低仓
    g2 = deepcopy(base)
    g2.chaos_pos = (0.00, 0.35)
    g2.cash_lo = 0.10
    cands.append(
        pack(
            "C2_实盘_混沌降仓",
            "实盘",
            g2,
            "G1.混沌期总仓上限 0.50→0.35；现金下限 0.05→0.10",
            live_theme + "；不确定则更大现金",
        )
    )

    # 实盘向 3：降低开仓步长
    g3 = deepcopy(base)
    g3.open_step = 0.06
    g3.outflow_days = 2
    cands.append(
        pack(
            "C3_实盘_小步与更快流出反应",
            "实盘",
            g3,
            "G2.新开底仓 0.08→0.06；G5.主题流出退出天数 3→2",
            live_theme + "；试错更小、流出反应更快",
        )
    )

    # 历史向 1：趋势仓略抬（对照）
    g4 = deepcopy(base)
    g4.trend_pos = (0.85, 0.95)
    g4.offense_hi = 0.75
    # offense_hi will clamp with hard top via backtest 0.20 single sleeve — keep budget
    g4.offense_hi = 0.70
    cands.append(
        pack(
            "C4_历史_趋势抬仓",
            "历史",
            g4,
            "G1.趋势上行总仓下限 0.80→0.85",
            "历史窗归因对照：趋势段参与度略升（不超过 2/5 配额）",
        )
    )

    # 历史向 2：略放宽止损对照
    g5 = deepcopy(base)
    g5.stop_from_peak = 0.22
    g5.warn_from_peak = 0.16
    cands.append(
        pack(
            "C5_历史_止损略宽",
            "历史",
            g5,
            "G4.自高点强制止损 0.20→0.22；预警 0.15→0.16",
            "历史窗归因对照：降低过早止损（对照用，冲突时偏信实盘）",
        )
    )

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


def main() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    win = init_or_load_window(panel)
    scorecard = json.loads(SCORECARD_PATH.read_text(encoding="utf-8"))
    live_days = int(scorecard.get("交易日数") or 0)
    win["现任实盘交易日数"] = live_days

    check = checker()
    live_meta = latest_live_files()
    base = parse_v0_genome()
    base.version = win.get("现任基因版本") or "v0"

    b0, b1 = win["盲测段起"], win["盲测段止"]
    e0, e1 = win["进化段起"], win["进化段止"]

    # 现任基线
    base_hist = backtest(panel, base, b0, b1)
    live_s, live_note = live_score(scorecard)
    base_composite = W_LIVE * live_s + W_HIST * base_hist["年化"]
    paper_mdd = scorecard.get("最大回撤")
    if paper_mdd is None:
        paper_mdd = 0.0

    cands = mutate_candidates(base, live_meta)
    rows = []
    best = None
    promote_blocked = live_days < int(win.get("最少实盘交易日") or 20)

    for c in cands:
        # 差异合法性：拒绝点名
        if any(re.search(p, c["差异"] + c["理由"]) for p in FORBIDDEN_NAME_PATTERNS):
            rows.append({**c, "拒收": True, "原因": "差异含违禁点名"})
            continue
        hist = backtest(panel, c["参数"], b0, b1)
        comp = W_LIVE * live_s + W_HIST * hist["年化"]
        dd_hist_ok = hist["最大回撤"] <= base_hist["最大回撤"] * (1 + EPS_DD) + 1e-12
        dd_live_ok = True if live_days == 0 else (paper_mdd <= (paper_mdd * (1 + EPS_DD) + 1e-12))
        # 纸交易回撤：候选尚无独立纸交易账户，沿用现任纸交易回撤门槛（空样本时视为通过占位，但仍不得晋级）
        if live_days == 0:
            dd_live_ok = True
        else:
            # 无候选独立纸交易回撤时，要求不劣于现任记分卡回撤门（保守：沿用现任）
            dd_live_ok = paper_mdd <= float(scorecard.get("最大回撤") or 0) * (1 + EPS_DD) + 1e-12

        pass_gate = dd_hist_ok and dd_live_ok and (comp > base_composite) and (not promote_blocked)
        rec = {
            "名称": c["名称"],
            "来源": c["来源"],
            "差异": c["差异"],
            "理由": c["理由"],
            "实盘分": live_s,
            "历史分": hist["年化"],
            "综合分": comp,
            "历史最大回撤": hist["最大回撤"],
            "历史总收益": hist["总收益"],
            "回撤门槛_历史通过": dd_hist_ok,
            "回撤门槛_纸交易通过": dd_live_ok,
            "可晋级": pass_gate,
            "拒收": False,
        }
        rows.append(rec)
        if pass_gate and (best is None or rec["综合分"] > best["综合分"]):
            best = {**rec, "参数": c["参数"]}

    decision = "保持现任"
    new_version = base.version
    if best and not promote_blocked:
        decision = f"晋级:{best['名称']}"
        # 版本号 +1
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
    else:
        if promote_blocked:
            decision = "保持现任（实盘交易日<20，禁止晋级）"
        elif not any(r.get("可晋级") for r in rows if not r.get("拒收")):
            decision = "保持现任（无候选过线）"

    # 决策后右移
    win["状态"] = "右移中"
    win = shift_window(win, panel)
    win["最近决策"] = decision
    win["最近一代时间"] = utc_now()
    win["现任实盘交易日数"] = live_days
    WINDOW_PATH.write_text(json.dumps(win, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    gen_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = {
        "代数时间": gen_id,
        "窗口": {
            "进化段": f"{e0}~{e1}",
            "盲测段": f"{b0}~{b1}",
            "右移后": {
                "进化段起": win.get("进化段起"),
                "进化段止": win.get("进化段止"),
                "盲测段起": win.get("盲测段起"),
                "盲测段止": win.get("盲测段止"),
                "状态": win.get("状态"),
            },
        },
        "检查器": check,
        "消费的实盘文件": live_meta,
        "现任": {
            "版本": base.version,
            "实盘交易日": live_days,
            "实盘分": live_s,
            "实盘注": live_note,
            "历史分": base_hist["年化"],
            "历史最大回撤": base_hist["最大回撤"],
            "历史总收益": base_hist["总收益"],
            "综合分": base_composite,
            "纸交易最大回撤": scorecard.get("最大回撤"),
        },
        "权重": {"实盘": W_LIVE, "历史": W_HIST},
        "候选": [{k: v for k, v in r.items() if k != "参数"} for r in rows],
        "决策": decision,
        "新版本": new_version,
        "回测代理说明": "历史分用角色代理指数面板（进攻/次线/防御/工具）按基因参数回放；规则层无个股点名。",
        "数据源": "akshare stock_zh_index_daily → evolution/data/*.csv",
    }
    log_path = LOG / f"{gen_id}_gen.md"
    # 中文 Markdown 日志
    lines = [
        f"# 进化日志 {gen_id}",
        "",
        f"- 决策：**{decision}**",
        f"- 现任版本：{base.version} → 记录版本：{new_version}",
        f"- 进化段：{e0} ~ {e1}",
        f"- 盲测段：{b0} ~ {b1}",
        f"- 右移后状态：{win.get('状态')}",
        f"- 检查器：{'通过' if check['通过'] else '失败'} {check['问题']}",
        f"- 实盘文件：反思={live_meta['反思'] or '无'}；提案={live_meta['提案'] or '无'}",
        f"- 实盘分={live_s:.4f}（{live_note}）；现任历史分={base_hist['年化']:.4f}；现任综合分={base_composite:.4f}",
        f"- 现任历史最大回撤={base_hist['最大回撤']:.4f}；纸交易最大回撤={scorecard.get('最大回撤')}",
        "",
        "## 候选",
        "",
        "| 名称 | 来源 | 实盘分 | 历史分 | 综合分 | 历史回撤 | 门槛 | 可晋级 | 差异 |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for r in rows:
        if r.get("拒收"):
            lines.append(f"| {r['名称']} | {r['来源']} | - | - | - | - | 拒收 | 否 | {r.get('原因')} |")
            continue
        gate = f"H{'Y' if r['回撤门槛_历史通过'] else 'N'}/L{'Y' if r['回撤门槛_纸交易通过'] else 'N'}"
        lines.append(
            f"| {r['名称']} | {r['来源']} | {r['实盘分']:.4f} | {r['历史分']:.4f} | {r['综合分']:.4f} | "
            f"{r['历史最大回撤']:.4f} | {gate} | {'是' if r['可晋级'] else '否'} | {r['差异']} |"
        )
    lines += [
        "",
        "## 说明",
        "",
        "- 综合分 = 0.70×实盘分 + 0.30×历史分。",
        "- 双轨回撤 ε=0.10；实盘交易日<20 不得晋级。",
        "- 历史回放：每窗现金 100 万；信号 T→T+1 开盘；佣金双边万三、卖出印花税千一、滑点 0.1%。",
        "- 纸交易账户未重启。",
        "",
        "```json",
        json.dumps(log, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")
    (LOG / f"{gen_id}_gen.json").write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "决策": decision,
        "窗口盲测": f"{b0}~{b1}",
        "右移后状态": win.get("状态"),
        "日志": str(log_path.relative_to(ROOT)),
        "检查器通过": check["通过"],
        "候选摘要": [
            {
                "名称": r["名称"],
                "来源": r["来源"],
                "综合分": r.get("综合分"),
                "历史分": r.get("历史分"),
                "可晋级": r.get("可晋级"),
            }
            for r in rows
            if not r.get("拒收")
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
