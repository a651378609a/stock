---
name: strategy-genome
version: v0
generation: 0
description: 可进化战术基因。禁止个股名与细分行业/概念名。幅度与触发形态可被进化修改；不得突破 skill.md 硬顶与宪法。
---

# Strategy Genome v0

> 进化代理只能改本文件。任何点名股票/细分行业/概念的条款视为非法，检查器拒收。

## G0. 元数据

- `version`: v0
- `parent`: null
- `notes`: 从旧执行稿抽象蒸馏；已剥离品牌、个股与细分优先级。

## G1. 角色预算目标（在硬顶内）

单位：占组合净值比例。进化可改区间端点，但不得超过壳硬顶。

| 角色 | 目标下限 | 目标上限 | 说明 |
|------|----------|----------|------|
| 现金 | 0.05 | 1.00 | 不确定时抬升 |
| 主线进攻 | 0.00 | 0.70 | 当日动态主线，非预设名单 |
| 次线配置 | 0.00 | 0.30 | 低位配置、降低换手 |
| 防御 | 0.00 | 0.25 | 降组合波动时启用 |
| 周期 | 0.00 | 0.30 | 与进攻波动互补 |
| 工具（宽基/货币类 ETF） | 0.00 | 0.40 | 角色级，不点名代码 |

**总仓目标 recs（软目标，受硬顶约束）**：

- `trend_up_total`: 0.80–0.95  
- `range_rotate_total`: 0.60–0.75  
- `chaos_total`: 0.00–0.50  
- `pre_event_risk_total`: 0.60–0.75  

## G2. 单笔步长

| 动作 | 默认步长 | 允许区间 |
|------|----------|----------|
| 新开底仓 | 0.08 | 0.05–0.10 |
| 试错加仓 | 0.02 | 0.01–0.03 |
| 趋势加仓 | 0.02–0.03 | 单票加后仍 ≤ 壳硬顶与 G3 |
| 减压至底仓 | 目标回到 0.08–0.09 | — |
| 平换目标仓位 | 0.08–0.10 | 可略高于被替换仓位 |
| 日内 T 加仓层 | 0.02 | 隔日不强则去掉加仓层 |

## G3. 单票软顶（硬顶仍为 20%）

- `soft_cap_reduce_band`: 0.12–0.15（进入后优先把加仓层降回底仓带）  
- `soft_cap_force_cut`: 0.15（主线未破也可连底仓下调）  
- `base_band`: 0.08–0.09  

## G4. 风控触发（抽象）

- `hard_stop_from_peak`: 0.20（自持仓期高点回撤）  
- `warn_from_peak`: 0.15  
- `two_day_crash_stop`: 连续 2 日累计跌幅 > 0.20 → 强制退出优先  
- `mid_term_add_only_if_drawdown_ge`: 0.10（中线角色；短线角色禁止亏损加仓）  
- `never_average_down_weak`: true（弱势标的禁止补仓）  
- `never_average_down_short_term_role`: true  

## G5. 资金连续性（主题级，非点名）

- `theme_outflow_days_to_exit`: 3（连续净流出天数阈值，需同时偏弱于基准才清）  
- `theme_dead_weeks_to_abandon`: 4（连续大流出周数 → 放弃该动态主题）  
- `single_day_outflow_not_enough`: true  

## G6. 买入触发权重（形态开关）

值为 `on|off` 或权重 0–1；进化可调，不可改写成点名清单。

- `new_theme_base`: on  
- `theme_diffusion_second_tier`: on  
- `defense_for_vol_control`: on  
- `catchup_relative_laggard`: on  
- `rotate_higher_moat_same_theme`: on  
- `rotate_earnings_verified_same_theme`: on  
- `rotate_capacity_same_theme`: on  
- `post_corp_action_capacity_entry`: on  
- `dip_trial_on_divergence`: on  
- `ma_pullback_add`: on  
- `morning_weakness_entry`: on  
- `chase_extended_breakout`: **off**（默认禁止追高）  

## G7. 卖出触发权重

- `target_reached_scale_out`: on  
- `catalyst_disappointment_cut`: on  
- `overweight_band_reduce`: on  
- `intraday_t_fail_remove_add_layer`: on  
- `fundamental_miss_exit`: on  
- `valuation_cap_exit`: on  
- `theme_flow_exit`: on  
- `weak_vs_theme_peers_exit`: on  
- `risk_stop_exit`: on  

## G8. 时机偏好（抽象）

- `prefer_enter_on_divergence_near_close`: true  
- `prefer_add_on_morning_weakness`: true  
- `no_chase_gap_up_extension`: true  
- `reduce_on_bounce_not_on_cascade`: true  
- `chaos_minimize_trades`: true  
- `min_reward_risk_trial`: 3.0  

## G9. 催化密度（多笔同步动作门槛）

当下列条件在**盘前可见数据**中至少满足 `catalytic_min_hits` 条时，允许「先清背离角色 → 再开主线底仓」的多笔同步；否则默认少动。

- `catalytic_min_hits`: 2  
- `catalytic_signals`（抽象，运行时实例化）：  
  - 动态主线出现广度高潮（多标的大幅上涨/涨停潮类现象）  
  - 动态主线出现可验证超预期催化（业绩/订单/指引）  
  - 持仓中存在与当前主线背离的防御/弱仓需腾位  

## G10. 换手与交易频率软约束

- `max_names_cut_per_day_without_distinct_motives`: 2  
- `prefer_hold_when_thesis_intact`: true  
- `intraday_t_enabled`: true  

## G11. 进化约束（写给突变器）

合法突变示例：

- 调整 G1–G3 数值端点（不破壳硬顶）  
- 开关 G6/G7 某一形态  
- 微调 G4/G5/G8/G9 阈值  
- 响应 `evolution/live_proposals/` 中字段级提案（高优先级）  

非法突变（检查器拒收）：

- 加入任何股票名、代码、细分行业、概念名  
- 修改 `skill.md`  
- 放宽评估成本/成交/ε/walk-forward 窗长 / `w_live`/`w_hist`  
- 引用某盲测段「事后涨跌」作为改参理由  
- 忽略 live 配额（每代至少 3/5 候选须响应 live）而纯历史刷分  

## G12. 评分权重提示（只读，真源在 skill 壳）

- `w_live`: 0.70  
- `w_hist`: 0.30  
- `min_live_trading_days`: 20  
- 日更反思权重大于历史回溯；本文件不得改写上述权重。  

## G13. 版本记录

| version | generation | change |
|---------|------------|--------|
| v0 | 0 | 初始蒸馏；纳入 live 优先进化约束 |
