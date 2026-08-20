"""Documented public systems. Parameters fixed from published recipes, not fitted here.

Sources (rules only; none claimed profitable on our bar):
- BabyPips Cowabunga (EMA 5/10 + RSI 9 + Stoch 10,3,3 + MACD hist) on this 1h clock
- SuperTrend ATR 10x3 (common forex default) and nearby published settings
- Ichimoku TK / Kijun / cloud (cloud at t uses values from t-26; no lookahead)
- Turtle Donchian S1=20 / S2=55
- ADX + DI trend (Wilder)
- CCI, Williams %R, Stochastic fade/cross
- Keltner breakout
- Alligator (Williams SMMA 13/8/5)
- Awesome Oscillator zero-cross
- Connors 2-period RSI
- Admiral/common: MACD zero-cross 1h, EMA 34/55 pullback
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.catalog.helpers import (
    adx_di,
    atr,
    cci,
    close_arr,
    ema,
    heikin_ashi_close,
    prior_rolling_max,
    prior_rolling_min,
    rsi,
    sign_pos,
    sma,
    smma,
    stochastic_k,
    supertrend,
    williams_r,
)
from src.signals.catalog.spec import RuleSpec, register


def _stoch_fade(df: pd.DataFrame, period: int, smooth: int, lo: float, hi: float, **_: object) -> np.ndarray:
    k = stochastic_k(df, period, smooth)
    ok = np.isfinite(k)
    return sign_pos(ok & (k < lo), ok & (k > hi))


def _stoch_cross50(df: pd.DataFrame, period: int = 14, smooth: int = 3, **_: object) -> np.ndarray:
    k = stochastic_k(df, period, smooth)
    ok = np.isfinite(k)
    return sign_pos(ok & (k > 50.0), ok & (k < 50.0))


def _willr_fade(df: pd.DataFrame, period: int, lo: float, hi: float, **_: object) -> np.ndarray:
    w = williams_r(df, period)
    ok = np.isfinite(w)
    return sign_pos(ok & (w < lo), ok & (w > hi))


def _cci_fade(df: pd.DataFrame, period: int, thresh: float, **_: object) -> np.ndarray:
    x = cci(df, period)
    ok = np.isfinite(x)
    t = float(thresh)
    return sign_pos(ok & (x < -t), ok & (x > t))


def _cci_trend(df: pd.DataFrame, period: int, thresh: float, **_: object) -> np.ndarray:
    x = cci(df, period)
    ok = np.isfinite(x)
    t = float(thresh)
    return sign_pos(ok & (x > t), ok & (x < -t))


def _supertrend_pos(df: pd.DataFrame, atr_window: int, mult: float, **_: object) -> np.ndarray:
    return supertrend(df, atr_window, mult)


def _dual_supertrend(
    df: pd.DataFrame,
    fast_n: int,
    fast_m: float,
    slow_n: int,
    slow_m: float,
    **_: object,
) -> np.ndarray:
    a = supertrend(df, fast_n, fast_m)
    b = supertrend(df, slow_n, slow_m)
    pos = np.zeros(len(a))
    pos[(a > 0) & (b > 0)] = 1.0
    pos[(a < 0) & (b < 0)] = -1.0
    return pos


def _ichimoku_tk(df: pd.DataFrame, **_: object) -> np.ndarray:
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    tenkan = (pd.Series(high).rolling(9, min_periods=9).max() + pd.Series(low).rolling(9, min_periods=9).min()) / 2.0
    kijun = (pd.Series(high).rolling(26, min_periods=26).max() + pd.Series(low).rolling(26, min_periods=26).min()) / 2.0
    t = tenkan.to_numpy()
    k = kijun.to_numpy()
    ok = np.isfinite(t) & np.isfinite(k)
    return sign_pos(ok & (t > k), ok & (t < k))


def _ichimoku_price_kijun(df: pd.DataFrame, **_: object) -> np.ndarray:
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = close_arr(df)
    kijun = (
        pd.Series(high).rolling(26, min_periods=26).max() + pd.Series(low).rolling(26, min_periods=26).min()
    ) / 2.0
    k = kijun.to_numpy()
    ok = np.isfinite(k)
    return sign_pos(ok & (close > k), ok & (close < k))


def _ichimoku_cloud(df: pd.DataFrame, **_: object) -> np.ndarray:
    """Cloud at bar t is Senkou computed 26 bars earlier (standard shift, no future)."""
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = close_arr(df)
    tenkan = (
        pd.Series(high).rolling(9, min_periods=9).max() + pd.Series(low).rolling(9, min_periods=9).min()
    ) / 2.0
    kijun = (
        pd.Series(high).rolling(26, min_periods=26).max() + pd.Series(low).rolling(26, min_periods=26).min()
    ) / 2.0
    senkou_a = ((tenkan + kijun) / 2.0).to_numpy()
    senkou_b = (
        (pd.Series(high).rolling(52, min_periods=52).max() + pd.Series(low).rolling(52, min_periods=52).min()) / 2.0
    ).to_numpy()
    # values formed at t-26 sit at the cloud at t
    sa = np.roll(senkou_a, 26)
    sb = np.roll(senkou_b, 26)
    sa[:26] = np.nan
    sb[:26] = np.nan
    top = np.maximum(sa, sb)
    bot = np.minimum(sa, sb)
    ok = np.isfinite(top) & np.isfinite(bot)
    return sign_pos(ok & (close > top), ok & (close < bot))


def _adx_di_trend(df: pd.DataFrame, period: int, adx_min: float, **_: object) -> np.ndarray:
    adx, pdi, mdi = adx_di(df, period)
    ok = np.isfinite(adx) & np.isfinite(pdi) & np.isfinite(mdi) & (adx >= float(adx_min))
    return sign_pos(ok & (pdi > mdi), ok & (mdi > pdi))


def _keltner_break(df: pd.DataFrame, window: int, mult: float, fade: bool = False, **_: object) -> np.ndarray:
    c = close_arr(df)
    mid = ema(c, window)
    band = float(mult) * atr(df, window)
    up = mid + band
    dn = mid - band
    ok = np.isfinite(up) & np.isfinite(dn)
    if fade:
        return sign_pos(ok & (c < dn), ok & (c > up))
    return sign_pos(ok & (c > up), ok & (c < dn))


def _alligator(df: pd.DataFrame, **_: object) -> np.ndarray:
    c = close_arr(df)
    jaw = smma(c, 13)
    teeth = smma(c, 8)
    lips = smma(c, 5)
    ok = np.isfinite(jaw) & np.isfinite(teeth) & np.isfinite(lips)
    return sign_pos(ok & (lips > teeth) & (teeth > jaw), ok & (lips < teeth) & (teeth < jaw))


def _ao_zero(df: pd.DataFrame, **_: object) -> np.ndarray:
    mid = (df["high"].to_numpy(dtype=np.float64) + df["low"].to_numpy(dtype=np.float64)) / 2.0
    ao = sma(mid, 5) - sma(mid, 34)
    ok = np.isfinite(ao)
    return sign_pos(ok & (ao > 0), ok & (ao < 0))


def _connors_rsi2(df: pd.DataFrame, lo: float, hi: float, **_: object) -> np.ndarray:
    r = rsi(close_arr(df), 2)
    ok = np.isfinite(r)
    return sign_pos(ok & (r < lo), ok & (r > hi))


def _macd_zero(df: pd.DataFrame, fast: int, slow: int, signal: int, **_: object) -> np.ndarray:
    c = close_arr(df)
    line = ema(c, fast) - ema(c, slow)
    sig = ema(line, signal)
    hist = line - sig
    ok = np.isfinite(hist)
    return sign_pos(ok & (hist > 0), ok & (hist < 0))


def _ema_pullback(df: pd.DataFrame, fast: int, slow: int, **_: object) -> np.ndarray:
    c = close_arr(df)
    f = ema(c, fast)
    s = ema(c, slow)
    ok = np.isfinite(f) & np.isfinite(s)
    up = ok & (f > s) & (c <= f) & (c >= s)
    dn = ok & (f < s) & (c >= f) & (c <= s)
    return sign_pos(up, dn)


def _cowabunga_1h(df: pd.DataFrame, **_: object) -> np.ndarray:
    """BabyPips Cowabunga filters on this 1h series (not 15m/4h dual chart)."""
    c = close_arr(df)
    e5 = ema(c, 5)
    e10 = ema(c, 10)
    r = rsi(c, 9)
    k = stochastic_k(df, 10, 3)
    line = ema(c, 12) - ema(c, 26)
    hist = line - ema(line, 9)
    ok = np.isfinite(e5) & np.isfinite(e10) & np.isfinite(r) & np.isfinite(k) & np.isfinite(hist)
    long = ok & (e5 > e10) & (r > 50.0) & (k < 80.0) & (hist > 0)
    short = ok & (e5 < e10) & (r < 50.0) & (k > 20.0) & (hist < 0)
    return sign_pos(long, short)


def _ha_trend(df: pd.DataFrame, **_: object) -> np.ndarray:
    ha_c, ha_o = heikin_ashi_close(df)
    return sign_pos(ha_c > ha_o, ha_c < ha_o)


def _turtle(df: pd.DataFrame, entry: int, **_: object) -> np.ndarray:
    close = close_arr(df)
    hi = prior_rolling_max(df["high"].to_numpy(dtype=np.float64), entry)
    lo = prior_rolling_min(df["low"].to_numpy(dtype=np.float64), entry)
    ok = np.isfinite(hi) & np.isfinite(lo)
    return sign_pos(ok & (close > hi), ok & (close < lo))


def _aroon(df: pd.DataFrame, period: int = 25, **_: object) -> np.ndarray:
    high = pd.Series(df["high"].to_numpy(dtype=np.float64))
    low = pd.Series(df["low"].to_numpy(dtype=np.float64))
    up = 100.0 * high.rolling(int(period), min_periods=int(period)).apply(
        lambda x: float(int(period) - 1 - np.argmax(x)) / float(int(period)), raw=True
    )
    down = 100.0 * low.rolling(int(period), min_periods=int(period)).apply(
        lambda x: float(int(period) - 1 - np.argmin(x)) / float(int(period)), raw=True
    )
    u = up.to_numpy()
    d = down.to_numpy()
    ok = np.isfinite(u) & np.isfinite(d)
    return sign_pos(ok & (u > d), ok & (d > u))


def register_all(hold: int = 12) -> None:
    for p, sm, lo, hi in (
        (14, 3, 20.0, 80.0),
        (14, 3, 30.0, 70.0),
        (10, 3, 20.0, 80.0),
        (21, 3, 20.0, 80.0),
        (5, 3, 20.0, 80.0),
    ):
        register(
            RuleSpec(
                rule_id=f"stoch_{p}_{sm}_{int(lo)}_{int(hi)}_h{hold}",
                family="public",
                hold=hold,
                fn=_stoch_fade,
                kwargs={"period": p, "smooth": sm, "lo": lo, "hi": hi},
                note="Stochastic fade (BabyPips / common)",
            )
        )
    register(
        RuleSpec(
            rule_id=f"stoch14_cross50_h{hold}",
            family="public",
            hold=hold,
            fn=_stoch_cross50,
            kwargs={"period": 14, "smooth": 3},
        )
    )
    for p, lo, hi in ((14, -80.0, -20.0), (14, -70.0, -30.0), (21, -80.0, -20.0)):
        register(
            RuleSpec(
                rule_id=f"willr_{p}_{int(abs(lo))}_{int(abs(hi))}_h{hold}",
                family="public",
                hold=hold,
                fn=_willr_fade,
                kwargs={"period": p, "lo": lo, "hi": hi},
            )
        )
    for p, t, mode in ((14, 100.0, "fade"), (20, 100.0, "fade"), (20, 200.0, "fade"), (14, 100.0, "trend"), (20, 100.0, "trend")):
        fn = _cci_fade if mode == "fade" else _cci_trend
        register(
            RuleSpec(
                rule_id=f"cci_{p}_{int(t)}_{mode}_h{hold}",
                family="public",
                hold=hold,
                fn=fn,
                kwargs={"period": p, "thresh": t},
            )
        )
    for aw, m in ((10, 3.0), (14, 3.0), (7, 3.0), (10, 2.0), (14, 2.0), (10, 3.5), (20, 3.0)):
        tag = str(m).replace(".", "p")
        register(
            RuleSpec(
                rule_id=f"supertrend_{aw}_{tag}_h{hold}",
                family="public",
                hold=hold,
                fn=_supertrend_pos,
                kwargs={"atr_window": aw, "mult": m},
                note="SuperTrend (ATR x multiplier)",
            )
        )
    register(
        RuleSpec(
            rule_id=f"supertrend_dual_10x3_30x9_h{hold}",
            family="public",
            hold=hold,
            fn=_dual_supertrend,
            kwargs={"fast_n": 10, "fast_m": 3.0, "slow_n": 30, "slow_m": 9.0},
            note="Published dual SuperTrend 10/3 + 30/9",
        )
    )
    register(RuleSpec(rule_id=f"ichimoku_tk_h{hold}", family="public", hold=hold, fn=_ichimoku_tk, note="Ichimoku Tenkan/Kijun cross"))
    register(RuleSpec(rule_id=f"ichimoku_price_kijun_h{hold}", family="public", hold=hold, fn=_ichimoku_price_kijun))
    register(RuleSpec(rule_id=f"ichimoku_cloud_h{hold}", family="public", hold=hold, fn=_ichimoku_cloud, note="Price vs lagged cloud"))
    for per, amin in ((14, 20.0), (14, 25.0), (14, 30.0), (20, 20.0), (20, 25.0)):
        register(
            RuleSpec(
                rule_id=f"adx{per}_min{int(amin)}_h{hold}",
                family="public",
                hold=hold,
                fn=_adx_di_trend,
                kwargs={"period": per, "adx_min": amin},
                note="ADX + DI trend",
            )
        )
    for w, m, fade in ((20, 2.0, False), (20, 1.5, False), (20, 2.0, True), (14, 2.0, False), (14, 2.0, True)):
        kind = "fade" if fade else "brk"
        tag = str(m).replace(".", "p")
        register(
            RuleSpec(
                rule_id=f"keltner_{w}_{tag}_{kind}_h{hold}",
                family="public",
                hold=hold,
                fn=_keltner_break,
                kwargs={"window": w, "mult": m, "fade": fade},
            )
        )
    register(RuleSpec(rule_id=f"alligator_h{hold}", family="public", hold=hold, fn=_alligator, note="Williams Alligator"))
    register(RuleSpec(rule_id=f"ao_zero_h{hold}", family="public", hold=hold, fn=_ao_zero, note="Awesome Oscillator zero"))
    for lo, hi in ((10.0, 90.0), (5.0, 95.0), (20.0, 80.0)):
        register(
            RuleSpec(
                rule_id=f"connors_rsi2_{int(lo)}_{int(hi)}_h{hold}",
                family="public",
                hold=hold,
                fn=_connors_rsi2,
                kwargs={"lo": lo, "hi": hi},
                note="Connors 2-period RSI",
            )
        )
    register(
        RuleSpec(
            rule_id=f"macd_zero_12_26_9_h{hold}",
            family="public",
            hold=hold,
            fn=_macd_zero,
            kwargs={"fast": 12, "slow": 26, "signal": 9},
            note="Admiral-style MACD 1h zero-cross",
        )
    )
    register(
        RuleSpec(
            rule_id=f"ema_pb_34_55_h{hold}",
            family="public",
            hold=hold,
            fn=_ema_pullback,
            kwargs={"fast": 34, "slow": 55},
            note="EMA 34/55 pullback (Admiral 4h recipe on 1h)",
        )
    )
    register(
        RuleSpec(
            rule_id=f"ema_pb_8_21_h{hold}",
            family="public",
            hold=hold,
            fn=_ema_pullback,
            kwargs={"fast": 8, "slow": 21},
        )
    )
    register(
        RuleSpec(
            rule_id=f"cowabunga_1h_h{hold}",
            family="public",
            hold=hold,
            fn=_cowabunga_1h,
            note="BabyPips Cowabunga filters on 1h",
        )
    )
    register(RuleSpec(rule_id=f"heikin_ashi_h{hold}", family="public", hold=hold, fn=_ha_trend))
    register(RuleSpec(rule_id=f"turtle_s1_20_h{hold}", family="public", hold=hold, fn=_turtle, kwargs={"entry": 20}, note="Turtle S1 Donchian 20"))
    register(RuleSpec(rule_id=f"turtle_s2_55_h{hold}", family="public", hold=hold, fn=_turtle, kwargs={"entry": 55}, note="Turtle S2 Donchian 55"))
    register(RuleSpec(rule_id=f"aroon_25_h{hold}", family="public", hold=hold, fn=_aroon, kwargs={"period": 25}))
    register(RuleSpec(rule_id=f"aroon_14_h{hold}", family="public", hold=hold, fn=_aroon, kwargs={"period": 14}))
