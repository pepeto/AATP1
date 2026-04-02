"""
Martingale Strategy for OP/USDT Futures
========================================
Lógica de posición:
  - Capital base configurable
  - Tras cada PÉRDIDA: capital *= 4/3
  - Tras cada GANANCIA: capital vuelve al capital base

Risk management:
  - Stop Loss:   1%  desde precio de entrada
  - Take Profit: 3%  desde precio de entrada

Indicadores:
  - EMA(9) / EMA(21)       → Tendencia de corto plazo
  - RSI(14)                → Momentum (filtro sobrecompra/sobreventa)
  - MACD(12, 26, 9)        → Confirmación de dirección
  - Bollinger Bands(20, 2) → Zonas de reversión / expansión de volatilidad
  - ATR(14)                → Referencia de volatilidad del mercado
"""

import os
import time
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

import ccxt
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Credenciales (se leen de variables de entorno)
    api_key:    str = field(default_factory=lambda: os.getenv("EXCHANGE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("EXCHANGE_API_SECRET", ""))

    # Par y mercado
    symbol:    str = "OP/USDT:USDT"   # futuros perpetuos
    timeframe: str = "15m"
    leverage:  int = 5

    # Capital y martingala
    base_capital: float = 100.0        # USDT por operación inicial
    martingale_factor: float = 4 / 3   # multiplicador tras pérdida

    # Gestión de riesgo
    stop_loss_pct:   float = 0.01      # 1 %
    take_profit_pct: float = 0.03      # 3 %

    # Indicadores
    ema_fast:   int = 9
    ema_slow:   int = 21
    rsi_period: int = 14
    macd_fast:  int = 12
    macd_slow:  int = 26
    macd_sig:   int = 9
    bb_period:  int = 20
    bb_std:     float = 2.0
    atr_period: int = 14

    # RSI umbrales
    rsi_overbought: float = 65.0
    rsi_oversold:   float = 35.0

    # Backtesting
    backtest_limit: int = 500          # velas a descargar para backtest


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("MartingaleOP")


# ──────────────────────────────────────────────────────────────────────────────
# INDICADORES TÉCNICOS
# ──────────────────────────────────────────────────────────────────────────────

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int, slow: int, signal: int):
    ema_fast   = compute_ema(series, fast)
    ema_slow   = compute_ema(series, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(series: pd.Series, period: int, num_std: float):
    sma   = series.rolling(period).mean()
    std   = series.rolling(period).std(ddof=0)
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # EMA
    df["ema_fast"] = compute_ema(close, cfg.ema_fast)
    df["ema_slow"] = compute_ema(close, cfg.ema_slow)

    # RSI
    df["rsi"] = compute_rsi(close, cfg.rsi_period)

    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(
        close, cfg.macd_fast, cfg.macd_slow, cfg.macd_sig
    )

    # Bollinger Bands
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = compute_bollinger_bands(
        close, cfg.bb_period, cfg.bb_std
    )

    # Ancho de banda relativo (%)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100

    # ATR
    df["atr"] = compute_atr(high, low, close, cfg.atr_period)

    return df.dropna()


# ──────────────────────────────────────────────────────────────────────────────
# SEÑALES DE ENTRADA
# ──────────────────────────────────────────────────────────────────────────────

def generate_signal(row: pd.Series, cfg: Config) -> Optional[str]:
    """
    Retorna 'long', 'short' o None.

    LONG  cuando se cumplen 3 de 4 condiciones:
      1. EMA rápida > EMA lenta         (tendencia alcista)
      2. RSI entre 40-65                (momentum alcista, no sobrecomprado)
      3. MACD histograma > 0            (impulso comprador activo)
      4. Precio en mitad inferior de BB (precio < BB_mid)

    SHORT cuando se cumplen 3 de 4 condiciones:
      1. EMA rápida < EMA lenta         (tendencia bajista)
      2. RSI entre 35-60                (momentum bajista, no sobrevendido)
      3. MACD histograma < 0            (impulso vendedor activo)
      4. Precio en mitad superior de BB (precio > BB_mid)

    Filtro adicional: BB Width > 1% (evita mercados sin volatilidad)
    """
    # Filtro de volatilidad mínima
    if row["bb_width"] < 1.0:
        return None

    # Tendencia
    trend_up   = row["ema_fast"] > row["ema_slow"]
    trend_down = row["ema_fast"] < row["ema_slow"]

    # RSI
    rsi_long  = 40.0 <= row["rsi"] <= 65.0
    rsi_short = 35.0 <= row["rsi"] <= 60.0

    # MACD dirección (histograma, no solo cruce)
    macd_bull = row["macd_hist"] > 0
    macd_bear = row["macd_hist"] < 0

    # Bollinger posición
    price_below_mid = row["close"] < row["bb_mid"]
    price_above_mid = row["close"] > row["bb_mid"]

    # Score: necesita 3/4 condiciones para entrar
    long_score  = sum([trend_up, rsi_long, macd_bull, price_below_mid])
    short_score = sum([trend_down, rsi_short, macd_bear, price_above_mid])

    if long_score >= 3:
        return "long"
    if short_score >= 3:
        return "short"
    return None


# ──────────────────────────────────────────────────────────────────────────────
# GESTIÓN DE CAPITAL (MARTINGALA)
# ──────────────────────────────────────────────────────────────────────────────

class MartingaleManager:
    def __init__(self, base_capital: float, factor: float):
        self.base_capital    = base_capital
        self.factor          = factor
        self.current_capital = base_capital
        self.trade_count     = 0
        self.wins            = 0
        self.losses          = 0
        self.total_pnl       = 0.0

    def get_position_size(self) -> float:
        return self.current_capital

    def register_result(self, won: bool, pnl: float):
        self.trade_count += 1
        self.total_pnl   += pnl
        if won:
            self.wins            += 1
            self.current_capital  = self.base_capital      # reset
        else:
            self.losses          += 1
            self.current_capital *= self.factor            # escalar 4/3

    def stats(self) -> dict:
        wr = self.wins / self.trade_count * 100 if self.trade_count else 0
        return {
            "trades":      self.trade_count,
            "wins":        self.wins,
            "losses":      self.losses,
            "win_rate_%":  round(wr, 2),
            "total_pnl":   round(self.total_pnl, 4),
            "current_cap": round(self.current_capital, 4),
        }


# ──────────────────────────────────────────────────────────────────────────────
# BACKTESTING
# ──────────────────────────────────────────────────────────────────────────────

def backtest(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Simula la estrategia sobre datos históricos.
    Devuelve un DataFrame con el registro de cada operación.
    """
    mgr    = MartingaleManager(cfg.base_capital, cfg.martingale_factor)
    trades = []

    # Pre-calcular la columna del histograma anterior
    df = df.copy()
    df["macd_hist_prev"] = df["macd_hist"].shift(1)
    df = df.dropna(subset=["macd_hist_prev"])

    in_trade      = False
    direction     = None
    entry_price   = 0.0
    capital_used  = 0.0
    tp_price      = 0.0
    sl_price      = 0.0
    entry_time    = None

    for idx, row in df.iterrows():
        if not in_trade:
            sig = generate_signal(row, cfg)
            if sig is None:
                continue

            direction    = sig
            entry_price  = row["close"]
            capital_used = mgr.get_position_size()
            entry_time   = row.name

            if direction == "long":
                tp_price = entry_price * (1 + cfg.take_profit_pct)
                sl_price = entry_price * (1 - cfg.stop_loss_pct)
            else:
                tp_price = entry_price * (1 - cfg.take_profit_pct)
                sl_price = entry_price * (1 + cfg.stop_loss_pct)

            in_trade = True

        else:
            # Evaluar TP / SL en la vela actual (usamos high/low)
            hit_tp = hit_sl = False

            if direction == "long":
                if row["high"] >= tp_price:
                    hit_tp = True
                elif row["low"] <= sl_price:
                    hit_sl = True
            else:
                if row["low"] <= tp_price:
                    hit_tp = True
                elif row["high"] >= sl_price:
                    hit_sl = True

            if hit_tp or hit_sl:
                exit_price = tp_price if hit_tp else sl_price
                pct_change = (
                    (exit_price - entry_price) / entry_price
                    if direction == "long"
                    else (entry_price - exit_price) / entry_price
                )
                pnl = capital_used * pct_change * cfg.leverage
                won = hit_tp

                mgr.register_result(won, pnl)
                trades.append({
                    "entry_time":   entry_time,
                    "exit_time":    row.name,
                    "direction":    direction,
                    "entry_price":  entry_price,
                    "exit_price":   exit_price,
                    "capital_used": round(capital_used, 4),
                    "pnl":          round(pnl, 4),
                    "result":       "WIN" if won else "LOSS",
                    "next_capital": round(mgr.current_capital, 4),
                })

                in_trade = False

    results = pd.DataFrame(trades)
    if not results.empty:
        log.info("─── Backtest completado ──────────────────────────")
        for k, v in mgr.stats().items():
            log.info(f"  {k:20s}: {v}")
        log.info("─────────────────────────────────────────────────")
    else:
        log.warning("No se generaron operaciones en el backtest.")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# TRADING EN VIVO
# ──────────────────────────────────────────────────────────────────────────────

class LiveTrader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mgr = MartingaleManager(cfg.base_capital, cfg.martingale_factor)
        self.exchange = self._init_exchange()
        self.in_trade     = False
        self.direction    = None
        self.entry_price  = 0.0
        self.tp_price     = 0.0
        self.sl_price     = 0.0
        self.position_qty = 0.0
        self.order_id     = None

    # ── Inicialización ──────────────────────────────────────────────────────

    def _init_exchange(self) -> ccxt.Exchange:
        exchange = ccxt.binance({
            "apiKey":  self.cfg.api_key,
            "secret":  self.cfg.api_secret,
            "options": {"defaultType": "future"},
            "enableRateLimit": True,
        })
        exchange.set_leverage(self.cfg.leverage, self.cfg.symbol)
        log.info(f"Exchange inicializado: {exchange.id} | leverage={self.cfg.leverage}x")
        return exchange

    # ── Datos de mercado ────────────────────────────────────────────────────

    def fetch_ohlcv(self, limit: int = 100) -> pd.DataFrame:
        ohlcv = self.exchange.fetch_ohlcv(
            self.cfg.symbol, self.cfg.timeframe, limit=limit
        )
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        return df

    # ── Ejecución de órdenes ────────────────────────────────────────────────

    def _qty_from_capital(self, price: float) -> float:
        capital = self.mgr.get_position_size()
        return round((capital * self.cfg.leverage) / price, 4)

    def open_position(self, direction: str, price: float):
        qty  = self._qty_from_capital(price)
        side = "buy" if direction == "long" else "sell"
        try:
            order = self.exchange.create_market_order(self.cfg.symbol, side, qty)
            self.in_trade    = True
            self.direction   = direction
            self.entry_price = order.get("average", price)
            self.position_qty = qty

            if direction == "long":
                self.tp_price = self.entry_price * (1 + self.cfg.take_profit_pct)
                self.sl_price = self.entry_price * (1 - self.cfg.stop_loss_pct)
            else:
                self.tp_price = self.entry_price * (1 - self.cfg.take_profit_pct)
                self.sl_price = self.entry_price * (1 + self.cfg.stop_loss_pct)

            log.info(
                f"OPEN {direction.upper()} | qty={qty} | entry={self.entry_price:.4f}"
                f" | TP={self.tp_price:.4f} | SL={self.sl_price:.4f}"
                f" | capital_usado={self.mgr.get_position_size():.2f} USDT"
            )
        except Exception as e:
            log.error(f"Error abriendo posición: {e}")

    def close_position(self, exit_price: float, won: bool):
        side = "sell" if self.direction == "long" else "buy"
        try:
            self.exchange.create_market_order(
                self.cfg.symbol, side, self.position_qty,
                params={"reduceOnly": True}
            )
            pct   = abs(exit_price - self.entry_price) / self.entry_price
            pnl   = self.mgr.get_position_size() * pct * self.cfg.leverage * (1 if won else -1)
            label = "WIN ✓" if won else "LOSS ✗"

            log.info(
                f"CLOSE {self.direction.upper()} [{label}] | exit={exit_price:.4f}"
                f" | PnL={pnl:+.4f} USDT"
            )
            self.mgr.register_result(won, pnl)
            log.info(f"Próximo capital: {self.mgr.current_capital:.4f} USDT")

            self.in_trade  = False
            self.direction = None
        except Exception as e:
            log.error(f"Error cerrando posición: {e}")

    # ── Loop principal ──────────────────────────────────────────────────────

    def check_exit(self, current_price: float):
        if not self.in_trade:
            return
        won = None
        if self.direction == "long":
            if current_price >= self.tp_price:
                won = True
            elif current_price <= self.sl_price:
                won = False
        else:
            if current_price <= self.tp_price:
                won = True
            elif current_price >= self.sl_price:
                won = False

        if won is not None:
            self.close_position(current_price, won)

    def run(self):
        log.info("▶  Iniciando loop de trading en vivo...")
        sleep_sec = self._timeframe_to_seconds(self.cfg.timeframe)
        prev_hist = None

        while True:
            try:
                df  = self.fetch_ohlcv(limit=60)
                df  = add_indicators(df, self.cfg)
                df["macd_hist_prev"] = df["macd_hist"].shift(1)
                df.dropna(inplace=True)

                last  = df.iloc[-1]
                price = last["close"]

                # Verificar cierre de posición
                self.check_exit(price)

                # Buscar nueva entrada
                if not self.in_trade:
                    sig = generate_signal(last, self.cfg)
                    if sig:
                        self.open_position(sig, price)

                log.info(
                    f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
                    f"price={price:.4f} | RSI={last['rsi']:.1f} | "
                    f"in_trade={self.in_trade} | capital={self.mgr.current_capital:.2f}"
                )

            except KeyboardInterrupt:
                log.info("⏹  Detenido por el usuario.")
                log.info(str(self.mgr.stats()))
                break
            except Exception as e:
                log.error(f"Error en loop: {e}")

            time.sleep(sleep_sec)

    @staticmethod
    def _timeframe_to_seconds(tf: str) -> int:
        unit = tf[-1]
        val  = int(tf[:-1])
        return val * {"m": 60, "h": 3600, "d": 86400}.get(unit, 60)


# ──────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest():
    """Descarga datos y ejecuta el backtesting."""
    cfg = Config()

    # Usar Binance sin credenciales para datos públicos
    exchange = ccxt.binance({"options": {"defaultType": "future"}})
    log.info(f"Descargando {cfg.backtest_limit} velas de {cfg.symbol} ({cfg.timeframe})…")
    ohlcv = exchange.fetch_ohlcv(cfg.symbol, cfg.timeframe, limit=cfg.backtest_limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")

    df = add_indicators(df, cfg)
    results = backtest(df, cfg)

    if not results.empty:
        out = "backtest_results.csv"
        results.to_csv(out)
        log.info(f"Resultados guardados en {out}")

        # Resumen
        wins   = (results["result"] == "WIN").sum()
        losses = (results["result"] == "LOSS").sum()
        total  = len(results)
        pnl    = results["pnl"].sum()
        print("\n" + "=" * 50)
        print(f"  Total operaciones : {total}")
        print(f"  Wins / Losses     : {wins} / {losses}")
        print(f"  Win rate          : {wins/total*100:.1f}%")
        print(f"  PnL total         : {pnl:+.4f} USDT  (leverage {cfg.leverage}x)")
        print("=" * 50)

    return results


def run_live():
    """Inicia el bot de trading en vivo (requiere API keys en env vars)."""
    cfg = Config()
    if not cfg.api_key or not cfg.api_secret:
        raise EnvironmentError(
            "Define EXCHANGE_API_KEY y EXCHANGE_API_SECRET como variables de entorno."
        )
    trader = LiveTrader(cfg)
    trader.run()


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "backtest"

    if mode == "live":
        run_live()
    else:
        run_backtest()
