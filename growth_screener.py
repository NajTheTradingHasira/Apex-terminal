"""
growth_screener.py — CANSLIM Growth Screener
═══════════════════════════════════════════════
Screens US index constituents for high-growth stocks using yfinance.

Modes:
  2020  →  ≥20% revenue growth AND ≥20% EPS growth (QoQ or YoY)
  4040  →  ≥40% revenue growth AND ≥40% EPS growth (QoQ or YoY)

Usage:
  python growth_screener.py --mode 2020
  python growth_screener.py --mode 4040
  python growth_screener.py --mode 2020 --slack --config channel_config.json
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install yfinance pandas")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# INDEX CONSTITUENTS
# ═══════════════════════════════════════════════════════════════

def _fetch_wiki_table(url: str, match: str) -> list[str]:
    """Pull ticker list from a Wikipedia table."""
    try:
        tables = pd.read_html(url, match=match)
        if not tables:
            return []
        df = tables[0]
        # Find the column that looks like tickers
        for col in ["Symbol", "Ticker", "Ticker symbol"]:
            if col in df.columns:
                tickers = df[col].astype(str).str.strip().str.replace(".", "-", regex=False).tolist()
                return [t for t in tickers if t and t != "nan"]
        # Fallback: first column
        tickers = df.iloc[:, 0].astype(str).str.strip().str.replace(".", "-", regex=False).tolist()
        return [t for t in tickers if t and t != "nan"]
    except Exception as e:
        print(f"  ⚠ Failed to fetch from {url}: {e}")
        return []


def get_sp500() -> list[str]:
    """S&P 500 constituents from Wikipedia."""
    print("  Fetching S&P 500...")
    return _fetch_wiki_table(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "Symbol"
    )


def get_nasdaq100() -> list[str]:
    """Nasdaq-100 constituents from Wikipedia."""
    print("  Fetching Nasdaq 100...")
    return _fetch_wiki_table(
        "https://en.wikipedia.org/wiki/Nasdaq-100",
        "Ticker"
    )


def get_dow30() -> list[str]:
    """Dow 30 constituents from Wikipedia."""
    print("  Fetching Dow 30...")
    return _fetch_wiki_table(
        "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
        "Symbol"
    )


def get_russell2000() -> list[str]:
    """Russell 2000 — use IWM holdings as proxy (top ~200) or a static fallback."""
    print("  Fetching Russell 2000 (IWM proxy)...")
    try:
        iwm = yf.Ticker("IWM")
        holdings = iwm.get_holdings()
        if holdings is not None and not holdings.empty:
            # holdings index or 'Symbol' column
            if "Symbol" in holdings.columns:
                return holdings["Symbol"].tolist()
            return holdings.index.tolist()
    except Exception:
        pass
    # Fallback: skip Russell if we can't get holdings
    print("  ⚠ Russell 2000 holdings unavailable — using S&P 500 + Nasdaq 100 + Dow 30 only")
    return []


def build_universe() -> list[str]:
    """Deduplicated universe from all US indexes."""
    sp500 = get_sp500()
    ndx100 = get_nasdaq100()
    dow30 = get_dow30()
    russell = get_russell2000()

    all_tickers = set(sp500 + ndx100 + dow30 + russell)
    # Filter out obvious non-tickers
    all_tickers = {t for t in all_tickers if t.isalpha() or "-" in t or "." in t}
    universe = sorted(all_tickers)
    print(f"\n  Universe: {len(universe)} unique tickers")
    print(f"    S&P 500: {len(sp500)} | Nasdaq 100: {len(ndx100)} | Dow 30: {len(dow30)} | Russell 2000: {len(russell)}")
    return universe


# ═══════════════════════════════════════════════════════════════
# GROWTH ANALYSIS
# ═══════════════════════════════════════════════════════════════

def calc_growth_pct(current, previous) -> float | None:
    """Calculate growth percentage. Returns None if invalid."""
    if current is None or previous is None:
        return None
    if previous == 0:
        return None
    return ((current - previous) / abs(previous)) * 100


def analyze_ticker(ticker: str) -> dict | None:
    """
    Pull quarterly financials from yfinance and compute:
      - Revenue growth (QoQ and YoY)
      - EPS growth (QoQ and YoY)
    Returns a dict with growth metrics or None if data insufficient.
    """
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}

        # Quarterly financials
        q_income = tk.quarterly_income_stmt
        if q_income is None or q_income.empty or q_income.shape[1] < 2:
            return None

        # Sort columns by date descending (most recent first)
        q_income = q_income.sort_index(axis=1, ascending=False)
        dates = q_income.columns.tolist()

        # Revenue (Total Revenue or Operating Revenue)
        rev_row = None
        for label in ["Total Revenue", "Operating Revenue", "Revenue"]:
            if label in q_income.index:
                rev_row = q_income.loc[label]
                break

        # Net Income / EPS
        eps_row = None
        for label in ["Diluted EPS", "Basic EPS"]:
            if label in q_income.index:
                eps_row = q_income.loc[label]
                break

        # If no EPS row, try computing from net income + shares
        ni_row = None
        if eps_row is None:
            for label in ["Net Income", "Net Income Common Stockholders"]:
                if label in q_income.index:
                    ni_row = q_income.loc[label]
                    break

        if rev_row is None and eps_row is None and ni_row is None:
            return None

        result = {
            "ticker": ticker,
            "company": info.get("shortName", info.get("longName", "")),
            "sector": info.get("sector", ""),
            "market_cap": info.get("marketCap", 0),
            "price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
        }

        # ── Revenue growth ──
        if rev_row is not None:
            rev_current = _safe_float(rev_row.iloc[0])
            rev_prev_q = _safe_float(rev_row.iloc[1]) if len(dates) >= 2 else None
            rev_prev_y = _safe_float(rev_row.iloc[4]) if len(dates) >= 5 else None

            result["rev_current"] = rev_current
            result["rev_growth_qoq"] = calc_growth_pct(rev_current, rev_prev_q)
            result["rev_growth_yoy"] = calc_growth_pct(rev_current, rev_prev_y)
            # Best revenue growth (prefer YoY if available)
            result["rev_growth"] = result["rev_growth_yoy"] if result["rev_growth_yoy"] is not None else result["rev_growth_qoq"]
        else:
            result["rev_current"] = None
            result["rev_growth_qoq"] = None
            result["rev_growth_yoy"] = None
            result["rev_growth"] = None

        # ── EPS growth ──
        if eps_row is not None:
            eps_current = _safe_float(eps_row.iloc[0])
            eps_prev_q = _safe_float(eps_row.iloc[1]) if len(dates) >= 2 else None
            eps_prev_y = _safe_float(eps_row.iloc[4]) if len(dates) >= 5 else None

            result["eps_current"] = eps_current
            result["eps_growth_qoq"] = calc_growth_pct(eps_current, eps_prev_q)
            result["eps_growth_yoy"] = calc_growth_pct(eps_current, eps_prev_y)
            result["eps_growth"] = result["eps_growth_yoy"] if result["eps_growth_yoy"] is not None else result["eps_growth_qoq"]
        elif ni_row is not None:
            ni_current = _safe_float(ni_row.iloc[0])
            ni_prev_q = _safe_float(ni_row.iloc[1]) if len(dates) >= 2 else None
            ni_prev_y = _safe_float(ni_row.iloc[4]) if len(dates) >= 5 else None

            result["eps_current"] = ni_current  # Using net income as proxy
            result["eps_growth_qoq"] = calc_growth_pct(ni_current, ni_prev_q)
            result["eps_growth_yoy"] = calc_growth_pct(ni_current, ni_prev_y)
            result["eps_growth"] = result["eps_growth_yoy"] if result["eps_growth_yoy"] is not None else result["eps_growth_qoq"]
        else:
            result["eps_current"] = None
            result["eps_growth_qoq"] = None
            result["eps_growth_yoy"] = None
            result["eps_growth"] = None

        return result

    except Exception:
        return None


def _safe_float(val) -> float | None:
    """Convert a value to float, handling NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        if pd.isna(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def screen(universe: list[str], min_rev: float, min_eps: float) -> list[dict]:
    """Screen universe for stocks meeting growth thresholds."""
    hits = []
    total = len(universe)

    for i, ticker in enumerate(universe, 1):
        pct = (i / total) * 100
        sys.stdout.write(f"\r  Screening: {i}/{total} ({pct:.0f}%) — {ticker:<6}")
        sys.stdout.flush()

        result = analyze_ticker(ticker)
        if result is None:
            continue

        rev_g = result.get("rev_growth")
        eps_g = result.get("eps_growth")

        if rev_g is not None and eps_g is not None:
            if rev_g >= min_rev and eps_g >= min_eps:
                hits.append(result)

        # Rate limit: ~2 requests per ticker, stay under yfinance limits
        if i % 10 == 0:
            time.sleep(0.5)

    sys.stdout.write("\r" + " " * 60 + "\r")
    return hits


# ═══════════════════════════════════════════════════════════════
# OUTPUT — TERMINAL + CSV
# ═══════════════════════════════════════════════════════════════

def fmt_pct(val) -> str:
    if val is None:
        return "—"
    return f"{val:+.1f}%"


def fmt_mcap(val) -> str:
    if not val:
        return "—"
    if val >= 1e12:
        return f"${val/1e12:.1f}T"
    if val >= 1e9:
        return f"${val/1e9:.1f}B"
    if val >= 1e6:
        return f"${val/1e6:.0f}M"
    return f"${val:,.0f}"


def print_table(hits: list[dict], mode: str):
    """Print results as a formatted terminal table."""
    if not hits:
        print(f"\n  No stocks passed the {mode} screen.")
        return

    # Sort by revenue growth descending
    hits.sort(key=lambda x: x.get("rev_growth") or 0, reverse=True)

    header = f"{'Ticker':<7} {'Company':<28} {'Sector':<18} {'Mkt Cap':>9} {'Rev YoY':>9} {'Rev QoQ':>9} {'EPS YoY':>9} {'EPS QoQ':>9}"
    sep = "─" * len(header)

    print(f"\n  ══ {mode} GROWTH SCREEN — {len(hits)} hits ══")
    print(f"  {sep}")
    print(f"  {header}")
    print(f"  {sep}")

    for h in hits:
        line = (
            f"  {h['ticker']:<7} "
            f"{h.get('company','')[:27]:<28} "
            f"{h.get('sector','')[:17]:<18} "
            f"{fmt_mcap(h.get('market_cap')):>9} "
            f"{fmt_pct(h.get('rev_growth_yoy')):>9} "
            f"{fmt_pct(h.get('rev_growth_qoq')):>9} "
            f"{fmt_pct(h.get('eps_growth_yoy')):>9} "
            f"{fmt_pct(h.get('eps_growth_qoq')):>9}"
        )
        print(line)

    print(f"  {sep}")
    print(f"  {len(hits)} stocks with ≥{int(min_from_mode(mode))}% revenue + ≥{int(min_from_mode(mode))}% EPS growth")


def write_csv(hits: list[dict], mode: str) -> str:
    """Write results to a timestamped CSV file."""
    if not hits:
        return ""

    hits.sort(key=lambda x: x.get("rev_growth") or 0, reverse=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"growth_screen_{mode}_{ts}.csv"

    fields = [
        "ticker", "company", "sector", "market_cap", "price",
        "rev_growth_yoy", "rev_growth_qoq", "eps_growth_yoy", "eps_growth_qoq",
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for h in hits:
            row = {k: h.get(k, "") for k in fields}
            # Format percentages for CSV
            for pct_field in ["rev_growth_yoy", "rev_growth_qoq", "eps_growth_yoy", "eps_growth_qoq"]:
                val = row.get(pct_field)
                if val is not None and val != "":
                    row[pct_field] = f"{val:.2f}"
            writer.writerow(row)

    print(f"\n  CSV saved: {filename}")
    return filename


# ═══════════════════════════════════════════════════════════════
# SLACK INTEGRATION
# ═══════════════════════════════════════════════════════════════

def send_to_slack(hits: list[dict], mode: str, config_path: str):
    """Post screen results to Slack channels defined in config."""
    try:
        from slack_sdk import WebClient
        from slack_sdk.errors import SlackApiError
    except ImportError:
        print("\n  ⚠ slack-sdk not installed. Run: pip install slack-sdk")
        return

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("\n  ⚠ SLACK_BOT_TOKEN not set. Export it before using --slack.")
        return

    # Load channel config
    try:
        with open(config_path) as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"\n  ⚠ Failed to load {config_path}: {e}")
        return

    channels = config.get("channels", {})
    routing = config.get("routing", {})

    # Determine target channel
    route_key = routing.get(mode)
    channel_id = channels.get(route_key) if route_key else None
    firehose_id = channels.get("firehose")

    if not channel_id and not firehose_id:
        print(f"\n  ⚠ No channel configured for mode '{mode}'")
        return

    client = WebClient(token=token)

    # Build message
    threshold = int(min_from_mode(mode))
    header = f"*{mode} Growth Screen* — {len(hits)} hits (≥{threshold}% Rev + ≥{threshold}% EPS)"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    if not hits:
        text = f"{header}\n_{ts} — No stocks passed the screen._"
    else:
        hits.sort(key=lambda x: x.get("rev_growth") or 0, reverse=True)
        lines = [f"{header}\n_{ts}_\n"]
        lines.append("```")
        lines.append(f"{'Ticker':<7} {'Rev YoY':>9} {'EPS YoY':>9} {'Mkt Cap':>9}  {'Company'}")
        lines.append("─" * 60)
        for h in hits[:50]:  # Cap at 50 for Slack message limits
            lines.append(
                f"{h['ticker']:<7} "
                f"{fmt_pct(h.get('rev_growth_yoy')):>9} "
                f"{fmt_pct(h.get('eps_growth_yoy')):>9} "
                f"{fmt_mcap(h.get('market_cap')):>9}  "
                f"{h.get('company', '')[:30]}"
            )
        lines.append("```")
        if len(hits) > 50:
            lines.append(f"_...and {len(hits) - 50} more (see CSV)_")
        text = "\n".join(lines)

    # Post to routed channel
    targets = []
    if channel_id:
        targets.append((route_key, channel_id))
    if firehose_id:
        targets.append(("firehose", firehose_id))

    for name, cid in targets:
        try:
            client.chat_postMessage(channel=cid, text=text, mrkdwn=True)
            print(f"  ✓ Posted to #{name} ({cid})")
        except SlackApiError as e:
            print(f"  ✗ Failed to post to #{name}: {e.response['error']}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def min_from_mode(mode: str) -> float:
    if mode == "4040":
        return 40.0
    return 20.0


def main():
    parser = argparse.ArgumentParser(description="CANSLIM Growth Screener")
    parser.add_argument("--mode", choices=["2020", "4040"], required=True,
                        help="Screen mode: 2020 (≥20%/20%) or 4040 (≥40%/40%)")
    parser.add_argument("--slack", action="store_true",
                        help="Post results to Slack")
    parser.add_argument("--config", default="channel_config.json",
                        help="Path to channel_config.json (default: channel_config.json)")
    args = parser.parse_args()

    threshold = min_from_mode(args.mode)
    print(f"\n{'═' * 60}")
    print(f"  APEX Growth Screener — {args.mode} Mode")
    print(f"  Threshold: ≥{threshold:.0f}% Revenue + ≥{threshold:.0f}% EPS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 60}\n")

    # Build universe
    print("  Building universe...")
    universe = build_universe()

    if not universe:
        print("\n  ✗ Failed to build universe. Check your internet connection.")
        sys.exit(1)

    # Screen
    print(f"\n  Screening {len(universe)} tickers (this may take 10-20 minutes)...\n")
    hits = screen(universe, min_rev=threshold, min_eps=threshold)

    # Output — terminal table
    print_table(hits, args.mode)

    # Output — CSV
    csv_file = write_csv(hits, args.mode)

    # Output — Slack
    if args.slack:
        print(f"\n  Sending to Slack...")
        send_to_slack(hits, args.mode, args.config)

    print(f"\n  Done. {len(hits)} stocks passed the {args.mode} screen.\n")


if __name__ == "__main__":
    main()
