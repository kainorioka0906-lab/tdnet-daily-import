from __future__ import annotations

import re
import time
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf
from tqdm import tqdm
import tdnet


SAVE_ROOT = Path("tdnet_pdfs")
SHARES_CACHE = Path("shares_cache.csv")

MARKET_CAP_THRESHOLD_JPY = 500_000_000_000
JST = timezone(timedelta(hours=9))

DOWNLOAD_SLEEP_SEC = 0.5

EXCLUDE_ETF_REIT_BY_TITLE = True
EXCLUDE_ENGLISH_DISCLOSURE = True
EXCLUDE_CORRECTION = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def today_jst_yyyymmdd() -> str:
    return datetime.now(JST).strftime("%Y%m%d")


def safe_filename(s: str, max_len: int = 150) -> str:
    s = re.sub(r'[\\/:*?"<>|]', "_", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len]


def normalize_tse_code(company_code: str | int) -> str | None:
    digits = re.sub(r"\D", "", str(company_code))
    if len(digits) >= 4:
        return digits[:4]
    return None


def is_excluded_title(title: str) -> bool:
    title = str(title)

    if EXCLUDE_ETF_REIT_BY_TITLE:
        etf_reit_keywords = [
            "ETF", "ＥＴＦ", "投資信託", "上場投信",
            "REIT", "リート", "投資法人",
        ]
        if any(k in title for k in etf_reit_keywords):
            return True

    if EXCLUDE_ENGLISH_DISCLOSURE:
        english_keywords = [
            "English", "英文", "英訳", "Summary",
        ]
        if any(k in title for k in english_keywords):
            return True

    if EXCLUDE_CORRECTION:
        correction_keywords = [
            "訂正", "Correction", "一部訂正",
        ]
        if any(k in title for k in correction_keywords):
            return True

    return False


def load_shares_cache() -> dict[str, int]:
    if not SHARES_CACHE.exists():
        return {}

    df = pd.read_csv(SHARES_CACHE, dtype={"code": str})
    return dict(zip(df["code"], df["shares_outstanding"]))


def save_shares_cache(cache: dict[str, int]) -> None:
    df = pd.DataFrame(
        [{"code": k, "shares_outstanding": v} for k, v in sorted(cache.items())]
    )
    df.to_csv(SHARES_CACHE, index=False, encoding="utf-8-sig")


def fetch_latest_close_prices(codes: list[str]) -> dict[str, float]:
    codes = sorted(set(codes))
    tickers = [f"{code}.T" for code in codes]

    if not tickers:
        return {}

    logging.info(f"Downloading close prices for {len(tickers)} tickers...")

    data = yf.download(
        tickers=tickers,
        period="5d",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    close_map: dict[str, float] = {}

    for code in codes:
        ticker = f"{code}.T"

        try:
            if len(codes) == 1:
                close_series = data["Close"].dropna()
            else:
                close_series = data[ticker]["Close"].dropna()

            if len(close_series) > 0:
                close_map[code] = float(close_series.iloc[-1])

        except Exception as e:
            logging.warning(f"Failed to get close price for {code}: {e}")

    return close_map


def fetch_shares_outstanding(code: str, cache: dict[str, int]) -> int | None:
    if code in cache:
        return cache[code]

    ticker = yf.Ticker(f"{code}.T")

    shares = None

    try:
        fast_info = ticker.fast_info
        shares = getattr(fast_info, "shares", None)
    except Exception:
        pass

    if shares is None:
        try:
            info = ticker.info
            shares = info.get("sharesOutstanding")
        except Exception:
            pass

    if shares is None:
        logging.warning(f"Could not get shares outstanding for {code}")
        return None

    shares = int(shares)
    cache[code] = shares
    return shares


def build_large_cap_universe(codes: list[str]) -> pd.DataFrame:
    codes = sorted(set(codes))

    close_map = fetch_latest_close_prices(codes)
    shares_cache = load_shares_cache()

    rows = []

    for code in tqdm(codes, desc="Calculating market cap"):
        close = close_map.get(code)

        if close is None:
            continue

        shares = fetch_shares_outstanding(code, shares_cache)

        if shares is None:
            continue

        market_cap = close * shares

        rows.append(
            {
                "code": code,
                "close": close,
                "shares_outstanding": shares,
                "market_cap_jpy": market_cap,
                "market_cap_billion_jpy": market_cap / 1_000_000_000,
            }
        )

        time.sleep(0.1)

    save_shares_cache(shares_cache)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    return df[df["market_cap_jpy"] >= MARKET_CAP_THRESHOLD_JPY].copy()


def download_tdnet_pdfs(date_yyyymmdd: str | None = None) -> pd.DataFrame:
    if date_yyyymmdd is None:
        date_yyyymmdd = today_jst_yyyymmdd()

    logging.info(f"Fetching TDnet filings for {date_yyyymmdd}...")

    filings = tdnet.documents(date_yyyymmdd)

    if not filings:
        logging.info("No filings found.")
        return pd.DataFrame()

    filing_rows = []
    codes = []

    for f in filings:
        code = normalize_tse_code(getattr(f, "company_code", ""))
        title = getattr(f, "title", "")
        company_name = getattr(f, "company_name", "")
        pubdate = getattr(f, "pubdate", "")
        doc_id = getattr(f, "doc_id", "")

        if code is None:
            continue

        if is_excluded_title(title):
            continue

        codes.append(code)

        filing_rows.append(
            {
                "code": code,
                "company_name": company_name,
                "title": title,
                "pubdate": pubdate,
                "doc_id": doc_id,
                "filing_obj": f,
            }
        )

    if not filing_rows:
        logging.info("No stock filings after filters.")
        return pd.DataFrame()

    large_cap_df = build_large_cap_universe(codes)

    if large_cap_df.empty:
        logging.info("No companies above market cap threshold.")
        return pd.DataFrame()

    large_cap_codes = set(large_cap_df["code"])

    save_dir = SAVE_ROOT / date_yyyymmdd
    save_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []

    for row in tqdm(filing_rows, desc="Downloading PDFs"):
        code = row["code"]

        if code not in large_cap_codes:
            continue

        f = row["filing_obj"]

        company_name = row["company_name"]
        title = row["title"]
        pubdate = row["pubdate"]
        doc_id = row["doc_id"]

        cap_row = large_cap_df[large_cap_df["code"] == code].iloc[0]

        filename = safe_filename(
            f"{pubdate}_{code}_{company_name}_{title}_{doc_id}.pdf"
        )
        filepath = save_dir / filename

        if filepath.exists():
            logging.info(f"Already exists: {filepath.name}")
            continue

        try:
            result = f.fetch_pdf()
            pdf_bytes = result.data
            source_url = getattr(result, "source_url", "")

            with open(filepath, "wb") as out:
                out.write(pdf_bytes)

            metadata_rows.append(
                {
                    "date": date_yyyymmdd,
                    "pubdate": pubdate,
                    "code": code,
                    "company_name": company_name,
                    "title": title,
                    "doc_id": doc_id,
                    "close": cap_row["close"],
                    "shares_outstanding": cap_row["shares_outstanding"],
                    "market_cap_jpy": cap_row["market_cap_jpy"],
                    "market_cap_billion_jpy": cap_row["market_cap_billion_jpy"],
                    "source_url": source_url,
                    "saved_path": str(filepath),
                }
            )

            logging.info(f"Saved: {filepath.name}")

        except Exception as e:
            logging.error(f"Failed to download {code} {title}: {e}")

        time.sleep(DOWNLOAD_SLEEP_SEC)

    meta_df = pd.DataFrame(metadata_rows)

    if not meta_df.empty:
        meta_path = save_dir / f"tdnet_metadata_{date_yyyymmdd}.csv"
        meta_df.to_csv(meta_path, index=False, encoding="utf-8-sig")
        logging.info(f"Metadata saved: {meta_path}")

    logging.info(f"Downloaded {len(meta_df)} PDFs.")

    return meta_df


if __name__ == "__main__":
    download_tdnet_pdfs()
