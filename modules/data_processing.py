import io
import re

import numpy as np
import pandas as pd
import streamlit as st


# RTO tags ("RTO Initiated" / "RTO Delivered") were not consistently applied to
# orders shipped before this month in this dataset (RTO rate is ~0% for every
# month before this, then jumps to 20-50% from this month onward) -- so any
# RTO-rate analysis should exclude orders shipped earlier than this to avoid
# showing falsely perfect "100% delivered" results for older/niche categories.
RTO_TRACKING_START = pd.Period("2024-04", freq="M")


# ---------------------------------------------------------------------------
# File loading & merging
# ---------------------------------------------------------------------------

# Only these columns are used anywhere in the dashboard. Restricting to them
# at load time keeps memory usage low even for large multi-hundred-MB exports.
USEFUL_COLUMNS = {
    "Name", "Financial Status", "Paid at", "Fulfillment Status", "Fulfilled at",
    "Created at", "Cancelled at", "Total", "Subtotal", "Shipping", "Taxes",
    "Discount Code", "Discount Amount", "Refunded Amount", "Payment Method",
    "Accepts Marketing", "Tags", "Vendor", "Source", "Outstanding Balance",
    "Billing Name", "Billing Zip", "Shipping City", "Shipping Zip", "Shipping Province",
    "Lineitem name", "Lineitem quantity", "Lineitem price", "Lineitem sku",
    "Note Attributes",
}

# Matches utm_* params from either `"utm_source":"facebook"` (JSON-style) or
# `utm_source: facebook` (flat key-value) forms found in Note Attributes.
def _utm_re(key: str) -> re.Pattern:
    return re.compile(rf'"{key}"\s*:\s*"([^"]*)"|{key}:\s*([^\n,&|]+)')


_UTM_SOURCE_RE = _utm_re("utm_source")
_UTM_MEDIUM_RE = _utm_re("utm_medium")
_UTM_CAMPAIGN_RE = _utm_re("utm_campaign")


def _extract_utm(note, pattern: re.Pattern):
    if not isinstance(note, str):
        return np.nan
    m = pattern.search(note)
    if not m:
        return np.nan
    val = (m.group(1) or m.group(2) or "").strip()
    return val or np.nan


def _trim_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if c in USEFUL_COLUMNS]
    return df[keep]


def _extract_and_drop_note_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Extract UTM source/medium/campaign from the (PII-heavy, memory-heavy)
    Note Attributes column, then drop the raw column so it isn't retained
    in memory."""
    if "Note Attributes" in df.columns:
        df["UTM Source"] = df["Note Attributes"].apply(_extract_utm, args=(_UTM_SOURCE_RE,))
        df["UTM Medium"] = df["Note Attributes"].apply(_extract_utm, args=(_UTM_MEDIUM_RE,))
        df["UTM Campaign"] = df["Note Attributes"].apply(_extract_utm, args=(_UTM_CAMPAIGN_RE,))
        df = df.drop(columns=["Note Attributes"])
    return df


@st.cache_data(show_spinner=False)
def load_local_files(paths_with_mtime: list[tuple[str, float]]) -> pd.DataFrame:
    """Load files directly from disk by path (cached on path + mtime)."""
    frames = []
    for path, _mtime in paths_with_mtime:
        lower = path.lower()
        if lower.endswith(".csv"):
            frames.append(pd.read_csv(path, low_memory=False, usecols=lambda c: c in USEFUL_COLUMNS))
        elif lower.endswith((".xlsx", ".xls")):
            sheets = pd.read_excel(path, sheet_name=None)
            frames.extend(_trim_columns(df) for df in sheets.values())

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates()
    combined = _extract_and_drop_note_attributes(combined)
    return combined


@st.cache_data(show_spinner=False)
def load_and_merge(file_bundle: list[tuple[str, bytes]]) -> pd.DataFrame:
    """Load any number of uploaded CSV/XLSX files (xlsx = all sheets) and
    concatenate them into a single raw orders dataframe, deduplicated."""
    frames = []
    for name, content in file_bundle:
        lower = name.lower()
        if lower.endswith(".csv"):
            frames.append(pd.read_csv(io.BytesIO(content), low_memory=False, usecols=lambda c: c in USEFUL_COLUMNS))
        elif lower.endswith((".xlsx", ".xls")):
            sheets = pd.read_excel(io.BytesIO(content), sheet_name=None)
            for sheet_df in sheets.values():
                frames.append(_trim_columns(sheet_df))

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates()

    if "Name" in combined.columns:
        combined = combined.drop_duplicates(subset=combined.columns[combined.columns != "Name"].tolist() + ["Name"])

    combined = _extract_and_drop_note_attributes(combined)
    return combined


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _payment_type(pm) -> str:
    pm = str(pm).lower()
    if "cod" in pm or "cash_on_delivery" in pm or "cash on delivery" in pm:
        return "COD"
    return "Prepaid"


def _traffic_type(medium) -> str:
    if pd.isna(medium):
        return "Unspecified"
    m = str(medium).lower()
    if "paid" in m or m == "cpc" or "banner" in m:
        return "Paid"
    if "organic" in m or "instagram" in m or "bio" in m:
        return "Organic Social"
    if "product_sync" in m or "feed" in m:
        return "Catalog / Feed"
    if "automation" in m:
        return "Automation / Retention"
    return "Other"


def _normalize_source_channel(source) -> str:
    """Collapse messy/inconsistent raw `UTM Source` values into a handful of
    marketing channels. The raw values come straight from ad platforms, the
    checkout app (GoKwik), and affiliate/influencer tools and were never
    standardized -- e.g. 'facebook', 'FB', 'fb', 'Meta', 'Meta_fb', 'Meta_ig',
    'META', 'instagarm' all really mean "Meta (Facebook/Instagram) traffic".
    """
    if pd.isna(source):
        return "Unspecified"
    s = str(source).strip().lower()
    if s in ("", "na", "nan", "{{site_source_name}}"):
        return "Unspecified"
    if s == "direct":
        return "Direct"
    if s == "bio":
        return "Organic Social (Bio Link)"
    if "whatsapp" in s or "whatapp" in s:
        return "WhatsApp"
    if re.search(r"facebook|instagram|instagarm|meta|^fb$|^ig|social media|^sm$|^an$", s):
        return "Meta (Facebook/Instagram)"
    if "google" in s or s == "gpay":
        return "Google"
    if re.search(
        r"uniqoemedia|kwik|adzck|arka media|kosmc|analytics clouds|affnads|affilienet|"
        r"affoy media|clickonik|pautm|bitespeed|coupons|wishlink|mediaxpedia|skyzenads|"
        r"convertway|grabon|couponzguru|inrdeals|xomofomo|popcoins|trustpilot|run machine|"
        r"flexiable|engage_360|checkmate|paytm",
        s,
    ):
        return "Affiliate / Influencer Network"
    if s in ("shopify_email", "pushowl", "judgeme", "copilot.com", "chatgpt.com"):
        return "Email / App Tools"
    # Free-text names with a separator (e.g. "sagar saini", "chinmay.palav_") look
    # like individual influencer/affiliate codes rather than a platform.
    if re.fullmatch(r"[a-z][a-z._]*(?:[\s._][a-z][a-z._]*)+", s):
        return "Affiliate / Influencer Network"
    return "Other / Unclassified"


def _normalize_source_platform(source) -> str:
    """Finer-grained sibling of `_normalize_source_channel`: keeps Facebook and
    Instagram as separate platforms (instead of one 'Meta' bucket) and splits
    'Email / App Tools' into 'Shopify Email' vs other app tools, while still
    merging spelling variants within each platform -- e.g. 'facebook', 'FB',
    'fb', 'Meta_fb', 'FACEBOOK_ADS_BYOB-B5G5' -> 'Facebook'; 'Meta_ig',
    'instagram', 'ig', 'instagarm', 'IGShopping' -> 'Instagram'.
    """
    if pd.isna(source):
        return "Unspecified"
    s = str(source).strip().lower()
    if s in ("", "na", "nan", "{{site_source_name}}"):
        return "Unspecified"
    if s == "direct":
        return "Direct"
    if s == "bio":
        return "Organic Social (Bio Link)"
    if "whatsapp" in s or "whatapp" in s:
        return "WhatsApp"
    if re.search(
        r"uniqoemedia|kwik|adzck|arka media|kosmc|analytics clouds|affnads|affilienet|"
        r"affoy media|clickonik|pautm|bitespeed|coupons|wishlink|mediaxpedia|skyzenads|"
        r"convertway|grabon|couponzguru|inrdeals|xomofomo|popcoins|trustpilot|run machine|"
        r"flexiable|engage_360|checkmate",
        s,
    ):
        return "Affiliate / Influencer Network"
    if "instagram" in s or "instagarm" in s or re.search(r"^ig\b|_ig\b|igshopping", s):
        return "Instagram"
    if "facebook" in s or s == "fb" or "meta_fb" in s:
        return "Facebook"
    if s in ("social media", "sm", "an") or "meta" in s:
        return "Meta - Other / Unspecified"
    if "google" in s or s == "gpay":
        return "Google"
    if s == "paytm":
        return "Affiliate / Influencer Network"
    if s == "shopify_email":
        return "Shopify Email"
    if s in ("pushowl", "judgeme", "copilot.com", "chatgpt.com"):
        return "App Tools"
    if re.fullmatch(r"[a-z][a-z._]*(?:[\s._][a-z][a-z._]*)+", s):
        return "Affiliate / Influencer Network"
    return "Other / Unclassified"


def _categorize_discount(code) -> str:
    if pd.isna(code) or str(code).strip() == "":
        return "No Discount"
    c = str(code).upper()
    if "PREPAID" in c and "AUTOMATIC" in c:
        return "Automatic + Prepaid"
    if "PREPAID" in c:
        return "Prepaid Discount"
    if "RECOVERY" in c:
        return "Recovery Discount"
    if "AUTOMATIC" in c:
        return "Automatic Discount"
    if "SURPRISE" in c or "GIFT" in c:
        return "Surprise Gift"
    return "Coupon Code"


def _get_category(name) -> str:
    m = re.search(r"^(.*?\bEDITION\b)", str(name), re.IGNORECASE)
    if m:
        return m.group(1).strip().upper()
    return str(name).split(" - ")[0].strip().upper()


def _get_pack_size(name):
    m = re.search(r"(\d+)\s*[- ]?\s*PAIRS?", str(name), re.IGNORECASE)
    return int(m.group(1)) if m else np.nan


# ---------------------------------------------------------------------------
# Product taxonomy: "Product Family" (consolidated category) + "Style Tags"
# ---------------------------------------------------------------------------
#
# The raw `Category` (everything before "EDITION", or before " - ") yields
# ~790 distinct values because the same product line is sold under many
# color/pattern variants (e.g. "BLACK- ADDY EDITION", "WHITE- ADDY EDITION",
# "ULTRA NEYONE- BLUE 1 PAIR" are all the same product line as "ADDY EDITION" /
# "ULTRA NEYONE EDITION"). `_get_product_family` strips those leading/trailing
# color descriptors (and a few known spelling variants) so charts group by
# product *line* rather than by color. `_get_style_tags` separately extracts
# cross-cutting style attributes (No-Show, Loafer, Gift, Kids, etc.) that a
# single family can carry, so charts can slice "all No-Show styles" etc.
# regardless of which family they belong to.

_COLOR_WORDS = (
    r"(?:BLACK|WHITE|GREY|GRAY|RED|BLUE|GREEN|YELLOW|ORANGE|PINK|PURPLE|BROWN|BEIGE|CREAM|"
    r"NAVY(?:\s*BLUE)?|SKY\s*BLUE|MID\s*BLUE|LIGHT\s*(?:GREY|GRAY|BLUE|GREEN|BROWN)|"
    r"DARK\s*(?:GREY|GRAY|BLUE|BROWN)|STEEL\s*GREY|MUSTARD|MAROON|OLIVE\s*GREEN|"
    r"CHOCOLATE\s*BROWN|BOMBAY\s*GREY|FULL\s*WHITE|OFF\s*WHITE|CYAN|MULTICOLOR)"
)
_COLOR_COMBO = (
    rf"{_COLOR_WORDS}(?:\s*(?:/|&|,|\sAND\s)\s*{_COLOR_WORDS})*"
    rf"(?:\s*(?:FLAME|FLAMES|STRIPE|EDGE))?"
)
_LEADING_COLOR_RE = re.compile(rf"^{_COLOR_COMBO}\s*[-–]\s*", re.IGNORECASE)
_TRAILING_COLOR_RE = re.compile(
    rf"\s*[-–]?\s*{_COLOR_COMBO}(?:\s+(?:COLOR|COLOUR|\d+\s*PAIRS?))*\s*$", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Catalog-driven Product Family lookup (the "mentor's way")
# ---------------------------------------------------------------------------
#
# `CATALOG_FAMILY_MAP` was generated once from the live Shopify product export
# ("Socks products.xlsx", 508 catalog products). Each key is a normalized
# "core" name -- the catalog product's title with the brand/edition boilerplate
# words (EDITION, SOCKS, COZY, UNISEX, ...), pack-size/box descriptors, and
# color words stripped out -- mapped to a clean, human-readable canonical
# family name taken from that catalog product's title. `_get_product_family`
# tries this catalog lookup first (so families match what's actually listed
# for sale), and only falls back to the older color-stripping regex heuristic
# for line items whose product has since been discontinued/renamed and is no
# longer in the catalog export.
_CATALOG_GENERIC_WORDS = (
    r"SOCKSCARVING|EDITION|SOCKS?|LIMITED|FULL[\s-]?LENGTH|COZY|COMFORT(?:ABLE)?|UNISEX|"
    r"WOMEN'?S?|MEN'?S?|KIDS?|CREW|ANKLE|NO[\s-]?SHOW|WINTER|CUSHION(?:ED)?|EXTRA|PREMIUM|"
    r"ORGANIC|COTTON|TERRY|WOOL(?:EN)?|BAMBOO|FURRY|LOAFER|YOGA|PILATES|STUDIO|"
    r"PACK|BOX|OF|PAIRS?|AND|WITH|THE|A|FOR"
)
_CATALOG_COLOR_WORDS = (
    r"BLACK|WHITE|GREY|GRAY|RED|BLUE|GREEN|YELLOW|ORANGE|PINK|PURPLE|BROWN|BEIGE|CREAM|"
    r"NAVY|SKY|MID|LIGHT|DARK|STEEL|MUSTARD|MAROON|OLIVE|CHOCOLATE|BOMBAY|FULL|OFF|CYAN|"
    r"MULTICOLOR|MULTI|NEON|LIME|ROYAL|RUSTIC|FLAME|FLAMES|STRIPE|EDGE|ANY"
)
_CATALOG_PAREN_RE = re.compile(r"\(.*?\)")
_CATALOG_DECIMAL_RE = re.compile(r"(\d+)\.(\d+)")
_CATALOG_NONALNUM_RE = re.compile(r"[^A-Z0-9_]+")
_CATALOG_COLOR_RE = re.compile(rf"\b(?:{_CATALOG_COLOR_WORDS})\b")
_CATALOG_GENERIC_RE = re.compile(rf"\b(?:{_CATALOG_GENERIC_WORDS})\b")
_CATALOG_WS_RE = re.compile(r"\s+")


def _catalog_core_key(s) -> str:
    """Normalize a product name to its distinguishing "core" words for
    matching against `CATALOG_FAMILY_MAP` (strips boilerplate/color words,
    punctuation, and pack-size descriptors)."""
    s = str(s).upper()
    s = _CATALOG_PAREN_RE.sub(" ", s)
    s = _CATALOG_DECIMAL_RE.sub(r"\1_\2", s)
    s = _CATALOG_NONALNUM_RE.sub(" ", s)
    s = _CATALOG_COLOR_RE.sub(" ", s)
    s = _CATALOG_GENERIC_RE.sub(" ", s)
    s = s.replace("_", ".")
    return _CATALOG_WS_RE.sub(" ", s).strip()


CATALOG_FAMILY_MAP = {
    'ADDY 2.0': 'Addy 2.0 Edition',
    'ADDY 2.0 ULTRABREEZE': 'Addy 2.0 Ultrabreeze Edition',
    'ADDY CLOUDTOUCH': 'Addy Edition',
    'ADDY EVERYDAY': 'Addy Edition',
    'AEROGRIP': 'Aerogrip Edition',
    'AFFAIR': 'Socks Affair Furry Edition',
    'AIROFIT': 'Airofit Cushioned Ankle Socks',
    'AIRSKIN': 'Airskin Edition',
    'ALPHA 2.0': 'Alpha 2.0 Edition',
    'ALPHA 2.0 COMBED SPANDEX': 'Alpha 2.0 Edition',
    'ANIMAL': 'The Animal Socks',
    'ARGILE': 'Argile Edition',
    'ARISTO': 'Aristo Edition',
    'ATHLETICA': 'Athletica Edition',
    'AVIK': 'Avik Edition',
    'AVOCADO': 'Avocado Socks',
    'AWNING': 'Awning Edition',
    'AXIS': 'Axis Edition',
    'BANG POP LENGTH': 'Bang Pop Full-Length Edition',
    'BESPOKE': 'Bespoke Edition',
    'BOLTFLEX': 'Boltflex Edition',
    'BOLTFLEX COURT': 'Boltflex Court Socks',
    'BOMBOJA BOMBASTIC': 'Bomboja',
    'BOMBOJA KALIII PEELIII': 'Bomboja',
    'BOMBOJA LOCAL': 'Bomboja - Local Edition',
    'BOMBOJA MUMBAI MERI JAAN': 'Bomboja',
    'BOMBOJA MUMBAI SEH HU BOSS': 'Bomboja',
    'BOND CHAOS CHARACTER': 'Bond Of Chaos- Character Edition',
    'BREEZEGRIP': 'Breezegrip Edition',
    'CADEN': 'Caden Edition',
    'CHARACTER DESIGNER 5': 'Character Designer Socks',
    'CHICKEN LEGS': 'Chicken Legs Socks',
    'CHRISTMAS TREE': 'Christmas Tree Edition',
    'CHROME': 'Chrome Edition',
    'CLASSIC LINE': 'Bamboo - The Classic Line',
    'CLASSY': 'Classy Loafer Socks',
    'CORPORATE CLASSIC': 'Bamboo - Corporate Classic',
    'CORPORATE CLASSIC LENGTH COLOR': 'Bamboo - Corporate Classic',
    'CRAZY POP LENGTH': 'Crazy Pop Full-Length Edition',
    'CRICKET': 'Cricket Socks',
    'CROCODILE': 'Crocodile Socks',
    'CROSSON': 'Crosson Wool Edition',
    'DERP': 'Derp Edition',
    'DERP LENGTH': 'Derp Cushion Full-Length Edition',
    'DRIZY': 'Drizy Edition',
    'DRIZY 5': 'Drizy Edition',
    'DRUMDARK': 'Drumdark Edition',
    'DRUMDARK 2.0': 'Drumdark 2.0 Edition',
    'ECHO': 'Echo Edition',
    'ELEGANCE': 'Elegance Edition',
    'ELEGANCE ESSENTIALS': 'Bamboo - Elegance Essentials',
    'ELEGANCE LUXURY LENGTH': 'Elegance Luxury Full-Length Edition',
    'ELEGANT CHECK': 'Elegant Check',
    'EMOTIONAL BLACKMAIL DIALOGUE': 'Emotional Blackmail Edition',
    'EVOQUE': 'Evoque Edition',
    'EYE SEE YOU PANDA': 'Eye See You Panda Edition',
    'FESTIVAL HARMONY COUPLE': 'Festival Harmony Couple Edition',
    'FIRE': 'Fire Socks',
    'FIRE 2.0': 'Fire 2.0 Socks',
    'FIRE 2.0 STATEMENT': 'Fire 2.0 Socks',
    'FIRE 3.0': 'Fire 3.0 Edition',
    'FLASH PERFORMER KNEE LENGTH': 'Flash Performer Knee Length Socks',
    'FLASH POP LENGTH': 'Flash Pop Full-Length Edition',
    'FLORAL PASTEL': 'Floral Pastel Edition',
    'FUN EXPRESSIONS POP': 'Fun Expressions Pop Ankle Edition',
    'GEN Z': 'Gen-Z Socks',
    'GEN Z POP': 'Gen-Z Pop Edition',
    'GIRRAFE': 'Kids Girrafe Edition',
    'GOAL GRABBERS': 'Goal Grabbers Socks',
    'GRIP GYM SPORTS ANTI SLIP SILICON SOLE': 'Grip Socks For Yoga, Gym & Sports',
    'GRIPFLEX': 'Grip-Flex Edition',
    'HAWK EYE': 'Hawk-Eye Edition',
    'HEART': 'Red Heart Furry Edition',
    'HEART GLORY': 'Heart Glory Red Furry Socks',
    'HEARTLINE': 'Heartline Furry Edition',
    'HEARTS': 'Hearts Edition',
    'HEARTSTRINGS': 'Heartstrings Edition',
    'HELLO POP LENGTH': 'Hello Pop Full-Length Edition',
    'HERRINGBONE 5': 'Herringbone Edition',
    'HIDDEN PANDA': 'Hidden Panda Edition',
    'HOLIDAY SMILE': 'Holiday Smile Edition',
    'HOUNDSTOOTH': 'Houndstooth Edition',
    'HYPE ULTRA': 'Hype Edition',
    'IGNITE SERIES': 'Ignite Series',
    'KATTI BATTI DIALOGUE': 'Katti Batti Edition',
    'KHOL DARWAZA CHARACTER': 'Khol Darwaza- Character Edition',
    'KOOL': 'Kool Edition',
    'KOOL LENGTH': 'Kool Comfort Full-Length Edition',
    'LEGACY': 'Legacy Edition',
    'LENGTH': 'Multicolor Cushion Full-Length Wool Edition',
    'LENGTH DRUMDARK': 'Full Length Cotton Crew Socks - Drumdark Edition',
    'LEVEL UP': 'Level-Up Edition',
    'LEVEL UP SPORTS': 'Level-Up Sports Edition',
    'LIGHTNING': 'Lightning Socks',
    'LIGHTNING 2.0 BOLT': 'Lightning 2.0 Bolt Socks',
    'LIGHTNING 2.0 STREET': 'Lightning 2.0 Street Socks',
    'LIGHTNING 2.0 THUNDER': 'Lightning 2.0 Thunder Socks',
    'LIKE 2.0 POP LENGTH': 'Like 2.0 Pop Full-Length Edition',
    'LIKE POP LENGTH': 'Like Pop Full-Length Edition',
    'LOVE KNOTES': 'Love Knotes Furry Edition',
    'LOVE POP LENGTH': 'Love Pop Full-Length Edition',
    'LOW CUT': 'Stripe Low-Cut Edition',
    'LUXE S': "Cozy Luxe Women's Edition",
    'MADE BY DIDI DIALOGUE': 'Made By Didi- Dialogue Socks',
    'MAJESTIC': 'Majestic Edition',
    'MELLOW': 'Mellow Comfort No-Show Edition',
    'MEOW POP LENGTH': 'Meow Pop Full-Length Edition',
    'MICRO ARMED': 'Micro-Armed No-Show Edition',
    'MILD S WARM SOFT 4': "Mild Furry Cozy Cushion Women's Crew Socks",
    'MINIMAL LUV': 'Minimal Luv Edition',
    'MONEY POP LENGTH': 'Money Pop Full-Length Edition',
    'MONOCHROME MASTER': 'Bamboo - Monochrome Master',
    'MUMMY ME NAHI CHOTU DIALOGUE': 'Mummy Me Nahi Chotu- Dialogue Socks',
    'MUSE': 'Muse Edition',
    'MUSIC POP LENGTH': 'Music Pop Full-Length Edition',
    'MY SEAT CHARACTER': 'My Seat Edition',
    'MYSTERY': 'Mystery Pair',
    'NEOS': 'Neos Edition',
    'NEOS LENGTH': 'Neos Cozy Full-Length Winter Edition',
    'NETSHOT': 'Netshot Edition',
    'NEW REINDEER': 'New Reindeer Edition',
    'NEXUS': 'Nexus Edition',
    'NO GIFT NO RAKHI CHARACTER': 'No Gift, No Rakhi- Character Edition',
    'NO SLIP': 'No-Slip Loafer',
    'NOEL': 'Noel Edition',
    'NOMAD SCRIPT': 'Nomad Script Edition',
    'OMG POP LENGTH': 'Omg Pop Full-Length Edition',
    'ONFIELD': 'Onfield Socks',
    'OOPS POP LENGTH': 'Oops Pop Full-Length Edition',
    'PALMORA': 'Palmora Socks',
    'PAWFECT LOVE PANDA': 'Pawfect Love Panda Edition',
    'PEMBREY': 'Pembrey Edition',
    'PERFORMANCE BOOSTER': 'Performance Booster Edition',
    'PERFORMANCE BOOSTER SPORTS': 'Performance Booster Sports Edition',
    'PERSONAL BOUNCER DIALOGUE': 'Personal Bouncer Edition',
    'POPHEART POP LENGTH': 'Popheart Pop Full-Length Edition',
    'PULSE': 'Pulse Edition',
    'PUMPED UP HEART': 'Pumped Up Red Heart Furry Socks',
    'PUMPED UP HEART LENGTH': 'Pumped Up Heart Cozy Full-Length Furry Edition',
    'REINDEER': 'Reindeer Ankle Edition',
    'REINDEER LENGTH': 'Reindeer Cozy Full-Length Winter Edition',
    'REMOTE WAR CHARACTER': 'Remote War- Character Edition',
    'RETRO GLASSES POP LENGTH': 'Retro Glasses Pop Full-Length Edition',
    'RUN MACHINE': 'Run Machine',
    'RUNNER': 'Flame Runner Socks',
    'RUNNERS': 'Flame Runners Edition',
    'S CHRISTMAS SANTA': 'Christmas Santa Limited Edition',
    'S HARMONIC': 'Harmonic Limited Edition',
    'S REIGN': 'Reign Limited Edition',
    'S SNOW MAN': 'Snow-Man Limited Edition',
    'SHARK': 'Shark Socks',
    'SIBLINGS BOND FUN DIALOGUE': 'Siblings Bond Fun Dialogue Ankle Edition',
    'SIBLINGS FOREVER FUN EXPRESSION': 'Siblings Forever Edition',
    'SILICON GRIP SPORTS': 'Silicon Grip Sports Edition',
    'SOLEMATE': 'Solemate Red Furry Socks',
    'SOLEMATE LENGTH': 'Solemate Cozy Full-Length Furry Edition',
    'SPORTS SOLID KNEE LENGTH': 'Sports Solid Knee Length Socks',
    'STRETCH': 'Stretch Edition',
    'SUPER': 'Super Wool Edition',
    'SUPER SPORTY': 'Super Sporty Edition',
    'SUPER SPORTY 2.0': 'Super Sporty 2.0 - Limited Edition',
    'SUPER SPORTY 5': 'Super Sporty Cushioned Ankle Socks',
    'SWEETHEART': 'Sweetheart Edition',
    'TIKHI MIRCHI FUN EXPRESSION': 'Tikhi Mirchi- Fun Expression',
    'TIMELESS TOES': 'Bamboo - Timeless Toes',
    'TRAGIC': 'Tragic Edition',
    'TRAGIC LENGTH': 'Tragic Cushion Full-Length Edition',
    'TWOSTRIP': 'Two-Stripe Edition',
    'TWOSTRIPE': 'Two-Stripe Edition',
    'ULTIMATE ROAST FUN EXPRESSION': 'Ultimate Roast Socks- Fun Expression',
    'ULTRA NEYONE': 'Ultra Neyone Edition',
    'URBAN SHOE': 'Urban Shoe Socks',
    'VIBRANT': 'Vibrant',
    'VIBRANT 180 AIRFLOW DESIGN': 'Vibrant Edition',
    'WAKE UP FUNKY POP': 'Wake Up Funky Pop Edition',
    'WANDERSOCKS': 'Wandersocks Limited Edition',
    'WANDERSOCKS DESERT': 'Wandersocks Socks- Desert',
    'WANDERSOCKS HIGHWAY': 'Wandersocks Socks- Highway',
    'WANDERSOCKS MOUNTAIN': 'Wandersocks Socks- Mountain',
    'WANDERSOCKS SNOWY MOUNTAIN': 'Wandersocks Socks- Snowy Mountain',
    'WEEKLY': 'Weekly Edition',
    'WHAT POP LENGTH': 'What Pop Full-Length Edition',
    'WINNER POP LENGTH': 'Winner Comfort Pop Full-Length Edition',
    'WTF POP LENGTH': 'Wtf Pop Full-Length Edition',
    'ZACK': 'Zack Edition',
    'ZIGZAG': 'Zigzag Edition',
}

# Sorted longest-first so e.g. "ADDY 2.0 ULTRABREEZE" is tried before "ADDY 2.0".
_CATALOG_KEYS_BY_LEN = sorted(CATALOG_FAMILY_MAP.keys(), key=len, reverse=True)
_CATALOG_KEY_RES = {k: re.compile(rf"\b{re.escape(k)}\b") for k in _CATALOG_KEYS_BY_LEN}


def _match_catalog_family(category):
    """Look up `category`'s product family in the live Shopify catalog
    (`CATALOG_FAMILY_MAP`). Returns None if no catalog product matches, so the
    caller can fall back to the regex-based heuristic for older/discontinued
    products no longer in the catalog export."""
    ck = _catalog_core_key(category)
    if not ck:
        return None
    if ck in CATALOG_FAMILY_MAP:
        return CATALOG_FAMILY_MAP[ck]
    for catk in _CATALOG_KEYS_BY_LEN:
        if _CATALOG_KEY_RES[catk].search(ck):
            return CATALOG_FAMILY_MAP[catk]
    return None


def _get_product_family(category) -> str:
    """Collapse color/pattern variants of a `Category` into one product line,
    preferring the live Shopify catalog's product names (see
    `_match_catalog_family`) and falling back to a regex heuristic for
    products no longer in the catalog."""
    catalog_match = _match_catalog_family(category)
    if catalog_match:
        return catalog_match

    cat = str(category).strip().upper()

    # Spelling/spacing variants of the same word -> normalize before grouping.
    cat = re.sub(r"ADDY\s*2\.0", "ADDY 2.0", cat)
    cat = re.sub(r"NO[\s-]*SHOW", "NO-SHOW", cat)
    cat = re.sub(r"GRIP[\s-]*FLEX", "GRIP-FLEX", cat)
    cat = re.sub(r"ZONE[\s-]*UP", "ZONE-UP", cat)
    cat = re.sub(r"TWO[\s-]*STRIPE?S?\b", "TWO-STRIPE", cat)
    cat = re.sub(r"VIBERANT", "VIBRANT", cat)
    cat = re.sub(r"MICRO[\s-]*ARMED", "MICRO-ARMED", cat)
    cat = re.sub(r"\bLIGHTENING\b", "LIGHTNING", cat)

    # Bamboo line: group by sub-collection (e.g. "ELEGANCE ESSENTIALS"), not color.
    if "BAMBOO" in cat:
        after = re.split(r"BAMBOO", cat, maxsplit=1)[1]
        after = re.sub(r"^[\s\-–&]+", "", after)
        after = re.sub(r"\b(CREW|ANKLE|HIGH ANKLE)\s+LENGTH.*$", "", after)
        after = re.sub(r"\bEDITION\b.*$", "", after)
        after = re.sub(r"^(SOLID|EDGE COLOR)\s*[-–]?\s*", "", after).strip(" -–")
        return f"BAMBOO - {after}" if after else "BAMBOO"

    if re.match(r"^SOLID\s*COLOR\b", cat):
        return "SOLID COLOR"

    if re.match(r"^ADDY 2\.0 ULTRABREEZE", cat):
        return "ADDY 2.0 ULTRABREEZE EDITION"

    stripped = _LEADING_COLOR_RE.sub("", cat)
    if stripped and stripped != cat:
        cat = stripped.strip()

    stripped = _TRAILING_COLOR_RE.sub("", cat)
    if stripped and stripped != cat and len(stripped.strip()) >= 3:
        cat = stripped.strip()

    return cat.strip(" -–") or str(category).strip().upper()


# Cross-cutting style tags: a product can carry more than one. Order matters
# only for display; matching is independent per tag.
STYLE_TAG_RULES = {
    "No-Show": r"NO[\s-]*SHOW",
    "Ankle": r"ANKLE",
    "Loafer": r"LOAFER",
    "Crew Length": r"CREW",
    "Bamboo": r"BAMBOO",
    "Organic Cotton": r"ORGANIC",
    "Limited Edition": r"LIMITED",
    "Gift / Combo": r"GIFT|COMBO|SURPRISE|MYSTERY|BOX OF",
    "Solid Color": r"SOLID\s*COLOR|SOLIDCOLOR",
    "Kids": r"KIDS|KID'S",
    "Women's": r"WOMEN",
    "Grip / Sports": r"GRIP|SPORT|YOGA|PILATES",
    "Woolen": r"WOOL",
    "Multicolor": r"MULTI[\s-]*COLOR",
    "Furry": r"FURRY",
}


def _get_style_tags(name) -> str:
    """Return a comma-separated list of style tags matched in a Lineitem name."""
    text = str(name).upper()
    tags = [tag for tag, pattern in STYLE_TAG_RULES.items() if re.search(pattern, text)]
    return ", ".join(tags)


# ---------------------------------------------------------------------------
# Core derivation pipeline
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def build_datasets(raw: pd.DataFrame) -> dict:
    """Given the merged raw orders dataframe, derive every dataset used
    across the dashboard's analytics tabs."""

    if raw.empty:
        return {"orders": pd.DataFrame(), "paid": pd.DataFrame(),
                "li_all": pd.DataFrame(), "li": pd.DataFrame(),
                "customers": pd.DataFrame()}

    orders = raw.drop_duplicates(subset="Name").copy() if "Name" in raw.columns else raw.copy()

    # --- Dates ---
    # Source timestamps carry a +0530 (IST) offset. Convert to IST (rather than
    # leaving them in UTC) so derived fields like "Order Hour" / "Order DOW"
    # reflect the time the customer actually placed the order, not a time
    # shifted ~5.5 hours earlier (e.g. a 9:30am IST order showing as 4am).
    for col in ["Created at", "Paid at", "Cancelled at", "Fulfilled at"]:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")

    if "Created at" in orders.columns:
        orders = orders.dropna(subset=["Created at"]).copy()
        orders["Order Month"] = orders["Created at"].dt.to_period("M")
        orders["Order Date"] = orders["Created at"].dt.date
        orders["Order Hour"] = orders["Created at"].dt.hour
        orders["Order DOW"] = orders["Created at"].dt.day_name()
        orders["Order Year"] = orders["Created at"].dt.year

    # --- Geography ---
    if "Shipping City" in orders.columns:
        orders["Shipping City"] = orders["Shipping City"].astype(str).str.strip().str.upper()
    if "Shipping Province" in orders.columns:
        orders["Shipping Province"] = orders["Shipping Province"].astype(str).str.strip().str.upper()
    if "Shipping Zip" in orders.columns:
        orders["Shipping Zip"] = orders["Shipping Zip"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    # --- RTO / delivery proxies ---
    if "Tags" in orders.columns:
        tags = orders["Tags"].astype(str)
        orders["RTO Initiated"] = tags.str.contains("RTO Initiated", case=False, na=False)
        orders["RTO Delivered"] = tags.str.contains("RTO Delivered", case=False, na=False)
        orders["Any RTO"] = orders["RTO Initiated"] | orders["RTO Delivered"]
    else:
        orders["Any RTO"] = False

    if "Fulfillment Status" in orders.columns:
        orders["Is Shipped"] = orders["Fulfillment Status"] == "fulfilled"
    else:
        orders["Is Shipped"] = False
    orders["Is Delivered"] = orders["Is Shipped"] & ~orders["Any RTO"]

    # --- Cancellation ---
    if "Cancelled at" in orders.columns:
        orders["Is Cancelled"] = orders["Cancelled at"].notna()
    else:
        orders["Is Cancelled"] = False

    # --- Payment / discounts ---
    if "Payment Method" in orders.columns:
        orders["Payment Type"] = orders["Payment Method"].apply(_payment_type)
    else:
        orders["Payment Type"] = "Unknown"

    if "UTM Medium" in orders.columns:
        orders["Traffic Type"] = orders["UTM Medium"].apply(_traffic_type)

    if "UTM Source" in orders.columns:
        orders["Source Channel"] = orders["UTM Source"].apply(_normalize_source_channel)
        orders["Source Platform"] = orders["UTM Source"].apply(_normalize_source_platform)

    if "Discount Code" in orders.columns and "Discount Amount" in orders.columns:
        orders["Discount Category"] = orders["Discount Code"].apply(_categorize_discount)
        orders["Has Discount"] = orders["Discount Amount"].fillna(0) > 0
    else:
        orders["Discount Category"] = "Unknown"
        orders["Has Discount"] = False

    # --- Refunds / net revenue ---
    if "Refunded Amount" in orders.columns:
        orders["Refunded Amount"] = orders["Refunded Amount"].fillna(0)
    else:
        orders["Refunded Amount"] = 0
    if "Total" in orders.columns:
        orders["Net Revenue"] = orders["Total"] - orders["Refunded Amount"]

    # --- Customer key (Billing Name + Zip) ---
    if "Billing Name" in orders.columns and "Billing Zip" in orders.columns:
        orders["Customer Key"] = (
            orders["Billing Name"].astype(str).str.strip().str.lower()
            + " | " + orders["Billing Zip"].astype(str).str.strip()
        )

    # --- Paid orders subset (date-anchored on Paid at) ---
    paid = pd.DataFrame()
    if "Financial Status" in orders.columns and "Paid at" in orders.columns:
        paid = orders[orders["Financial Status"] == "paid"].dropna(subset=["Paid at"]).copy()
        paid["Paid Month"] = paid["Paid at"].dt.to_period("M")
        if "Customer Key" in paid.columns:
            paid = paid.sort_values("Paid at")
            paid["Order Seq"] = paid.groupby("Customer Key").cumcount() + 1
            paid["Is New Customer Order"] = paid["Order Seq"] == 1

    # --- Line items ---
    li_all, li = pd.DataFrame(), pd.DataFrame()
    needed = {"Name", "Lineitem name", "Lineitem quantity", "Lineitem price"}
    if needed.issubset(raw.columns):
        li_all = raw[raw["Lineitem name"].notna()].copy()
        li_all["Revenue"] = li_all["Lineitem quantity"] * li_all["Lineitem price"]
        li_all["Category"] = li_all["Lineitem name"].apply(_get_category)
        li_all["Pack Size"] = li_all["Lineitem name"].apply(_get_pack_size)

        # "Pack of 1-Pair" SKUs are often sold as part of a multi-buy offer
        # (e.g. "Buy 5 / Buy 6" promos), where the customer adds N separate
        # 1-pair line items to the same order rather than buying one N-pair
        # SKU. Treat the order-level total quantity of 1-pair items as the
        # "Effective Pack Size" so bundle-size analysis reflects what the
        # customer actually bought, not just the individual SKU.
        ones_per_order = (
            li_all.loc[li_all["Pack Size"] == 1].groupby("Name")["Lineitem quantity"].sum()
        )
        li_all["Effective Pack Size"] = li_all["Pack Size"]
        one_mask = li_all["Pack Size"] == 1
        li_all.loc[one_mask, "Effective Pack Size"] = li_all.loc[one_mask, "Name"].map(ones_per_order)

        li_all["Product Family"] = li_all["Category"].apply(_get_product_family)
        li_all["Style Tags"] = li_all["Lineitem name"].apply(_get_style_tags)

        merge_cols = ["Name"]
        for c in ["Shipping City", "Shipping Province", "Shipping Zip", "Order Month", "Is Shipped", "Any RTO",
                  "Financial Status", "Created at", "Paid at"]:
            if c in orders.columns:
                merge_cols.append(c)
        drop_cols = [c for c in merge_cols if c != "Name" and c in li_all.columns]
        li_all = li_all.drop(columns=drop_cols).merge(orders[merge_cols], on="Name", how="left")

        if "Financial Status" in li_all.columns:
            li = li_all[li_all["Financial Status"] == "paid"].copy()
        else:
            li = li_all.copy()

    # --- Customer summary / tiers ---
    customers = pd.DataFrame()
    if not paid.empty and "Customer Key" in paid.columns and "Total" in paid.columns:
        customers = paid.groupby("Customer Key").agg(
            Orders=("Name", "nunique"),
            Total_Spend=("Total", "sum"),
            First_Order=("Paid at", "min"),
            Last_Order=("Paid at", "max"),
        ).reset_index()
        customers["AOV"] = customers["Total_Spend"] / customers["Orders"]
        customers["Repeat_Buyer"] = customers["Orders"] >= 2
        try:
            customers["Tier"] = pd.qcut(
                customers["Total_Spend"], q=[0, 0.5, 0.8, 0.95, 1.0],
                labels=["Bronze", "Silver", "Gold", "Platinum"], duplicates="drop"
            )
        except ValueError:
            customers["Tier"] = "Bronze"

    return {"orders": orders, "paid": paid, "li_all": li_all, "li": li, "customers": customers}


def build_overview_metrics(data: dict) -> dict:
    """Compact JSON-safe metrics dict, used as input to the Claude summary."""
    orders, paid, li, customers = data["orders"], data["paid"], data["li"], data["customers"]
    metrics: dict = {}

    if not orders.empty:
        metrics["total_orders"] = int(len(orders))
        if "Is Cancelled" in orders.columns:
            metrics["cancellation_rate_pct"] = round(orders["Is Cancelled"].mean() * 100, 2)
        if "Payment Type" in orders.columns:
            metrics["cod_share_pct"] = round((orders["Payment Type"] == "COD").mean() * 100, 2)
        if "Is Shipped" in orders.columns:
            shipped = orders[orders["Is Shipped"]]
            if not shipped.empty:
                metrics["delivery_rate_pct"] = round(shipped["Is Delivered"].mean() * 100, 2)
                metrics["rto_rate_pct"] = round(shipped["Any RTO"].mean() * 100, 2)
        if "Shipping City" in orders.columns:
            metrics["top_cities_by_orders"] = orders["Shipping City"].value_counts().head(5).to_dict()
        if "Shipping Province" in orders.columns and "Is Shipped" in orders.columns:
            shipped = orders[orders["Is Shipped"]]
            state_rto = shipped.groupby("Shipping Province")["Any RTO"].agg(["mean", "count"])
            state_rto = state_rto[state_rto["count"] >= 30]
            if not state_rto.empty:
                worst = state_rto["mean"].sort_values(ascending=False).head(3) * 100
                metrics["highest_rto_states_pct"] = worst.round(2).to_dict()

    if not paid.empty:
        metrics["paid_orders"] = int(len(paid))
        if "Total" in paid.columns:
            metrics["total_revenue"] = round(float(paid["Total"].sum()), 2)
            metrics["avg_order_value"] = round(float(paid["Total"].mean()), 2)
        if "Discount Amount" in paid.columns and "Subtotal" in paid.columns and paid["Subtotal"].sum() > 0:
            metrics["discount_pct_of_subtotal"] = round(paid["Discount Amount"].sum() / paid["Subtotal"].sum() * 100, 2)

    if not li.empty:
        top_cat = li.groupby("Category")["Revenue"].sum().sort_values(ascending=False).head(5)
        metrics["top_categories_by_revenue"] = {k: round(v, 2) for k, v in top_cat.items()}

    if not customers.empty:
        metrics["unique_customers"] = int(customers["Customer Key"].nunique())
        metrics["repeat_customer_rate_pct"] = round(customers["Repeat_Buyer"].mean() * 100, 2)
        total_spend = customers["Total_Spend"].sum()
        if total_spend > 0:
            repeat_spend = customers.loc[customers["Repeat_Buyer"], "Total_Spend"].sum()
            metrics["revenue_share_from_repeat_pct"] = round(repeat_spend / total_spend * 100, 2)

    return metrics
