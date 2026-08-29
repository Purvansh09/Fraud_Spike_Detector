"""Generator constants. Anything a judge might ask "why that number?" lives here, not inline."""

from dataclasses import dataclass, field

SEED = 7
START = "2026-06-01"

CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Kochi",
]

CHANNELS = ["card", "upi", "netbanking", "wallet"]
CHANNEL_WEIGHTS = [0.42, 0.38, 0.10, 0.10]

# Everyday categories an account uses without raising an eyebrow.
COMMON_CATEGORIES = [
    "groceries", "food_delivery", "fuel", "utilities",
    "apparel", "pharmacy", "transport", "entertainment",
]
# Liquidation-friendly categories: rare for most accounts, favoured by takeover fraud.
HIGH_RISK_CATEGORIES = ["digital_goods", "gift_cards", "electronics", "crypto"]
ALL_CATEGORIES = COMMON_CATEGORIES + HIGH_RISK_CATEGORIES

MERCHANTS_PER_CATEGORY = 12


@dataclass(frozen=True)
class GenConfig:
    """Every knob in the generator. Defaults are the frozen SPEC.md values."""

    n_accounts: int = 600
    n_days: int = 60
    seed: int = SEED
    start: str = START

    # --- normal behaviour ---
    daily_rate_min: float = 0.15          # Poisson lambda, transactions per account-day
    daily_rate_max: float = 1.40
    amount_log_mu_min: float = 5.9        # exp(5.9) ~ INR 365 median spend
    amount_log_mu_max: float = 8.2        # exp(8.2) ~ INR 3641 median spend
    amount_log_sigma_min: float = 0.55
    amount_log_sigma_max: float = 1.05
    night_owl_share: float = 0.12         # accounts whose *normal* peak hour is 00:00-05:00

    # --- confounders (SPEC.md section 2) ---
    p_legit_session: float = 0.06         # chance an active account-day is a burst shopping session
    legit_session_min: int = 3
    legit_session_max: int = 6
    legit_session_span_min_s: int = 120   # 2 minutes
    legit_session_span_max_s: int = 600   # 10 minutes
    p_new_device_legit: float = 0.03      # device churn; this device PERSISTS afterwards
    p_novel_category: float = 0.04        # legitimate exploration of an unused category
    trips_per_account_max: int = 3        # legitimate travel: people transact away from home
    trip_len_min_days: int = 1
    trip_len_max_days: int = 5

    # --- fraud episodes (SPEC.md section 1) ---
    compromised_share: float = 0.20       # share of accounts that suffer one episode
    burst_min: int = 4
    burst_max: int = 8
    burst_span_min_s: int = 180           # 3 minutes
    burst_span_max_s: int = 720           # 12 minutes
    n_probes_min: int = 1                 # card-testing probes before the ramp
    n_probes_max: int = 2
    probe_amount_min: float = 10.0
    probe_amount_max: float = 150.0
    ramp_factor_min: float = 2.0          # multiples of the account's median spend
    ramp_factor_max: float = 12.0
    p_night_episode: float = 0.60         # 40% deliberately happen in daylight
    p_foreign_city: float = 0.70
    p_high_risk_category: float = 0.80
    earliest_episode_day: int = 3         # leave the account some history to be judged against

    # --- temporal split boundaries, in days from START (SPEC.md section 3) ---
    train_end_day: int = 40               # [0, 40)
    val_end_day: int = 50                 # [40, 50)  -> test is [50, n_days)

    # Columns a real processor has at authorisation time. Nothing else may reach the model.
    feature_side_columns: tuple = (
        "txn_id", "account_id", "timestamp", "amount",
        "merchant_id", "merchant_category", "device_id", "city", "channel",
    )
    label_side_columns: tuple = ("txn_id", "is_fraud", "episode_id")
