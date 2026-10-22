"""Build one row of features per labelled subscriber, for two cohorts.

Training cohort (Feb):  users in train.csv, membership expiring in February 2017.
                        Features use data up to 31 Jan 2017.
Out-of-time cohort (Mar): users in train_v2.csv, membership expiring in March 2017.
                        Features use data up to 28 Feb 2017.

Label: is_churn = 1 if the user did not renew within 30 days of expiry.
Each cohort's features stop before its churn window opens, so nothing from
the window leaks into the model.

Transactions: transactions_v2.csv holds rows (some dated before Feb 2017)
that transactions.csv lacks. That file was released for the March cohort,
so merely having rows in it hints at renewing into March. The Feb cohort
therefore uses transactions.csv only. The March cohort is built twice: with
both files (everything known at the cutoff) and with transactions.csv only
(the same sources the model was trained on).

Outputs in data/:
    features.parquet           Feb cohort
    features_oot.parquet       Mar cohort, both transaction files
    features_oot_v1tx.parquet  Mar cohort, transactions.csv only
"""

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def ymd(col):
    return f"strptime(CAST({col} AS VARCHAR), '%Y%m%d')::DATE"


def build(labels_file, cutoff, tx_files, out_file):
    con = duckdb.connect()
    con.execute("SET threads = 8")
    con.execute("SET memory_limit = '10GB'")
    con.execute(f"SET temp_directory = '{ROOT / '.duckdb_tmp'}'")
    cutoff_int = cutoff.replace("-", "")

    con.execute(f"CREATE TABLE labels AS SELECT * FROM '{DATA / labels_file}'")

    # Members: demographics and sign-up channel. bd (age) has junk values, so
    # anything outside 10-80 becomes NULL.
    con.execute(f"""
    CREATE TABLE members AS
    SELECT
        msno,
        city,
        CASE WHEN bd BETWEEN 10 AND 80 THEN bd END            AS age,
        NULLIF(gender, '')                                    AS gender,
        registered_via,
        DATE '{cutoff}' - {ymd('registration_init_time')}     AS account_age_days
    FROM '{DATA / 'members_v3.csv'}'
    WHERE msno IN (SELECT msno FROM labels)
    """)

    # Transactions up to the cutoff.
    tx_source = " UNION ".join(f"SELECT * FROM '{DATA / f}'" for f in tx_files)
    con.execute(f"""
    CREATE TABLE tx AS
    SELECT
        msno,
        payment_method_id,
        payment_plan_days,
        plan_list_price,
        actual_amount_paid,
        is_auto_renew,
        is_cancel,
        {ymd('transaction_date')}        AS tx_date,
        {ymd('membership_expire_date')}  AS expire_date
    FROM ({tx_source})
    WHERE msno IN (SELECT msno FROM labels)
      AND transaction_date <= {cutoff_int}
    """)

    # History across all transactions, plus the state of the most recent one.
    # Same-day transactions with the same expiry need a full tie-break, or the
    # "most recent" row changes between runs. A cancellation counts as the
    # later event on its day.
    con.execute(f"""
    CREATE TABLE tx_feats AS
    WITH ranked AS (
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY msno
            ORDER BY tx_date DESC, expire_date DESC, is_cancel DESC,
                     payment_plan_days DESC, actual_amount_paid DESC,
                     plan_list_price DESC, is_auto_renew DESC, payment_method_id DESC
        ) AS rn
        FROM tx
    ),
    history AS (
        SELECT
            msno,
            COUNT(*)                                   AS n_transactions,
            SUM(is_cancel)                             AS n_cancels,
            DATE '{cutoff}' - MIN(tx_date)             AS subscriber_tenure_days,
            AVG(is_auto_renew)                         AS share_auto_renew
        FROM tx
        GROUP BY msno
    )
    SELECT
        h.*,
        r.payment_plan_days                            AS plan_days,
        r.is_auto_renew                                AS auto_renew,
        r.is_cancel                                    AS last_tx_cancel,
        r.payment_method_id                            AS payment_method,
        r.plan_list_price                              AS list_price,
        r.actual_amount_paid                           AS amount_paid,
        r.plan_list_price - r.actual_amount_paid       AS discount,
        r.expire_date - DATE '{cutoff}'                AS days_to_expiry
    FROM history h
    JOIN ranked r ON r.msno = h.msno AND r.rn = 1
    """)

    # Listening logs: stream the 30 GB CSV once and aggregate per user.
    # total_secs has corrupt rows (negative or huge), so clip to [0, 1 day].
    # "Last 30" = the 30 days ending at the cutoff; "previous 30" = the 30 before.
    # user_logs.csv ends on 28 Feb 2017, which covers both cutoffs.
    con.execute(f"""
    CREATE TABLE log_feats AS
    WITH logs AS (
        SELECT
            msno,
            {ymd('date')}                                        AS d,
            num_25 + num_50 + num_75 + num_985 + num_100         AS plays,
            num_100,
            num_unq,
            LEAST(GREATEST(total_secs, 0), 86400)                AS secs
        FROM '{DATA / 'user_logs.csv'}'
        WHERE msno IN (SELECT msno FROM labels)
          AND date <= {cutoff_int}
    )
    SELECT
        msno,
        COUNT(*)                                                       AS log_rows,
        DATE '{cutoff}' - MAX(d)                                       AS days_since_last_listen,
        COUNT(*)  FILTER (WHERE d >  DATE '{cutoff}' - 30)             AS active_days_30,
        COUNT(*)  FILTER (WHERE d <= DATE '{cutoff}' - 30
                            AND d >  DATE '{cutoff}' - 60)             AS active_days_prev30,
        COALESCE(SUM(secs) FILTER (WHERE d > DATE '{cutoff}' - 30), 0) / 3600.0
                                                                       AS hours_30,
        COALESCE(SUM(secs) FILTER (WHERE d <= DATE '{cutoff}' - 30
                                     AND d >  DATE '{cutoff}' - 60), 0) / 3600.0
                                                                       AS hours_prev30,
        AVG(num_unq) FILTER (WHERE d > DATE '{cutoff}' - 30)           AS avg_unique_songs_30,
        SUM(num_100) FILTER (WHERE d > DATE '{cutoff}' - 30)
            / NULLIF(SUM(plays) FILTER (WHERE d > DATE '{cutoff}' - 30), 0)
                                                                       AS completion_rate_30
    FROM logs
    GROUP BY msno
    """)
    log_rows = con.execute("SELECT SUM(log_rows) FROM log_feats").fetchone()[0]

    df = con.execute("""
    SELECT l.msno, l.is_churn, m.* EXCLUDE (msno), t.* EXCLUDE (msno), g.* EXCLUDE (msno, log_rows)
    FROM labels l
    LEFT JOIN members   m USING (msno)
    LEFT JOIN tx_feats  t USING (msno)
    LEFT JOIN log_feats g USING (msno)
    """).df()

    df["has_logs"] = df["days_since_last_listen"].notna().astype(int)
    df["has_profile"] = df["registered_via"].notna().astype(int)
    for c in ["active_days_30", "active_days_prev30", "hours_30", "hours_prev30"]:
        df[c] = df[c].fillna(0)
    df["hours_change"] = df["hours_30"] - df["hours_prev30"]

    df.to_parquet(DATA / out_file, index=False)

    n_tx = con.execute("SELECT COUNT(*) FROM tx").fetchone()[0]
    print(f"{out_file}")
    print(f"  cohort: {labels_file}, features up to {cutoff}, transactions from {', '.join(tx_files)}")
    print(f"  subscribers:          {len(df):,}")
    print(f"  churn rate:           {df.is_churn.mean():.2%}")
    print(f"  transactions used:    {n_tx:,}")
    print(f"  log rows aggregated:  {log_rows:,}")


if __name__ == "__main__":
    build("train.csv", "2017-01-31", ["transactions.csv"], "features.parquet")
    build("train_v2.csv", "2017-02-28", ["transactions.csv", "transactions_v2.csv"], "features_oot.parquet")
    build("train_v2.csv", "2017-02-28", ["transactions.csv"], "features_oot_v1tx.parquet")
