"""Exploratory analysis on the TRAINING set only.

The test set is split off first and never looked at here, so nothing we learn
during exploration can leak into the final evaluation.

Outputs: four figures in reports/figures/ and reports/eda_insights.md.
"""
import matplotlib.pyplot as plt
import pandas as pd

import config
import plotting
from data import load_train_test


def churn_rate_by(train: pd.DataFrame, column: str) -> pd.DataFrame:
    """Churn rate (%) and customer count for each level of `column`."""
    out = train.groupby(column, observed=True)[config.TARGET].agg(["mean", "size"])
    out["rate_pct"] = out["mean"] * 100
    return out.rename(columns={"size": "customers"})[["rate_pct", "customers"]]


def bar_chart(table: pd.DataFrame, title: str, overall_pct: float, filename: str,
              horizontal: bool = False, xlabel: str = ""):
    """Single-series bar chart of churn rate with a dashed overall-rate reference."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = [str(i) for i in table.index]
    values = table["rate_pct"].to_numpy()
    if horizontal:
        bars = ax.barh(labels, values, color=plotting.BLUE, height=0.55)
        ax.invert_yaxis()
        ax.axvline(overall_pct, color=plotting.GREY, ls="--", lw=1.2)
        ax.text(overall_pct + 0.6, 0.5, f"overall {overall_pct:.1f}%",
                color=plotting.INK_SOFT, fontsize=9, va="center")
        ax.set_xlabel("Churn rate (%)")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(values) * 1.18)
        for bar, v, n in zip(bars, values, table["customers"]):
            ax.text(v + 0.6, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%  (n={n:,})", va="center", fontsize=9.5,
                    # surface-coloured box so the dashed reference line never
                    # runs through a label
                    bbox=dict(facecolor=plotting.SURFACE, edgecolor="none", pad=1.5))
    else:
        bars = ax.bar(labels, values, color=plotting.BLUE, width=0.55)
        ax.axhline(overall_pct, color=plotting.GREY, ls="--", lw=1.2)
        ax.text(len(labels) - 0.5, overall_pct + 0.8, f"overall {overall_pct:.1f}%",
                color=plotting.INK_SOFT, fontsize=9, ha="right")
        ax.set_ylabel("Churn rate (%)")
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(values) * 1.18)
        for bar, v, n in zip(bars, values, table["customers"]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8,
                    f"{v:.1f}%\nn={n:,}", ha="center", fontsize=9.5)
    ax.set_title(title)
    plotting.save(fig, filename)


def main():
    plotting.apply_style()
    X_train, _, y_train, _ = load_train_test()
    train = X_train.assign(**{config.TARGET: y_train})
    overall = train[config.TARGET].mean() * 100
    print(f"[EDA] training rows: {len(train):,}  churn rate: {overall:.2f}%")

    # --- Churn by contract type ---------------------------------------------
    contract = churn_rate_by(train, "Contract").loc[
        ["Month-to-month", "One year", "Two year"]]
    bar_chart(contract, "Churn rate by contract type (training set)", overall,
              "01_churn_by_contract.png")

    # --- Churn by tenure band -----------------------------------------------
    bands = pd.cut(train["tenure"], bins=[-1, 6, 12, 24, 48, 72],
                   labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
    tenure = churn_rate_by(train.assign(tenure_band=bands), "tenure_band")
    bar_chart(tenure, "Churn rate by tenure (training set)", overall,
              "02_churn_by_tenure.png", xlabel="Tenure (months)")

    # --- Churn by payment method --------------------------------------------
    payment = churn_rate_by(train, "PaymentMethod").sort_values("rate_pct",
                                                               ascending=False)
    bar_chart(payment, "Churn rate by payment method (training set)", overall,
              "03_churn_by_payment_method.png", horizontal=True)

    # --- Churn by internet service ------------------------------------------
    internet = churn_rate_by(train, "InternetService").sort_values(
        "rate_pct", ascending=False)
    bar_chart(internet, "Churn rate by internet service (training set)", overall,
              "04_churn_by_internet_service.png")

    # --- Write the insights down, with the numbers computed above -----------
    charges = train.groupby(config.TARGET)["MonthlyCharges"].median()
    m2m_share = (train["Contract"] == "Month-to-month").mean() * 100

    def table_md(t):
        rows = "\n".join(f"| {i} | {r.rate_pct:.1f}% | {int(r.customers):,} |"
                         for i, r in t.iterrows())
        return "| Segment | Churn rate | Customers |\n|---|---|---|\n" + rows

    md = f"""# EDA insights (training set only, {len(train):,} customers)

Overall churn rate: **{overall:.1f}%**. A model that always says "no churn" is
therefore {100 - overall:.1f}% accurate, which is why accuracy is not used as a metric.

## 1. Contract type is the strongest single split
{table_md(contract)}

Month-to-month customers ({m2m_share:.0f}% of the base) churn at
{contract.loc['Month-to-month', 'rate_pct']:.1f}%, versus
{contract.loc['Two year', 'rate_pct']:.1f}% on two-year contracts.

## 2. Churn is concentrated in the first year
{table_md(tenure)}

## 3. Electronic-check payers churn far more than the rest
{table_md(payment)}

## 4. Fiber optic customers churn more than DSL customers
{table_md(internet)}

Median monthly charge is ${charges[1]:.2f} for churners vs ${charges[0]:.2f} for
customers who stayed.

These are associations in one dataset. They show where churn concentrates, not
what causes it.
"""
    path = config.REPORTS_DIR / "eda_insights.md"
    path.write_text(md, encoding="utf-8")
    print(f"  saved {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
