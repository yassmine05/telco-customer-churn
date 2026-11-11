# EDA insights (training set only, 5,634 customers)

Overall churn rate: **26.5%**. A model that always says "no churn" is
therefore 73.5% accurate, which is why accuracy is not used as a metric.

## 1. Contract type is the strongest single split
| Segment | Churn rate | Customers |
|---|---|---|
| Month-to-month | 42.7% | 3,102 |
| One year | 11.1% | 1,173 |
| Two year | 2.9% | 1,359 |

Month-to-month customers (55% of the base) churn at
42.7%, versus
2.9% on two-year contracts.

## 2. Churn is concentrated in the first year
| Segment | Churn rate | Customers |
|---|---|---|
| 0-6 | 52.2% | 1,170 |
| 7-12 | 37.2% | 567 |
| 13-24 | 28.6% | 818 |
| 25-48 | 20.2% | 1,290 |
| 49-72 | 9.9% | 1,789 |

## 3. Electronic-check payers churn far more than the rest
| Segment | Churn rate | Customers |
|---|---|---|
| Electronic check | 45.7% | 1,891 |
| Mailed check | 19.3% | 1,286 |
| Bank transfer (automatic) | 16.2% | 1,244 |
| Credit card (automatic) | 14.9% | 1,213 |

## 4. Fiber optic customers churn more than DSL customers
| Segment | Churn rate | Customers |
|---|---|---|
| Fiber optic | 42.1% | 2,483 |
| DSL | 18.7% | 1,937 |
| No | 7.2% | 1,214 |

Median monthly charge is $79.95 for churners vs $64.40 for
customers who stayed.

These are associations in one dataset. They show where churn concentrates, not
what causes it.
