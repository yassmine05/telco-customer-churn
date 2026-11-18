"""Reproduce the whole project with one command:  python src/run_all.py

Order matters: the test set is only read in the last three steps, after every
modelling decision has been frozen by train.py. The last step evaluates uplift
modelling on a SIMULATED retention campaign (about 5 seconds).
"""
import eda
import evaluate
import explain
import train
from uplift import evaluate as uplift_evaluate

if __name__ == "__main__":
    for step in (eda, train, evaluate, explain, uplift_evaluate):
        print(f"\n=== {step.__name__} ===")
        step.main()
    print("\nDone. Figures: reports/figures/  |  metrics: results/metrics.csv")
