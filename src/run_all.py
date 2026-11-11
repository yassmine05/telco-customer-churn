"""Reproduce the whole project with one command:  python src/run_all.py

Order matters: the test set is only read in the last two steps, after every
modelling decision has been frozen by train.py.
"""
import eda
import evaluate
import explain
import train

if __name__ == "__main__":
    for step in (eda, train, evaluate, explain):
        print(f"\n=== {step.__name__} ===")
        step.main()
    print("\nDone. Figures: reports/figures/  |  metrics: results/metrics.csv")
