from src.collect_data import collect_data
from src.clean_data import clean_data
from src.features import main as run_features
from src.visualize import main as run_visualizations
from src.train_model import main as run_training
from src.zillow_analysis import main as run_zillow_analysis


def main():
    print("Step 1: Collecting Boston Property Assessment FY2025 data...")
    collect_data()

    print("\nStep 2: Cleaning data...")
    clean_data()

    print("\nStep 3: Engineering features...")
    run_features()

    print("\nStep 4: Creating visualizations...")
    run_visualizations()

    print("\nStep 5: Training models...")
    run_training()

    print("\nStep 6: Running Zillow analysis...")
    run_zillow_analysis()


if __name__ == "__main__":
    main()
