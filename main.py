from src.clean_data import clean_data
from src.features import create_features
from src.train_model import main as run_training
from src.zillow_analysis import main as run_zillow_analysis

def main():
    print("Step 1: Cleaning data...")
    clean_data()

    print("\nStep 2: Training model...")
    run_training()

    print("\nStep 3: Running Zillow analysis...")
    run_zillow_analysis()

if __name__ == "__main__":
    main()