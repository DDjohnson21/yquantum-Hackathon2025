# phase1.py
import pandas as pd
import matplotlib.pyplot as plt

def load_and_inspect_data(csv_file):
    """
    Load the CSV file and inspect its contents.
    """
    try:
        # Load CSV data into a DataFrame
        df = pd.read_csv(csv_file)
        print("CSV file loaded successfully.\n")
    except Exception as e:
        print("Error loading CSV file:", e)
        return None

    # Display first few rows to see the structure
    print("First 5 rows of the dataset:")
    print(df.head(), "\n")
    
    # Display dataset information (data types, missing values, etc.)
    print("Dataset Information:")
    df.info()
    print()
    
    # Display descriptive statistics for numeric columns
    print("Descriptive Statistics:")
    print(df.describe(), "\n")
    
    return df

def analyze_severity_distribution(df, severity_column='CAT Severity Code'):
    """
    Analyze and visualize the distribution of tornado severity levels.
    """
    # Check if the severity column exists in the dataset
    if severity_column not in df.columns:
        print(f"The column '{severity_column}' was not found in the dataset. Please verify the column name.")
        return
    
    # Compute the frequency of each severity level
    severity_counts = df[severity_column].value_counts().sort_index()
    print(f"Severity Level Distribution (using '{severity_column}'):")
    print(severity_counts, "\n")
    
    # Create a bar chart to visualize the severity distribution
    plt.figure(figsize=(8, 5))
    # Convert index to int (if needed) and then to string for a clearer x-axis label
    plt.bar(severity_counts.index.astype(int).astype(str), severity_counts.values, color='skyblue')
    plt.xlabel('Tornado Severity Level')
    plt.ylabel('Number of Claims')
    plt.title('Distribution of Tornado Claim Severities')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def main():
    # Specify your CSV file name (adjust path if needed)
    csv_file = 'tornado_severity_data.csv'
    
    # Step 1: Load and inspect the dataset
    df = load_and_inspect_data(csv_file)
    
    # Proceed only if the data loaded successfully
    if df is not None:
        # Step 2: Analyze the severity distribution
        analyze_severity_distribution(df, severity_column='CAT Severity Code')
        
        # Additional steps can be added here as part of further data preparation or visualization.
        # For example, you could parse date fields or visualize geographic locations.
    
if __name__ == "__main__":
    main()