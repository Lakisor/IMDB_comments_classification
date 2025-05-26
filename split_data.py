import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_directory(directory):
    os.makedirs(directory, exist_ok=True)

def split_data(data_path, test_size=0.2, random_state=42):
    df = pd.read_csv(data_path)
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['sentiment'] if 'sentiment' in df.columns else None
    )
    
    return train_df, test_df

def save_datasets(train_df, test_df, output_dir):
    create_directory(output_dir)
    
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Training set saved to: {train_path}")
    print(f"Test set saved to: {test_path}")

if __name__ == "__main__":
    input_data = "IMDB Dataset.csv"
    output_dir = "data/raw"
    
    try:
        train_df, test_df = split_data(input_data)
        
        save_datasets(train_df, test_df, output_dir)
        
        print("Data splitting completed successfully!")
        print(f"Training samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        
    except FileNotFoundError:
        print(f"Error: Input data file not found at {input_data}")
        print("Please ensure the data file exists at the specified location.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
