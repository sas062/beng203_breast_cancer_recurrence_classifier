import pandas as pd

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def main():
    bc_meta = 'data/validation_bc_meta.csv'
    normal_meta = 'data/validation_normal_data.csv'

if __name__ == "__main__":
    main()    
