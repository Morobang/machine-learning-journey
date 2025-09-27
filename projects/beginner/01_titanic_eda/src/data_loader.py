"""
Titanic Dataset - Data Loading Utilities

This module provides functions for loading and initial processing of the Titanic dataset.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

class TitanicDataLoader:
    """
    A class to handle loading and basic processing of Titanic dataset
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the data loader
        
        Args:
            data_path (str): Path to the data directory
        """
        if data_path is None:
            self.data_path = Path(__file__).parent.parent / "data"
        else:
            self.data_path = Path(data_path)
    
    def load_raw_data(self, filename: str = "titanic.csv") -> pd.DataFrame:
        """
        Load raw Titanic dataset
        
        Args:
            filename (str): Name of the CSV file
            
        Returns:
            pd.DataFrame: Raw Titanic dataset
        """
        file_path = self.data_path / "raw" / filename
        
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Dataset loaded successfully from {file_path}")
            print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            print("Please download the Titanic dataset from Kaggle and place it in the data/raw/ folder")
            return None
    
    def get_dataset_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the dataset
        
        Args:
            df (pd.DataFrame): The dataset
            
        Returns:
            dict: Dictionary containing dataset information
        """
        if df is None:
            return {}
        
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns)
        }
        
        return info
    
    def get_missing_value_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a summary of missing values
        
        Args:
            df (pd.DataFrame): The dataset
            
        Returns:
            pd.DataFrame: Missing value summary
        """
        if df is None:
            return pd.DataFrame()
        
        missing_summary = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum(),
            'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
            'Data_Type': df.dtypes
        })
        
        return missing_summary.sort_values('Missing_Percentage', ascending=False)
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "titanic_processed.csv"):
        """
        Save processed dataset
        
        Args:
            df (pd.DataFrame): Processed dataset
            filename (str): Output filename
        """
        if df is None:
            print("❌ No data to save")
            return
        
        output_path = self.data_path / "processed" / filename
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"💾 Processed data saved to {output_path}")

def load_titanic_data(data_path: str = None) -> pd.DataFrame:
    """
    Convenience function to load Titanic dataset
    
    Args:
        data_path (str): Path to data directory
        
    Returns:
        pd.DataFrame: Titanic dataset
    """
    loader = TitanicDataLoader(data_path)
    return loader.load_raw_data()

if __name__ == "__main__":
    # Example usage
    loader = TitanicDataLoader()
    df = loader.load_raw_data()
    
    if df is not None:
        info = loader.get_dataset_info(df)
        print(f"\n📋 Dataset Info:")
        print(f"Shape: {info['shape']}")
        print(f"Columns: {len(info['columns'])}")
        print(f"Missing values: {sum(info['missing_values'].values())}")
        
        missing_summary = loader.get_missing_value_summary(df)
        print(f"\n🔍 Missing Values:")
        print(missing_summary[missing_summary['Missing_Count'] > 0])