"""
COVID-19 Data Processor

This module provides utilities for cleaning, processing, and aggregating COVID-19 data
from various sources for analysis and visualization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CovidDataProcessor:
    """
    A class to process and clean COVID-19 data for analysis.
    """
    
    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        """
        Initialize the data processor.
        
        Args:
            raw_data_dir: Directory containing raw data files
            processed_data_dir: Directory to save processed data
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Country mappings and filters
        self.country_replacements = {
            'United States': 'USA',
            'United Kingdom': 'UK',
            'Russian Federation': 'Russia',
            'Korea, South': 'South Korea',
            'Iran (Islamic Republic of)': 'Iran',
        }
        
        # List of major countries for focused analysis
        self.major_countries = [
            'USA', 'China', 'India', 'Brazil', 'Russia', 'UK', 'France', 
            'Germany', 'Italy', 'Spain', 'Iran', 'South Korea', 'Japan',
            'Canada', 'Australia', 'Mexico', 'Indonesia', 'Turkey', 'Ukraine',
            'South Africa'
        ]
    
    def load_covid_data(self, filename: str = 'covid_19_clean_complete.csv') -> pd.DataFrame:
        """
        Load the main COVID-19 dataset.
        
        Args:
            filename: Name of the COVID data file
            
        Returns:
            DataFrame with COVID-19 data
        """
        try:
            filepath = self.raw_data_dir / filename
            logger.info(f"Loading COVID data from {filepath}")
            
            df = pd.read_csv(filepath, low_memory=False)
            
            # Handle different date column names and formats
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.rename(columns={'Date': 'date'})
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Standardize column names
            column_mapping = {
                'Country/Region': 'location',
                'Province/State': 'province_state',
                'Confirmed': 'total_cases',
                'Deaths': 'total_deaths',
                'Recovered': 'total_recovered', 
                'Active': 'active_cases'
            }
            
            df = df.rename(columns=column_mapping)
            
            logger.info(f"Loaded COVID data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading COVID data: {str(e)}")
            raise
    
    def load_country_latest_data(self, filename: str = 'country_wise_latest.csv') -> pd.DataFrame:
        """
        Load the latest country-wise COVID-19 data.
        
        Args:
            filename: Name of the country latest data file
            
        Returns:
            DataFrame with latest country data
        """
        try:
            filepath = self.raw_data_dir / filename
            logger.info(f"Loading country latest data from {filepath}")
            
            df = pd.read_csv(filepath, low_memory=False)
            
            # Standardize column names
            column_mapping = {
                'Country/Region': 'location',
                'Confirmed': 'total_cases',
                'Deaths': 'total_deaths',
                'Recovered': 'total_recovered',
                'Active': 'active_cases',
                'New cases': 'new_cases',
                'New deaths': 'new_deaths',
                'New recovered': 'new_recovered'
            }
            
            df = df.rename(columns=column_mapping)
            
            logger.info(f"Loaded country latest data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading country latest data: {str(e)}")
            raise
    
    def clean_covid_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize COVID-19 data.
        
        Args:
            df: Raw COVID DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning COVID-19 data...")
        
        # Create a copy
        df_clean = df.copy()
        
        # Ensure we have the required columns
        if 'location' not in df_clean.columns:
            logger.error("Missing 'location' column in data")
            return df_clean
        
        # Standardize country names
        df_clean['location'] = df_clean['location'].replace(self.country_replacements)
        
        # Handle province/state data - aggregate to country level if needed
        if 'province_state' in df_clean.columns:
            # Group by country and date, summing the numeric columns
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            groupby_cols = ['location', 'date'] if 'date' in df_clean.columns else ['location']
            
            df_clean = df_clean.groupby(groupby_cols)[numeric_cols].sum().reset_index()
        
        # Sort by location and date
        sort_cols = ['location', 'date'] if 'date' in df_clean.columns else ['location']
        df_clean = df_clean.sort_values(sort_cols).reset_index(drop=True)
        
        # Fill missing values for cumulative columns
        cumulative_cols = ['total_cases', 'total_deaths', 'total_recovered', 'active_cases']
        
        for col in cumulative_cols:
            if col in df_clean.columns:
                # Replace negative values with 0 (data corrections)
                df_clean[col] = df_clean[col].clip(lower=0)
                
                if 'date' in df_clean.columns:
                    # Forward fill missing values within each country
                    df_clean[col] = df_clean.groupby('location')[col].fillna(method='ffill')
        
        # Calculate daily values from cumulative data if date column exists
        if 'date' in df_clean.columns:
            df_clean = self._calculate_daily_values(df_clean)
        
        # Add derived metrics
        df_clean = self._add_derived_metrics(df_clean)
        
        logger.info(f"Cleaned COVID data: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        return df_clean
    
    def _calculate_daily_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily values from cumulative data.
        
        Args:
            df: DataFrame with cumulative data
            
        Returns:
            DataFrame with daily values added
        """
        df = df.copy()
        
        # Calculate daily new cases, deaths, tests, vaccinations
        cumulative_to_daily = {
            'total_cases': 'new_cases_calculated',
            'total_deaths': 'new_deaths_calculated',
            'total_tests': 'new_tests_calculated',
            'total_vaccinations': 'new_vaccinations_calculated'
        }
        
        for cumulative_col, daily_col in cumulative_to_daily.items():
            if cumulative_col in df.columns:
                df[daily_col] = df.groupby('location')[cumulative_col].diff()
                # Replace negative values with 0 (corrections in data)
                df[daily_col] = df[daily_col].clip(lower=0)
        
        return df
    
    def _add_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived metrics like case fatality rate, test positivity rate, etc.
        
        Args:
            df: DataFrame with basic COVID-19 data
            
        Returns:
            DataFrame with derived metrics
        """
        df = df.copy()
        
        # Case fatality rate (%)
        df['case_fatality_rate'] = (df['total_deaths'] / df['total_cases'] * 100).round(2)
        
        # Test positivity rate (%) using 7-day average
        if 'new_cases' in df.columns and 'new_tests' in df.columns:
            df['test_positivity_rate'] = (
                df.groupby('location')['new_cases'].rolling(7, min_periods=1).mean() /
                df.groupby('location')['new_tests'].rolling(7, min_periods=1).mean() * 100
            ).round(2)
        
        # Cases per million (if population data available)
        if 'population' in df.columns:
            df['cases_per_million'] = (df['total_cases'] / df['population'] * 1_000_000).round(1)
            df['deaths_per_million'] = (df['total_deaths'] / df['population'] * 1_000_000).round(1)
        
        # Moving averages (7-day)
        for col in ['new_cases', 'new_deaths']:
            if col in df.columns:
                df[f'{col}_7day_avg'] = (
                    df.groupby('location')[col].rolling(7, min_periods=1).mean().round(1)
                )
        
        return df
    
    def create_country_summary(self, df: pd.DataFrame, countries: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create a summary table with key metrics for each country.
        
        Args:
            df: Cleaned COVID-19 DataFrame
            countries: List of countries to include (default: major countries)
            
        Returns:
            DataFrame with country summaries
        """
        if countries is None:
            countries = self.major_countries
        
        logger.info(f"Creating country summary for {len(countries)} countries...")
        
        # Filter for specified countries
        df_filtered = df[df['location'].isin(countries)].copy()
        
        # Get latest data for each country
        latest_data = df_filtered.groupby('location').tail(1)
        
        # Create summary
        summary = pd.DataFrame({
            'Country': latest_data['location'],
            'Total_Cases': latest_data['total_cases'],
            'Total_Deaths': latest_data['total_deaths'],
            'Case_Fatality_Rate': latest_data['case_fatality_rate'],
            'Cases_Per_Million': latest_data.get('cases_per_million', np.nan),
            'Deaths_Per_Million': latest_data.get('deaths_per_million', np.nan),
            'Population': latest_data.get('population', np.nan),
            'Latest_Date': latest_data['date']
        })
        
        # Sort by total cases
        summary = summary.sort_values('Total_Cases', ascending=False).reset_index(drop=True)
        
        return summary
    
    def create_time_series_data(self, df: pd.DataFrame, countries: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create time series data for specified countries.
        
        Args:
            df: Cleaned COVID-19 DataFrame
            countries: List of countries to include
            
        Returns:
            DataFrame with time series data
        """
        if countries is None:
            countries = self.major_countries
        
        logger.info(f"Creating time series data for {len(countries)} countries...")
        
        # Filter data
        df_ts = df[df['location'].isin(countries)].copy()
        
        # Select key columns for time series analysis
        ts_columns = [
            'location', 'date', 'total_cases', 'new_cases', 'total_deaths', 
            'new_deaths', 'case_fatality_rate', 'new_cases_7day_avg', 'new_deaths_7day_avg'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in ts_columns if col in df_ts.columns]
        df_ts = df_ts[available_columns]
        
        return df_ts
    
    def aggregate_global_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data to create global totals by date.
        
        Args:
            df: Country-level COVID-19 DataFrame
            
        Returns:
            DataFrame with global aggregates
        """
        logger.info("Creating global aggregated data...")
        
        # Group by date and sum
        global_data = df.groupby('date').agg({
            'new_cases': 'sum',
            'new_deaths': 'sum',
            'total_cases': 'sum',
            'total_deaths': 'sum'
        }).reset_index()
        
        # Calculate cumulative totals (in case summing doesn't work perfectly)
        global_data['total_cases'] = global_data['new_cases'].cumsum()
        global_data['total_deaths'] = global_data['new_deaths'].cumsum()
        
        # Add 7-day moving averages
        global_data['new_cases_7day_avg'] = global_data['new_cases'].rolling(7, min_periods=1).mean()
        global_data['new_deaths_7day_avg'] = global_data['new_deaths'].rolling(7, min_periods=1).mean()
        
        # Add global case fatality rate
        global_data['case_fatality_rate'] = (global_data['total_deaths'] / global_data['total_cases'] * 100).round(2)
        
        return global_data
    
    def process_jhu_time_series(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Process Johns Hopkins time series data into long format.
        
        Args:
            df: JHU time series DataFrame
            data_type: Type of data ('confirmed', 'deaths', 'recovered')
            
        Returns:
            DataFrame in long format
        """
        logger.info(f"Processing JHU {data_type} time series data...")
        
        # Identify date columns (everything after 'Long')
        date_columns = df.columns[4:]  # Assuming first 4 are Province/State, Country/Region, Lat, Long
        
        # Melt the dataframe
        df_melted = df.melt(
            id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
            value_vars=date_columns,
            var_name='date',
            value_name=data_type
        )
        
        # Clean up
        df_melted['date'] = pd.to_datetime(df_melted['date'])
        df_melted = df_melted.rename(columns={'Country/Region': 'location'})
        
        # Aggregate by country (sum provinces/states)
        df_country = df_melted.groupby(['location', 'date'])[data_type].sum().reset_index()
        
        # Standardize country names
        df_country['location'] = df_country['location'].replace(self.country_replacements)
        
        return df_country
    
    def save_processed_data(self, datasets: Dict[str, pd.DataFrame]) -> None:
        """
        Save processed datasets to CSV files.
        
        Args:
            datasets: Dictionary of DataFrames to save
        """
        logger.info("Saving processed datasets...")
        
        for name, df in datasets.items():
            filepath = self.processed_data_dir / f'{name}.csv'
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {name}: {df.shape[0]} rows, {df.shape[1]} columns -> {filepath}")
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a data quality report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with data quality metrics
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A',
                'end': df['date'].max().strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A'
            },
            'countries': df['location'].nunique() if 'location' in df.columns else 'N/A',
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        return report

def main():
    """
    Example usage of the CovidDataProcessor class.
    """
    # Initialize processor
    processor = CovidDataProcessor()
    
    try:
        # Load and clean COVID data
        print("Loading and processing COVID-19 data...")
        covid_data = processor.load_covid_data()
        clean_data = processor.clean_covid_data(covid_data)
        
        # Create various processed datasets
        print("Creating processed datasets...")
        datasets = {}
        
        # Country summary
        datasets['country_summary'] = processor.create_country_summary(clean_data)
        
        # Time series for major countries (if date column exists)
        if 'date' in clean_data.columns:
            datasets['major_countries_timeseries'] = processor.create_time_series_data(clean_data)
            datasets['global_timeseries'] = processor.aggregate_global_data(clean_data)
        
        # Save processed data
        processor.save_processed_data(datasets)
        
        # Generate data quality report
        quality_report = processor.get_data_quality_report(clean_data)
        print("\nData Quality Report:")
        print(f"Total rows: {quality_report['total_rows']:,}")
        print(f"Total columns: {quality_report['total_columns']}")
        print(f"Date range: {quality_report['date_range']['start']} to {quality_report['date_range']['end']}")
        print(f"Countries: {quality_report['countries']}")
        print(f"Duplicate rows: {quality_report['duplicate_rows']}")
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()