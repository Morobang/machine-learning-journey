"""
COVID-19 Data Fetcher

This module provides utilities to download COVID-19 data from various reliable sources
including Our World in Data, Johns Hopkins University, and other public health organizations.
"""

import pandas as pd
import requests
from pathlib import Path
import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CovidDataFetcher:
    """
    A class to fetch COVID-19 data from various sources.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the data fetcher.
        
        Args:
            data_dir: Directory to save downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Data sources - Updated for Kaggle COVID-19 dataset structure
        self.sources = {
            'full_grouped': {
                'url': 'https://raw.githubusercontent.com/imdevskp/covid_19_jhu_data_web_scrap_and_cleaning/master/csv/full_grouped.csv',
                'description': 'Complete daily data with state/province level details'
            },
            'covid_clean_complete': {
                'url': 'https://raw.githubusercontent.com/imdevskp/covid_19_jhu_data_web_scrap_and_cleaning/master/csv/covid_19_clean_complete.csv',
                'description': 'Clean daily country-level data without provinces'
            },
            'country_wise_latest': {
                'url': 'https://raw.githubusercontent.com/imdevskp/covid_19_jhu_data_web_scrap_and_cleaning/master/csv/country_wise_latest.csv',
                'description': 'Latest statistics for each country'
            },
            'day_wise': {
                'url': 'https://raw.githubusercontent.com/imdevskp/covid_19_jhu_data_web_scrap_and_cleaning/master/csv/day_wise.csv',
                'description': 'Global daily aggregated data'
            },
            'usa_county_wise': {
                'url': 'https://raw.githubusercontent.com/imdevskp/covid_19_jhu_data_web_scrap_and_cleaning/master/csv/usa_county_wise.csv',
                'description': 'US county-level detailed data'
            },
            'worldometer_data': {
                'url': 'https://raw.githubusercontent.com/imdevskp/covid_19_jhu_data_web_scrap_and_cleaning/master/csv/worldometer_data.csv',
                'description': 'Latest data from Worldometer'
            }
        }
    
    def fetch_full_grouped_data(self, save_local: bool = True) -> pd.DataFrame:
        """
        Fetch the complete grouped COVID-19 dataset with province/state level data.
        
        Args:
            save_local: Whether to save the data locally
            
        Returns:
            DataFrame with COVID-19 data
        """
        try:
            logger.info("Fetching full grouped COVID-19 dataset...")
            url = self.sources['full_grouped']['url']
            
            # Download data
            df = pd.read_csv(url, low_memory=False)
            
            # Convert date column (handle different date formats)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            if save_local:
                filepath = self.data_dir / 'full_grouped.csv'
                df.to_csv(filepath, index=False)
                logger.info(f"Data saved to {filepath}")
            
            logger.info(f"Successfully fetched full grouped data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching full grouped data: {str(e)}")
            raise
    
    def fetch_clean_complete_data(self, save_local: bool = True) -> pd.DataFrame:
        """
        Fetch the clean complete COVID-19 dataset (country level only).
        
        Args:
            save_local: Whether to save the data locally
            
        Returns:
            DataFrame with COVID-19 data
        """
        try:
            logger.info("Fetching clean complete COVID-19 dataset...")
            url = self.sources['covid_clean_complete']['url']
            
            # Download data
            df = pd.read_csv(url, low_memory=False)
            
            # Convert date column
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            if save_local:
                filepath = self.data_dir / 'covid_19_clean_complete.csv'
                df.to_csv(filepath, index=False)
                logger.info(f"Data saved to {filepath}")
            
            logger.info(f"Successfully fetched clean complete data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching clean complete data: {str(e)}")
            raise
    
    def fetch_country_wise_latest(self, save_local: bool = True) -> pd.DataFrame:
        """
        Fetch latest country-wise COVID-19 statistics.
        
        Args:
            save_local: Whether to save the data locally
            
        Returns:
            DataFrame with latest country data
        """
        try:
            logger.info("Fetching latest country-wise data...")
            url = self.sources['country_wise_latest']['url']
            
            # Download data
            df = pd.read_csv(url, low_memory=False)
            
            if save_local:
                filepath = self.data_dir / 'country_wise_latest.csv'
                df.to_csv(filepath, index=False)
                logger.info(f"Country-wise latest data saved to {filepath}")
            
            logger.info(f"Successfully fetched country-wise latest: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching country-wise latest data: {str(e)}")
            raise
    
    def fetch_day_wise_data(self, save_local: bool = True) -> pd.DataFrame:
        """
        Fetch global day-wise aggregated COVID-19 data.
        
        Args:
            save_local: Whether to save the data locally
            
        Returns:
            DataFrame with day-wise global data
        """
        try:
            logger.info("Fetching day-wise global data...")
            url = self.sources['day_wise']['url']
            
            # Download data
            df = pd.read_csv(url, low_memory=False)
            
            # Convert date column
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            if save_local:
                filepath = self.data_dir / 'day_wise.csv'
                df.to_csv(filepath, index=False)
                logger.info(f"Day-wise data saved to {filepath}")
            
            logger.info(f"Successfully fetched day-wise data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching day-wise data: {str(e)}")
            raise
    
    def fetch_jhu_data(self, data_type: str = 'confirmed', save_local: bool = True) -> pd.DataFrame:
        """
        Fetch Johns Hopkins University time series data.
        
        Args:
            data_type: Type of data ('confirmed', 'deaths', 'recovered')
            save_local: Whether to save the data locally
            
        Returns:
            DataFrame with JHU time series data
        """
        try:
            logger.info(f"Fetching JHU {data_type} data...")
            
            if data_type not in ['confirmed', 'deaths', 'recovered']:
                raise ValueError("data_type must be 'confirmed', 'deaths', or 'recovered'")
            
            url = self.sources[f'jhu_{data_type}']['url']
            
            # Download data
            df = pd.read_csv(url)
            
            if save_local:
                filepath = self.data_dir / f'jhu_{data_type}_timeseries.csv'
                df.to_csv(filepath, index=False)
                logger.info(f"JHU {data_type} data saved to {filepath}")
            
            logger.info(f"Successfully fetched JHU {data_type} data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching JHU {data_type} data: {str(e)}")
            raise
    
    def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch all available COVID-19 datasets from the Kaggle source.
        
        Returns:
            Dictionary containing all datasets
        """
        datasets = {}
        
        try:
            # Fetch main datasets
            datasets['full_grouped'] = self.fetch_full_grouped_data()
            datasets['clean_complete'] = self.fetch_clean_complete_data()
            datasets['country_latest'] = self.fetch_country_wise_latest()
            datasets['day_wise'] = self.fetch_day_wise_data()
            
            # Fetch additional datasets
            try:
                datasets['usa_counties'] = self.fetch_usa_county_data()
                datasets['worldometer'] = self.fetch_worldometer_data()
            except Exception as e:
                logger.warning(f"Could not fetch additional datasets: {str(e)}")
            
            logger.info("Successfully fetched COVID-19 datasets!")
            return datasets
            
        except Exception as e:
            logger.error(f"Error fetching datasets: {str(e)}")
            raise
    
    def fetch_usa_county_data(self, save_local: bool = True) -> pd.DataFrame:
        """
        Fetch US county-level COVID-19 data.
        
        Args:
            save_local: Whether to save the data locally
            
        Returns:
            DataFrame with US county data
        """
        try:
            logger.info("Fetching US county-level data...")
            url = self.sources['usa_county_wise']['url']
            
            # Download data
            df = pd.read_csv(url, low_memory=False)
            
            if save_local:
                filepath = self.data_dir / 'usa_county_wise.csv'
                df.to_csv(filepath, index=False)
                logger.info(f"US county data saved to {filepath}")
            
            logger.info(f"Successfully fetched US county data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching US county data: {str(e)}")
            raise
    
    def fetch_worldometer_data(self, save_local: bool = True) -> pd.DataFrame:
        """
        Fetch latest Worldometer COVID-19 data.
        
        Args:
            save_local: Whether to save the data locally
            
        Returns:
            DataFrame with Worldometer data
        """
        try:
            logger.info("Fetching Worldometer data...")
            url = self.sources['worldometer_data']['url']
            
            # Download data
            df = pd.read_csv(url, low_memory=False)
            
            if save_local:
                filepath = self.data_dir / 'worldometer_data.csv'
                df.to_csv(filepath, index=False)
                logger.info(f"Worldometer data saved to {filepath}")
            
            logger.info(f"Successfully fetched Worldometer data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Worldometer data: {str(e)}")
            raise
    
    def get_data_info(self) -> None:
        """
        Print information about available data sources.
        """
        print("Available COVID-19 Data Sources:")
        print("=" * 50)
        
        for source_key, source_info in self.sources.items():
            print(f"\n{source_key.upper()}:")
            print(f"  Description: {source_info['description']}")
            print(f"  URL: {source_info['url']}")
    
    def get_latest_update_info(self) -> Dict:
        """
        Get information about when the data was last updated.
        
        Returns:
            Dictionary with update information
        """
        update_info = {
            'fetch_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_sources': list(self.sources.keys()),
            'local_files': []
        }
        
        # Check for existing local files
        for file_path in self.data_dir.glob('*.csv'):
            update_info['local_files'].append({
                'filename': file_path.name,
                'size_mb': round(file_path.stat().st_size / (1024 * 1024), 2),
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return update_info

def main():
    """
    Example usage of the CovidDataFetcher class.
    """
    # Initialize fetcher
    fetcher = CovidDataFetcher()
    
    # Show available data sources
    fetcher.get_data_info()
    
    # Fetch main clean complete dataset
    print("\nFetching main COVID-19 dataset...")
    covid_data = fetcher.fetch_clean_complete_data()
    
    print(f"\nDataset overview:")
    print(f"Shape: {covid_data.shape}")
    
    # Handle different date column names
    date_col = 'Date' if 'Date' in covid_data.columns else 'date'
    if date_col in covid_data.columns:
        print(f"Date range: {covid_data[date_col].min()} to {covid_data[date_col].max()}")
    
    # Handle different country column names
    country_col = 'Country/Region' if 'Country/Region' in covid_data.columns else 'Country'
    if country_col in covid_data.columns:
        print(f"Countries: {covid_data[country_col].nunique()}")
    
    # Show some basic statistics
    print(f"\nBasic statistics:")
    if 'Confirmed' in covid_data.columns:
        print(f"Total cases worldwide: {covid_data['Confirmed'].max():,.0f}")
    if 'Deaths' in covid_data.columns:
        print(f"Total deaths worldwide: {covid_data['Deaths'].max():,.0f}")
    if 'Recovered' in covid_data.columns:
        print(f"Total recovered worldwide: {covid_data['Recovered'].max():,.0f}")

if __name__ == "__main__":
    main()