"""
COVID-19 Visualization Utilities

This module provides custom plotting functions for COVID-19 data analysis,
including time series plots, geographic visualizations, and comparative charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pathlib import Path
import logging
from typing import List, Optional, Dict, Tuple, Union
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CovidVisualizer:
    """
    A class for creating COVID-19 data visualizations.
    """
    
    def __init__(self, figures_dir: str = "reports/figures"):
        """
        Initialize the visualizer.
        
        Args:
            figures_dir: Directory to save generated figures
        """
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting styles
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Configure matplotlib for high-quality figures
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.facecolor'] = 'white'
        
        # Color palettes
        self.colors = {
            'cases': '#1f77b4',      # Blue
            'deaths': '#d62728',     # Red
            'recovered': '#2ca02c',  # Green
            'tests': '#ff7f0e',      # Orange
            'vaccinations': '#9467bd' # Purple
        }
        
        # Country color palette for multiple countries
        self.country_colors = px.colors.qualitative.Set3
    
    def plot_global_trends(self, df: pd.DataFrame, save_fig: bool = True) -> go.Figure:
        """
        Create a comprehensive global trends dashboard.
        
        Args:
            df: Global time series DataFrame
            save_fig: Whether to save the figure
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating global trends visualization...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Global Daily Cases', 'Global Daily Deaths', 
                          'Cumulative Cases', 'Cumulative Deaths'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Daily cases with 7-day average
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['new_cases'], 
                      name='Daily Cases', opacity=0.3, 
                      line=dict(color=self.colors['cases'])),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['new_cases_7day_avg'], 
                      name='7-day Average', 
                      line=dict(color=self.colors['cases'], width=3)),
            row=1, col=1
        )
        
        # Daily deaths with 7-day average
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['new_deaths'], 
                      name='Daily Deaths', opacity=0.3,
                      line=dict(color=self.colors['deaths'])),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['new_deaths_7day_avg'], 
                      name='7-day Average', 
                      line=dict(color=self.colors['deaths'], width=3)),
            row=1, col=2
        )
        
        # Cumulative cases
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['total_cases'], 
                      name='Total Cases', 
                      line=dict(color=self.colors['cases'], width=2)),
            row=2, col=1
        )
        
        # Cumulative deaths
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['total_deaths'], 
                      name='Total Deaths', 
                      line=dict(color=self.colors['deaths'], width=2)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="COVID-19 Global Trends Dashboard",
            title_font_size=20,
            showlegend=False,
            height=800,
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Number of Cases", row=1, col=1)
        fig.update_yaxes(title_text="Number of Deaths", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Cases", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Deaths", row=2, col=2)
        
        if save_fig:
            filepath = self.figures_dir / 'global_trends_dashboard.html'
            fig.write_html(filepath)
            # Also save as PNG
            filepath_png = self.figures_dir / 'global_trends_dashboard.png'
            fig.write_image(filepath_png, width=1200, height=800)
            logger.info(f"Global trends dashboard saved to {filepath}")
        
        return fig
    
    def plot_country_comparison(self, df: pd.DataFrame, countries: List[str], 
                              metric: str = 'new_cases_7day_avg', 
                              save_fig: bool = True) -> go.Figure:
        """
        Compare trends across multiple countries.
        
        Args:
            df: Country time series DataFrame
            countries: List of countries to compare
            metric: Metric to plot
            save_fig: Whether to save the figure
            
        Returns:
            Plotly figure object
        """
        logger.info(f"Creating country comparison for {len(countries)} countries...")
        
        # Filter data
        df_filtered = df[df['location'].isin(countries)].copy()
        
        fig = go.Figure()
        
        for i, country in enumerate(countries):
            country_data = df_filtered[df_filtered['location'] == country]
            
            fig.add_trace(
                go.Scatter(
                    x=country_data['date'],
                    y=country_data[metric],
                    name=country,
                    line=dict(color=self.country_colors[i % len(self.country_colors)], width=2),
                    mode='lines'
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"COVID-19 {metric.replace('_', ' ').title()} - Country Comparison",
            xaxis_title="Date",
            yaxis_title=metric.replace('_', ' ').title(),
            template='plotly_white',
            height=600,
            hovermode='x unified'
        )
        
        if save_fig:
            filename = f'country_comparison_{metric}.html'
            filepath = self.figures_dir / filename
            fig.write_html(filepath)
            logger.info(f"Country comparison saved to {filepath}")
        
        return fig
    
    def plot_country_rankings(self, df: pd.DataFrame, metric: str = 'Total_Cases', 
                            top_n: int = 20, save_fig: bool = True) -> plt.Figure:
        """
        Create a horizontal bar chart showing country rankings.
        
        Args:
            df: Country summary DataFrame
            metric: Metric to rank by
            top_n: Number of top countries to show
            save_fig: Whether to save the figure
            
        Returns:
            Matplotlib figure object
        """
        logger.info(f"Creating country rankings for {metric}...")
        
        # Get top N countries
        df_top = df.nlargest(top_n, metric).copy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar chart
        bars = ax.barh(df_top['Country'], df_top[metric], 
                      color=self.colors.get(metric.lower().split('_')[0], '#1f77b4'),
                      alpha=0.7)
        
        # Customize the plot
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_title(f'Top {top_n} Countries by {metric.replace("_", " ").title()}')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:,.0f}', ha='left', va='center', 
                   fontweight='bold', fontsize=9)
        
        # Invert y-axis so highest values are at top
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_fig:
            filename = f'country_rankings_{metric.lower()}.png'
            filepath = self.figures_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Country rankings saved to {filepath}")
        
        return fig
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, save_fig: bool = True) -> plt.Figure:
        """
        Create a correlation heatmap of COVID-19 metrics.
        
        Args:
            df: DataFrame with COVID-19 metrics
            save_fig: Whether to save the figure
            
        Returns:
            Matplotlib figure object
        """
        logger.info("Creating correlation heatmap...")
        
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Remove ID columns and dates
        exclude_cols = ['date', 'iso_code', 'population']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax, fmt='.2f', cbar_kws={'shrink': 0.8})
        
        ax.set_title('COVID-19 Metrics Correlation Matrix', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_fig:
            filepath = self.figures_dir / 'correlation_heatmap.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation heatmap saved to {filepath}")
        
        return fig
    
    def plot_time_series_decomposition(self, df: pd.DataFrame, country: str, 
                                     metric: str = 'new_cases_7day_avg',
                                     save_fig: bool = True) -> plt.Figure:
        """
        Create a time series decomposition plot for a specific country.
        
        Args:
            df: Time series DataFrame
            country: Country to analyze
            metric: Metric to decompose
            save_fig: Whether to save the figure
            
        Returns:
            Matplotlib figure object
        """
        logger.info(f"Creating time series decomposition for {country}...")
        
        # Filter data for specific country
        country_data = df[df['location'] == country].copy()
        country_data = country_data.sort_values('date')
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Original time series
        axes[0].plot(country_data['date'], country_data[metric], 
                    color=self.colors['cases'], linewidth=2)
        axes[0].set_title(f'{country} - {metric.replace("_", " ").title()}')
        axes[0].grid(True, alpha=0.3)
        
        # 30-day rolling average (trend)
        trend = country_data[metric].rolling(30, center=True).mean()
        axes[1].plot(country_data['date'], trend, 
                    color='red', linewidth=2)
        axes[1].set_title('30-Day Trend')
        axes[1].grid(True, alpha=0.3)
        
        # Residuals (original - trend)
        residuals = country_data[metric] - trend
        axes[2].plot(country_data['date'], residuals, 
                    color='gray', alpha=0.7)
        axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[2].set_title('Residuals')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            filename = f'timeseries_decomposition_{country.lower().replace(" ", "_")}.png'
            filepath = self.figures_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Time series decomposition saved to {filepath}")
        
        return fig
    
    def plot_case_fatality_analysis(self, df: pd.DataFrame, countries: List[str],
                                   save_fig: bool = True) -> go.Figure:
        """
        Analyze case fatality rates across countries and time.
        
        Args:
            df: Time series DataFrame with case fatality rates
            countries: List of countries to analyze
            save_fig: Whether to save the figure
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating case fatality rate analysis...")
        
        # Filter data
        df_filtered = df[df['location'].isin(countries)].copy()
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Case Fatality Rate Over Time', 'Latest CFR by Country'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Time series of CFR
        for i, country in enumerate(countries):
            country_data = df_filtered[df_filtered['location'] == country]
            
            fig.add_trace(
                go.Scatter(
                    x=country_data['date'],
                    y=country_data['case_fatality_rate'],
                    name=country,
                    line=dict(color=self.country_colors[i % len(self.country_colors)]),
                    mode='lines'
                ),
                row=1, col=1
            )
        
        # Latest CFR by country (bar chart)
        latest_cfr = df_filtered.groupby('location')['case_fatality_rate'].last().sort_values(ascending=True)
        
        fig.add_trace(
            go.Bar(
                x=latest_cfr.values,
                y=latest_cfr.index,
                orientation='h',
                name='Latest CFR',
                marker_color='red',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="COVID-19 Case Fatality Rate Analysis",
            showlegend=True,
            height=600,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Case Fatality Rate (%)", row=1, col=2)
        fig.update_yaxes(title_text="Case Fatality Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Country", row=1, col=2)
        
        if save_fig:
            filepath = self.figures_dir / 'case_fatality_analysis.html'
            fig.write_html(filepath)
            logger.info(f"Case fatality analysis saved to {filepath}")
        
        return fig
    
    def create_dashboard_summary(self, global_df: pd.DataFrame, country_df: pd.DataFrame,
                                summary_df: pd.DataFrame, save_fig: bool = True) -> go.Figure:
        """
        Create a comprehensive dashboard with key metrics.
        
        Args:
            global_df: Global time series data
            country_df: Country time series data
            summary_df: Country summary data
            save_fig: Whether to save the figure
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating comprehensive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Global Daily Cases Trend', 'Top 10 Countries - Total Cases',
                'Case Fatality Rates', 'Global Daily Deaths Trend',
                'Cases vs Deaths Scatter', 'Recent Trends (Last 30 Days)'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Global daily cases trend
        fig.add_trace(
            go.Scatter(x=global_df['date'], y=global_df['new_cases_7day_avg'],
                      name='Global Cases (7-day avg)', 
                      line=dict(color=self.colors['cases'], width=2)),
            row=1, col=1
        )
        
        # 2. Top 10 countries bar chart
        top_10 = summary_df.head(10)
        fig.add_trace(
            go.Bar(x=top_10['Total_Cases'], y=top_10['Country'],
                  orientation='h', name='Total Cases',
                  marker_color=self.colors['cases']),
            row=1, col=2
        )
        
        # 3. Case fatality rates
        fig.add_trace(
            go.Bar(x=top_10['Country'], y=top_10['Case_Fatality_Rate'],
                  name='CFR (%)', marker_color=self.colors['deaths']),
            row=1, col=3
        )
        
        # 4. Global daily deaths trend
        fig.add_trace(
            go.Scatter(x=global_df['date'], y=global_df['new_deaths_7day_avg'],
                      name='Global Deaths (7-day avg)',
                      line=dict(color=self.colors['deaths'], width=2)),
            row=2, col=1
        )
        
        # 5. Cases vs Deaths scatter
        fig.add_trace(
            go.Scatter(x=summary_df['Total_Cases'], y=summary_df['Total_Deaths'],
                      mode='markers', name='Countries',
                      marker=dict(size=8, color=self.colors['cases'], opacity=0.6),
                      text=summary_df['Country'], textposition='top center'),
            row=2, col=2
        )
        
        # 6. Recent trends (last 30 days) - showing top 5 countries
        recent_date = global_df['date'].max() - timedelta(days=30)
        recent_global = global_df[global_df['date'] >= recent_date]
        
        fig.add_trace(
            go.Scatter(x=recent_global['date'], y=recent_global['new_cases_7day_avg'],
                      name='Recent Global Trend',
                      line=dict(color='orange', width=3)),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title_text="COVID-19 Comprehensive Dashboard",
            title_font_size=20,
            showlegend=False,
            height=1000,
            template='plotly_white'
        )
        
        if save_fig:
            filepath = self.figures_dir / 'comprehensive_dashboard.html'
            fig.write_html(filepath)
            logger.info(f"Comprehensive dashboard saved to {filepath}")
        
        return fig
    
    def save_all_static_plots(self) -> None:
        """
        Convert all HTML plots to static PNG images for reports.
        """
        logger.info("Converting interactive plots to static images...")
        
        html_files = list(self.figures_dir.glob('*.html'))
        
        for html_file in html_files:
            try:
                # Read the HTML file and convert to PNG
                png_file = html_file.with_suffix('.png')
                # This would require additional setup for headless browser
                logger.info(f"Would convert {html_file.name} to PNG")
            except Exception as e:
                logger.warning(f"Could not convert {html_file.name}: {str(e)}")

def main():
    """
    Example usage of the CovidVisualizer class.
    """
    # This would typically be called from a notebook
    visualizer = CovidVisualizer()
    
    print("COVID-19 Visualizer initialized successfully!")
    print(f"Figures will be saved to: {visualizer.figures_dir}")
    print("Ready to create visualizations!")

if __name__ == "__main__":
    main()