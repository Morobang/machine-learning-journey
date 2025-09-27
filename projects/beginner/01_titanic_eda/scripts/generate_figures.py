"""
Titanic EDA - Figure Generation Script
=====================================

This script generates all visualizations for the Titanic EDA project.
Run this script to create high-quality figures for reports and presentations.

Usage:
    python generate_figures.py

Requirements:
    - pandas, numpy, matplotlib, seaborn, scipy
    - Cleaned dataset: '../data/processed/titanic_cleaned.csv'
    - Output directory: '../reports/figures/'

Author: Data Science Team
Date: September 27, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import os
import warnings
warnings.filterwarnings('ignore')

def setup_matplotlib():
    """Configure matplotlib for high-quality output"""
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

def load_data():
    """Load the cleaned Titanic dataset"""
    try:
        df_clean = pd.read_csv('../data/processed/titanic_cleaned.csv')
        print(f"✅ Loaded cleaned dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        return df_clean
    except FileNotFoundError:
        print("❌ Error: Could not find cleaned dataset at '../data/processed/titanic_cleaned.csv'")
        print("Please run the main analysis notebook first to generate the cleaned dataset.")
        return None

def create_output_directory():
    """Create output directory for figures"""
    os.makedirs('../reports/figures', exist_ok=True)
    print("✅ Output directory ready: '../reports/figures/'")

def generate_survival_overview(df_clean):
    """Generate overall survival statistics visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Titanic Survival Overview', fontsize=16, fontweight='bold')

    # Survival count
    survival_counts = df_clean['Survived'].value_counts()
    axes[0].bar(['Did not survive', 'Survived'], survival_counts.values, color=['#ff7f7f', '#7fbf7f'])
    axes[0].set_title('Survival Count')
    axes[0].set_ylabel('Number of Passengers')

    # Survival pie chart
    axes[1].pie(survival_counts.values, labels=['Did not survive', 'Survived'], 
               autopct='%1.1f%%', colors=['#ff7f7f', '#7fbf7f'])
    axes[1].set_title('Survival Distribution')

    # Survival by passenger class
    survival_by_class = df_clean.groupby('Pclass')['Survived'].mean()
    axes[2].bar(survival_by_class.index, survival_by_class.values, color='skyblue')
    axes[2].set_title('Survival Rate by Class')
    axes[2].set_xlabel('Passenger Class')
    axes[2].set_ylabel('Survival Rate')
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('../reports/figures/01_overview_survival_stats.png', dpi=300, bbox_inches='tight')
    plt.close()
    return "01_overview_survival_stats.png"

def generate_demographics_breakdown(df_clean):
    """Generate demographics overview visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Passenger Demographics Overview', fontsize=16, fontweight='bold')

    # Age distribution
    axes[0,0].hist(df_clean['Age'].dropna(), bins=30, color='lightblue', alpha=0.7, edgecolor='black')
    axes[0,0].set_title('Age Distribution')
    axes[0,0].set_xlabel('Age (years)')
    axes[0,0].set_ylabel('Frequency')

    # Gender distribution
    gender_counts = df_clean['Sex'].value_counts()
    axes[0,1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
                  colors=['lightcoral', 'lightblue'])
    axes[0,1].set_title('Gender Distribution')

    # Class distribution
    class_counts = df_clean['Pclass'].value_counts().sort_index()
    axes[1,0].bar(class_counts.index, class_counts.values, color=['gold', 'silver', 'bronze'])
    axes[1,0].set_title('Passenger Class Distribution')
    axes[1,0].set_xlabel('Passenger Class')
    axes[1,0].set_ylabel('Number of Passengers')

    # Embarkation port distribution
    embark_counts = df_clean['Embarked'].value_counts()
    axes[1,1].bar(embark_counts.index, embark_counts.values, color='lightgreen')
    axes[1,1].set_title('Embarkation Port Distribution')
    axes[1,1].set_xlabel('Port')
    axes[1,1].set_ylabel('Number of Passengers')

    plt.tight_layout()
    plt.savefig('../reports/figures/03_demographics_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    return "03_demographics_breakdown.png"

def generate_comprehensive_analysis(df_clean):
    """Generate 6-panel comprehensive survival analysis dashboard"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Survival Analysis Dashboard', fontsize=16, fontweight='bold')

    # Panel 1: Survival by Gender
    survival_sex = df_clean.groupby(['Sex', 'Survived']).size().unstack()
    survival_sex.plot(kind='bar', ax=axes[0,0], color=['#ff7f7f', '#7fbf7f'], alpha=0.8)
    axes[0,0].set_title('Survival by Gender')
    axes[0,0].set_ylabel('Count')
    axes[0,0].legend(['Did not survive', 'Survived'])
    axes[0,0].tick_params(axis='x', rotation=0)

    # Panel 2: Survival by Class
    survival_pclass = df_clean.groupby(['Pclass', 'Survived']).size().unstack()
    survival_pclass.plot(kind='bar', ax=axes[0,1], color=['#ff7f7f', '#7fbf7f'], alpha=0.8)
    axes[0,1].set_title('Survival by Passenger Class')
    axes[0,1].set_ylabel('Count')
    axes[0,1].legend(['Did not survive', 'Survived'])
    axes[0,1].tick_params(axis='x', rotation=0)

    # Panel 3: Survival by Age Group
    if 'AgeGroup' in df_clean.columns:
        survival_age = df_clean.groupby(['AgeGroup', 'Survived']).size().unstack()
        survival_age.plot(kind='bar', ax=axes[0,2], color=['#ff7f7f', '#7fbf7f'], alpha=0.8)
        axes[0,2].set_title('Survival by Age Group')
        axes[0,2].set_ylabel('Count')
        axes[0,2].legend(['Did not survive', 'Survived'])
        axes[0,2].tick_params(axis='x', rotation=45)

    # Panel 4: Survival Rate Heatmap
    survival_rate_sex_class = df_clean.groupby(['Sex', 'Pclass'])['Survived'].mean().unstack()
    sns.heatmap(survival_rate_sex_class, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1,0])
    axes[1,0].set_title('Survival Rate: Gender × Class')

    # Panel 5: Family Size vs Survival
    if 'FamilySize' in df_clean.columns:
        family_survival = df_clean.groupby('FamilySize')['Survived'].mean()
        axes[1,1].bar(family_survival.index, family_survival.values, color='skyblue', alpha=0.8)
        axes[1,1].set_title('Survival Rate by Family Size')
        axes[1,1].set_xlabel('Family Size')
        axes[1,1].set_ylabel('Survival Rate')

    # Panel 6: Fare Distribution by Survival
    survived_fares = df_clean[df_clean['Survived'] == 1]['Fare']
    not_survived_fares = df_clean[df_clean['Survived'] == 0]['Fare']
    axes[1,2].hist([not_survived_fares, survived_fares], bins=30, alpha=0.7, 
                   color=['#ff7f7f', '#7fbf7f'], label=['Did not survive', 'Survived'])
    axes[1,2].set_title('Fare Distribution by Survival')
    axes[1,2].set_xlabel('Fare')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].legend()

    plt.tight_layout()
    plt.savefig('../reports/figures/04_comprehensive_survival_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    return "04_comprehensive_survival_analysis.png"

def generate_statistical_summary(df_clean):
    """Generate statistical significance summary visualization"""
    categorical_features = ['Sex', 'Pclass']
    
    # Add optional features if they exist
    optional_features = ['AgeGroup', 'Title', 'IsAlone', 'HasCabin', 'Embarked']
    for feature in optional_features:
        if feature in df_clean.columns:
            categorical_features.append(feature)
    
    chi2_results = []
    p_values = []
    effect_sizes = []

    for feature in categorical_features:
        contingency_table = pd.crosstab(df_clean[feature], df_clean['Survived'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Calculate Cramér's V (effect size)
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
        
        chi2_results.append(chi2)
        p_values.append(p_value)
        effect_sizes.append(cramers_v)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Statistical Significance Analysis', fontsize=16, fontweight='bold')

    # P-values comparison
    colors = ['red' if p > 0.05 else 'green' for p in p_values]
    axes[0].bar(categorical_features, [-np.log10(p) for p in p_values], color=colors, alpha=0.8)
    axes[0].axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='α = 0.05')
    axes[0].set_title('P-values (-log10 scale)')
    axes[0].set_ylabel('-log10(p-value)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Effect sizes (Cramér's V)
    effect_colors = ['lightgreen' if v < 0.1 else 'yellow' if v < 0.3 else 'orange' if v < 0.5 else 'red' 
                    for v in effect_sizes]
    axes[1].bar(categorical_features, effect_sizes, color=effect_colors, alpha=0.8)
    axes[1].set_title('Effect Sizes (Cramér\'s V)')
    axes[1].set_ylabel('Cramér\'s V')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../reports/figures/06_statistical_significance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    return "06_statistical_significance_summary.png"

def main():
    """Main function to generate all figures"""
    print("🎨 Titanic EDA - Figure Generation Script")
    print("=" * 50)
    
    # Setup
    setup_matplotlib()
    create_output_directory()
    
    # Load data
    df_clean = load_data()
    if df_clean is None:
        return
    
    # Generate figures
    print("\n📊 Generating visualizations...")
    generated_figures = []
    
    try:
        fig1 = generate_survival_overview(df_clean)
        generated_figures.append(fig1)
        print(f"✅ {fig1}")
        
        fig2 = generate_demographics_breakdown(df_clean)
        generated_figures.append(fig2)
        print(f"✅ {fig2}")
        
        fig3 = generate_comprehensive_analysis(df_clean)
        generated_figures.append(fig3)
        print(f"✅ {fig3}")
        
        fig4 = generate_statistical_summary(df_clean)
        generated_figures.append(fig4)
        print(f"✅ {fig4}")
        
    except Exception as e:
        print(f"❌ Error generating figures: {e}")
        return
    
    # Summary
    print("\n" + "=" * 60)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 60)
    print(f"\n📁 Location: '../reports/figures/'")
    print(f"📊 Total figures: {len(generated_figures)}")
    print(f"🎯 Quality: 300 DPI (publication-ready)")
    print("\n🚀 Ready for use in reports, presentations, and portfolio!")

if __name__ == "__main__":
    main()