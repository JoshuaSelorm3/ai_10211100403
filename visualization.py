# Name:  Joshua Jerry Selorm Yegbe

# Index Number: 10211100403


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

class ElectionDataVisualizer:
    def __init__(self, data=None):
        """Initialize with optional dataframe"""
        self.data = data
    
    def set_data(self, data):
        """Set or update the dataframe"""
        self.data = data
        return self
    
    def plot_party_votes(self, top_n=5):
        """Plot the top N parties by votes"""
        if self.data is None or 'party' not in self.data.columns:
            return None
        
        if 'valid_votes' in self.data.columns:
            # Aggregate votes by party
            party_votes = self.data.groupby('party')['valid_votes'].sum().reset_index()
            # Sort and get top N
            party_votes = party_votes.sort_values('valid_votes', ascending=False).head(top_n)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(party_votes['party'], party_votes['valid_votes'], color='skyblue')
            ax.set_title(f'Top {top_n} Parties by Total Votes')
            ax.set_xlabel('Political Party')
            ax.set_ylabel('Total Valid Votes')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5000,
                        f'{int(height):,}',
                        ha='center', va='bottom', rotation=0)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            return fig
        
        return None
    
    def plot_regional_distribution(self):
        """Plot vote distribution by region"""
        if self.data is None or 'region' not in self.data.columns:
            return None
        
        if 'valid_votes' in self.data.columns:
            # Aggregate votes by region
            region_votes = self.data.groupby('region')['valid_votes'].sum().reset_index()
            region_votes = region_votes.sort_values('valid_votes', ascending=False)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(region_votes['region'], region_votes['valid_votes'], color='lightgreen')
            ax.set_title('Vote Distribution by Region')
            ax.set_xlabel('Region')
            ax.set_ylabel('Total Valid Votes')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5000,
                        f'{int(height):,}',
                        ha='center', va='bottom', rotation=0)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            return fig
        
        return None
    
    def plot_party_comparison_by_region(self, parties=None):
        """Plot comparison of specified parties across regions"""
        if self.data is None or 'region' not in self.data.columns or 'party' not in self.data.columns:
            return None
        
        if parties is None:
            # Get top 3 parties by votes
            top_parties = self.data.groupby('party')['valid_votes'].sum().nlargest(3).index.tolist()
        else:
            top_parties = parties
        
        if 'valid_votes' in self.data.columns:
            # Filter for selected parties
            party_data = self.data[self.data['party'].isin(top_parties)]
            
            # Aggregate votes by region and party
            pivot_table = pd.pivot_table(
                party_data, 
                values='valid_votes', 
                index=['region'], 
                columns=['party'], 
                aggfunc=np.sum,
                fill_value=0
            )
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 10))
            pivot_table.plot(kind='bar', ax=ax)
            ax.set_title('Party Performance by Region')
            ax.set_xlabel('Region')
            ax.set_ylabel('Total Valid Votes')
            ax.legend(title='Party')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            return fig
        
        return None
    
    def plot_voter_turnout(self):
        """Plot voter turnout by region"""
        if self.data is None or 'region' not in self.data.columns:
            return None
        
        if 'valid_votes' in self.data.columns and 'registered_voters' in self.data.columns:
            # Aggregate by region
            region_data = self.data.groupby('region').agg({
                'valid_votes': 'sum',
                'registered_voters': 'sum'
            }).reset_index()
            
            # Calculate turnout percentage
            region_data['turnout_percent'] = (region_data['valid_votes'] / region_data['registered_voters'] * 100).round(2)
            region_data = region_data.sort_values('turnout_percent', ascending=False)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(region_data['region'], region_data['turnout_percent'], color='coral')
            ax.set_title('Voter Turnout by Region')
            ax.set_xlabel('Region')
            ax.set_ylabel('Turnout Percentage (%)')
            ax.set_ylim(0, 100)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%',
                        ha='center', va='bottom', rotation=0)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            return fig
        
        return None
    
    @st.cache_data
    def create_streamlit_visualizations(self):
        """Generate all visualizations for Streamlit"""
        visualizations = {}
        
        visualizations['party_votes'] = self.plot_party_votes()
        visualizations['regional_distribution'] = self.plot_regional_distribution()
        visualizations['party_comparison'] = self.plot_party_comparison_by_region()
        visualizations['voter_turnout'] = self.plot_voter_turnout()
        
        return visualizations
