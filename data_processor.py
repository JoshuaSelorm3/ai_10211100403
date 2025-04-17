# Name:  Joshua Jerry Selorm Yegbe

# Index Number: 10211100403


import pandas as pd
import re
import numpy as np

class GhanaElectionDataProcessor:
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.chunks = []
        
    def load_data(self, file_path):
        """Load the Ghana election results CSV file"""
        self.raw_data = pd.read_csv(file_path)
        return self.raw_data
    
    def preprocess_data(self):
        """Clean and prepare data for embedding"""
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data first.")
        
        # Make a copy to avoid modifying the original
        self.processed_data = self.raw_data.copy()
        
        # Fill NaN values with appropriate placeholders
        self.processed_data = self.processed_data.fillna("Not Available")
        
        # Clean text columns - replace special characters, extra spaces
        for col in self.processed_data.select_dtypes(include=['object']).columns:
            self.processed_data[col] = self.processed_data[col].astype(str)
            self.processed_data[col] = self.processed_data[col].apply(self._clean_text)
        
        return self.processed_data
    
    def _clean_text(self, text):
        """Clean individual text fields"""
        # Remove special characters and extra spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def create_text_chunks(self):
        """Convert dataframe rows to text chunks for embedding"""
        if self.processed_data is None:
            raise ValueError("Data not processed. Call preprocess_data first.")
        
        self.chunks = []
        
        # Create a text representation for each row with column names as context
        for idx, row in self.processed_data.iterrows():
            chunk_text = f"Election data record {idx}: "
            for col in self.processed_data.columns:
                chunk_text += f"{col}: {row[col]}. "
            
            # Also create region-specific chunk
            if 'region' in row and 'constituency' in row:
                region_text = f"In {row['region']}, constituency {row['constituency']}: "
                for col in self.processed_data.columns:
                    if col not in ['region', 'constituency']:
                        region_text += f"{col}: {row[col]}. "
                self.chunks.append({"text": region_text, "metadata": {"row_idx": idx, "type": "region_specific"}})
            
            # Add the full row chunk
            self.chunks.append({"text": chunk_text, "metadata": {"row_idx": idx, "type": "full_row"}})
        
        # Create aggregated statistics chunks for better question answering
        self._create_stat_chunks()
        
        return self.chunks
    
    def _create_stat_chunks(self):
        """Create statistical summary chunks for common queries"""
        if 'party' in self.processed_data.columns and 'valid_votes' in self.processed_data.columns:
            # Party vote summaries
            party_votes = self.processed_data.groupby('party')['valid_votes'].sum().reset_index()
            for idx, row in party_votes.iterrows():
                chunk_text = f"The party {row['party']} received a total of {row['valid_votes']} valid votes across all constituencies."
                self.chunks.append({"text": chunk_text, "metadata": {"type": "party_summary"}})
            
            # Winner information
            winner = party_votes.loc[party_votes['valid_votes'].idxmax()]
            chunk_text = f"The party with the most votes was {winner['party']} with {winner['valid_votes']} total valid votes."
            self.chunks.append({"text": chunk_text, "metadata": {"type": "winner_summary"}})
        
        if 'region' in self.processed_data.columns:
            # Regional summaries
            region_stats = self.processed_data.groupby('region').agg({
                'valid_votes': 'sum',
                'registered_voters': 'sum' if 'registered_voters' in self.processed_data.columns else 'count'
            }).reset_index()
            
            for idx, row in region_stats.iterrows():
                chunk_text = f"In the region of {row['region']}, there were {row['valid_votes']} valid votes"
                if 'registered_voters' in region_stats.columns:
                    chunk_text += f" out of {row['registered_voters']} registered voters."
                self.chunks.append({"text": chunk_text, "metadata": {"type": "region_summary", "region": row['region']}})