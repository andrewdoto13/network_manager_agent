"""Schema mapping utility for handling inconsistent column names across datasets."""

from typing import Optional
import pandas as pd

class SchemaMapper:
    """Maps generic field names to actual DataFrame column names case-insensitively."""
    
    def __init__(self, df: pd.DataFrame = None):
        self.mapping = {}
        if df is not None:
            self.add_dataframe(df)

    def add_dataframe(self, df: pd.DataFrame):
        """Add a dataframe to the mapper to discover more column names."""
        requirements = {
            "lat": ["latitude", "lat"],
            "lon": ["longitude", "lon"],
            "county": ["county"],
            "specialty": ["specialty"],
            "entity": ["primary contract entity", "entity"],
            "effectiveness": ["effectiveness"],
            "efficiency": ["efficiency"],
            "claims_amount": ["total claims amount"],
            "medicare_claims": ["medicare total claims amount"],
            "confidence": ["location confidence score"],
            "new_patient": ["medicare new patient claims"],
            "claims_volume": ["total claims volume"],
            "city": ["city"],
        }

        for key, synonyms in requirements.items():
            if key in self.mapping:
                continue
            for col in df.columns:
                if col.lower() in synonyms:
                    self.mapping[key] = col
                    break

    def get(self, key: str, df: pd.DataFrame) -> str:
        """Return the actual column name for a given key in the provided DataFrame.
        
        If the mapped name for 'key' is not in df.columns, it attempts to find 
        the column using the synonyms for that key.
        """
        mapped_col = self.mapping.get(key)
        if mapped_col and mapped_col in df.columns:
            return mapped_col
        
        # Fallback: search for synonyms in this specific dataframe
        requirements = {
            "lat": ["latitude", "lat"],
            "lon": ["longitude", "lon"],
            "county": ["county"],
            "specialty": ["specialty"],
            "entity": ["primary contract entity", "entity"],
            "effectiveness": ["effectiveness"],
            "efficiency": ["efficiency"],
            "claims_amount": ["total claims amount"],
            "medicare_claims": ["medicare total claims amount"],
            "confidence": ["location confidence score"],
            "new_patient": ["medicare new patient claims"],
            "claims_volume": ["total claims volume"],
            "city": ["city"],
        }
        
        synonyms = requirements.get(key, [])
        for col in df.columns:
            if col.lower() in synonyms:
                return col
                
        raise KeyError(f"Column mapping for '{key}' not found in DataFrame. Available columns: {list(df.columns)}")

