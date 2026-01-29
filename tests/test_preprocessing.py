"""
Tests unitaires pour les modules de preprocessing.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.clean_and_merge import clean_application_data
from src.data.feature_engineering import create_domain_features, engineer_features


class TestCleanApplicationData:
    """Tests pour la fonction clean_application_data."""
    
    def test_days_birth_absolute(self):
        """Test que DAYS_BIRTH est converti en valeur absolue."""
        df = pd.DataFrame({
            'DAYS_BIRTH': [-10000, -5000, -20000],
            'TARGET': [0, 1, 0]
        })
        
        result = clean_application_data(df)
        
        assert all(result['DAYS_BIRTH'] > 0)
        assert result['DAYS_BIRTH'].tolist() == [10000, 5000, 20000]
    
    def test_days_employed_anomaly(self):
        """Test que la valeur aberrante 365243 est remplacée par NaN."""
        df = pd.DataFrame({
            'DAYS_BIRTH': [10000, 5000, 20000],
            'DAYS_EMPLOYED': [365243, -1000, -2000],
            'TARGET': [0, 1, 0]
        })
        
        result = clean_application_data(df)
        
        assert pd.isna(result.loc[0, 'DAYS_EMPLOYED'])
        assert result.loc[1, 'DAYS_EMPLOYED'] == -1000
        assert result.loc[2, 'DAYS_EMPLOYED'] == -2000


class TestFeatureEngineering:
    """Tests pour l'ingénierie des features."""
    
    def test_create_domain_features(self):
        """Test de création des features métier."""
        df = pd.DataFrame({
            'AMT_CREDIT': [100000, 200000],
            'AMT_INCOME_TOTAL': [50000, 100000],
            'AMT_ANNUITY': [5000, 10000],
            'DAYS_BIRTH': [10000, 15000],
            'DAYS_EMPLOYED': [-2000, -3000],
            'CNT_FAM_MEMBERS': [2, 4]
        })
        
        result = create_domain_features(df)
        
        # Vérifier que les nouvelles colonnes existent
        assert 'CREDIT_INCOME_RATIO' in result.columns
        assert 'ANNUITY_INCOME_RATIO' in result.columns
        assert 'CREDIT_TERM' in result.columns
        assert 'AGE_YEARS' in result.columns
        
        # Vérifier les valeurs
        assert result.loc[0, 'CREDIT_INCOME_RATIO'] == pytest.approx(2.0)
        assert result.loc[0, 'AGE_YEARS'] == pytest.approx(10000 / 365, rel=1e-2)
    
    def test_no_infinite_values(self):
        """Test qu'il n'y a pas de valeurs infinies après engineering."""
        df = pd.DataFrame({
            'AMT_CREDIT': [100000, 200000],
            'AMT_INCOME_TOTAL': [0, 100000],  # Une valeur 0 pour tester
            'AMT_ANNUITY': [5000, 10000],
            'DAYS_BIRTH': [10000, 15000],
            'DAYS_EMPLOYED': [-2000, -3000],
            'CNT_FAM_MEMBERS': [2, 4],
            'NAME_CONTRACT_TYPE': ['Cash loans', 'Revolving loans']
        })
        
        result = engineer_features(df)
        
        # Vérifier qu'il n'y a pas d'infinis
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
