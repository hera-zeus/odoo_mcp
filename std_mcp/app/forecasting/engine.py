import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class ForecastEngine:
    """Moteur Prédictif Universel basé sur le lissage exponentiel (ETS)"""
    
    def forecast_ets(self, series: pd.Series, periods: int = 6) -> Dict:
        """
        Prévision par lissage exponentiel (ETS)
        """
        try:
            # Détection automatique de la saisonnalité
            n = len(series)
            if n >= 24:
                seasonal_periods = 12
                use_seasonal = 'add'
            elif n >= 8:
                seasonal_periods = 4
                use_seasonal = 'add'
            else:
                seasonal_periods = None
                use_seasonal = None

            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal=use_seasonal,
                seasonal_periods=seasonal_periods
            )
            
            fit = model.fit(optimized=True)
            forecast = fit.forecast(periods)
            fitted = fit.fittedvalues
            
            # Générer les dates futures — on utilise la fréquence RÉELLE
            # de la série (mensuelle), indépendamment de la saisonnalité du modèle
            last_date  = series.index[-1]
            real_freq  = pd.infer_freq(series.index) or 'MS'
            forecast_dates = pd.date_range(start=last_date, periods=periods+1, freq=real_freq)[1:]
            
            return {
                'forecast': forecast,
                'fitted': fitted,
                'forecast_dates': forecast_dates,
                'model_info': {
                    'trend': fit.params.get('trend', 'add'),
                    'seasonal': fit.params.get('seasonal', 'add'),
                    'aic': fit.aic
                }
            }
        except Exception as e:
            logger.error(f"❌ Erreur de prévision ETS: {e}")
            raise
    
    def calculate_metrics(self, actual: pd.Series, predicted: pd.Series) -> Dict:
        """Calculer MAE, RMSE, MAPE"""
        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            
            # Aligner les indices
            common_idx = actual.index.intersection(predicted.index)
            actual_aligned = actual.loc[common_idx]
            predicted_aligned = predicted.loc[common_idx]
            
            mae = mean_absolute_error(actual_aligned, predicted_aligned)
            rmse = np.sqrt(mean_squared_error(actual_aligned, predicted_aligned))
            mape = np.mean(np.abs((actual_aligned - predicted_aligned) / actual_aligned)) * 100
            
            return {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape)
            }
        except Exception as e:
            logger.error(f"❌ Erreur calcul métriques: {e}")
            return {'mae': 0, 'rmse': 0, 'mape': 0}
