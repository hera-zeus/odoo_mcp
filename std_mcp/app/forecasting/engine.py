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
            # Remplacer les zéros par NaN puis interpoler linéairement
            # Les zéros de mois sans activité faussent la tendance et la saisonnalité
            series = series.copy().astype(float)
            series = series.replace(0, np.nan).interpolate(method='linear').bfill().ffill()

            n = len(series)

            # Saisonnalité annuelle uniquement si assez de données (>= 2 cycles complets)
            # Pour des données mensuelles : il faut >= 24 points pour seasonal_periods=12
            # En dessous, on désactive la saisonnalité plutôt que d'utiliser une
            # saisonnalité trimestrielle (4) incorrecte sur du mensuel.
            if n >= 24:
                seasonal_periods = 12
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

            # Générer les dates futures en utilisant la fréquence réelle de la série
            last_date      = series.index[-1]
            real_freq      = pd.infer_freq(series.index) or 'MS'
            forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq=real_freq)[1:]

            logger.info(
                f"📈 ETS ajusté: n={n}, saisonnalité={use_seasonal}, "
                f"seasonal_periods={seasonal_periods}, AIC={fit.aic:.1f}"
            )

            return {
                'forecast': forecast,
                'fitted': fitted,
                'forecast_dates': forecast_dates,
                'model_info': {
                    'n_points': n,
                    'trend': 'add',
                    'seasonal': use_seasonal,
                    'seasonal_periods': seasonal_periods,
                    'aic': fit.aic
                }
            }
        except Exception as e:
            logger.error(f"❌ Erreur de prévision ETS: {e}")
            raise

    def calculate_metrics(self, actual: pd.Series, predicted: pd.Series) -> Dict:
        """
        Calculer MAE, RMSE, MAPE et sMAPE.
        - Les NaN (période d'initialisation ETS) sont exclus.
        - Les zéros dans actual sont exclus du MAPE (division par zéro).
        - sMAPE est fourni comme métrique de substitution robuste.
        """
        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error

            # Aligner les indices
            common_idx        = actual.index.intersection(predicted.index)
            actual_aligned    = actual.loc[common_idx]
            predicted_aligned = predicted.loc[common_idx]

            # Exclure les NaN (valeurs d'initialisation ETS) et les infinis
            valid_mask        = ~actual_aligned.isna() & ~predicted_aligned.isna() \
                                & np.isfinite(actual_aligned) & np.isfinite(predicted_aligned)
            actual_clean      = actual_aligned[valid_mask]
            predicted_clean   = predicted_aligned[valid_mask]

            if len(actual_clean) == 0:
                logger.warning("⚠️ Aucune valeur valide pour calculer les métriques")
                return {'mae': 0, 'rmse': 0, 'mape': None, 'smape': 0, 'n_points': 0}

            mae  = mean_absolute_error(actual_clean, predicted_clean)
            rmse = np.sqrt(mean_squared_error(actual_clean, predicted_clean))

            # sMAPE — robuste aux zéros et valeurs faibles
            smape = float(np.mean(
                2 * np.abs(actual_clean - predicted_clean)
                / (np.abs(actual_clean) + np.abs(predicted_clean) + 1e-9)
            ) * 100)

            # MAPE classique — uniquement sur les valeurs non-nulles
            nonzero_mask      = actual_clean != 0
            actual_nz         = actual_clean[nonzero_mask]
            predicted_nz      = predicted_clean[nonzero_mask]

            if len(actual_nz) > 0:
                mape = float(np.mean(np.abs((actual_nz - predicted_nz) / actual_nz)) * 100)
            else:
                mape = None  # Toutes les valeurs réelles sont à zéro

            logger.info(
                f"📊 Métriques: MAE={mae:.2f}, RMSE={rmse:.2f}, "
                f"MAPE={mape:.1f}% (sur {len(actual_nz)} pts non-nuls), "
                f"sMAPE={smape:.1f}% (sur {len(actual_clean)} pts)"
            )

            return {
                'mae':     float(mae),
                'rmse':    float(rmse),
                'mape':    round(mape, 2) if mape is not None else None,
                'smape':   round(smape, 2),
                'n_points': len(actual_clean)
            }
        except Exception as e:
            logger.error(f"❌ Erreur calcul métriques: {e}")
            return {'mae': 0, 'rmse': 0, 'mape': None, 'smape': 0, 'n_points': 0}
