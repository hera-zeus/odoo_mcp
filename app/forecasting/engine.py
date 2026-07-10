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
        Prévision par lissage exponentiel (ETS).
        Sélectionne automatiquement la meilleure configuration (add/mul) via AIC.
        Retourne aussi series_clean pour que les appelants calculent les métriques
        sur la même série que celle utilisée pour l'ajustement.
        """
        try:
            # Interpolation des zéros — les mois sans activité faussent trend et saisonnalité
            series_clean = series.copy().astype(float)
            series_clean = series_clean.replace(0, np.nan).interpolate(method='linear').bfill().ffill()

            n = len(series_clean)

            last_date  = series_clean.index[-1]
            real_freq  = pd.infer_freq(series_clean.index) or 'MS'

            if n >= 24:
                seasonal_periods = 12
                # Tester additif vs multiplicatif et garder le meilleur AIC
                best_fit = None
                best_aic = np.inf
                for trend in ('add',):
                    for seasonal in ('add', 'mul'):
                        try:
                            m = ExponentialSmoothing(
                                series_clean,
                                trend=trend,
                                seasonal=seasonal,
                                seasonal_periods=seasonal_periods
                            )
                            f = m.fit(optimized=True)
                            if f.aic < best_aic:
                                best_aic = f.aic
                                best_fit = f
                                best_seasonal = seasonal
                        except Exception:
                            continue

                if best_fit is None:
                    raise ValueError("Aucune configuration ETS n'a convergé")

                fit          = best_fit
                use_seasonal = best_seasonal

            else:
                seasonal_periods = None
                use_seasonal     = None
                model = ExponentialSmoothing(series_clean, trend='add', seasonal=None)
                fit   = model.fit(optimized=True)

            forecast       = fit.forecast(periods)
            fitted         = fit.fittedvalues
            forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq=real_freq)[1:]

            logger.info(
                f"ETS: n={n}, seasonal={use_seasonal}, "
                f"seasonal_periods={seasonal_periods}, AIC={fit.aic:.1f}"
            )

            return {
                'forecast':      forecast,
                'fitted':        fitted,
                'series_clean':  series_clean,   # série préprocessée pour calculate_metrics
                'forecast_dates': forecast_dates,
                'model_info': {
                    'n_points':        n,
                    'trend':           'add',
                    'seasonal':        use_seasonal,
                    'seasonal_periods': seasonal_periods,
                    'aic':             fit.aic
                }
            }
        except Exception as e:
            logger.error(f"Erreur de prévision ETS: {e}")
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
