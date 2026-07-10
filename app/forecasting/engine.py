import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Dict
import logging

logger = logging.getLogger(__name__)

# Seuil de CV au-delà duquel on bascule sur la WMA
# CV > 1.2 = données très irrégulières, ETS peu fiable
CV_THRESHOLD = 1.2


class ForecastEngine:
    """Moteur Prédictif Universel — ETS pour données régulières, WMA pour demande sporadique."""

    def _weighted_moving_average(
        self, series: pd.Series, periods: int, real_freq: str
    ) -> Dict:
        """
        Moyenne mobile pondérée (poids croissants vers le présent).
        Utilisée quand CV > CV_THRESHOLD (demande sporadique / projet par projet).
        Retourne aussi un intervalle de confiance basé sur l'écart-type historique.
        """
        n = len(series)
        # Poids linéaires : le mois le plus récent pèse n fois plus que le premier
        weights    = np.arange(1, n + 1, dtype=float)
        weights   /= weights.sum()
        wma_value  = float(np.dot(weights, series.values))
        std        = float(series.std())

        last_date      = series.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq=real_freq)[1:]

        # Fitted = WMA sur une fenêtre glissante de 3 mois (in-sample approximation)
        window = min(3, n)
        fitted = series.rolling(window=window, min_periods=1).mean().shift(1)
        fitted = fitted.fillna(series.mean())

        forecast_series = pd.Series(
            [wma_value] * periods,
            index=forecast_dates
        )

        logger.info(
            f"WMA: valeur={wma_value:.0f}, std={std:.0f}, "
            f"intervalle=[{max(0, wma_value - std):.0f}, {wma_value + std:.0f}]"
        )

        return {
            'forecast':       forecast_series,
            'fitted':         fitted,
            'series_clean':   series,
            'forecast_dates': forecast_dates,
            'model_info': {
                'algo':         'WMA',
                'n_points':     n,
                'wma_value':    round(wma_value),
                'std':          round(std),
                'lower_bound':  round(max(0, wma_value - std)),
                'upper_bound':  round(wma_value + std),
                'cv':           round(series.std() / series.mean() if series.mean() else 0, 3),
                'reason':       'Demande sporadique détectée (CV élevé) — ETS non adapté'
            }
        }

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

            n          = len(series_clean)
            last_date  = series_clean.index[-1]
            real_freq  = pd.infer_freq(series_clean.index) or 'MS'

            # Diagnostics
            cv           = series_clean.std() / series_clean.mean() if series_clean.mean() != 0 else 0
            n_zeros_orig = int((series == 0).sum())
            logger.info(
                f"Série: n={n}, min={series_clean.min():.0f}, max={series_clean.max():.0f}, "
                f"mean={series_clean.mean():.0f}, CV={cv:.2f}, zéros_originaux={n_zeros_orig}"
            )

            # Détection demande sporadique : CV > seuil → WMA plus adaptée qu'ETS
            if cv > CV_THRESHOLD:
                logger.info(f"CV={cv:.2f} > {CV_THRESHOLD} — basculement sur WMA (demande sporadique)")
                return self._weighted_moving_average(series_clean, periods, real_freq)

            # Transformation log pour données financières à forte variance (CV > 0.5)
            # Normalise la variance et améliore drastiquement la convergence ETS
            use_log = cv > 0.5 and series_clean.min() > 0
            if use_log:
                series_fit = np.log(series_clean)
                logger.info("Transformation log appliquée (CV élevé)")
            else:
                series_fit = series_clean

            if n >= 24:
                seasonal_periods = 12
                best_fit     = None
                best_aic     = np.inf
                best_seasonal = 'add'
                for seasonal in ('add', 'mul'):
                    try:
                        m = ExponentialSmoothing(
                            series_fit,
                            trend='add',
                            seasonal=seasonal,
                            seasonal_periods=seasonal_periods
                        )
                        f = m.fit(optimized=True)
                        if f.aic < best_aic:
                            best_aic      = f.aic
                            best_fit      = f
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
                model = ExponentialSmoothing(series_fit, trend='add', seasonal=None)
                fit   = model.fit(optimized=True)

            # Retransformation si log appliqué
            if use_log:
                forecast = np.exp(fit.forecast(periods))
                fitted   = np.exp(fit.fittedvalues)
            else:
                forecast = fit.forecast(periods)
                fitted   = fit.fittedvalues

            forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq=real_freq)[1:]

            logger.info(
                f"ETS: seasonal={use_seasonal}, log={use_log}, AIC={fit.aic:.1f}"
            )

            return {
                'forecast':       forecast,
                'fitted':         fitted,
                'series_clean':   series_clean,
                'forecast_dates': forecast_dates,
                'model_info': {
                    'algo':             'ETS',
                    'n_points':         n,
                    'trend':            'add',
                    'seasonal':         use_seasonal,
                    'seasonal_periods': seasonal_periods,
                    'log_transform':    use_log,
                    'cv':               round(cv, 3),
                    'aic':              fit.aic
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
