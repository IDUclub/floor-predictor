import libpysal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, ParameterGrid, train_test_split

from tqdm.auto import tqdm
from esda.moran import Moran_Local, Moran
from libpysal.weights import KNN, lag_spatial
from splot.esda import moran_scatterplot, lisa_cluster

import joblib, sklearn
from datetime import datetime
from types import SimpleNamespace



class BuildingPreprocessor:
    def __init__(self, df):
        self.df = df.copy()

    def filter_residential(self):
        # Отбор только жилых зданий и этажей > 2
        self.df = self.df[(self.df["is_living"] == 1) & (self.df["storey"] > 2)].copy()
        self.df.reset_index(drop=True, inplace=True)

    def get(self):
        return self.df


class GeometryFeatureGenerator:
    def __init__(self, df):
        self.df = df.to_crs(3857)

    def compute_geometry_features(self):
        df = self.df
        df["shape_area"] = df.geometry.area
        df["shape_length"] = df.geometry.length
        df["sqft"] = df["shape_area"] * 10.7639
        bounds = df.geometry.bounds
        df["lat_dif"] = bounds["maxy"] - bounds["miny"]
        df["long_dif"] = bounds["maxx"] - bounds["minx"]
        df["envel_area"] = df["lat_dif"] * df["long_dif"]
        df["vertex_count"] = df.geometry.apply(
            lambda geom: (
                sum(len(ring.coords) for ring in ([geom.exterior] + list(geom.interiors)))
                if geom.geom_type == "Polygon"
                else sum(
                    sum(len(ring.coords) for ring in ([g.exterior] + list(g.interiors)))
                    for g in geom.geoms
                )
            )
        )
        df["geom_count"] = df.geometry.apply(
            lambda geom: 1 if geom.geom_type == "Polygon" else len(geom.geoms)
        )
        df["complexity_ratio"] = df["shape_length"] / df["shape_area"]
        df["iasl"] = df["shape_length"] / df["vertex_count"]
        df["vpa"] = df["vertex_count"] / df["shape_area"]
        df["complexity_ps"] = df["complexity_ratio"] / df["vertex_count"]
        df["ipq"] = (4 * np.pi * df["shape_area"]) / (df["shape_length"] ** 2)
        df["sqmeters"] = df["shape_area"]
        df["centroid"] = df.geometry.centroid
        self.df = df
        return self.df


class SpatialNeighborhoodAnalyzer:
    def __init__(self, df, radius=500, ):
        self.df = df
        self.radius = radius
        self.spatial_artifacts = None

    def compute_neighborhood_metrics(self, k_neighbors=10, plot=False, show_progress=True):
        """
        Возвращает:
          - self.df (обновлённый GeoDataFrame),
          - artifacts: dict[str, {"data": ..., "plots": ...}]
        """
        df = self.df.copy()
        coords = np.array(df["centroid"].apply(lambda pt: (pt.x, pt.y)).to_list())
        tree = KDTree(coords)
        emda_val = np.sqrt(df["shape_area"].mean()) * 0.5

        results = {
            "n_counta": [], "omda": [], "nnd": [], "nnia": [], "intensitya": [],
            "n_size_meana": [], "n_size_stda": [], "n_size_mina": [],
            "n_size_maxa": [], "n_size_cva": [],
        }

        for i, (x, y) in tqdm(enumerate(coords), total=len(coords), desc="Neighborhood stats"):
            idx = tree.query_ball_point([x, y], r=self.radius)
            idx = [j for j in idx if j != i]
            dists = np.linalg.norm(coords[idx] - [x, y], axis=1) if idx else np.array([])
            areas = df.iloc[idx]["shape_area"].values if idx else np.array([])

            results["n_counta"].append(len(idx))
            results["omda"].append(np.mean(dists) if len(dists) > 0 else np.nan)
            results["nnd"].append(np.min(dists) if len(dists) > 0 else np.nan)
            results["nnia"].append(np.min(dists) / emda_val if len(dists) > 0 else np.nan)
            results["intensitya"].append(np.min(dists) / emda_val if len(dists) > 0 else np.nan)
            results["n_size_meana"].append(np.mean(areas) if len(areas) > 0 else np.nan)
            results["n_size_stda"].append(np.std(areas) if len(areas) > 0 else np.nan)
            results["n_size_mina"].append(np.min(areas) if len(areas) > 0 else np.nan)
            results["n_size_maxa"].append(np.max(areas) if len(areas) > 0 else np.nan)
            results["n_size_cva"].append(
                np.std(areas) / np.mean(areas) if len(areas) > 0 and np.mean(areas) > 0 else np.nan
            )

        df = df.join(pd.DataFrame(results, index=df.index))
        geom_features = (
            df.select_dtypes(include="number")
              .drop(columns=[
                  "storey", "lisa_cluster", "pval", "sig",
                  "cluster_High-High", "storey_lag", "cluster_Low-Low",
                  "cluster_High-Low", "cluster_Low-High", "cluster_Not Significant",
              ], errors="ignore")
              .columns
        )

        # ↓ теперь функция возвращает (gdf, artifacts)
        df, artifacts = self.generate_spatial_lags_and_morans(
            df.set_geometry("centroid"),
            geom_features,
            k_neighbors=k_neighbors,
            plot=plot,
            show_progress=show_progress
        )

        self.df = df
        self.spatial_artifacts = artifacts  # можно обращаться после вызова
        return self.df, artifacts

    def generate_spatial_lags_and_morans(
        self,
        gdf,
        feature_columns,
        k_neighbors: int = 10,
        plot: bool = True,
        show_progress: bool = True,
    ):
        """
        Добавляет spatial-lag, считает Moran/LISA и возвращает (gdf, artifacts),
        где artifacts[feature] = {"data": {...}, "plots": {"scatter": fig|None, "lisa": fig|None}}
        """
        artifacts = {}

        gdf_meter = gdf.to_crs(epsg=3857)
        w = KNN.from_dataframe(gdf_meter, k=k_neighbors, silence_warnings=True)
        w.transform = "r"

        # копим новые колонки здесь и присоединяем одним join
        lag_cols = {}
        lisa_cols = {}

        iterator = tqdm(feature_columns, desc="Moran/LISA per feature", leave=False) if show_progress else feature_columns

        for col in iterator:
            entry = {"data": {}, "plots": {"scatter": None, "lisa": None}}
            x = gdf[col].to_numpy()
            finite = np.isfinite(x)

            # защита от NaN/нулевой дисперсии
            if finite.sum() < 3 or np.nanstd(x) == 0:
                lag_name = f"{col}_lag"
                lag_cols[lag_name] = lag_spatial(w, gdf[col])
                entry["data"].update({
                    "skipped": True,
                    "reason": "constant_or_too_few_values",
                    "lag_name": lag_name,
                    "moran_I": np.nan,
                    "moran_p": np.nan,
                })
                if show_progress:
                    iterator.set_postfix_str(f"{col}: skipped")
                artifacts[col] = entry
                continue

            # 1) spatial lag
            lag_name = f"{col}_lag"
            lag_cols[lag_name] = lag_spatial(w, gdf[col])
            entry["data"]["lag_name"] = lag_name

            # 2) global Moran
            moran = Moran(gdf[col].values, w)
            I = float(moran.I) if np.isfinite(moran.I) else np.nan
            p = float(moran.p_sim)
            entry["data"]["moran_I"] = I
            entry["data"]["moran_p"] = p

            if show_progress:
                # лаконичный статус в прогрессе
                postfix = f"{col}: I={I:.3f}" if np.isfinite(I) else f"{col}: I=nan"
                iterator.set_postfix_str(postfix)

            if plot:
                fig1, ax1 = plt.subplots(figsize=(5, 5))
                moran_scatterplot(moran, ax=ax1, zstandard=True)
                ax1.set_title(f"{col}: Moran's I = {I:.2f} (p={p:.2g})")
                plt.tight_layout()
                entry["plots"]["scatter"] = fig1
                plt.show()

            # 3) LISA — только если Moran достаточно высокий
            if np.isfinite(I) and I >= 0.3:
                lisa = Moran_Local(gdf[col].values, w)
                lisa_name = f"{col}_lisa_cluster"
                cluster = lisa.q.copy()
                cluster[lisa.p_sim > 0.05] = 0
                lisa_cols[lisa_name] = cluster
                entry["data"].update({
                    "lisa_name": lisa_name,
                    "lisa_counts": pd.Series(cluster).value_counts().sort_index().to_dict(),
                })

                if plot:
                    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
                    lisa_cluster(lisa, gdf, p=0.05, ax=ax2, legend=True, markersize=5)
                    ax2.set_title(f"{col}: LISA clusters")
                    plt.tight_layout()
                    entry["plots"]["lisa"] = fig2
                    plt.show()
            else:
                entry["data"]["lisa_name"] = None

            artifacts[col] = entry

        # одно присоединение вместо множества вставок
        if lag_cols:
            gdf = gdf.join(pd.DataFrame(lag_cols, index=gdf.index))
        if lisa_cols:
            gdf = gdf.join(pd.DataFrame(lisa_cols, index=gdf.index))

        # дефрагментация фрейма (ускоряет последующие операции)
        gdf = gdf.copy()

        return gdf, artifacts



class SpatialStatisticsComputer:
    def __init__(self, df):
        self.df = df.to_crs(3857)

    def compute_moran_and_lisa(self, col="storey", k=10):
        w = libpysal.weights.KNN.from_dataframe(self.df, k=k, silence_warnings=True)
        w.transform = "r"
        y = self.df[col]
        lisa = Moran_Local(y, w, n_jobs=8)
        global_moran = Moran(y, w)

        self.df["lisa_cluster"] = lisa.q
        self.df["pval"] = lisa.p_sim
        self.df["sig"] = (self.df["pval"] < 0.05).astype(int)
        self.df["lisa_cluster"] *= self.df["sig"]

        cluster_labels = {
            0: "Not Significant",
            1: "High-High",
            2: "Low-High",
            3: "Low-Low",
            4: "High-Low",
        }

        self.df["cluster_name"] = self.df["lisa_cluster"].map(cluster_labels).fillna("Not Significant")

        self.df["storey_lag"] = lag_spatial(w, self.df["storey"])
        # one-hot encode
        dummies = pd.get_dummies(
            self.df["cluster_name"],
            prefix="cluster",  # column names: cluster_0, cluster_1, …
            dtype=int,
        )

        # 4. stitch back onto your GeoDataFrame
        self.df = pd.concat([self.df, dummies], axis=1).drop(columns=["cluster_name"])


        return self.df, global_moran, lisa


class StoreyModelTrainer:
    def __init__(self, df, target_col="storey"):
        self.df = df
        self.target_col = target_col
        self.model = None
        self.search_results = {}
        self.feature_names_ = None
    
    def prepare_data(self):
        self.df = self.df.dropna().reset_index(drop=True)
        self.df = self.df[(self.df["storey"] > 0) & (self.df["storey"] < 35)]

        y = self.df[self.target_col]
        drop_cols = [self.target_col, "lisa_cluster", "pval", "sig", "geometry", "centroid"]
        X = self.df.drop(columns=drop_cols, errors="ignore")

        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            dummies = pd.get_dummies(X[cat_cols], prefix=cat_cols, dtype=int)
            X = pd.concat([X.drop(columns=cat_cols), dummies], axis=1)

        # ✅ безопасная стратификация
        stratify = None
        if "lisa_cluster" in self.df.columns and self.df["lisa_cluster"].nunique() > 1:
            stratify = self.df["lisa_cluster"]

        return train_test_split(X, y, test_size=0.3, stratify=stratify, random_state=42)


    @staticmethod
    def custom_score_func(y_true, y_pred, rel_thr=0.15, abs_thr=2):
        rel_err = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1)
        abs_err = np.abs(y_pred - y_true)
        ok = (rel_err <= rel_thr) | (abs_err <= abs_thr)
        return np.mean(ok)

    def train_rf(self, X_train, y_train, param_dist=None, scorer_name="Rel15abs2"):
        if param_dist is None:
            param_dist = {
                "n_estimators": [100, 300, 500, 1000],
                "max_depth": [4, 6, 8, 10, 12, None],
                "max_features": [1.0, "sqrt", "log2", 0.5, 0.7],
                "min_samples_split": [2, 4, 8, 16],
                "min_samples_leaf": [1, 2, 4, 8],
            }

        scorer = make_scorer(self.custom_score_func, rel_thr=0.15, abs_thr=2, greater_is_better=True)
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)

        # ✅ FIX: корректная логика выбора поиска и n_iter
        try:
            space_size = len(ParameterGrid(param_dist))  # получится, если значения — конечные списки
            finite_space = True
        except Exception:
            space_size = None
            finite_space = False  # например, распределения scipy.stats

        if finite_space and space_size <= 1:
            search = GridSearchCV(
                rf, param_grid=param_dist, cv=5, scoring=scorer, n_jobs=-1, verbose=0  # ✅ verbose=0
            )
        else:
            n_iter = 20 if not finite_space else min(20, space_size)
            search = RandomizedSearchCV(
                rf,
                param_distributions=param_dist,
                n_iter=n_iter,
                cv=5,
                scoring=scorer,
                n_jobs=-1,
                random_state=42,
                verbose=0,  # ✅ FIX: без лишних сообщений
            )

        search.fit(X_train, y_train)

        self.model = search.best_estimator_
        self.feature_names_ = list(X_train.columns)
        self.search_results[scorer_name] = {
            "best_score": search.best_score_,
            "best_params": search.best_params_,
        }

        return self.model

    def predict(self, df):
        """
        Принимает либо сырой DataFrame/GeoDataFrame, либо уже подготовленный X.
        Автоматически распознаёт и выравнивает признаки под self.feature_names_.
        """
        if self.model is None:
            return None
        if not self.feature_names_:
            raise ValueError("Отсутствует feature_names_. Обучите модель или загрузите её через load_model().")

        # если это уже X с нужными колонками — не делаем OHE повторно
        if isinstance(df, pd.DataFrame) and set(self.feature_names_).issubset(df.columns):
            X = df.reindex(columns=self.feature_names_, fill_value=0)
        else:
            X = self._prepare_features(df, feature_names=self.feature_names_)

        return self.model.predict(X)


    def get_model(self):
        return self.model

    def get_search_results(self):
        return self.search_results
    
    def _prepare_features(self, df, *, feature_names: list[str] | None = None):
        """
        Готовит матрицу X: дропаeт служебные колонки, делает one-hot и,
        при переданном feature_names, выравнивает матрицу под нужный порядок/состав.
        """
        drop_cols = [self.target_col, "lisa_cluster", "pval", "sig", "geometry", "centroid"]
        X = df.drop(columns=drop_cols, errors="ignore")
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            dummies = pd.get_dummies(X[cat_cols], prefix=cat_cols, dtype=int)
            X = pd.concat([X.drop(columns=cat_cols), dummies], axis=1)

        if feature_names is not None:
            # выравнивание под сохранённый порядок/состав
            X = X.reindex(columns=feature_names, fill_value=0)
        return X

    def save_model(self, path: str) -> None:
        """
        Сохраняет обученную модель и метаданные в один .joblib файл.

        Сохраняем:
        - model: обученный estimator
        - target_col: целевая переменная
        - feature_names: порядок/состав признаков, ожидаемый моделью
        - drop_cols: служебные колонки, которые удаляем при подготовке
        - search_results: результаты подбора
        - sklearn_version, timestamp: для аудита
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Вызовите train_rf(...) перед сохранением.")
        if not hasattr(self, "feature_names_"):
            # на всякий случай восстановим из self.df
            X_tmp = self._prepare_features(self.df)
            self.feature_names_ = list(X_tmp.columns)

        payload = {
            "model": self.model,
            "target_col": self.target_col,
            "feature_names": self.feature_names_,
            "drop_cols": [self.target_col, "lisa_cluster", "pval", "sig", "geometry", "centroid"],
            "search_results": getattr(self, "search_results", {}),
            "sklearn_version": sklearn.__version__,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        joblib.dump(payload, path)

    @staticmethod
    def load_model(path: str):
        """
        Загружает модель из .joblib и возвращает объект с методами:
        - predict(df) -> np.ndarray
        - info -> dict метаданных
        Объект не зависит от StoreyModelTrainer и готов к инференсу «как есть».
        """
        payload = joblib.load(path)
        model = payload["model"]
        feature_names = payload["feature_names"]
        target_col = payload["target_col"]
        drop_cols = payload.get("drop_cols", [target_col, "lisa_cluster", "pval", "sig", "geometry", "centroid"])

        # маленький обёрточный предиктор
        def _prepare(df):
            X = df.drop(columns=drop_cols, errors="ignore")
            cat_cols = X.select_dtypes(include=["object", "category"]).columns
            if len(cat_cols) > 0:
                dummies = pd.get_dummies(X[cat_cols], prefix=cat_cols, dtype=int)
                X = pd.concat([X.drop(columns=cat_cols), dummies], axis=1)
            X = X.reindex(columns=feature_names, fill_value=0)
            return X

        def predict(df):
            X = _prepare(df)
            return model.predict(X)

        info = {
            "target_col": target_col,
            "feature_names": feature_names,
            "search_results": payload.get("search_results", {}),
            "sklearn_version_saved": payload.get("sklearn_version"),
            "timestamp": payload.get("timestamp"),
        }

        return SimpleNamespace(predict=predict, info=info, raw=model)




class ResidualAnalyzer:
    def __init__(self, df, true_col="storey", pred_col="pred_storey", k=10):
        self.df = df
        self.true_col = true_col
        self.pred_col = pred_col
        self.k = k

    def analyze(self):
        self.df["residuals"] = self.df[self.true_col] - self.df[self.pred_col]
        w = libpysal.weights.KNN.from_dataframe(self.df, k=self.k)
        w.transform = "r"
        moran = Moran(self.df["residuals"], w)
        return moran.I, moran.p_sim
