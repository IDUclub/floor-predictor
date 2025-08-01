import libpysal
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.spatial import KDTree
from tqdm import tqdm
from esda.moran import Moran_Local, Moran
import matplotlib.pyplot as plt
from libpysal.weights import KNN, lag_spatial
from splot.esda import moran_scatterplot, lisa_cluster


class BuildingPreprocessor:
    def __init__(self, df):
        self.df = df.copy()

    def filter_residential(self):
        # Отбор только жилых зданий и этажей > 2
        self.df = self.df[self.df["is_living"] == 1].copy()
        self.df = self.df[self.df["storey"] > 2].copy()
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
                if geom.type == "Polygon"
                else sum(
                    sum(len(ring.coords) for ring in ([g.exterior] + list(g.interiors)))
                    for g in geom.geoms
                )
            )
        )
        df["geom_count"] = df.geometry.apply(
            lambda geom: 1 if geom.type == "Polygon" else len(geom.geoms)
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

    def compute_neighborhood_metrics(self, k_neighbors=10, plot=False):
        df = self.df.copy()
        coords = np.array(df["centroid"].apply(lambda pt: (pt.x, pt.y)).to_list())
        tree = KDTree(coords)
        emda_val = np.sqrt(df["shape_area"].mean()) * 0.5

        results = {
            "n_counta": [],
            "omda": [],
            "nnd": [],
            "nnia": [],
            "intensitya": [],
            "n_size_meana": [],
            "n_size_stda": [],
            "n_size_mina": [],
            "n_size_maxa": [],
            "n_size_cva": [],
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
            results["n_size_cva"].append(np.std(areas) / np.mean(areas) if len(areas) > 0 and np.mean(areas) > 0 else np.nan)

        for k, v in results.items():
            df[k] = v


        geom_features = (
            df.select_dtypes(include="number")
            .drop(
                columns=[
                    "storey",
                    "lisa_cluster",
                    "pval",
                    "sig",
                    "cluster_High-High",
                    "storey_lag",
                    "cluster_Low-Low",
                    "cluster_High-Low",
                    "cluster_Low-High",
                    "cluster_Not Significant",
                    # "geometry",
                ]
            )
            .columns
        )

        df = self.generate_spatial_lags_and_morans(
            df.set_geometry("centroid"), geom_features, plot=plot, k_neighbors=k_neighbors
        )

        self.df = df
        return self.df
    
    def generate_spatial_lags_and_morans(gdf, feature_columns, k_neighbors=10, plot=True):
        """
        Генерирует spatial-lag признаки и считает Moran's I/LISA для заданных признаков.
        Рисует scatterplot глобального Moran’s I и карту LISA кластеров для каждого признака.

        Parameters
        ----------
        gdf : GeoDataFrame
            Исходный GeoDataFrame (geometry обязателен!)
        feature_columns : list of str
            Список названий признаков для spatial lag и Moran's/LISA
        k_neighbors : int
            Число ближайших соседей для KNN weights
        plot : bool
            Рисовать ли графики
        lisa_to_feature : bool
            Добавлять ли LISA кластеры как новую фичу

        Returns
        -------
        gdf : GeoDataFrame
            Исходный GeoDataFrame + spatial lag и, опционально, LISA кластеры как новые колонки
        """

        # Проекция в метры для корретного поиска соседей
        gdf_meter = gdf.to_crs(epsg=3857)
        w = KNN.from_dataframe(gdf_meter, k=k_neighbors)
        w.transform = "r"

        for col in feature_columns:
            print(f"\nProcessing feature: {col}")
            # 1. Spatial lag
            lag_name = f"{col}_lag"
            gdf[lag_name] = lag_spatial(w, gdf[col])

            # 2. Глобальный Moran's I
            moran = Moran(gdf[col].values, w)
            print(
                f"\n--- {col} ---\n"
                f"Global Moran's I: {moran.I:.4f}, {type(moran.I)}\n"
                f"p-value: {moran.p_sim:.4e}\n"
            )
            if np.isnan(moran.I) or moran.I < 0.3:
                continue

            if plot:
                fig, ax = plt.subplots(figsize=(5, 5))
                moran_scatterplot(moran, ax=ax, zstandard=True)
                ax.set_title(f"{col}: Moran's I = {moran.I:.2f} (p={moran.p_sim:.2g})")
                plt.tight_layout()
                plt.show()

            # 3. Локальный Moran's I (LISA)
            lisa = Moran_Local(gdf[col].values, w)

            # 4. LISA clusters (1=HH, 2=LL, 3=LH, 4=HL)
            lisa_name = f"{col}_lisa_cluster"
            # По умолчанию: not significant=0, HH=1, LL=2, LH=3, HL=4
            # 📌 Вычисляем локальную автокорреляцию

            cluster = lisa.q.copy()
            cluster[lisa.p_sim > 0.05] = 0
            gdf[lisa_name] = cluster

            if plot:
                fig, ax = plt.subplots(1, 1, figsize=(7, 6))
                lisa_cluster(lisa, gdf, p=0.05, ax=ax, legend=True, markersize=5)
                ax.set_title(f"{col}: LISA clusters")
                plt.tight_layout()
                plt.show()

            # Покажи count по кластерам для каждой фичи
            print(
                f"{col} LISA cluster counts:\n{gdf[lisa_name].value_counts().sort_index()}\n"
            )

        return gdf


class SpatialStatisticsComputer:
    def __init__(self, df):
        self.df = df.to_crs(3857)

    def compute_moran_and_lisa(self, col="storey", k=10):
        w = libpysal.weights.KNN.from_dataframe(self.df, k=k)
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
    
    def prepare_data(self):
        self.df = self.df.dropna().reset_index(drop=True)
        self.df = self.df[(self.df["storey"] > 0) & (self.df["storey"] < 35)]

        # Выделяем целевую переменную
        y = self.df[self.target_col]

        # Удаляем неинформативные колонки
        drop_cols = [self.target_col, "lisa_cluster", "pval", "sig", "geometry", "centroid"]
        X = self.df.drop(columns=drop_cols, errors="ignore")

        # Преобразуем строковые признаки в one-hot
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            dummies = pd.get_dummies(X[cat_cols], prefix=cat_cols, dtype=int)
            X = pd.concat([X.drop(columns=cat_cols), dummies], axis=1)

        return train_test_split(X, y, test_size=0.3, stratify=self.df["lisa_cluster"], random_state=42)

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
                "max_features": ["auto", "sqrt", "log2", 0.5, 0.7, 1.0],
                "min_samples_split": [2, 4, 8, 16],
                "min_samples_leaf": [1, 2, 4, 8],
            }

        scorer = make_scorer(
            self.custom_score_func,
            rel_thr=0.15,
            abs_thr=2,
            greater_is_better=True,
        )

        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        search = RandomizedSearchCV(
            rf,
            param_distributions=param_dist,
            n_iter=20,
            cv=5,
            scoring=scorer,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)

        self.model = search.best_estimator_
        self.search_results[scorer_name] = {
            "best_score": search.best_score_,
            "best_params": search.best_params_,
        }

        print(f"[{scorer_name}] Best score: {search.best_score_:.4f}")
        print(f"[{scorer_name}] Best parameters: {search.best_params_}")
        return self.model

    def predict(self, X):
        return self.model.predict(X) if self.model else None

    def get_model(self):
        return self.model

    def get_search_results(self):
        return self.search_results


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
