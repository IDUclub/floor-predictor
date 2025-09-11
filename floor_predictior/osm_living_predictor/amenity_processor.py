
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import jenkspy
from dataclasses import dataclass, field
from .constants import POLYGONS

DEFAULT_RADII = [30, 60, 90]


@dataclass
class AmenityProcessor:
    """
    A class for processing amenity data, specifically focusing on parking facilities and their spatial relationships with building features.
    
    This class loads amenity data, classifies parking areas, creates buffers around them, and performs spatial joins with a base layer of buildings or features.
    It's designed to work with GeoDataFrames and utilizes geopandas and osmnx for geospatial operations.
    
    Key Features:
    - Loads amenities from OpenStreetMap within specified bounds
    - Classifies parking areas into bins based on size
    - Creates buffers around classified parking areas
    - Performs spatial joins between parking buffers and base features
    - Optionally cleans building levels data
    
    Constructor Parameters:
    - bounds (Polygon/MultiPolygon): The geographic area to extract amenities from
    - base_gdf (gpd.GeoDataFrame): The base layer of buildings or features to analyze
    - local_crs (int, optional): The local coordinate reference system in meters (default: 3857)
    - radii (list[int], optional): List of buffer radii in meters (default: predefined DEFAULT_RADII)
    
    Example Usage:
    
    Note: The class expects the base_gdf to be in a projected coordinate system or will transform it to local_crs.
    """
    bounds: object                   # Polygon/MultiPolygon
    base_gdf: gpd.GeoDataFrame      # GeoDataFrame зданий/фич (с geometry)
    local_crs: int = 3857
    radii: list[int] = field(default_factory=lambda: DEFAULT_RADII.copy())

    amenities: gpd.GeoDataFrame | None = field(default=None, init=False)
    buffered_layers: dict[str, gpd.GeoDataFrame] = field(default_factory=dict, init=False)
    features: gpd.GeoDataFrame | None = field(default=None, init=False)

    # ---------- публичный сценарный метод ----------
    def run(self) -> gpd.GeoDataFrame:
        """Полный шаг: загрузка -> классификация парковок -> буферы -> джоин -> возврат фичей."""
        self.load_amenities()
        bins = self._classify_parking_bins(n_classes=3)
        self._build_parking_buffers(bins)
        self.features = self._join_parking_buffers()
        self._clean_building_levels_inplace(self.features)   # если колонка есть
        return self.features

    # ---------- загрузка ----------
    def load_amenities(self) -> gpd.GeoDataFrame:
        a = ox.features_from_polygon(self.bounds, {"amenity": True})
        if a.empty:
            # создаём пустой корректный GeoDataFrame
            self.amenities = gpd.GeoDataFrame({"amenity": [], "geometry": []}, geometry="geometry", crs=4326)
            return self.amenities

        # оставим только нужное
        keep = [c for c in ("amenity", "geometry") if c in a.columns]
        if not keep:
            keep = ["geometry"]
            a["amenity"] = np.nan
            keep.insert(0, "amenity")

        a = a[keep].copy()
        a = a.set_geometry("geometry")
        # единожды приводим к метрам
        a = a.to_crs(self.local_crs)
        # валидные геометрии
        a = a[~a.geometry.is_empty & a.geometry.notna()].copy()

        self.amenities = a.rename(columns={"amenity": "amenity_key"})
        return self.amenities

    # ---------- классификация парковок по размерам ----------
    def _classify_parking_bins(self, n_classes: int = 3) -> gpd.GeoDataFrame:
        """Возвращает GeoDataFrame парковок с колонкой 'parking_bin' ∈ {'bin1','bin2','bin3'} (или меньше)."""
        if self.amenities is None:
            raise ValueError("Call load_amenities() first")

        a = self.amenities
        # берём только площадные парковки (Point нам тут не нужен)
        parking = a[(a["amenity_key"].isin(["parking", "parking_space"])) & a.geom_type.isin(POLYGONS)].copy()
        if parking.empty:
            # создаём пустой каркас, downstream код обработает
            parking = gpd.GeoDataFrame({"amenity_key": [], "geometry": []}, geometry="geometry", crs=a.crs)
            parking["parking_area"] = pd.Series(dtype="float64")
            parking["parking_bin"] = pd.Series(dtype="object")
            return parking

        # площадь в м^2 (уже в local_crs)
        parking["parking_area"] = parking.geometry.area

        # защита от маленьких выборок/повторяющихся значений
        uniq_vals = np.unique(parking["parking_area"])
        if len(uniq_vals) < n_classes:
            # fallback: квантильные бины
            q = np.linspace(0, 1, min(len(uniq_vals), n_classes) + 1)
            breaks = parking["parking_area"].quantile(q).to_numpy()
            breaks[0] = parking["parking_area"].min() - 1e-6
            breaks[-1] = parking["parking_area"].max() + 1e-6
        else:
            # Jenks natural breaks
            breaks = np.array(jenkspy.jenks_breaks(parking["parking_area"].to_numpy(), n_classes=n_classes), dtype="float64")
            # подправим крайние границы, чтобы pd.cut включил крайние значения
            breaks[0] -= 1e-6
            breaks[-1] += 1e-6

        labels = [f"bin{i+1}" for i in range(len(breaks) - 1)]
        parking["parking_bin"] = pd.cut(parking["parking_area"], bins=breaks, labels=labels, include_lowest=True)

        return parking

    # ---------- буферизация парковок по бинам ----------
    def _build_parking_buffers(self, parking: gpd.GeoDataFrame) -> None:
        """Строим dissolve‑буферы по каждому бину и радиусу. Сохраняем в self.buffered_layers."""
        self.buffered_layers.clear()
        if parking.empty or parking["parking_bin"].dropna().empty:
            return

        # dissolve по бину, чтобы резко сократить число геометрий перед buffer
        # получим по одной (мульти)геометрии на каждый bin
        grouped = parking.dropna(subset=["parking_bin"]).dissolve(by="parking_bin", as_index=False)[["parking_bin", "geometry"]]

        # буферим и складываем
        for bin_label, geom in zip(grouped["parking_bin"], grouped["geometry"]):
            gdf_bin = gpd.GeoDataFrame({"parking_bin": [bin_label], "geometry": [geom]}, geometry="geometry", crs=parking.crs)
            for r in self.radii:
                key = f"parking_{bin_label}_r{r}"
                buf = gdf_bin.copy()
                buf["geometry"] = buf.buffer(r)
                self.buffered_layers[key] = buf

    # ---------- spatial join буферов с базовой таблицей ----------
    def _join_parking_buffers(self) -> gpd.GeoDataFrame:
        """Делает sjoin со всеми буферами и возвращает GeoDataFrame базовых объектов с фичами near_parking_*."""
        b = self.base_gdf
        if not isinstance(b, gpd.GeoDataFrame):
            raise TypeError("base_gdf must be a GeoDataFrame with a valid geometry column")
        if b.crs is None or b.crs.to_epsg() != self.local_crs:
            b = b.to_crs(self.local_crs)

        out = b.copy()
        out["bix"] = np.arange(len(out), dtype=np.int64)

        # если буферов нет — просто вернём исходную таблицу
        if not self.buffered_layers:
            return out.drop(columns=["bix"])

        for key, buf in self.buffered_layers.items():
            # быстрый sjoin: берём только необходимые поля
            j = gpd.sjoin(out[["bix", "geometry"]], buf[["geometry"]], how="left", predicate="intersects")

            # флаг: пересеклось ли с буфером
            flag = j["index_right"].notna().astype("int8")
            # агрегируем (если одна запись попала в несколько полигонов буфера — все равно 1)
            f = flag.groupby(j["bix"], sort=False).max()
            col = f"near_{key}"   # например: near_parking_bin1_r30

            # присоединяем
            out = out.join(f.rename(col), on="bix")
            out[col] = out[col].fillna(0).astype("int8")

        return out.drop(columns=["bix"])

    # ---------- чистка этажности (опционально, если колонка присутствует) ----------
    @staticmethod
    def _clean_building_levels_inplace(df: pd.DataFrame, col: str = "building:levels") -> None:
        if col not in df.columns:
            return
        s = df[col].astype("string")
        # извлекаем первое число (можно взять max, но первое обычно корректнее для формата '3;4')
        num = s.str.extract(r"(\d+(?:[.,]\d+)?)", expand=False).str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(num, errors="coerce").clip(lower=0, upper=60).fillna(0).astype("float32")
