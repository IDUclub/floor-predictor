from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import base as shp_base


RoadClass = Literal["residential", "primary", "secondary_tertiary", "motorway_trunk", "service"]


def classify_roads(roads: gpd.GeoDataFrame, highway_col: str = "highway") -> gpd.GeoDataFrame:
    """
    Classifies roads into different categories based on their highway type.
    
    This function takes a GeoDataFrame containing road data and adds a new column 
    'road_class' that categorizes roads into predefined classes. The original 
    highway column remains unchanged in the output.
    
    Parameters
    ----------
    roads : gpd.GeoDataFrame
        A GeoDataFrame containing road data with a highway column specifying road types.
    highway_col : str, optional
        The name of the column containing highway/road type information (default is "highway").
        
    Returns
    -------
    gpd.GeoDataFrame
        A copy of the input GeoDataFrame with an additional 'road_class' column.
        The 'road_class' values are strings from the RoadClass enumeration.
        
    Notes
    -----
    The function handles cases where the highway value might be a list, tuple, or set
    by taking the first element. If no matching class is found, roads are classified
    as "service" by default.
    
    Road classification rules:
    - "residential" and "living_street" -> "residential"
    - "primary" -> "primary"
    - "secondary" and "tertiary" -> "secondary_tertiary"
    - "motorway" and "trunk" -> "motorway_trunk"
    - "service" -> "service"
    - All other types -> "service" (default)
    """
    rd = roads.copy()

    def _class(v) -> RoadClass:
        # osmnx may return list/tuple values; take the first one
        if isinstance(v, (list, tuple, set)):
            v = next(iter(v), None)
        if v in {"residential", "living_street"}:
            return "residential"
        if v in {"primary"}:
            return "primary"
        if v in {"secondary", "tertiary"}:
            return "secondary_tertiary"
        if v in {"motorway", "trunk"}:
            return "motorway_trunk"
        if v in {"service"}:
            return "service"
        return "service"  # fallback to the most basic class

    rd["road_class"] = rd[highway_col].map(_class)
    return rd


def _parse_building_levels(val) -> float | np.nan:
    """
    Parse building levels from various input formats.
    
    This function attempts to convert input values to a float representing
    the number of floors in a building. It handles various string formats
    and returns NaN for invalid inputs.
    
    Args:
        val: Input value to be parsed, can be of any type
        
    Returns:
        float | np.nan: 
            - If parsing is successful and value is within valid range (0, 60]:
              returns the parsed float value
            - If input is NaN or cannot be parsed to a valid number:
              returns np.nan
            - If parsed value is outside valid range (0, 60]:
              returns np.nan
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace(",", ".")
    try:
        x = float(s)
        # допустимый диапазон этажей
        if 0 < x <= 60:
            return x
    except ValueError:
        pass
    return np.nan


@dataclass
class RoadProcessor:
    """
    A class for processing road data and joining it with building data based on proximity.
    
    Attributes:
        bounds (shp_base.BaseGeometry): The geographic boundary for data processing.
        buildings (gpd.GeoDataFrame): The building data to be processed.
        local_crs (int): The local coordinate reference system (default: 3857).
        radius_list (Iterable[int]): List of radii for buffering roads (default: (30, 60, 90)).
        roads (gpd.GeoDataFrame | None): Processed road data (default: None).
        buffered_roads (dict[str, gpd.GeoDataFrame]): Dictionary of buffered roads by radius.
        joined_buildings (dict[str, gpd.GeoDataFrame]): Dictionary of buildings joined with road buffers.
        backup_data (pd.DataFrame | None): Final processed feature table (default: None).
    """
    bounds: shp_base.BaseGeometry
    buildings: gpd.GeoDataFrame
    local_crs: int = 3857
    radius_list: Iterable[int] = (30, 60, 90)

    roads: gpd.GeoDataFrame | None = field(default=None, init=False)
    buffered_roads: dict[str, gpd.GeoDataFrame] = field(default_factory=dict, init=False)
    joined_buildings: dict[str, gpd.GeoDataFrame] = field(default_factory=dict, init=False)
    backup_data: pd.DataFrame | None = field(default=None, init=False)

    def load_roads(self) -> gpd.GeoDataFrame:
        """
        Load road data (lines only) within the specified bounds.
        
        Converts the data to the local CRS and performs basic cleaning.
        
        Returns:
            gpd.GeoDataFrame: Processed road data with essential columns.
        """
        r = ox.features_from_polygon(self.bounds, {"highway": True})
        r = r[r.geom_type.isin(["LineString", "MultiLineString"])].copy()
        r = r.reset_index(drop=False)
        r = r.to_crs(self.local_crs)
        # оставим только полезные колонки, если есть
        keep = [c for c in ("id", "osmid", "highway", "geometry") if c in r.columns]
        if len(keep) < 2:
            keep = ["highway", "geometry"]
        self.roads = r[keep].copy()
        return self.roads

    def _buffer_roads(self) -> None:
        """
        Classify and buffer roads by specified radii.
        
        Stores the buffered roads in self.buffered_roads.
        
        Raises:
            ValueError: If roads haven't been loaded yet.
        """
        if self.roads is None:
            raise ValueError("Call load_roads() first")

        rd = classify_roads(self.roads)

        for radius in self.radius_list:
            # Буферим все дороги, затем dissolve по road_class, чтобы получить одну зону на класс
            buf = rd.copy()
            buf["geometry"] = buf.geometry.buffer(radius)
            buf = buf.dissolve(by="road_class", as_index=False)
            key = f"r{radius}"
            self.buffered_roads[key] = buf

    def _join_buildings_to_buffers(self) -> dict[str, gpd.GeoDataFrame]:
        """
        Perform spatial join between buildings and road buffers for each radius.
        
        Returns:
            dict[str, gpd.GeoDataFrame]: Dictionary with keys 'r{radius}' and values as building tables
                with flag columns indicating proximity to different road classes.
        """
        if not self.buffered_roads:
            self._buffer_roads()

        # гарантируем нужную СК
        b = self.buildings
        if b.crs is None or b.crs.to_epsg() != self.local_crs:
            b = b.to_crs(self.local_crs)

        # базовые столбцы, которые протаскиваем (если они есть)
        base_keep = [c for c in (
            "geometry", "id", "building", "landuse", "land_building",
            "all_tag_keys", "all_tags"
        ) if c in b.columns]

        # «скелет» + собственный ключ строк
        base = b[base_keep].copy()
        base["bix"] = np.arange(len(base), dtype=np.int64)

        # список всех возможных классов дорог (фиксируем схему)
        all_rc = ["residential", "primary", "secondary_tertiary", "motorway_trunk", "service"]

        results: dict[str, gpd.GeoDataFrame] = {}

        for key, buf in self.buffered_roads.items():
            # spatial join: для каждого здания — попало ли оно в буфер того или иного road_class
            j = gpd.sjoin(
                base[["bix", "geometry"]],       # только нужные поля для join
                buf[["road_class", "geometry"]],
                how="left",
                predicate="intersects",
            )

            # one-hot по road_class и агрегация по зданию
            d = pd.get_dummies(j["road_class"], dtype="int8")
            # добьём отсутствующие классы, чтобы схема была стабильной
            for rc in all_rc:
                if rc not in d.columns:
                    d[rc] = 0
            d = d[all_rc]
            d = d.groupby(j["bix"], sort=False).max()

            # финальные имена колонок
            d.columns = [f"near_{rc}_{key}" for rc in d.columns]

            # присоединяем к базе по "bix"
            out = base.join(d, on="bix")

            # NaN -> 0 (если здание не попало ни в один буфер)
            new_cols = list(d.columns)
            out[new_cols] = out[new_cols].fillna(0).astype("int8")

            # убираем служебный ключ, индекс оставляем как есть (или делаем RangeIndex по желанию)
            out = out.drop(columns=["bix"]).reset_index(drop=True)

            results[key] = out

        self.joined_buildings = results
        return results

    def build_feature_table(
        self,
        *,
        onehot_tag_keys: list[str] = ("source", "addr:street", "name", "amenity", "brand", "website", "shop"),
        value_tags: list[str] = ("roof:shape", "building:levels"),
    ) -> pd.DataFrame:
        """
        Build a unified feature table using the smallest radius and add all near_* flags from all radii.
        
        Processes building levels as numeric features and performs final data cleaning.
        
        Args:
            onehot_tag_keys: List of tag keys to one-hot encode.
            value_tags: List of tag keys to extract as values.
            
        Returns:
            pd.DataFrame: Final processed feature table with geographic data.
        """
        if not self.joined_buildings:
            self._join_buildings_to_buffers()

        keys = list(self.joined_buildings.keys())
        base_key = keys[0]  # первый радиус как базовый набор строк
        base = self.joined_buildings[base_key].copy()

        # площадь полигона (в метрах^2)
        base["area"] = base.geometry.area

        # one-hot по наличию ключей в all_tag_keys (ожидаем set/list)
        if "all_tag_keys" in base.columns:
            def has_key_set(keys, k):  # keys может быть list, делаем set один раз
                if isinstance(keys, set):
                    return k in keys
                if isinstance(keys, (list, tuple)):
                    return k in set(keys)
                return False

            for k in onehot_tag_keys:
                base[f"tag_{k}"] = [1 if has_key_set(ks, k) else 0 for ks in base["all_tag_keys"]]
        # value‑теги: вытаскиваем значения из словаря all_tags
        if "all_tags" in base.columns:
            for k in value_tags:
                base[k] = base["all_tags"].map(lambda d: d.get(k) if isinstance(d, dict) else np.nan)

        # этажность -> числовая
        if "building:levels" in base.columns:
            base["building:levels"] = base["building:levels"].map(_parse_building_levels)

        # добавляем все флаги near_* со всех радиусов
        # (ключи в joined_buildings имеют одинаковый набор строк и порядок, так как основаны на base)
        for k in keys:
            flags = [c for c in self.joined_buildings[k].columns if c.startswith("near_")]
            for c in flags:
                base[c] = self.joined_buildings[k][c].astype("int8")

        # финальная чистка, выброс служебных полей
        drop_cols = [c for c in ("all_tag_keys", "all_tags") if c in base.columns]
        base = base.drop(columns=drop_cols)

        self.backup_data = gpd.GeoDataFrame(base, geometry="geometry", crs=self.local_crs)
        return self.backup_data
