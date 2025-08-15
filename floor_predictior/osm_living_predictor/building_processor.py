import osmnx as ox
import pandas as pd
import geopandas as gpd
import numpy as np

class BuildingProcessor:
    """
    A class for processing building and landuse data from OpenStreetMap.
    
    This class provides functionality to download, process, and merge building and landuse data
    within specified geographical bounds. It uses OSMnx for data retrieval and GeoPandas for
    spatial operations.
    
    Core functionality:
    - load_buildings(): Downloads and processes building polygons from OpenStreetMap
    - load_landuse(): Downloads and classifies landuse polygons
    - merge_landuse(): Merges landuse data with building data based on spatial intersection
    
    Parameters:
        bounds: The geographical bounds (minx, miny, maxx, maxy) to extract data within
        local_crs (int, optional): The local coordinate reference system to use for processing. 
                                  Defaults to 3857 (Web Mercator).
        requests_timeout (int, optional): Timeout in seconds for HTTP requests to Overpass API. 
                                         Defaults to 180 seconds.
        overpass_endpoint (str | None, optional): Custom Overpass API endpoint URL. 
                                               If None, uses the default OSMnx endpoint.
    """
    
    def __init__(
        self,
        bounds,
        local_crs: int = 3857,
        requests_timeout: int = 180,             # ← больше времени на запрос
        overpass_endpoint: str | None = None,    # ← можно указать альтернативный эндпоинт
    ):
        self.bounds = bounds
        self.buildings: gpd.GeoDataFrame | None = None
        self.landuse: gpd.GeoDataFrame | None = None
        self.local_crs = local_crs
        self.requests_timeout = requests_timeout
        self.overpass_endpoint = overpass_endpoint


    # --- настройки OSMnx ---
    def _configure_osmnx(self) -> None:
        if self.overpass_endpoint:
            # в разных версиях osmnx поле называлось и так, и так — выставим оба
            if hasattr(ox.settings, "overpass_endpoint"):
                ox.settings.overpass_endpoint = self.overpass_endpoint
            if hasattr(ox.settings, "overpass_url"):
                ox.settings.overpass_url = self.overpass_endpoint
        ox.settings.overpass_rate_limit = True
        ox.settings.requests_timeout = self.requests_timeout # таймаут requests

    def load_buildings(self) -> gpd.GeoDataFrame:
        """Загружает здания по тегу 'building' и подготавливает их."""
        self._configure_osmnx()
        gdf = ox.features_from_polygon(self.bounds, {"building": True})
        gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        gdf.reset_index(inplace=True)

        # Собираем словари всех тегов кроме системных
        skip_cols = {"geometry", "osmid", "element", "id"}
        gdf["all_tags"] = [
            {col: row[col] for col in gdf.columns if col not in skip_cols and pd.notnull(row[col])}
            for _, row in gdf.iterrows()
        ]
        gdf["all_tag_keys"] = gdf["all_tags"].apply(list)

        gdf["building"] = gdf["building"].fillna("")
        gdf = gdf[["id", "building", "geometry", "all_tag_keys", "all_tags"]]

        self.buildings = gdf
        return gdf

    def load_landuse(self) -> gpd.GeoDataFrame:
        """Загружает landuse полигоны и классифицирует non_res."""
        lu = ox.features_from_polygon(self.bounds, {"landuse": True})
        lu = lu[lu.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        lu.reset_index(drop=True, inplace=True)

        non_res = {
            "commercial", "retail", "industrial", "grass", "plant_nursery",
            "quarry", "railway", "government", "institutional"
        }
        lu["landuse"] = lu["landuse"].apply(lambda x: "non_res" if x in non_res else x)

        lu = lu[['geometry', 'landuse']]

        self.landuse = lu
        return lu

    def merge_landuse(self, tags_list: list[str] | None = None) -> gpd.GeoDataFrame:
        """Присоединяет landuse к зданиям и обогащает land_building."""
        if self.buildings is None or self.landuse is None:
            raise ValueError("Сначала вызови load_buildings и load_landuse")

        b = self.buildings.to_crs(self.local_crs)
        lu = self.landuse.to_crs(self.local_crs)

        merged = b.sjoin(lu, how="left", predicate="intersects")
        merged.drop_duplicates(subset="id", inplace=True)
        merged.rename(columns={"index_right": "index_landuse"}, inplace=True)
        merged["landuse"] = merged["landuse"].fillna("no_landuse")

        # land_building: если building == 'yes', берём landuse
        merged["land_building"] = np.where(
            merged["building"] == "yes",
            merged["landuse"],
            merged["building"]
        )

        if tags_list:
            merged["land_building"] = np.select(
                [merged["landuse"] == t for t in tags_list],
                tags_list,
                default=merged["land_building"]
            )

        self.buildings = merged
        return merged
