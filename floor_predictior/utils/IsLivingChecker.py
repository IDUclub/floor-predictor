# file: floor_predictor/is_living_annotator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

# Правила разметки
POS_BUILDING = {
    "house",
    "apartments",
    "residential",
    "detached",
    "semidetached_house",
    "semidetached",
    "terrace",
    "terraced_house",
    "bungalow",
    "flats",
    "static_caravan",
    "hut",
    "cabin",
}
NEG_BUILDING = {
    "commercial",
    "retail",
    "industrial",
    "warehouse",
    "school",
    "university",
    "public",
    "service",
    "garage",
    "roof",
    "farm_auxiliary",
    "greenhouse",
    "church",
    "religious",
    "hospital",
    "kindergarten",
    "train_station",
    "transportation",
    "hotel",
}
NEG_AMENITY = {
    "school",
    "university",
    "hospital",
    "clinic",
    "public_building",
    "place_of_worship",
    "fire_station",
    "police",
}
NON_RES_LU = {"commercial", "retail", "industrial", "railway", "military", "government"}


@dataclass
class IsLivingAnnotator:
    """
    A class to annotate buildings with is_living using OSM tags fetched for the area that covers input gdf.
    
    This class processes building geometries and determines whether they are residential based on
    OpenStreetMap tags and landuse information. It provides multiple matching strategies and fallback
    mechanisms for accurate classification.
    
    Attributes:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame containing building geometries to annotate
        local_crs (int): Local coordinate reference system to use for processing (default: 3857)
        overpass_endpoint (Optional[str]): Custom Overpass API endpoint URL (default: None)
        timeout (int): Timeout for Overpass API requests in seconds (default: 180)
        match_strategy (Literal["iou", "centroid_within"]): Strategy for matching OSM buildings
            - "iou": Match based on Intersection over Union threshold
            - "centroid_within": Match if centroid of building is within OSM building
        iou_threshold (float): Threshold for IoU matching strategy (default: 0.3)
        fallback_landuse_only (bool): If True, use landuse classification as fallback when
            building-level matching fails (default: True)
    """
    gdf: gpd.GeoDataFrame
    local_crs: int = 3857
    overpass_endpoint: Optional[str] = None
    timeout: int = 180
    match_strategy: Literal["iou", "centroid_within"] = "iou"
    iou_threshold: float = 0.3  # reasonable threshold for intersection
    fallback_landuse_only: bool = True  # if no building matches, classify by landuse

    # internal attributes
    _bounds_geom: BaseGeometry | None = None
    _osm_buildings: gpd.GeoDataFrame | None = None
    _osm_landuse: gpd.GeoDataFrame | None = None

    def annotate(self) -> gpd.GeoDataFrame:
        """
        Main method to annotate buildings with is_living classification.
        
        Returns:
            gpd.GeoDataFrame: Input GeoDataFrame with added 'is_living' column and optionally
                other OSM-derived fields
            
        Process:
            1. Validates input GeoDataFrame
            2. Configures OSMnx settings
            3. Computes bounds for OSM data query
            4. Loads OSM building and landuse data
            5. Transfers OSM tags to input geometries
            6. Classifies buildings as living/non-living
            7. Applies fallback landuse classification if needed
        """
        self._validate_input()
        self._configure_osmnx()

        self._bounds_geom = self._compute_bounds(self.gdf)
        self._load_osm_layers(self._bounds_geom)

        # transfer OSM tags to your geometries
        tagged = self._transfer_tags(self.gdf, self._osm_buildings)

        # classify is_living based on rules + landuse fallback
        out = self._label_is_living(tagged, self._osm_landuse)

        # fallback: if all NaN and fallback mode is enabled, classify only by landuse
        if self.fallback_landuse_only and out["is_living"].isna().all():
            out = self._label_is_living_by_landuse_only(out, self._osm_landuse)

        return out

    # ---------- helper methods ----------

    def _validate_input(self) -> None:
        """
        Validates the input GeoDataFrame.
        
        Raises:
            TypeError: If gdf is not a GeoDataFrame
            ValueError: If gdf doesn't have geometry column or is empty
        """
        if not isinstance(self.gdf, gpd.GeoDataFrame):
            raise TypeError("gdf must be a GeoDataFrame")
        if "geometry" not in self.gdf.columns:
            raise ValueError("gdf must have a 'geometry' column")
        if self.gdf.empty:
            raise ValueError("gdf is empty")

    def _configure_osmnx(self) -> None:
        """
        Configures OSMnx settings including Overpass endpoint and timeout.
        """
        if self.overpass_endpoint:
            ox.settings.overpass_endpoint = self.overpass_endpoint
        ox.settings.overpass_rate_limit = True
        ox.settings.timeout = self.timeout

    def _compute_bounds(self, gdf: gpd.GeoDataFrame) -> BaseGeometry:
        """
        Computes bounding geometry for OSM data query.
        
        Args:
            gdf (gpd.GeoDataFrame): Input GeoDataFrame to compute bounds for
            
        Returns:
            BaseGeometry: Envelope geometry containing all input geometries
        """
        # combined geometry and its bounding box (minimal bbox)
        src = gdf if (gdf.crs and gdf.crs.to_epsg() == 4326) else gdf.to_crs(4326)
        geom = unary_union(src.geometry.values)
        return geom.envelope  # can be replaced with convex_hull if needed

    def _load_osm_layers(self, bounds: BaseGeometry) -> None:
        """
        Loads OSM building and landuse data for the given bounds.
        
        Args:
            bounds (BaseGeometry): Geometry defining the area to query OSM data for
            
        Note:
            Only relevant OSM tags are kept from the raw OSM data:
            - Buildings: osmid, building, building:use, building:usage, amenity, building:levels
            - Landuse: landuse
        """
        # Buildings
        b = ox.features_from_polygon(bounds, {"building": True})
        b = b[b.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        b = b.to_crs(self.local_crs)
        # safely keep only columns that actually exist
        want_b = ["osmid", "building", "building:use", "building:usage", "amenity", "building:levels", "geometry"]
        keep_b = [c for c in want_b if c in b.columns]
        if "geometry" not in keep_b:
            keep_b.append("geometry")
        self._osm_buildings = b[keep_b].reset_index(drop=True)

        # Landuse
        lu = ox.features_from_polygon(bounds, {"landuse": True})
        lu = lu[lu.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        lu = lu.to_crs(self.local_crs)
        keep_lu = [c for c in ("landuse", "geometry") if c in lu.columns]
        if "geometry" not in keep_lu:
            keep_lu.append("geometry")
        self._osm_landuse = lu[keep_lu].reset_index(drop=True)

    def _transfer_tags(self, gdf: gpd.GeoDataFrame, osm_b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Transfers OSM tags to input building geometries using specified matching strategy.
        
        Args:
            gdf (gpd.GeoDataFrame): Input GeoDataFrame to annotate
            osm_b (gpd.GeoDataFrame): OSM buildings data to match against
            
        Matching strategies:
            - 'iou' (default): Finds intersecting pairs and selects best match by IoU >= threshold
            - 'centroid_within': Finds OSM building that contains the centroid of input building
            
        Returns:
            gpd.GeoDataFrame: Input GeoDataFrame with added OSM tags
        """
        # work in local_crs
        a = gdf.copy()
        if a.crs is None or a.crs.to_epsg() != self.local_crs:
            a = a.to_crs(self.local_crs)
        a = a.copy()
        a["__idx"] = np.arange(len(a), dtype=np.int64)

        # If no OSM buildings, return without tags
        if osm_b is None or osm_b.empty:
            return a.drop(columns="__idx")

        # available columns in osm_b
        have_all = [c for c in ["osmid", "building", "building:use", "building:usage", "amenity", "building:levels", "geometry"] if c in osm_b.columns]

        if self.match_strategy == "centroid_within":
            cent = a[["__idx", "geometry"]].copy()
            cent["geometry"] = cent.geometry.centroid
            pairs = gpd.sjoin(cent, osm_b[have_all], how="left", predicate="within")
            # take first match
            best = pairs.drop_duplicates(subset="__idx")
            res = a.merge(best[["__idx", "index_right"]], on="__idx", how="left")

            tag_cols = [c for c in ["osmid", "building", "building:use", "building:usage", "amenity", "building:levels"] if c in osm_b.columns]
            # add only columns that DON'T exist in res to avoid overlap
            join_cols = [c for c in tag_cols if c not in res.columns]
            if join_cols:
                res = res.join(osm_b[join_cols], on="index_right")
            # for overlapping names, keep original values; optionally can fillna from OSM:
            # for c in set(tag_cols).intersection(res.columns):
            #     res[c] = res[c].fillna(res["index_right"].map(osm_b[c]))

            # ensure missing columns exist with NA values
            for c in ["osmid", "building", "building:use", "building:usage", "amenity", "building:levels"]:
                if c not in res.columns:
                    res[c] = pd.NA
            return res.drop(columns=["__idx", "index_right"])

        # IoU matching
        candidates = gpd.sjoin(
            a[["__idx", "geometry"]],
            osm_b[have_all],
            how="left",
            predicate="intersects",
        )
        if candidates.empty:
            return a.drop(columns="__idx")

        left = a.loc[candidates["__idx"]].reset_index(drop=True)
        right = osm_b.loc[candidates["index_right"]].reset_index(drop=True)

        inter = left.geometry.intersection(right.geometry).area
        union = left.geometry.union(right.geometry).area
        iou = (inter / union).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        candidates = candidates.reset_index(drop=True)
        candidates["__iou"] = iou

        best = candidates.sort_values("__iou", ascending=False).groupby("__idx", as_index=False).first()
        best = best[best["__iou"] >= float(self.iou_threshold)]

        res = a.merge(best[["__idx", "index_right"]], on="__idx", how="left")
        tag_cols = [c for c in ["osmid", "building", "building:use", "building:usage", "amenity", "building:levels"] if c in osm_b.columns]

        # add only missing columns to avoid overlap
        join_cols = [c for c in tag_cols if c not in res.columns]
        if join_cols:
            res = res.join(osm_b[join_cols], on="index_right")

        # (optionally can fill existing columns from OSM only where NaN)
        # for c in set(tag_cols).intersection(res.columns):
        #     res[c] = res[c].fillna(res["index_right"].map(osm_b[c]))

        # ensure missing columns exist with NA values
        for c in ["osmid", "building", "building:use", "building:usage", "amenity", "building:levels"]:
            if c not in res.columns:
                res[c] = pd.NA

        return res.drop(columns=["__idx", "index_right"])

    def _label_is_living(self, b: gpd.GeoDataFrame, landuse: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Classifies buildings as living/non-living based on building tags and explicit amenities.
        
        Args:
            b (gpd.GeoDataFrame): Input buildings with OSM tags
            landuse (gpd.GeoDataFrame): OSM landuse data (not used in strict mode)
            
        Classification rules:
            - Missing building tag or 'yes' -> NaN
            - Building in POS_BUILDING -> 1 (living)
            - Building in NEG_BUILDING or amenity in NEG_AMENITY -> 0 (non-living)
            - No fallback to landuse/usage in strict mode
            
        Returns:
            gpd.GeoDataFrame: Input buildings with added 'is_living' column
        """
        out = b.copy()

        # normalize fields
        for k in ["building", "amenity"]:
            if k not in out.columns:
                out[k] = None
            out[k] = out[k].astype("string").str.strip().str.lower()

        # flags
        building = out["building"]
        amenity = out["amenity"]

        building_missing = building.isna() | (building.eq("")) | (building.eq("yes"))
        pos_building = building.isin(POS_BUILDING)
        neg_building = building.isin(NEG_BUILDING)
        neg_amenity = amenity.isin(NEG_AMENITY)

        # base case - NaN
        is_living = pd.Series(np.nan, index=out.index, dtype="float32")

        # NEG takes priority over POS? Usually not, but you can change the order if needed
        is_living.loc[neg_building | neg_amenity] = 0
        is_living.loc[pos_building] = 1
        # missing/yes remains NaN
        is_living.loc[building_missing] = np.nan

        out["is_living"] = is_living.astype("float32")
        return out


    def _label_is_living_by_landuse_only(self, b: gpd.GeoDataFrame, landuse: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Fallback classification using only landuse information.
        
        Args:
            b (gpd.GeoDataFrame): Input buildings to classify
            landuse (gpd.GeoDataFrame): OSM landuse data
            
        Classification rules:
            - 1 if intersects with residential landuse (excluding NON_RES_LU)
            - 0 if intersects with explicit non-residential landuse
            - NaN otherwise
            
        Returns:
            gpd.GeoDataFrame: Input buildings with added 'is_living' column
        """
        out = b.copy()
        if landuse is None or landuse.empty or "landuse" not in landuse.columns:
            out["is_living"] = np.nan
            return out

        lu = landuse.copy()
        lu["landuse"] = lu["landuse"].astype("string").str.lower()

        # residential
        lu_res = lu[lu["landuse"].str.contains("residential", na=False) & (~lu["landuse"].isin(NON_RES_LU))][["geometry"]]
        has_res = np.zeros(len(out), dtype=bool)
        if not lu_res.empty:
            j = gpd.sjoin(out[["geometry"]].assign(_i=np.arange(len(out))), lu_res, how="left", predicate="intersects")
            has_res = j.groupby("_i").size().reindex(range(len(out)), fill_value=0).values > 0

        # non-res
        lu_non = lu[lu["landuse"].isin(NON_RES_LU)][["geometry"]]
        has_non = np.zeros(len(out), dtype=bool)
        if not lu_non.empty:
            j2 = gpd.sjoin(out[["geometry"]].assign(_i=np.arange(len(out))), lu_non, how="left", predicate="intersects")
            has_non = j2.groupby("_i").size().reindex(range(len(out)), fill_value=0).values > 0

        out["is_living"] = np.where(has_res, 1, np.where(has_non, 0, np.nan)).astype("float32")
        return out
