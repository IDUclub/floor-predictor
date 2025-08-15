from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import geopandas as gpd
import osmnx as ox

try:
    # shapely >= 2.0
    from shapely import make_valid as _make_valid
except Exception:  # fallback для старого shapely
    _make_valid = None

from shapely.ops import unary_union
from shapely.geometry.base import BaseGeometry


@dataclass
class BoundaryFetcher:
    """
    A utility class for retrieving geographical boundaries (Polygon/MultiPolygon) 
    from OpenStreetMap data using either OSM IDs or place names.
    
    This class provides methods to fetch administrative boundaries, city limits, 
    and other geographical areas by their OSM identifiers or names. The retrieved 
    geometries can be transformed to a specified coordinate reference system (CRS).
    
    Core functionality:
        - by_osmid: Fetch boundary using OpenStreetMap ID
        - by_name: Fetch boundary using place name
    
    Parameters:
        target_crs (int | str | None): Target coordinate reference system (CRS) 
            for the output geometry (e.g., 4326 or 3857). If None, preserves the 
            original CRS from OSM.
        which_result (int | None): Index of the result when multiple results are 
            returned by a name query. If None, selects the largest polygon by area.
    """
    target_crs: int | str | None = 4326
    which_result: int | None = None

    def by_osmid(self, osmid: int | str, osmid_type: Literal["R", "W", "N"] = "R") -> BaseGeometry:
        """
        Retrieve boundary geometry by OSM ID.
        
        This method can accept either a numeric OSM ID with an explicit type,
        or a complete OSM ID string including the type prefix.
        
        Parameters
        ----------
        osmid : int | str
            The OSM ID as either a number or a string (with or without type prefix)
        osmid_type : Literal["R", "W", "N"], optional
            The OSM element type when osmid is provided as a number:
            'R' for relation (default), 'W' for way, 'N' for node
            
        Returns
        -------
        BaseGeometry
            The geometry object representing the boundary
            
        Notes
        -----
        The method will format the query string appropriately based on input parameters,
        query the OSM database using osmnx, and return the best matching geometry.
        
        Examples
        --------
        >>> by_osmid(12345)  # Default relation type
        >>> by_osmid("W12345")  # Explicit way type
        >>> by_osmid(12345, "N")  # Explicit node type
        """
        q = str(osmid).strip()
        if not q or q[0] not in {"R", "W", "N"}:
            q = f"{osmid_type}{q}"

        gdf = ox.geocode_to_gdf(q, by_osmid=True)
        geom = self._pick_best_geometry(gdf)
        return self._finalize(geom, gdf.crs)

    def by_name(self, name: str) -> BaseGeometry:
        """
        Retrieve boundary geometry by place name (city, district, etc.).
        If multiple results are found, uses which_result or selects the largest polygon.
        
        Parameters
        ----------
        name : str
            The name of the place to search for
            
        Returns
        -------
        BaseGeometry
            The geometry object representing the boundary
            
        Raises
        ------
        ValueError
            If no results are found for the given name
            
        Notes
        -----
        The method queries the OSM database using osmnx and returns the best
        matching geometry based on the which_result parameter or polygon area.
        
        Examples
        --------
        >>> by_name("Paris")
        >>> by_name("Central Park, New York")
        """
        gdf = ox.geocode_to_gdf(name)
        if gdf.empty:
            raise ValueError(f"No results for name='{name}'")

        if self.which_result is not None:
            if not (0 <= self.which_result < len(gdf)):
                raise IndexError(
                    f"which_result={self.which_result} is out of range for {len(gdf)} results"
                )
            gdf = gdf.iloc[[self.which_result]]

        geom = self._pick_best_geometry(gdf)
        return self._finalize(geom, gdf.crs)

    # -------- helpers --------

    def _pick_best_geometry(self, gdf: gpd.GeoDataFrame) -> BaseGeometry:
        """
        Select and prepare the best geometry from a GeoDataFrame.
        
        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            The GeoDataFrame containing candidate geometries
            
        Returns
        -------
        BaseGeometry
            The selected and validated geometry
            
        Raises
        ------
        ValueError
            If no valid geometries are found
            
        Notes
        -----
        The method prefers Polygon and MultiPolygon geometries, selecting the
        largest by area. Falls back to combining all geometries if no polygons
        are found. Applies geometry validation to fix potential issues.
        """
        if gdf.empty:
            raise ValueError("No geometry returned from geocoder")

        geoms = gdf.geometry.dropna()
        if geoms.empty:
            raise ValueError("Returned geometries are empty/NaN")

        # Prefer (Multi)Polygon
        polys = geoms[geoms.geom_type.isin(["Polygon", "MultiPolygon"])]
        if not polys.empty:
            # Select the largest by area (in meters)
            tmp = gpd.GeoSeries(polys, crs=gdf.crs).to_crs(3857)
            idx = tmp.area.fillna(0).idxmax()
            geom = geoms.loc[idx]
        else:
            # Fallback: combine all available geometries (may be lines/points)
            geom = unary_union(list(geoms))

        # Make geometry valid if possible
        if _make_valid is not None:
            try:
                geom = _make_valid(geom)
            except Exception:
                pass
        else:
            # Old trick for fixing self-intersections
            try:
                geom = geom.buffer(0)
            except Exception:
                pass
        return geom

    def _finalize(self, geom: BaseGeometry, src_crs) -> BaseGeometry:
        """
        Finalize the geometry by setting CRS and transforming if needed.
        
        Parameters
        ----------
        geom : BaseGeometry
            The geometry to finalize
        src_crs : Any
            The source coordinate reference system
            
        Returns
        -------
        BaseGeometry
            The geometry with proper CRS
            
        Notes
        -----
        Creates a GeoSeries from the geometry, sets the source CRS,
        and transforms to the target CRS if specified.
        """
        gs = gpd.GeoSeries([geom], crs=src_crs if src_crs else 4326)
        if self.target_crs is not None:
            gs = gs.to_crs(self.target_crs)
        return gs.iloc[0]
