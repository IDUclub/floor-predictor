import numpy as np
import pandas as pd

from ..utils.constants import ALL_TAGS


class FeatureBuilder:
    """
    A class for building final feature sets for machine learning models from geospatial data.
    
    This class processes a DataFrame containing numerical/binary features (e.g., near_*) along with 
    building and landuse categorization columns to create a comprehensive feature set. It handles 
    categorical encoding, feature alignment, and ensures consistent feature output.
    
    Core functionality:
    - Categorizes rare building types into 'misc_buildings'
    - Processes landuse values with common mappings and renames
    - Performs one-hot encoding on categorical features
    - Aligns final features to a predefined feature list
    
    Example usage:
    
    Parameters:
        base_features_gdf (pd.DataFrame): A DataFrame containing base features. Expected to have 
            numerical/binary features and 'building'/'landuse' columns for categorization.
            
    Notes:
        - The input DataFrame should be preprocessed to include numerical/binary features
        - Rare building types are grouped into 'misc_buildings'
        - Uncommon landuse values are mapped to 'misc_landuse'
        - The final feature set is aligned with FULL_FEATURE_LIST
    """

    def __init__(self, base_features_gdf: pd.DataFrame):
        """
        Initialize the FeatureBuilder with a base DataFrame containing features.
        
        Parameters:
            base_features_gdf (pd.DataFrame): Input DataFrame containing geospatial features.
                Should include numerical/binary features and building/landuse columns.
        """
        # это должен быть df/gdf, возвращённый, например, AmenityProcessor.run()
        self.df_base = base_features_gdf.copy()

        # Фиксированный список финальных бинарных фич по building/landuse
        self.FULL_FEATURE_LIST = np.array([
            # building=*
            "apartments", "commercial", "construction", "garage",
            "greenhouse", "house", "industrial", "kindergarten", "misc_buildings",
            "public", "residential", "retail", "roof", "school", "service",
            "warehouse", "yes",
            # landuse=*
            "cemetery_landuse", "construction_landuse",
            "forest_landuse", "misc_landuse", "no_landuse",
            "non_res_landuse", "residential_landuse",
        ])

        # Частые landuse, которые маппим в *_landuse
        self._COMMON_LANDUSE = np.array([
            "non_res", "no_landuse", "residential", "construction",
            "forest", "recreation_ground", "cemetery", "farmyard",
        ])

        # Маппинг для переименования landuse -> *_landuse
        self._LANDUSE_RENAME = {
            "residential": "residential_landuse",
            "non_res": "non_res_landuse",
            "forest": "forest_landuse",
            "cemetery": "cemetery_landuse",
            "recreation_ground": "recreation_ground_landuse",
            "construction": "construction_landuse",
            "farmyard": "farmyard_landuse",
            "no_landuse": "no_landuse",
        }

    # ---------- ВНУТРЕННИЕ ХЕЛПЕРЫ ----------

    def _categorize_buildings_landuse(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize rare building types and landuse values.
        
        Parameters:
            df (pd.DataFrame): Input DataFrame with building and landuse columns.
            
        Returns:
            pd.DataFrame: DataFrame with categorized building and landuse values.
                Rare building types are grouped into 'misc_buildings'.
                Rare landuse values are grouped into 'misc_landuse'.
        """
        df = df.copy()

        # building: редкие в misc_buildings
        if "building" in df.columns:
            unique_buildings = np.setxor1d(df["building"].astype("string").unique(), self.FULL_FEATURE_LIST)
            df["building"] = df["building"].astype("string").apply(
                lambda x: "misc_buildings" if x in unique_buildings else x
            )

        # landuse: редкие в misc_landuse, потом переименовать в *_landuse
        if "landuse" in df.columns:
            lu = df["landuse"].astype("string")
            unique_landuse = np.setxor1d(lu.unique(), self._COMMON_LANDUSE)
            lu = lu.apply(lambda x: "misc_landuse" if x in unique_landuse else x)
            df["landuse"] = lu.replace(self._LANDUSE_RENAME)

        return df

    def _ohe_nominal(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """
        Perform one-hot encoding on nominal columns using pandas.get_dummies.
        
        Parameters:
            df (pd.DataFrame): Input DataFrame containing columns to encode.
            cols (list[str]): List of column names to one-hot encode.
            
        Returns:
            pd.DataFrame: DataFrame with one-hot encoded columns.
                Original columns are dropped and replaced with their encoded versions.
        """
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return df

        # get_dummies за один вызов — быстрее, чем по одному столбцу
        ohe = pd.get_dummies(df[cols].astype("string"), prefix="", prefix_sep="", dtype="int8")
        df = df.drop(columns=cols)
        # align по индексу — безопасно
        df = df.join(ohe)
        return df

    def _align_features(self, df: pd.DataFrame, expected_features: np.ndarray) -> pd.DataFrame:
        """
        Align features with expected feature list by adding missing features with zero values.
        
        Parameters:
            df (pd.DataFrame): Input DataFrame to align.
            expected_features (np.ndarray): Array of expected feature names.
            
        Returns:
            pd.DataFrame: DataFrame aligned with expected features.
                Missing features are added with zero values.
        """
        df = df.copy()
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0
        # имена столбцов строками
        df.columns = [str(c) for c in df.columns]
        return df

    # ---------- ПУБЛИЧНЫЕ МЕТОДЫ ----------

    def build_features(self) -> pd.DataFrame:
        """
        Build and return the final feature set.
        
        This method processes the base DataFrame through several steps:
        1. Drops unnecessary columns ('all_tag_keys', 'all_tags')
        2. Categorizes building and landuse values
        3. Performs one-hot encoding on categorical features
        4. Aligns features with the predefined feature list
        
        Returns:
            pd.DataFrame: Final feature set with aligned binary features.
        Возвращает таблицу признаков:
          - категоризация building/landuse (редкие -> misc_*),
          - OHE для ['building','landuse'],
          - добавление отсутствующих фич из FULL_FEATURE_LIST
        """
        df = self.df_base.copy()

        # - 'all_tag_keys', 'all_tags' — служебные словари/списки тегов, модели не нужны
        to_drop = [c for c in ALL_TAGS if c in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)

        # категоризация
        df = self._categorize_buildings_landuse(df)

        # OHE для двух столбцов (если есть)
        df = self._ohe_nominal(df, ["building", "landuse"])

        # привести к нужному набору бинарных колонок (добавить отсутствующие нулями)
        df = self._align_features(df, expected_features=self.FULL_FEATURE_LIST)

        return df
