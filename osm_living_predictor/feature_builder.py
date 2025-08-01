import numpy as np
import pandas as pd
from sklearn import preprocessing

class FeatureBuilder:
    def __init__(self, building_processor, road_processor, amenity_processor):
        self.buildings = building_processor.buildings
        self.backup_data = amenity_processor.backup_data.copy()
        self.col_names = [k for k in road_processor.joined_buildings.keys()]
        self.buffers_list = [v['buffered_highway'] for v in road_processor.joined_buildings.values()]
        self.selected_tags = ['source','addr:street','name','building:levels',
                              'roof:shape','amenity','brand','website','shop']
        self.value_tags = ['roof:shape','building:levels']
        self.FULL_FEATURE_LIST = np.array([
            'apartments', 'commercial', 'construction', 'garage',
       'greenhouse', 'house', 'industrial', 'kindergarten', 'misc_buildings',
       'public', 'residential', 'retail', 'roof', 'school', 'service',
       'warehouse', 'yes', 'cemetery_landuse', 'construction_landuse',
       'forest_landuse', 'misc_landuse', 'no_landuse', 'non_res_landuse',
       'residential_landuse'
        ])

    def _categorize_buildings(self, df):

        unique_buildings = np.setxor1d(df['building'].unique(), self.FULL_FEATURE_LIST)
        df['building'] = df['building'].apply(lambda x: 'misc_buildings' if x in unique_buildings else x)

        common_landuse = np.array([
            'non_res', 'no_landuse', 'residential', 'construction',
            'forest', 'recreation_ground', 'cemetery', 'farmyard'
        ])
        unique_landuse = np.setxor1d(df['landuse'].unique(), common_landuse)
        df['landuse'] = df['landuse'].apply(lambda x: 'misc_landuse' if x in unique_landuse else x)

        df['landuse'].replace([
            'residential','non_res','forest','cemetery','recreation_ground',
            'construction','farmyard'
        ],[
            'residential_landuse','non_res_landuse','forest_landuse',
            'cemetery_landuse','recreation_ground_landuse',
            'construction_landuse','farmyard_landuse'
        ], inplace=True)

        return df

    def _nominal_transform(self, df, value_tags):
        ohe = preprocessing.OneHotEncoder(sparse_output=False)
        for tag in value_tags:
            ohe_results = ohe.fit_transform(df[[tag]])
            df = pd.concat([df.reset_index(drop=True), pd.DataFrame(ohe_results, columns=ohe.categories_[0].tolist())], axis=1)
            df.drop(columns=[tag], inplace=True)
        return df

    def build_features(self):
        df = self.backup_data.copy()

        if 'land_building' in df.columns:
            df.drop(columns=['land_building'], inplace=True)
        if 'roof:shape' in df.columns:
            df.drop(columns=['roof:shape'], inplace=True)

        df = self._categorize_buildings(df)
        df = self._nominal_transform(df, ['building','landuse'])
        df = self.align_features(df, expected_features=self.FULL_FEATURE_LIST)
        return df

    def align_features(self, df, expected_features):
        """
        Приводит df к нужному набору признаков: добавляет отсутствующие с 0
        """
        df = df.copy()
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0  # Добавляем отсутствующий столбец

        df.columns = [str(col) for col in df.columns]
        return df