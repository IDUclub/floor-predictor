import osmnx as ox
import pandas as pd
import geopandas as gpd

class RoadProcessor:
    def __init__(self, bounds, buildings):
        self.bounds = bounds
        self.buildings = buildings
        self.local_crs = 3857
        self.roads = None
        self.buffered_roads = {}
        self.backup_data = None

    def load_roads(self):
        self.roads = ox.features_from_polygon(self.bounds, {'highway': True})
        self.roads = self.roads[~self.roads.geometry.geom_type.isin(['Point', 'Polygon', 'Multipolygon'])]
        self.roads.reset_index(inplace=True)
        self.roads.to_crs(self.local_crs, inplace=True)
        self.roads = self.roads[['id', 'highway', 'geometry']]
        self._buffer_roads()
        self.joined_buildings = self._join_buildings_to_buffers()
        self._create_backup_data()

    def _classify_roads(self):
        r = self.roads
        cat1 = r[r['highway'].isin(['residential', 'living_street'])].assign(highway='residential')
        cat2 = r[r['highway'].isin(['primary', 'secondary', 'tertiary'])].assign(highway='non_res_road')
        cat3 = r[r['highway'].isin(['motorway', 'trunk'])].assign(highway='roads_cat3')
        cat4 = r[r['highway'].isin(['service'])].assign(highway='service')
        return {'cat1': cat1, 'cat2': cat2, 'cat3': cat3, 'cat4': cat4}

    def _buffer_roads(self, radius_list=[30, 60, 90]):
        road_groups = self._classify_roads()
        for cat_key, gdf in road_groups.items():
            for r in radius_list:
                key = f'{cat_key}_{r}'
                buffered = gdf.copy()
                buffered['geometry'] = buffered['geometry'].buffer(r)
                buffered.rename(columns={'highway': 'buffered_highway'}, inplace=True)
                self.buffered_roads[key] = buffered.dissolve()

    def _join_buildings_to_buffers(self):
        results = {}
        for key, buffered in self.buffered_roads.items():
            road_type = key.split('_')[0]
            roads_category = 1 if road_type == 'cat1' else 0
            joined = self.buildings.sjoin(buffered, how='left')
            joined.rename(columns={'index_right': 'index_roads'}, inplace=True)

            if roads_category:
                joined['buffered_highway'] = joined['buffered_highway'].apply(
                    lambda x: 'not_residential' if pd.isnull(x) else x)
            else:
                joined['buffered_highway'] = joined['buffered_highway'].apply(
                    lambda x: 'residential' if pd.isnull(x) else x)

            joined.reset_index(drop=True, inplace=True)
            results[key] = joined
        return results

    def _create_backup_data(self):
        buffer_keys = list(self.joined_buildings.keys())
        base_df = self.joined_buildings['cat1_30'].copy()
        base_df = base_df[['geometry', 'building', 'landuse', 'land_building', 'all_tag_keys', 'all_tags', 'id_left']]
        base_df['area'] = base_df['geometry'].to_crs(self.local_crs).area

        # Параметры можно задать в конструкторе или здесь
        value_tags = ['roof:shape', 'building:levels']
        selected_tags = ['source', 'addr:street', 'name', 'building:levels',
                         'roof:shape', 'amenity', 'brand', 'website', 'shop']

        base_df = preprocess_data(base_df, value_tags, selected_tags)

        # Добавляем 12 буферов
        col_names = [f'buffer_{key}' for key in buffer_keys]
        buffers_list = [self.joined_buildings[key]['buffered_highway'] for key in buffer_keys]

        base_df = add_buffers(base_df, col_names, buffers_list)

        self.backup_data = base_df

def preprocess_data(input_data, value_tags, selected_tags):
    col_pos = 2
    for tag_key in selected_tags:
        input_data.insert(col_pos, tag_key, 0)
        input_data[tag_key] = input_data.apply(lambda x: 1 if tag_key in list(x['all_tag_keys']) else 0, axis=1)

    for tag_key in value_tags:
        input_data[tag_key] = input_data.apply(
            lambda x: x['all_tags'].get(tag_key) if tag_key in list(x['all_tag_keys']) else 0, axis=1)

    input_data.drop(input_data[(input_data['building:levels'] == 'K-6')].index, inplace=True)
    input_data.loc[(input_data['building:levels'] == '-1,1,2,3'), 'building:levels'] = 4
    input_data.drop(columns=['all_tag_keys', 'all_tags'], inplace=True)
    return input_data

def add_buffers(input_data, col_names, buffers_list):
    for i, col in enumerate(col_names):
        input_data.insert(i + 2, col, buffers_list[i])
        input_data[col] = input_data[col].apply(lambda x: 1 if x == 'residential' else 0)
    return input_data
