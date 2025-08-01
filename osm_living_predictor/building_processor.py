import osmnx as ox
import pandas as pd

class BuildingProcessor:
    def __init__(self, bounds):
        self.bounds = bounds
        self.buildings = None
        self.local_crs = 3857

    def load_buildings(self):
        # Загрузка по тегу building
        self.buildings = ox.features_from_polygon(self.bounds, {'building': True})
        self._clean_buildings()
        self._load_landuse()
        self._clean_landuse()


    def _clean_buildings(self):
        self.buildings.reset_index(inplace=True)
        self.buildings['all_tags'] = self.buildings.apply(
            lambda x: {col: x[col] for col in self.buildings.columns 
                    if col not in ['geometry', 'osmid', 'element', 'id', 'index'] 
                    and pd.notnull(x[col])}, 
            axis=1
        )
        self.buildings['all_tag_keys'] = self.buildings['all_tags'].apply(lambda x: list(x.keys()))
        self.buildings['building'] = self.buildings['building'].fillna('')
        self.buildings = self.buildings[['id', 'building', 'geometry', 
                      'all_tag_keys', 'all_tags']]
        self.buildings = self.buildings[self.buildings.geometry.geom_type != 'Point']
        self.buildings.reset_index(drop=True, inplace=True)

    def _load_landuse(self):
        self.landuse = ox.features_from_polygon(self.bounds, {'landuse': True})


    def _clean_landuse(self):
        self.landuse.reset_index(inplace=True)
        self.landuse = self.landuse[['landuse', 'geometry']]
        non_res = ['commercial', 'retail','industrial','grass','plant_nursery','quarry','railway','government',\
            'institutional']
        self.landuse['landuse'] = self.landuse['landuse'].apply(\
                                lambda x: 'non_res' if x in non_res else x)
        self.buildings = self.buildings.sjoin(self.landuse, how="left", predicate='intersects')
        self.buildings.drop_duplicates(subset='id',inplace=True)
        self.buildings.rename(columns={'index_right': 'index_landuse'},inplace=True)
        self.buildings['landuse'] = \
            self.buildings['landuse'].apply(lambda x: 'no_landuse' if pd.isnull(x) else x)
        self.buildings.reset_index(inplace=True, drop=True)
        self.buildings['land_building'] = self.buildings.apply(lambda x: \
    x['landuse'] if x['building'] == 'yes' else x['building'], axis=1)
        
        
        tags_list = ['residential','commercial','religious','recreation_ground','cemetery',\
            'construction','farmland','farmyard','forest','military']
        self._enrich_buildings(tags_list)

        self.buildings.to_crs(self.local_crs, inplace=True)
    
    def _enrich_buildings(self, tags_list):
        for tag_val in tags_list:
            self.buildings['land_building'] = self.buildings.apply(lambda x: \
                tag_val if x['landuse'] == tag_val else x['land_building'], axis=1)