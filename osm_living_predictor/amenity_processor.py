import re
import osmnx as ox
import pandas as pd
import geopandas as gpd
import jenkspy
import matplotlib.pyplot as plt

class AmenityProcessor:
    def __init__(self, bounds, backup_data, crs=3857):
        self.bounds = bounds
        self.local_crs = crs
        self.backup_data = backup_data  # содержит all_tag_keys, all_tags, id_left и т.п.
        self.amenities = None
        self.buffered_layers = {}

    def load_amenities(self):
        self.amenities = ox.features_from_polygon(self.bounds, {'amenity': True})
        self.amenities.reset_index(inplace=True)
        self.amenities.to_crs(self.local_crs, inplace=True)
        self.amenities = self.amenities[['amenity', 'geometry']]
        self.amenities.rename(columns={'amenity': 'amenity_key'}, inplace=True)
        self._classify_parking()
        self._clear_building_levels()

    def _classify_parking(self):
        if self.amenities is None:
            raise ValueError("Сначала вызовите load_amenities()")

        parking = self.amenities[
            self.amenities['amenity_key'].isin(['parking', 'parking_space'])
        ]
        parking = parking[~parking.geometry.geom_type.isin(['Point'])].copy()
        parking['amenity_key'] = 'parking'
        parking.reset_index(drop=True, inplace=True)
        parking['parking_area'] = parking.geometry.to_crs(self.local_crs).area

        # Jenks natural breaks
        breaks = jenkspy.jenks_breaks(parking['parking_area'], n_classes=3)
        bins_list = ['bin1', 'bin2', 'bin3']
        parking['cut_jenks'] = pd.cut(
            parking['parking_area'], bins=breaks,
            labels=bins_list, include_lowest=True
        )

        for bin in bins_list:
            parking[bin] = parking['cut_jenks'] == bin

        radius_list = [30, 60, 90]
        counter = 1

        for bin in bins_list:
            for radius in radius_list:
                col_name = f"parking_cat{counter}"
                counter += 1

                buffer = parking[parking[bin]].copy()
                buffer = buffer[['geometry', bin]].rename(columns={bin: col_name})
                buffer['geometry'] = buffer['geometry'].to_crs(self.local_crs).buffer(radius)
                buffer = gpd.GeoDataFrame(buffer, geometry='geometry', crs=self.local_crs)

                # Буфер пустой — пропускаем
                if buffer.empty:
                    self.backup_data[col_name] = 0
                    continue

                joined = self.backup_data.sjoin(buffer, how="left", predicate='intersects')
                joined[col_name] = joined[col_name].fillna(0).astype(int)
                joined.drop_duplicates(subset='id_left', inplace=True)
                joined.reset_index(drop=True, inplace=True)

                # Добавляем колонку
                self.backup_data.insert(len(self.backup_data.columns), col_name, list(joined[col_name]))

                self.buffered_layers[col_name] = buffer

        # Удалить вспомогательный столбец
        if 'id_left' in self.backup_data.columns:
            self.backup_data.drop(columns=['id_left'], inplace=True)

    def _clear_building_levels(self):
        """
        Преобразует поле 'building:levels' в числовой формат.
        Берёт максимум из найденных чисел, иначе 0.
        """
        def extract_max_or_zero(val):
            if pd.isna(val):
                return 0
            numbers = re.findall(r'\d+', str(val))
            return int(max(map(int, numbers))) if numbers else 0

        self.backup_data['building:levels'] = self.backup_data['building:levels'].apply(extract_max_or_zero)
