import osmnx as ox

class OSMDownloader:
    def __init__(self, osm_id):
        self.osm_id = osm_id
        self.bounds = None

    def load_boundary(self):
        self.bounds = ox.geocode_to_gdf(f'R{self.osm_id}', by_osmid=True).iloc[0].geometry
        return self.bounds
