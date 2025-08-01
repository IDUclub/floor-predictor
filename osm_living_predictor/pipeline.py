from .downloader import OSMDownloader
from .building_processor import BuildingProcessor
from .road_processor import RoadProcessor
from .amenity_processor import AmenityProcessor
from .feature_builder import FeatureBuilder
from .model_handler import ModelHandler

class LivingPredictionPipeline:
    def __init__(self, osm_id, model_path):
        self.osm_id = osm_id
        self.model_path = model_path

    def run(self, output_path):
        # Загрузка
        downloader = OSMDownloader(self.osm_id)
        bounds = downloader.load_boundary()

        # Обработка
        buildings = BuildingProcessor(bounds)
        buildings.load_buildings()

        roads = RoadProcessor(bounds, buildings.buildings)
        roads.load_roads()

        amenities = AmenityProcessor(bounds, roads.backup_data)
        amenities.load_amenities()

        # Признаки
        builder = FeatureBuilder(buildings, roads, amenities)
        data = builder.build_features()

        # Предсказание
        model = ModelHandler(self.model_path)
        self.data_with_predictions = model.predict(data)

        # Сохранение
        self.save_output(output_path)

    def save_output(self, output_path):
        self.data_with_predictions.to_csv(output_path, index=False)
