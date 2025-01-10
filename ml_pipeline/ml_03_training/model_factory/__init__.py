# Rodrigo's work
from model_factory.classes.tattoo_classification_model  import TattooClassifierModel
from model_factory.classes.tattoo_detection_model       import TattooDetectorModel
from model_factory.classes.tattoo_localization_model    import TattooLocaterModel
from model_factory.classes. people_detection_model      import PeopleDetectorModel

class ModelFactory:
    @staticmethod
    def model_generate(config, model_name):
        if config["model_type"] == "tattoo_classification":
            model = TattooClassifierModel(config, model_name)
            return model
        elif config["model_type"] == "tattoo_detection":
            model = TattooDetectorModel(config, model_name)
            return model
        elif config["model_type"] == "tattoo_localization":
            model = TattooLocaterModel(config, model_name)
            return model
        elif config["model_type"] == "people_detection":
            model = PeopleDetectorModel(config, model_name)
            return model