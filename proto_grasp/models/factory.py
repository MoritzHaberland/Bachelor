from .det import Det
from .det_seg import DetSeg
from .det_seg_ref import DetSegRef

class ModelFactory:
    def __init__(self):
        self.models = {
            'det' : Det,
            'detseg' : DetSeg,
            'detsegref' : DetSegRef
        }

    def get_model(self, model_name, **kwargs):
        model_class = self._models.get(model_name.lower())
        if model_class is None:
            raise ValueError(f"Model {model_name} is not registered in the factory.")
        
        return model_class(**kwargs)