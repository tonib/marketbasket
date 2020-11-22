from settings import settings
import numpy as np

class ClassWeights:

    CLASS_WEIGHTS_NAME = 'classweights.npy'

    def __init__(self, train_item_n_outputs: np.ndarray = None):
        if train_item_n_outputs is None:
            return

        # Convert train_item_n_outputs to weight factors
        self.class_weights: np.ndarray = train_item_n_outputs.astype(float)
        #print(self.class_weights)

        # Get max value
        max_val = np.amax(self.class_weights)
        if max_val == 0:
            max_val = 1 # Should not happen
        
        # Replace zeros with something to avoid division by zero
        self.class_weights[self.class_weights == 0] = max_val
        # Get the factors:
        self.class_weights = max_val / self.class_weights
        #print(self.class_weights)
        
        # For very unbalanced datasets, there will be very large weights. Limit weights from (1-max_weight) to (1-limit)
        limit = 3.
        max_weight = np.amax(self.class_weights)
        if max_weight > limit:
            self.class_weights = (self.class_weights - 1.) * ( (limit-1) / (max_weight-1) )
            self.class_weights = self.class_weights + 1
        #print(self.class_weights)

    def save(self, path: str):
        # Save
        np.save(path, self.class_weights)

    def keras_class_weights(self):
        return dict(enumerate(self.class_weights))
        
    @staticmethod
    def class_weights_path() -> str:
        """ Returns the class weights path in data directory """
        return settings.get_data_path( ClassWeights.CLASS_WEIGHTS_NAME )

    @staticmethod
    def load(path: str) -> 'ClassWeights':
        cw = ClassWeights()
        cw.class_weights = np.load(path)
        return cw
