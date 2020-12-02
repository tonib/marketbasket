import marketbasket.settings as settings
from marketbasket.feature import Feature
from typing import List, Dict
import tensorflow as tf

class ModelInputs:
    """ TF model inputs """

    def __init__(self):

        # Model inputs. ORDER IS IMPORTANT, as Keras determines inputs by position, and not by name...
        self.inputs: List[tf.keras.Input] = []

        # Inputs by feature
        self.by_feature: Dict[Feature, tf.keras.Input] = {}

        feature: Feature
        for feature in settings.settings.features:
            if feature.sequence:
                input = tf.keras.layers.Input(shape=[None], name=feature.name, dtype=tf.int64, ragged=True)
            else:
                input = tf.keras.layers.Input(shape=(), name=feature.name, dtype=tf.int64)
            self.inputs.append(input)
            self.by_feature[feature] = input

    def item_labels_input(self) -> tf.keras.Input:
        """ Get the items sequence input """
        return self.by_feature[settings.settings.features.items_sequence_feature()]

    