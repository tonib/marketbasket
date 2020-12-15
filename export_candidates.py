from marketbasket.settings import settings # Setup. Must to be first
from marketbasket.model.model import create_model
from marketbasket.labels import Labels
from datetime import datetime
from marketbasket.predict import Prediction
import tensorflow as tf

def export_model(rating_model: bool):
    settings.features.load_label_files()
    model = create_model(rating_model)

    # Load weights from last train checkpoint
    latest_cp = tf.train.latest_checkpoint( settings.get_model_path(rating_model, Prediction.CHECKPOINTS_DIR) )
    model.load_weights(latest_cp)

    model.summary()

    # Save full model
    model.save( settings.get_model_path(rating_model, Prediction.EXPORTED_MODEL_DIR), include_optimizer=False )

if __name__ == "__main__":
    print(datetime.now(), "Process start: Export candidates model")
    export_model(False)
    print(datetime.now(), "Process end: Export candidates model")
