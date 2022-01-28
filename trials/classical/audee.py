import tensorflow as tf
import time

from src.model_json import ModelJSON


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    tf.config.experimental_run_functions_eagerly(True)

    start_time = time.time()
    model = ModelJSON('./tf_lenet-5.json')
