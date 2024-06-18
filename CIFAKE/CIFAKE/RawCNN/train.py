import argparse
import keras, os
from keras import layers, losses
import tensorflow as tf
tf.random.set_seed(123)

def train_models(args):

    print("Loading Dataset")
    train_ds = tf.keras.utils.image_dataset_from_directory(
    args.database_path,
    seed=123,
    image_size=(32, 32),
    batch_size=32,
    label_mode='binary')


    print("Training Model")
    model = keras.Sequential()
    model.add(layers.Rescaling(1./255))
    model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dense(units=1, activation='sigmoid'))
    model.compile(loss=losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    model.fit(train_ds, epochs=100, verbose=1)

    model.save(os.path.join("../models/", f'{args.iteration_number}.h5'), save_format="h5")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--database_path', type=str, default='../../../../DataSets/CIFAKE/CIFAKE2575')
    parser.add_argument('--iteration_number', type=str, default='25-75')
    
    args = parser.parse_args()


    train_models(args)