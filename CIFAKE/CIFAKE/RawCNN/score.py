import argparse
import keras, os
import numpy as np
import tensorflow as tf
tf.random.set_seed(123)

def test_models(args):



    if args.designator == "pretrained":
        tmp = (args.pretrained_model).split(".h5")
        mod_name = f"{tmp[0]}_p.h5{tmp[1]}"
    else:
        mod_name = args.pretrained_model

    model = keras.models.load_model(mod_name)
    if args.designator == "pretrained":
        tmp = (args.output).split(".score")
        scores_file = f"{tmp[0]}_p.score{tmp[1]}"
    else:
        scores_file = args.output

    val_ds = tf.keras.utils.image_dataset_from_directory(
    args.database_path,
    seed=123,
    image_size=(32, 32),
    batch_size=32,
    label_mode='binary', 
    shuffle=False)
    
    fnames = [fname for fname in val_ds.file_paths]

    probs = model.predict(val_ds)
    y_pred = tf.where(probs<=0.5,0,1)

    out = zip(fnames, y_pred, probs)

    with open(scores_file, "w") as f:
        for fname, y_pred, probb in out:
            fname = os.path.basename(fname).split(".")[0]
            proba = 1 - probb[0]
            temp_out = f"{fname} - {y_pred[0]} [{proba}, {probb[0]}]\n"
            f.write(temp_out)


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--database_path', type=str, default='')
    parser.add_argument('--iteration_number', type=str, default='25-75')
    parser.add_argument('--designator', type=str, default='pretrained')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--pretrained_model', type=str, default='')
    
    args = parser.parse_args()


    test_models(args)