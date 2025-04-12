import json
import mlflow
import keras
from keras.layers import StringLookup
import tensorflow as tf
import numpy as np
from mlflow.tracking import MlflowClient
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def decode_batch_predictions(pred, num_to_char, max_len):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.ops.nn.ctc_decode(pred, sequence_lengths=input_len)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = (
            tf.strings.reduce_join(num_to_char(res))
            .numpy()
            .decode("utf-8")
            .replace("[UNK]", "")
        )
        output_text.append(res)
    return output_text


if __name__ == '__main__':
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    model_name = "tester"
    version = "9"
    client = MlflowClient()

    version_info = client.get_model_version(model_name, version)
    model_uri = f"models:/{model_name}/{version}"

    run = client.get_run(version_info.run_id)
    params = run.data.params

    max_len = int(params["max_len"])
    artifact_path = "vocab.json"
    local_path = mlflow.artifacts.download_artifacts(
        run_id=version_info.run_id,
        artifact_path=artifact_path
    )
    with open(local_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    num_to_char = StringLookup(
        vocabulary=vocab, mask_token=None, invert=True
    )
    pred = json.load(sys.stdin)

    out = np.array(pred["predictions"])

    print(decode_batch_predictions(out, num_to_char, max_len))
