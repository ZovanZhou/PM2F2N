import os
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


def calc_metrics(y_true, y_logits):
    auc = roc_auc_score(y_true, y_logits[:, 1])
    auprc = average_precision_score(y_true, y_logits[:, 1])

    result_dict = {}
    result_dict["auroc"] = auc
    result_dict["auprc"] = auprc

    return result_dict


def train_one_step(model, dataLoader, optimizer):
    for ts, doc, entity, y in dataLoader.Data("train"):
        with tf.GradientTape() as tape:
            logits = model(
                ts,
                doc,
                entity,
                dataLoader.Index(),
                dataLoader.Graph("train"),
                training=True,
            )[0]
            loss = tf.reduce_mean(
                tf.losses.categorical_crossentropy(tf.one_hot(y, 2), logits)
            )
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


def test_model_with_loss(model, dataLoader, dtype: str = "dev"):
    for train_ts, train_doc, train_entity, _ in dataLoader.Data("train"):
        for dev_ts, dev_doc, dev_entity, y in dataLoader.Data(dtype):
            logits = model(
                tf.concat([train_ts, dev_ts], axis=0),
                tf.concat([train_doc, dev_doc], axis=0),
                tf.concat([train_entity, dev_entity], axis=0),
                dataLoader.Index(),
                dataLoader.Graph(dtype),
            )[0]
            loss = tf.reduce_mean(
                tf.losses.categorical_crossentropy(
                    tf.one_hot(y, 2), logits[-y.shape[0] :]
                )
            )
    return loss.numpy()


def test_model_with_metrics(model, dataLoader, dtype: str = "test"):
    for train_ts, train_doc, train_entity, _ in dataLoader.Data("train"):
        for test_ts, test_doc, test_entity, y in dataLoader.Data(dtype):
            logits = model(
                tf.concat([train_ts, test_ts], axis=0),
                tf.concat([train_doc, test_doc], axis=0),
                tf.concat([train_entity, test_entity], axis=0),
                dataLoader.Index(),
                dataLoader.Graph(dtype),
            )[0]
    return calc_metrics(y.numpy(), logits[-y.shape[0] :].numpy())


def extract_model_features(model, dataLoader, dtype: str = "test"):
    for train_ts, train_doc, train_entity, _ in dataLoader.Data("train"):
        for test_ts, test_doc, test_entity, y in dataLoader.Data(dtype):
            logits, features = model(
                tf.concat([train_ts, test_ts], axis=0),
                tf.concat([train_doc, test_doc], axis=0),
                tf.concat([train_entity, test_entity], axis=0),
                dataLoader.Index(),
                dataLoader.Graph(dtype),
            )
    return (features[-y.shape[0] :].numpy(), y.numpy(), logits[-y.shape[0] :].numpy())


def load_model(model, weights_path, dataLoader, optimizer):
    if os.path.exists(weights_path):
        train_one_step(model, dataLoader, optimizer)
        model.load_weights(weights_path, by_name="True")
