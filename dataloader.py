import numpy as np
import pandas as pd
from tensorflow.data import Dataset
from spektral.layers import GCNConv


class DataLoader(object):
    def __init__(self, path: str, task: str, max_entity_num: int = 32):
        target_problems = ["mort_hosp", "mort_icu", "los_3", "los_7"]
        assert task in target_problems

        self.__task = task
        self.__path = path
        self.__max_entity_num = max_entity_num
        self._train_data = self.__load_data("train")
        self._train_graph = self.__load_graph("train")
        self._dev_data = self.__load_data("dev")
        self._dev_graph = self.__load_graph("dev")
        self._test_data = self.__load_data("test")
        self._test_graph = self.__load_graph("test")
        self._index_features = self.__load_index_name()

    def Index(self):
        return self._index_features

    def Graph(self, dtype: str):
        return getattr(self, f"_{dtype}_graph")

    def Data(self, dtype: str):
        return getattr(self, f"_{dtype}_data")

    def __load_index_name(self):
        return pd.read_pickle(f"{self.__path}/index_name_features.pkl")

    def __load_graph(self, dtype: str):
        (rows, cols) = pd.read_pickle(f"{self.__path}/{dtype}_graph_idxs.pkl")
        matrix = pd.read_pickle(f"{self.__path}/patients_graph.pkl").A
        size = int(np.sqrt(matrix.shape[1]))
        PMI = matrix.reshape((size, size))
        I = np.eye(size)
        A_tilde = GCNConv.preprocess(PMI - I) + I
        return A_tilde[rows, cols]

    def __load_data(self, dtype: str):
        ts = np.array(pd.read_pickle(f"{self.__path}/x_{dtype}.pkl"))
        y = pd.read_pickle(f"{self.__path}/y_{dtype}.pkl")

        tmp_doc, entity = [], []
        entity_features = pd.read_pickle(f"{self.__path}/entity_features.pkl")
        doc_features = pd.read_pickle(f"{self.__path}/clinical_notes_features.pkl")
        for i in y.itertuples():
            entity_feature = entity_features[i.Index[0]]
            entity.append(
                np.float32(entity_feature[: self.__max_entity_num])
                if len(entity_feature) > self.__max_entity_num
                else np.float32(
                    np.concatenate(
                        [
                            entity_feature,
                            np.zeros(
                                (
                                    self.__max_entity_num - len(entity_feature),
                                    entity_feature.shape[1],
                                )
                            ),
                        ],
                        axis=0,
                    )
                )
            )
            tmp_doc.append(np.expand_dims(doc_features[i.Index[0]]["tfidf"], 0))
        doc = np.float32(np.concatenate(tmp_doc, axis=0))
        return (
            Dataset.from_tensor_slices(
                (
                    ts,
                    doc,
                    entity,
                    np.array(y[self.__task], dtype=np.int),
                )
            )
            .batch(len(ts))
            .shuffle(len(ts))
        )
