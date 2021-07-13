import os
import pickle
from typing import List, Tuple, Dict

import numpy as np
import torch
from bs4 import BeautifulSoup
from scipy.spatial import distance
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.models.task import Task


class BertRecommender:
    __tasks: List[Task]
    __graph_distances: np.ndarray
    __distances_methods: Dict

    def __init__(
            self,
            from_file: bool = False,
            bert_model: str = "roberta-base",
            path_to_distances: str = "./data/graph/distances.pkl",
    ):
        # Distances load
        if not os.path.exists(path_to_distances):
            raise Exception(f"{path_to_distances} doesn't exist")
        else:
            with open(path_to_distances, "rb") as fin:
                self.__graph_distances = pickle.load(fin)
        # Vectors and tasks load
        if from_file:
            if os.path.exists("./bert_data.pkl"):
                with open("./bert_data.pkl", "rb") as fin:
                    self.__tasks = pickle.load(fin)
            else:
                raise Exception("Can't start from file because there are no data in ./bert_data.pkl")
        else:
            self.__tasks = []
        self.__tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.__model = AutoModel.from_pretrained(bert_model)
        self.__distance_methods = {
            "cityblock": distance.cityblock,
            "euclidean": distance.euclidean,
            "cosine": distance.cosine,
        }

    def train(self, path_to_train: str, preprocess: bool = False):
        if not os.path.exists(path_to_train):
            raise Exception(f"{path_to_train} doesn't exist")
        if preprocess:
            pass  # TODO add preprocessing

        with open(path_to_train, "r", encoding="utf-8") as fin:
            data = fin.readlines()[1:]

        for line in tqdm(data, desc="vectors calculating"):
            obj = line.split("\t")
            vector = self.__calculate_vector(obj[2])
            self.__tasks.append(
                Task(
                    step_id=int(obj[0]), topic_id=int(obj[1]), preprocessed_text=obj[2],
                    raw_text=BeautifulSoup(obj[3], "lxml").text, vector=vector
                )
            )

        with open("./bert_data.pkl", "wb") as fout:
            pickle.dump(self.__tasks, fout)

    def evaluate(self, path_to_test: str, k: int = 10, t: int = 0, preprocess: bool = False) -> Dict[str, float]:
        if not os.path.exists(path_to_test):
            raise Exception(f"{path_to_test} doesn't exist")
        if len(self.__tasks) == 0:
            raise Exception("Before evaluation you should use 'train' first")
        if preprocess:
            pass  # TODO add preprocessing

        with open(path_to_test, "r", encoding="utf-8") as fin:
            test_data = fin.readlines()[1:]
        aps = [0, 0, 0]
        dist_sum = 0
        counter = 0
        for line in tqdm(test_data, desc="evaluating"):
            obj = line.split("\t")
            vector = self.__calculate_vector(obj[2])
            test_task = Task(
                step_id=int(obj[0]), topic_id=int(obj[1]), preprocessed_text=obj[2], raw_text=obj[3], vector=vector
            )

            relevant_tasks = self.retrieve(task=test_task, k=k)

            aps[0] += self.__ap_at_k(relevant_tasks=relevant_tasks, test_task=test_task, k=1, t=t)
            aps[1] += self.__ap_at_k(relevant_tasks=relevant_tasks, test_task=test_task, k=3, t=t)
            aps[2] += self.__ap_at_k(relevant_tasks=relevant_tasks, test_task=test_task, k=5, t=t)
            dist_sum += self.__graph_distances[relevant_tasks[0][1].topic_id, test_task.topic_id]
            counter += 1

        results = list(map(lambda x: x / counter, aps + [dist_sum]))
        return {"map@1": results[0], "map@3": results[1], "map@5": results[2], "mean graph distance": results[3]}

    def retrieve(
            self, task_text: str = "", task: Task = None, k: int = 5, mode: str = "evaluation"
    ) -> List[Tuple[float, Task]]:
        if len(self.__tasks) == 0:
            raise Exception("Before evaluation you should use 'train' first")
        if mode == "evaluation" and task is None:
            raise Exception("Inappropriate use of 'evaluation' mode. You should pass task")
        if mode == "retrieve" and task_text == "":
            raise Exception("Inappropriate use of 'retrieve' mode. You should pass task_text")
        if mode != "evaluation" and mode != "retrieve":
            raise Exception(f"Inappropriate mode: {mode}")
        if mode == "retrieve":
            vector = self.__calculate_vector(task_text)
            task = Task(step_id=0, topic_id=0, preprocessed_text=task_text, raw_text=task_text, vector=vector)

        relevant_tasks = []
        for train_task in self.__tasks:
            dist = self.__calculate_vector_distance(train_task=train_task, test_task=task, method="cityblock")
            if len(relevant_tasks) < k:
                relevant_tasks.append((dist, train_task))
                relevant_tasks.sort(key=lambda x: x[0])
            elif relevant_tasks[-1][0] > dist:
                relevant_tasks[-1] = (dist, train_task)
                relevant_tasks.sort(key=lambda x: x[0])
        return relevant_tasks

    def __calculate_vector(self, text):
        input_ids = torch.tensor(self.__tokenizer.encode(text, truncation=True)).unsqueeze(0)
        outputs = self.__model(input_ids)
        last_hidden_states = outputs[0]
        return last_hidden_states[0].data.numpy().mean(axis=0).flatten()

    def __calculate_vector_distance(self, train_task: Task, test_task: Task, method: str = "cityblock"):
        if method not in self.__distance_methods:
            raise Exception(f"Incorrect distance method: {method}")
        return self.__distance_methods[method](train_task.vector, test_task.vector)

    def __ap_at_k(self, relevant_tasks: list, test_task: Task, k: int, t: int) -> float:
        tp = 0
        tp_fp = 0
        precision_sum = 0
        for i in range(k):
            tp_fp += 1
            tp = tp + 1 if self.__graph_distances[relevant_tasks[i][1].topic_id, test_task.topic_id] <= t else tp
            precision_sum += tp / tp_fp
        return precision_sum / k
