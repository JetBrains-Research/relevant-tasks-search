import pickle
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import random
from typing import List, Tuple
import torch


def read_data(paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not isinstance(paths, List):
        raise Exception(f"You passed {type(paths)} instead of List[str] type")

    data_train = pd.read_csv(paths[0])
    data_val = pd.read_csv(paths[1])
    data_test = pd.read_csv(paths[2])

    data_train = data_train.dropna().reset_index(drop=True)
    data_val = data_val.dropna().reset_index(drop=True)
    data_test = data_test.dropna().reset_index(drop=True)

    return data_train, data_val, data_test


def create_dataset(data_train: pd.DataFrame):
    # Positive examples
    lessons_set_train = set(list(data_train.lesson_id))
    k = 7
    dataset_train = []
    for lesson_id in lessons_set_train:
        data_train_sampled = data_train[data_train.lesson_id == lesson_id]

        if data_train_sampled.shape[0] < 2:
            continue

        tmp = []
        for index1 in data_train_sampled.index:
            for index2 in data_train_sampled.index:
                if index2 <= index1:
                    continue
                tmp.append(
                    InputExample(
                        texts=[
                            data_train_sampled.at[index1, "preprocessed_text"],
                            data_train_sampled.at[index2, "preprocessed_text"],
                        ],
                        label=1.0,
                    )
                )

        if len(tmp) >= k:
            dataset_train.extend(random.sample(tmp, k=k))
        else:
            dataset_train.extend(tmp)
    positive_size = len(dataset_train)

    # Negative examples (randomly)
    data_indexes = list(data_train.index)
    for i in range(len(dataset_train) * 2):
        indexes = random.sample(data_indexes, 2)
        a_lesson_id = data_train.at[indexes[0], "lesson_id"]
        b_lesson_id = data_train.at[indexes[1], "lesson_id"]
        if a_lesson_id != b_lesson_id:
            dataset_train.append(
                InputExample(
                    texts=[
                        data_train.at[indexes[0], "preprocessed_text"],
                        data_train.at[indexes[1], "preprocessed_text"],
                    ],
                    label=0.0,
                )
            )

    # Negative examples (from one course, but different sections)
    courses_set_train = set(list(data_train.course_id))
    all_negative = []
    for course_id in tqdm(courses_set_train):
        data_train_sampled = data_train[data_train.course_id == course_id]
        tmp = []
        for index1 in data_train_sampled.index:
            for index2 in data_train_sampled.index:
                if index2 <= index1:
                    continue
                section_id1 = data_train_sampled.at[index1, "section_id"]
                section_id2 = data_train_sampled.at[index2, "section_id"]
                if section_id1 == section_id2:
                    continue
                tmp.append(
                    InputExample(
                        texts=[
                            data_train_sampled.at[index1, "preprocessed_text"],
                            data_train_sampled.at[index2, "preprocessed_text"],
                        ],
                        label=0.5,
                    )
                )
        all_negative.extend(tmp)
    dataset_train += random.sample(all_negative, positive_size)

    # Negative examples (from one section, but different lessons)
    sections_set_train = set(list(data_train.section_id))
    all = []
    for section_id in tqdm(sections_set_train):
        data_train_sampled = data_train[data_train.section_id == section_id]
        tmp = []
        for index1 in data_train_sampled.index:
            for index2 in data_train_sampled.index:
                if index2 <= index1:
                    continue
                lesson_id1 = data_train_sampled.at[index1, "lesson_id"]
                lesson_id2 = data_train_sampled.at[index2, "lesson_id"]
                if lesson_id1 == lesson_id2:
                    continue
                tmp.append(
                    InputExample(
                        texts=[
                            data_train_sampled.at[index1, "preprocessed_text"],
                            data_train_sampled.at[index2, "preprocessed_text"],
                        ],
                        label=0.75,
                    )
                )
        all.extend(tmp)
    dataset_train += random.sample(all, positive_size)

    # Negative examples (same Stepik's block, different courses)
    with open("/src/data/output_data/stepik_blocks.pkl", "rb") as fin:
        stepik_blocks = pickle.load(fin)

    for block in tqdm(stepik_blocks.values()):
        data_train_sampled = data_train[data_train.course_id.isin(block)]
        tmp = []
        for index1 in data_train_sampled.index:
            for index2 in data_train_sampled.index:
                if random.random() > 0.5:
                    continue
                if index2 <= index1:
                    continue
                course_id1 = data_train_sampled.at[index1, "course_id"]
                course_id2 = data_train_sampled.at[index2, "course_id"]
                if course_id1 == course_id2:
                    continue
                tmp.append(
                    InputExample(
                        texts=[
                            data_train_sampled.at[index1, "preprocessed_text"],
                            data_train_sampled.at[index2, "preprocessed_text"],
                        ],
                        label=0.25,
                    )
                )
        all.extend(tmp)
    dataset_train += random.sample(all, positive_size)

    random.shuffle(dataset_train)
    test_size = 0.0005
    dataset_train, dataset_test = (
        dataset_train[int(len(dataset_train) * test_size) :],
        dataset_train[: int(len(dataset_train) * test_size)],
    )

    # Saving dataset to output_data dir
    with open("../data/output_data/dataset_train_for_sbert.pkl", "rb") as fout:
        pickle.dump(dataset_train, fout)
    with open("../data/output_data/dataset_test_for_sbert.pkl", "rb") as fout:
        pickle.dump(dataset_test, fout)

    return dataset_train, dataset_test


def train(dataset_train: List[InputExample], dataset_test: List[InputExample]):
    model_name = "paraphrase-multilingual-mpnet-base-v2"
    train_batch_size = 16
    num_epochs = 4
    model_save_path = "../data/output_data/sbert_finetuned"

    # Model init and other tools
    model = SentenceTransformer(model_name)
    train_dataloader = DataLoader(dataset_train, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dataset_test, name="test split")

    # Training
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=100,
        output_path=model_save_path,
        checkpoint_path="../data/output_data/checkpoints",
        checkpoint_save_steps=2000,
        checkpoint_save_total_limit=1,
    )


def main():
    paths = ["", "", ""]
    data_train, data_val, data_test = read_data(paths=paths)
    dataset_train, dataset_test = create_dataset(data_train)
    train(dataset_train, dataset_test)


if __name__ == "__main__":
    main()
