import os
import pickle
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from tqdm import tqdm
import pandas as pd
import random
from typing import List, Dict

from src.tools.rut5_finetuning import read_data, configure_arg_parser


def positive_samples(data_train: pd.DataFrame, label: float) -> List[InputExample]:
    """
    Used for sampling positive examples
    :param data_train:
    :param label: label for positive examples
    :return: list with positive InputExamples
    """
    lessons_set_train = set(list(data_train.lesson_id))
    k = 7
    sampled = []
    for lesson_id in tqdm(lessons_set_train, desc="positive examples"):
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
                        label=label,
                    )
                )

        if len(tmp) >= k:
            sampled.extend(random.sample(tmp, k=k))
        else:
            sampled.extend(tmp)

    return sampled


def random_samples(data_train: pd.DataFrame, label: float, size: int) -> List[InputExample]:
    """
    Used for sampling negative examples in random way
    :param data_train:
    :param label: label for random examples
    :param size: size of randomly sampled examples
    :return: list with random InputExamples
    """
    data_indexes = list(data_train.index)
    sampled = []
    for _ in tqdm(range(size), desc="negative random examples"):
        indexes = random.sample(data_indexes, 2)
        a_lesson_id = data_train.at[indexes[0], "lesson_id"]
        b_lesson_id = data_train.at[indexes[1], "lesson_id"]
        if a_lesson_id != b_lesson_id:
            sampled.append(
                InputExample(
                    texts=[
                        data_train.at[indexes[0], "preprocessed_text"],
                        data_train.at[indexes[1], "preprocessed_text"],
                    ],
                    label=label,
                )
            )

    return sampled


def different_sections_samples(data_train: pd.DataFrame, label: float, positive_size: int) -> List[InputExample]:
    """
    Used for sampling negative examples, from one course, different sections
    :param data_train:
    :param label:
    :param positive_size: size of positive part of dataset
    :return:
    """
    courses_set_train = set(list(data_train.course_id))
    sampled = []
    for course_id in tqdm(courses_set_train, desc="negative different sections"):
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
                        label=label,
                    )
                )
        sampled.extend(tmp)

    return random.sample(sampled, positive_size)


def different_lessons_samples(data_train: pd.DataFrame, label: float, positive_size: int) -> List[InputExample]:
    """
    Used for sampling negative examples, from one section, different lessons
    :param data_train:
    :param label:
    :param positive_size: size of positive part of dataset
    :return:
    """
    sections_set_train = set(list(data_train.section_id))
    sampled = []
    for section_id in tqdm(sections_set_train, desc="negative different lessons"):
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
                        label=label,
                    )
                )
        sampled.extend(tmp)

    return random.sample(sampled, positive_size)


def different_courses_samples(
    data_train: pd.DataFrame, stepik_blocks: Dict[str, List[int]], label: float, positive_size: int
) -> List[InputExample]:
    """
    Used for sampling negative examples, from one Stepik's block, different courses
    :param data_train:
    :param stepik_blocks: Stepik's courses blocks
    :param label:
    :param positive_size: size of positive part of dataset
    :return:
    """
    sampled = []
    for block in tqdm(stepik_blocks.values(), desc="negative different courses"):
        data_train_sampled = data_train[data_train.course_id.isin(block)]
        tmp = []
        for index1 in data_train_sampled.index:
            for index2 in data_train_sampled.index:
                if random.random() > 0.25:
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
                        label=label,
                    )
                )
        sampled.extend(tmp)

    return random.sample(sampled, positive_size)


def create_dataset(input_path: str, output_path: str):
    """
    Used for dataset creation. Should be called before training
    :param input_path: path to dir with train/val/test split and stepik_blocks file
    :param output_path: path to saving datasets
    :return:
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} doesn't exist")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    data_train, _, _ = read_data(input_path, load_courses=False)

    dataset_train = []

    # Positive examples
    dataset_train += positive_samples(data_train, label=1.0)
    positive_size = len(dataset_train)

    # Negative examples (randomly)
    multiplier = 2
    dataset_train += random_samples(data_train, label=0.0, size=len(dataset_train) * multiplier)

    # Negative examples (from one course, but different sections)
    dataset_train += different_sections_samples(data_train, label=0.5, positive_size=positive_size)

    # Negative examples (from one section, but different lessons)
    dataset_train += different_lessons_samples(data_train, label=0.75, positive_size=positive_size)

    # Negative examples (same Stepik's block, different courses)
    with open(input_path + "/stepik_blocks.pkl", "rb") as fin:
        stepik_blocks = pickle.load(fin)
    dataset_train += different_courses_samples(data_train, stepik_blocks, label=0.25, positive_size=positive_size)

    random.shuffle(dataset_train)
    test_size = 0.0005
    dataset_train, dataset_test = (
        dataset_train[int(len(dataset_train) * test_size) :],
        dataset_train[: int(len(dataset_train) * test_size)],
    )

    # Saving dataset to output_path dir
    with open(output_path + "/dataset_train_for_sbert.pkl", "wb") as fout:
        pickle.dump(dataset_train, fout)
    with open(output_path + "/dataset_test_for_sbert.pkl", "wb") as fout:
        pickle.dump(dataset_test, fout)


def train(input_path: str, output_path: str, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
    """
    Used for training. Should be called after dataset creation
    :param input_path: path to dir with datasets files
    :param output_path: where to save finetuned model
    :param model_name: pretrained model name
    :return:
    """
    # Loading dataset from input path
    with open(input_path + "/dataset_train_for_sbert.pkl", "rb") as fin:
        dataset_train = pickle.load(fin)
    with open(input_path + "/dataset_test_for_sbert.pkl", "rb") as fin:
        dataset_test = pickle.load(fin)

    train_batch_size = 16
    num_epochs = 1
    model_save_path = output_path + "/sbert_finetuned"

    # Model init and other tools
    model = SentenceTransformer(model_name)
    train_dataloader = DataLoader(dataset_train, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dataset_test, name="test-split")

    # Training
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=100,
        output_path=model_save_path,
        checkpoint_path=output_path + "/checkpoints",
        checkpoint_save_steps=500,
        checkpoint_save_total_limit=1,
    )

    model.save(model_save_path)


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    if __args.task == "create_dataset":
        create_dataset(__args.input, __args.output)
    elif __args.task == "train":
        train(__args.input, __args.output)
