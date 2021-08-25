from argparse import ArgumentParser
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer
import pandas as pd
from typing import Any
import os


def retrieve_to_file(
    tasks_existing: pd.DataFrame,
    tasks_to_retrieve: pd.DataFrame,
    output_path: str,
    encoder: Any,
    top_k: int = 5,
    batch_size=32,
    file_type="tsv",
):
    """
    Used for retrieving top_k closest tasks
    :param tasks_existing: pd.Dataframe with column 'preprocessed_text'
    :param tasks_to_retrieve: pd.Dataframe with column 'preprocessed_text'
    :param output_path: path to save file
    :param encoder: class with method 'encode' which takes list of problem statements and returns numpy list of vectors
    :param top_k: how many tasks to retrieve
    :param batch_size:
    :param file_type: this is for separator in output_file. If tsv, then sep='\t', else sep=','
    :return:
    """
    if not hasattr(encoder, "encode"):
        raise ValueError("Incorrect encoder object, should have 'encode' method")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # encoding tasks
    vectors_existing = encoder.encode(list(tasks_existing.preprocessed_text), batch_size=batch_size)
    vectors_to_retrieve = encoder.encode(list(tasks_to_retrieve.preprocessed_text), batch_size=batch_size)

    # calculating distances
    dists = pairwise_distances(vectors_to_retrieve, vectors_existing, "cosine")
    ans = pd.DataFrame({"step_text": [], "step_id": [], "course_id": [], "distance": [], "top_N": []})

    # retrieving
    for i, dist_vector in enumerate(dists):
        # append row, representing test task
        ans = ans.append(
            {
                "step_text": [tasks_to_retrieve.at[i, "preprocessed_text"]],
                "step_id": [tasks_to_retrieve.at[i, "id"]],
                "course_id": [tasks_to_retrieve.at[i, "course_id"]],
                "distance": [""],
                "top_N": [""],
            }
        )
        # sorted indexes
        indexes = dist_vector.argsort()
        j = 0
        while tasks_existing.at[indexes[j], "preprocessed_text"] == tasks_to_retrieve.at[i, "preprocessed_text"]:
            j += 1
        indexes = indexes[j : j + top_k]
        tmp = tasks_existing.iloc[indexes]
        # append top_k closest tasks
        ans = ans.append(
            {
                "step_text": tmp.preprocessed_text,
                "step_id": tmp.id,
                "course_id": tmp.course_id,
                "distance": dist_vector[indexes],
                "top_N": list(range(1, top_k + 1)),
            }
        )

    # saving
    if file_type == "tsv":
        sep = "\t"
    else:
        sep = ","
    ans.to_csv(os.path.join(output_path, "retrieval.csv"), sep=sep, index=False, header=True)


def main(existing_path: str, retrieve_path: str, model_path: str, output_path: str):
    tasks_existing = pd.read_csv(existing_path)
    tasks_to_retrieve = pd.read_csv(retrieve_path)

    encoder = SentenceTransformer(model_path)

    retrieve_to_file(tasks_existing, tasks_to_retrieve, output_path, encoder, top_k=5, batch_size=64, file_type="tsv")


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-e", "--existing_tasks", required=True, help="Path to existing tasks .csv file")
    arg_parser.add_argument("-r", "--retrieve_tasks", required=True, help="Path to tasks for which retrieve .csv file")
    arg_parser.add_argument("-m", "--model", required=True, help="Path to model file")
    arg_parser.add_argument("-o", "--output", required=True, help="Path to output directory")
    return arg_parser


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    if os.path.exists(__args.existing_tasks) and os.path.exists(__args.retrieve_tasks) and os.path.exists(__args.model):
        main(__args.existing_tasks, __args.retrieve_tasks, __args.__model, __args.output)
    else:
        raise FileNotFoundError("Can't find necessary files, please, check correctness")
