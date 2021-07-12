from argparse import ArgumentParser

import requests
import json
import re

import pickle
import os
import queue
from pyvis.network import Network


def export_hyperskill_tasks(output_path: str, start_page: int):
    """
    Function used for exporting Hyperskill's tasks using public API.
    Saving the result in ./output_path with in the following format:
    step_id \t topic_id \t title \t text \n
    :param output_path: string containing path to output file
    :param start_page: page to start work with API
    """
    if not os.path.isdir("./logs"):
        os.mkdir("./logs")
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    page_number = start_page
    while True:
        results_fout = open(f"{output_path}/steps.tsv", "a")
        lines_added = 0
        r = requests.get(f"https://hyperskill.org/api/steps?page={page_number}&format=json")
        info = json.loads(r.text)
        for step in info["steps"]:
            if step["type"] == "practice":
                text = re.sub(r"\n", " ", step["block"]["text"])
                step_id = str(step["id"])
                topic_id = str(step["topic"])
                title = step["title"]
                results_fout.write("\t".join([step_id, topic_id, title, text]))
                lines_added += 1
        results_fout.close()
        with open("./logs/logs_steps", "a") as logs_fout:
            logs_fout.write(f"{page_number} done, {lines_added} lines added\n")
        print(f"{page_number} done, {lines_added} lines added")
        if info["meta"]["has_next"]:
            page_number += 1
        else:
            break


def build_hyperskill_knowledge_graph(output_path: str, start_page: int):
    """
    This function working with Hyperskill's public API and getting the knowledge graph as Python's dictionary.
    Then saving it to output_path/data.pkl.
    :param output_path: string containing name of directory with output results
    :param start_page: page to start work with API
    """
    if not os.path.isdir("./logs"):
        os.mkdir("./logs")
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    page_number = start_page
    dependencies = {}
    id_to_name = {}
    name_to_id = {}
    while True:
        dependencies_added = 0
        r = requests.get(f"https://hyperskill.org/api/topics?page={page_number}&format=json")
        info = json.loads(r.text)
        for topic in info["topics"]:
            topic_id = topic["id"]
            if topic_id not in id_to_name:
                id_to_name[topic_id] = topic["title"]
                name_to_id[topic["title"]] = topic_id
            if topic_id not in dependencies:
                dependencies[topic_id] = []
            for candidate in topic["children"] + topic["followers"] + topic["prerequisites"]:
                if candidate not in dependencies[topic_id]:
                    dependencies_added += 1
                    dependencies[topic_id].append(candidate)
                    if candidate not in dependencies:
                        dependencies[candidate] = [topic_id]
                    else:
                        dependencies[candidate].append(topic_id)
        data = {"graph": dependencies, "id2name": id_to_name, "name2id": name_to_id}
        with open(f"{output_path}/data.pkl", "wb") as data_fout:
            pickle.dump(data, data_fout)
        with open("./logs/logs_graph", "a") as logs_fout:
            logs_fout.write(f"{page_number} done, {dependencies_added} dependencies added\n")
        print(f"{page_number} done, {dependencies_added} dependencies added")
        if info["meta"]["has_next"]:
            page_number += 1
        else:
            break


def draw_hyperskill_knowledge_graph(graph_path: str):
    """
    Function that draws knowledge graph, located in graph_path.
    Should be used after build_hyperskill_knowledge_graph.
    :param graph_path: string containing path to directory with data.pkl
    """
    if not os.path.exists(f"{graph_path}/data.pkl"):
        raise Exception(f"{graph_path}/data.pkl is not exist, use 'build_hyperskill_knowledge_graph' first")
    with open(f"{graph_path}/data.pkl", "rb") as fin:
        dependencies = pickle.load(fin)["dependencies"]
    net = Network(height=1080, width=1920, directed=False, notebook=False)
    net.barnes_hut(gravity=-10000, overlap=1, spring_length=1)
    for node in dependencies:
        net.add_node(node, label=node)
    for node in dependencies:
        for out in dependencies[node]:
            net.add_edge(node, out)
    net.show("graph.html")


def calculate_distances(graph_path: str):
    """
    Function to calculating all distances in graph, located in output_path directory, saving as Python's dictionary.
    For calculation all distances used BFS, applied for all nodes sequentially.
    After calculation saving distances in graph_path/distances.pkl.
    Should be used after build_hyperskill_knowledge_graph.
    :param graph_path: string containing path to directory with data.pkl
    """
    if not os.path.exists(f"{graph_path}/data.pkl"):
        raise Exception(f"{graph_path}/data.pkl is not exist, use 'build_hyperskill_knowledge_graph' first")
    with open(f"{graph_path}/data.pkl", "rb") as fin:
        dependencies = pickle.load(fin)["dependencies"]
    distances = {}
    count = 0
    for root in dependencies:
        visited = {root}
        q = queue.Queue()
        q.put((root, 0))
        distances[(root, root)] = 0
        while not q.empty():
            node, d = q.get()
            for child in dependencies[node]:
                if child not in visited:
                    visited.add(child)
                    q.put((child, d + 1))
                    if (root, child) not in distances:
                        distances[(root, child)] = d + 1
                        distances[(child, root)] = d + 1
        count += 1
        if count % 100 == 0:
            print(f"{int(100 * count / len(dependencies.keys()))}% completed. Distances calculated for {root}")
    with open(f"{graph_path}/distances.pkl", "wb") as fout:
        pickle.dump(distances, fout)


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-t", "--task", choices=["tasks", "graph", "draw", "distances"], required=True)
    arg_parser.add_argument("-o", "--output", required=True, help="Path to output file")
    arg_parser.add_argument("-sp", "--start_page", required=False)
    return arg_parser


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    if __args.start_page is None:
        __args.start_page = 1
    else:
        __args.start_page = int(__args.start_page)
    if __args.task == "tasks":
        export_hyperskill_tasks(__args.output, __args.start_page)
    elif __args.task == "graph":
        build_hyperskill_knowledge_graph(__args.output, __args.start_page)
    elif __args.task == "draw":
        draw_hyperskill_knowledge_graph(__args.output)
    elif __args.task == "distances":
        calculate_distances(__args.output)
