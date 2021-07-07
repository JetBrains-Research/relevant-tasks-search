import requests
import json
import re
import pickle
import os
from pyvis.network import Network
from argparse import ArgumentParser


def export_hyperskill_tasks(output_path: str, start_page: str):
    page_number = int(start_page) if start_page is not None else 1
    while True:
        results_fout = open(output_path, "a")
        lines_added = 0
        r = requests.get(f"https://hyperskill.org/api/steps?page={page_number}&format=json")
        info = json.loads(r.text)
        for step in info["steps"]:
            if step["type"] == "practice":
                text = re.sub(r"\n", " ", step["block"]["text"])
                step_id = str(step["id"])
                topic_id = str(step["topic"])
                title = step["title"]
                results_fout.write(step_id + "\t" + topic_id + "\t" + title + "\t" + text + "\n")
                lines_added += 1
        results_fout.close()
        with open("./logs/logs_steps", "a") as logs_fout:
            logs_fout.write(f"{page_number} done, {lines_added} lines added\n")
        print(f"{page_number} done, {lines_added} lines added")
        if info["meta"]["has_next"]:
            page_number += 1
        else:
            break


def build_hyperskill_knowledge_graph(output_path: str, start_page: str):
    if not os.path.isdir("./logs"):
        os.mkdir("./logs")
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    page_number = int(start_page) if start_page is not None else 1
    dependencies = {}
    id_to_name = {}
    name_to_id = {}
    while True:
        dependencies_added = 0
        r = requests.get(f"https://hyperskill.org/api/topics?page={page_number}&format=json")
        info = json.loads(r.text)
        for topic in info["topics"]:
            topic_id = topic["id"]
            if topic_id not in dependencies:
                dependencies[topic_id] = []
                id_to_name[topic_id] = topic["title"]
                name_to_id[topic["title"]] = topic_id
            for candidate in topic["children"] + topic["followers"]:
                dependencies_added += 1
                if candidate not in dependencies[topic_id]:
                    dependencies[topic_id].append(candidate)
                    if candidate not in dependencies:
                        dependencies[candidate] = [topic_id]
                    else:
                        dependencies[candidate].append(topic_id)
        with open(f"{output_path}/graph.pkl", "wb") as graph_fout:
            pickle.dump(dependencies, graph_fout)
        with open(f"{output_path}/id2name.pkl", "wb") as fout:
            pickle.dump(id_to_name, fout)
        with open(f"{output_path}/name2id.pkl", "wb") as fout:
            pickle.dump(name_to_id, fout)
        with open("./logs/logs_graph", "a") as logs_fout:
            logs_fout.write(f"{page_number} done, {dependencies_added} dependencies added\n")
        print(f"{page_number} done, {dependencies_added} dependencies added")
        if info["meta"]["has_next"]:
            page_number += 1
        else:
            break


def draw_hyperskill_knowledge_graph(output_path: str):
    if not os.path.exists(f"{output_path}/graph.pkl"):
        raise Exception(f"{output_path}/graph.pkl is not exist, use 'build_hyperskill_knowledge_graph' first")
    with open(f"{output_path}/graph.pkl", "rb") as fin:
        dependencies = pickle.load(fin)
    net = Network(height=1080, width=1920, directed=False, notebook=False)
    net.barnes_hut(gravity=-10000, overlap=1, spring_length=1)
    for node in dependencies:
        net.add_node(node, label=node)
    for node in dependencies:
        for out in dependencies[node]:
            net.add_edge(node, out)
    net.show("graph.html")


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-t", "--task", choices=["tasks", "graph", "draw"], required=True)
    arg_parser.add_argument("-o", "--output", required=True, help="Path to output file")
    arg_parser.add_argument("-sp", "--start_page", required=False)
    return arg_parser


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    if __args.task == "tasks":
        export_hyperskill_tasks(__args.output, __args.start_page)
    elif __args.task == "graph":
        build_hyperskill_knowledge_graph(__args.output, __args.start_page)
    elif __args.task == "draw":
        draw_hyperskill_knowledge_graph(__args.output)
