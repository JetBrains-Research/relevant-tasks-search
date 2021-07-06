import requests
import json
import re
from argparse import ArgumentParser


def export_hyperskill_tasks(output_path: str, start_page: str):
    page_number = int(start_page) if start_page is not None else 1
    while True:
        results_fout = open(output_path, "a")
        lines_added = 0
        r = requests.get(f"https://hyperskill.org/api/steps?page={page_number}&format=json")
        info = json.loads(r.text)
        for step in info["steps"]:
            if step["type"] == 'practice':
                text = re.sub(r"\n", " ", step["block"]["text"])
                step_id = str(step["id"])
                topic_id = str(step["topic"])
                title = step["title"]
                results_fout.write(step_id + '\t' + topic_id + '\t' + title + '\t' + text + '\n')
                lines_added += 1
        results_fout.close()
        with open("./logs", 'a') as logs_fout:
            logs_fout.write(f"{page_number} done, {lines_added} lines added\n")
        print(f"{page_number} done, {lines_added} lines added")
        if info["meta"]["has_next"]:
            page_number += 1
        else:
            break


def build_hyperskill_knowledge_graph(output_path: str):
    pass


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-t", "--task", choices=["tasks", "graph"], required=True)
    arg_parser.add_argument("-o", "--output", required=True, help="Path to output file")
    arg_parser.add_argument("-sp", "--start_page", required=False)
    return arg_parser


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    if __args.task == "tasks":
        export_hyperskill_tasks(__args.output, __args.start_page)
    else:
        build_hyperskill_knowledge_graph(__args.output)
