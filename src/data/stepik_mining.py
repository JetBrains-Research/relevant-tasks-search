from argparse import ArgumentParser

import requests
import json
import os
import pickle


def dump_stepik_blocks(output_path: str, start_page: int = 1):
    """
    Used for collecting Stepik's blocks of courses
    :param output_path: path to output dir
    :param start_page: page to start mining
    :return:
    """
    if os.path.exists(output_path):
        os.mkdir(output_path)

    page_number = start_page
    groups = {}
    while True:
        url = "https://stepik.org:443/api/catalog-blocks?page={}".format(page_number)
        response = requests.get(url=url)
        info = json.loads(response.text)
        for block in info["catalog-blocks"]:
            for group in block["content"]:
                if "title" not in group or "courses" not in group:
                    continue
                groups[group["title"]] = groups.setdefault(group["title"], []) + group["courses"]
        print("{} page done".format(page_number))
        if info["meta"]["has_next"]:
            page_number += 1
        else:
            break
    for key in groups:
        groups[key] = list(set(groups[key]))
    with open(os.path.join(output_path, "stepik_groups.pkl"), "wb") as fout:
        pickle.dump(groups, fout)


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-sp", "--start_page", type=int, required=True, help="Page to start mining")
    arg_parser.add_argument("-o", "--output", required=True, help="Path to output directory")
    return arg_parser


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    if __args.start_page.isdigit() > 0:
        dump_stepik_blocks(__args.output, start_page=__args.start_page)
    else:
        raise ValueError("Incorrect start page argument, should be positive")
