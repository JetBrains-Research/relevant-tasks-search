from argparse import ArgumentParser


def export_hyperskill_tasks(output_path: str):
    pass


def build_hyperskill_knowledge_graph(output_path: str):
    pass


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("task", choices=["tasks", "graph"], required=True)
    arg_parser.add_argument("-o", "--output", required=True, help="Path to output file")
    return arg_parser


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    if __args.task == "tasks":
        export_hyperskill_tasks(__args.output)
    else:
        build_hyperskill_knowledge_graph(__args.output)
