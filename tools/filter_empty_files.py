import os


def filter_empty_files(directory: str) -> int:
    empty_counter = 0
    remain_counter = 0

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as f:
                content = f.read().strip()

            if content == "" or content == "[\n]":
                new_filepath = os.path.join(directory, filename.replace(".json", ".empty"))
                print(f"Rename empty file {filepath} to {new_filepath}")
                os.rename(filepath, new_filepath)
                empty_counter += 1
            else:
                remain_counter += 1

    print(f"Remain {remain_counter} log files in {directory}")
    return remain_counter
