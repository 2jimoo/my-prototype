def print_dicts(dict):
    for key, values in dict.items():
        line = f"{key}: " + " ".join(map(str, values)) + "\n"
        print(line)


def print_dict(dict):
    for key, value in dict.items():
        line = f"{key} " + str(value) + "\n"
        print(line)
