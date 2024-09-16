def print_dicts(dict):
    for key, value in dict.items():
        print(f"{key}: {value}. {type(key)}: {type(value)}")


def print_dict(dict):
    for key, value in dict.items():
        line = f"{key} " + str(value) + "\n"
        print(line)
