def print_dicts(dict):
    for key, value in dict.items():
        print(f"{key}: {value}. {type(key)}: {type(value)}")


def print_dict(dict):
    for key, value in dict.items():
        line = f"{key} " + str(value) + "\n"
        print(line)


def print_assignment(cluster_manager):
    for key, value in cluster_manager.assignment_table.items():
        key = cluster_manager.centroid_memory[key]
        print(f"centroid {key}: {value}")
