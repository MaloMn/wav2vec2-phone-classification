from typing import List
import collections.abc

__all__ = ["override_configuration"]


def read_yaml(filename: str) -> List:
    with open(filename, "r") as f:
        output = [line.split(":", 1) for line in f.readlines()]

    output = list(filter(lambda x: x[0] != "\n" and x[0][0] != "#", output))
    return output


def yaml_to_json(data):
    return recursive(data, {})[0]


def recursive(data, current, current_indent = 0, i = 0):
    if i >= len(data):
        return current, i

    if i >= len(data) - 1:
        # print("# STOP 1")
        if len(data[i]) > 1:
            current[data[i][0]] = data[i][1]
        else:
            current["\n"] = data[i][0]

        return current, i + 1

    # Stop if next level of indent is smaller
    if data[i + 1][0].count("   ") < current_indent:
        # print("# STOP 2")
        # print(data[i], data[i + 1])

        if len(data[i]) > 1:
            current[data[i][0]] = data[i][1]
        else:
            current["\n"] = data[i][0]

        return current, i + 1

    # If next level of indent is bigger, then treat separately what's inside
    if data[i + 1][0].count("   ") > current_indent:
        # print("# BRANCH 1", data[i+1], data[i + 1][0].count("  "), current_indent)
        sub_data, next_i = recursive(data, {}, current_indent = current_indent + 1, i=i + 1)
        current[data[i][0]] = [data[i][1], sub_data]
        # print("branch 1 result", current)

        return recursive(data, current, current_indent=current_indent, i=next_i)

    # If we have the same level of indent, we keep going
    if data[i + 1][0].count("   ") == current_indent:
        # print("# BRANCH 2")

        if len(data[i]) > 1:
            current[data[i][0]] = data[i][1]
        else:
            current["\n"] = data[i][0]

        return recursive(data, current, current_indent=current_indent, i=i + 1)


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def write_json_to_yaml(d, file):
    for k, v in d.items():
        if isinstance(v, collections.abc.Mapping):
            write_json_to_yaml(v, file)
        if isinstance(v, list):
            file.write(f"{k}:{v[0]}")
            write_json_to_yaml(v[1], file)
        else:
            if k == "\n":
                file.write(v)
            else:
                file.write(f"{k}:{v}")


def override_configuration(base_filename: str, override_filename: str, output_filename: str):
    base_config = read_yaml(base_filename)
    override = read_yaml(override_filename)

    with open(output_filename, "w+") as f:
        write_json_to_yaml(update(yaml_to_json(base_config), yaml_to_json(override)), f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('base_configuration', type=str)
    parser.add_argument('override_configuration', type=str)
    parser.add_argument('output', type=str)

    args = parser.parse_args()

    override_configuration(args.base_configuration, args.override_configuration, args.output)
