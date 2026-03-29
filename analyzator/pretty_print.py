import json
from pprint import pformat

def pretty_print_dict(
    data,
    append=False, 
    console=True,
    file_path="output.txt",
    as_json=True,
    indent=2,
):
    if as_json:
        try:
            formatted = json.dumps(data, indent=indent, ensure_ascii=False)
        except TypeError:
            formatted = pformat(data, indent=indent)
    else:
        formatted = pformat(data, indent=indent)

    if console:
        print(formatted)

    # режим записи
    mode = "a" if append else "w"

    with open(file_path, mode, encoding="utf-8") as f:
        if append:
            f.write("\n\n===========================================================================\n\n")
        f.write(formatted)

    action = "добавлено в" if append else "сохранено в"
    print(f"\nДанные {action} файл: {file_path}")