import sys


def print_err(obj):
    """
    Prints to both stdout and stderr
    Mainly for model training in slurm environments
    :param obj: to print
    :return: None
    """
    print(obj)
    print(obj, file=sys.stderr)
    return None


def print_block(phrase, err=False):
    """
    Prints phrase surrounded in hashtags for block effect, can print to stderr if specified
    :param phrase: str, phrase
    :param err: bool, enable stderr
    :return: None
    """
    padding = 6
    block_width = min(len(phrase) + padding * 2 + 2, 100)

    if err:
        print_func = print_err
    else:
        print_func = print

    print_func("#" * block_width)
    print_func("#" + " " * padding + phrase + " " * padding + "#")
    print_func("#" * block_width)
    return None
