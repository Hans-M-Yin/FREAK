
import datasets

def load_freaK_dataset():
    freak_dataset = datasets.load_dataset("hansQAQ/FREAK", split='test')
    mcq_len = len(freak_dataset.filter(lambda x: x["type"] == "mcq"))
    fib_len = len(freak_dataset.filter(lambda x: x["type"] == "fib"))

    return freak_dataset, mcq_len, fib_len

def load_freaK_dataset_from_local():
    freak_dataset = datasets.load_dataset("./freak", split='test')
    mcq_len = len(freak_dataset.filter(lambda x: x["type"] == "mcq"))
    fib_len = len(freak_dataset.filter(lambda x: x["type"] == "fib"))

    return freak_dataset, mcq_len, fib_len