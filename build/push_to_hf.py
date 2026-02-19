import datasets

freak_dataset = datasets.load_dataset("./freak")

freak_dataset.push_to_hub("hansQAQ/FREAK")