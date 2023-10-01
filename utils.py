

def label_mapper(labels: list[str]):
    """
    Method to make a simple label encoding for pytorch
    :param labels: list of labels
    :return: dict with keys as labels (str) and their integers as values
    """
    labels_dict = {}
    for i in range(len(labels)):
        labels_dict[labels[i]] = i

    return labels_dict
