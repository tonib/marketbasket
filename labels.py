from typing import List, Dict

class Labels:

    UNKNOWN_LABEL = "[UNKNOWN]"

    def __init__(self, labels: List[str]):

        # Index to label access
        self.labels: List[str] = []

        # Label to index access
        self.indices: Dict[str, int] = {}

        for label in labels:
            self.append(label)

    def save(self, path: str):
        with open(path, 'w') as labels_file:
            for label in self.labels:
                labels_file.write(label + '\n')

    def contains(self, label: str) -> bool:
        return label in self.indices

    def append(self, label: str):
        self.indices[label] = len(self.labels)
        self.labels.append(label)

    def label_index(self, label: str) -> int:
        return self.indices[label]

    def labels_indices(self, labels: List[str]) -> List[int]:
        return [self.indices[label] for label in labels]

    @staticmethod
    def load(path: str) -> 'Labels':
        labels: List[str] = []
        with open(path) as labels_file:
            for label in labels_file:
                labels.append(label.strip())
        return Labels(labels)
