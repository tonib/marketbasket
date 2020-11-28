from .settings import settings
from typing import List, Dict
import numpy as np

class Labels:

    # Label for unknown customers
    UNKNOWN_LABEL = "[UNKNOWN]"
    
    def __init__(self, labels: List[str]):

        # Only unique labels
        assert len(set(labels)) == len(labels), "Labels are not unique"

        # Index to label access
        self.labels: np.ndarray = np.array( list(labels) )

        # Label to index access
        self.indices: Dict[str, int] = {}
        #print(self.labels)
        for i in range( len(self.labels) ):
            self.indices[self.labels[i]] = i

    def save(self, path: str):
        with open(path, 'w') as labels_file:
            for label in self.labels:
                labels_file.write(label + '\n')

    def contains(self, label: str) -> bool:
        return label in self.indices

    def append(self, label: str):
        self.indices[label] = len(self.labels)
        self.labels = np.append(self.labels, label)
        assert len(set(self.labels)) == len(self.labels), "Labels are not unique"

    def index_label(self, index: int) -> str:
        return self.labels[index]

    def label_index(self, label: str) -> int:
        return self.indices[label]

    def labels_indices(self, labels: List[str]) -> List[int]:
        return [self.indices[label] for label in labels]

    def length(self) -> int:
        return len(self.labels)

    @staticmethod
    def load(path: str) -> 'Labels':
        labels: List[str] = []
        with open(path) as labels_file:
            for label in labels_file:
                labels.append(label.strip())
        return Labels(labels)

    @staticmethod
    def item_labels_path() -> str:
        """ Returns item labels path """
        return settings.get_data_path('itemlabels.txt')
    
    @staticmethod
    def customer_labels_path() -> str:
        """ Returns item labels path """
        return settings.get_data_path('customerlabels.txt')