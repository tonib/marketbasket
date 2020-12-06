import marketbasket.settings
from typing import List, Dict, Iterable
import numpy as np
import marketbasket.settings as settings

class Labels:

    # Label for unknown labels
    UNKNOWN_LABEL = "[UNKNOWN]"
    
    def __init__(self, labels: List[str]):

        # Only unique labels
        assert len(set(labels)) == len(labels), "Labels are not unique"

        # Index to label access
        self.labels: np.ndarray = np.array( list(labels) )

        # Label to index access
        self.indices: Dict[str, int] = {}
        for i in range( len(self.labels) ):
            self.indices[self.labels[i]] = i

        assert len(set(self.labels)) == len(self.labels), "Labels are not unique"

    def save(self, path: str):
        with open(path, 'w') as labels_file:
            for label in self.labels:
                labels_file.write(label + '\n')

    def contains(self, label: str) -> bool:
        return label in self.indices

    def index_label(self, index: int) -> str:
        return self.labels[index]

    def indices_to_labels(self, indices: Iterable[int]) -> List[str]:
        return [ self.labels[index] for index in indices ]

    def label_index(self, label: str) -> int:
        if label in self.indices:
            return self.indices[label]
        elif Labels.UNKNOWN_LABEL in self.indices:
            return self.indices[Labels.UNKNOWN_LABEL]
        else:
            return -1

    def labels_indices(self, labels: Iterable[str]) -> List[int]:
        return [ self.label_index(label) for label in labels ]
        
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
        return settings.settings.get_data_path('itemlabels.txt')
    
    @staticmethod
    def customer_labels_path() -> str:
        """ Returns item labels path """
        return settings.settings.get_data_path('customerlabels.txt')
