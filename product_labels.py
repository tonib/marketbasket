from typing import List, Dict

class ProductLabels:

    def __init__(self, labels: List[str]):

        # Index to label access
        self.labels: List[str] = []

        # Label to index access
        self.indices: Dict[str, int] = {}

        for label in labels:
            self.indices[label] = len(self.labels)
            self.labels.append(label)

    def save(self):
        with open('data/itemcodes.txt', 'w') as item_codes_file:
            for item_code in self.labels:
                item_codes_file.write(item_code + '\n')

    @staticmethod
    def load() -> 'ProductLabels':
        labels: List[str] = []
        with open('data/itemcodes.txt') as item_codes_file:
            for label in item_codes_file:
                labels.append(label)
        return ProductLabels(labels)
