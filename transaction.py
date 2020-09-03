from typing import List

class Transaction:

    def __init__(self, text_line: str):
        words: List[str] = text_line.split()

        # First word is the customer code
        self.customer_label: str = words[0]
        del words[0]

        # Others are item labels
        self.item_labels: List[str] = words


    def __repr__(self):
        return self.customer_label + ' ' + ' '.join(self.item_labels)

    def remove_duplicated_items(self):
        # As Python 3.7+, dict indices keep insertion order...
        self.item_labels = list(dict.fromkeys( self.item_labels ))
