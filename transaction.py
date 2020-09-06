from typing import List

class Transaction:

    # Top items transactions file path
    TRANSACTIONS_TOP_ITEMS_PATH = 'data/transactions_top_items.txt'
    
    # Raw eval transactions file path
    TRANSACTIONS_EVAL_DATASET_FILE = 'data/eval_transactions.txt'

    def __init__(self, text_line: str = None):

        if not text_line:
            return

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

    def assert_no_duplicates(self):
        assert len(set(self.item_labels)) == len(self.item_labels), "Transaction with duplicated items (" + str(self) + "), Transaction.remove_duplicated_items() failed"

    @staticmethod
    def from_labels(item_labels: List[str], customer_label: str):
        transaction = Transaction()
        transaction.item_labels = item_labels
        transaction.customer_label = customer_label
        return transaction
