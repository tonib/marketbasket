import marketbasket.settings as settings
from typing import Iterable, List, Tuple, Dict
import marketbasket.transactions_file as transactions_file
import marketbasket.transaction as transaction
from collections import Counter
import random
import real_eval
from collections import defaultdict

# Fix seed to  get reproducible datasets
random.seed(1)

class Skipgrams:

    def __init__(self, n_skips = 0):
        self.n_skips = n_skips
        self.rows = {}
        self.n_item_instances = Counter()
        self.n_sequences = 0

    def _add_pair(self, item_0, item_1):
        if item_0 in self.rows:
            column = self.rows[item_0]
        else:
            column = self.rows[item_0] = Counter()
        column[item_1] += 1

    def _add_gram(self, item_0, item_1):
        self._add_pair(item_0, item_1)
        self._add_pair(item_1, item_0)

    def add_sequence_grams(self, items: List):
        length = len(items)
        window = length if self.n_skips <= 0 else self.n_skips
        # Add grams
        for source_item_idx in range(0, length-1):
            source_item = items[source_item_idx]
            end_idx = min(length, source_item_idx + 1 + window)
            for target_item_idx in range(source_item_idx + 1, end_idx):
                self._add_gram(source_item, items[target_item_idx])
        # Items instances count
        for item in items:
            self.n_item_instances[item] += 1
        # Total number of sequences
        self.n_sequences += 1

    def most_common(self, item, n_top: int) -> List[Tuple[object, float]]:
        n_instances = self.n_item_instances[item]
        if n_instances == 0:
            return []
        
        related_top_items = self.rows[item].most_common(n_top)
        return [ (item, related_instances / n_instances) for (item, related_instances) in related_top_items ]

# Get transactions
WINDOW_SIZE = 5
eval_transactions = []
with transactions_file.TransactionsFile(transactions_file.TransactionsFile.top_items_path(), 'r') as trn_file:
    skipgrams = Skipgrams(WINDOW_SIZE)
    for trn in trn_file:
        if random.random() <= settings.settings.evaluation_ratio:
            eval_transactions.append( trn )
        else:
            skipgrams.add_sequence_grams(trn.item_labels)

# print(skipgrams.n_item_instances['21131'])
# print(skipgrams.rows['21131'].most_common(10))
# print( skipgrams.most_common('21131', 10) )

PROBABILITY_DECAY = 0.5
def predict(skipgrams: Skipgrams, prior_items: List, n_top: int):

    probable_items: Dict[object, float] = defaultdict(lambda: 0.0)

    # Traverse prior items, the most recently requested first
    # Reduce the predicted probability by older items exponentially with decay variable
    decay = 1.0
    for prior_item in reversed(prior_items):
        # Get top items for each individual item in prior items
        probables_for_prior = skipgrams.most_common(prior_item, n_top * 3)

        # Remove items already in prior
        probables_for_prior = filter(lambda x: x[0] not in prior_items, probables_for_prior)

        # Traverse predicted items
        for probable_item, probability in probables_for_prior:
            probability *= decay
            current_probability = probable_items[probable_item]
            if probability > current_probability:
                probable_items[probable_item] = probability
        decay *= PROBABILITY_DECAY

    # Sort by most probable
    probable_items = sorted(probable_items.items(), key=lambda probable_item: probable_item[1], reverse=True)
    probable_items = probable_items[0:n_top]
    # Return (items, probabilities)
    return tuple(zip(*probable_items))

def predict_with_voting(skipgrams: Skipgrams, prior_items: List, n_top: int):

    # Get candidades purposed by each item in prior
    candidate_items_set = set()
    prior_votes = []
    for prior_item in prior_items:

        # Get top items for each individual item in prior items
        vote = skipgrams.most_common(prior_item, n_top * 3)

        # Remove items already in prior
        vote = filter(lambda x: x[0] not in prior_items, vote)

        # prior_item votes. key = item, value = probability. Default value is 0.0
        vote = defaultdict(lambda: 0.0, vote)

        # Add all candidate items voted by this prior
        for candidate_item in vote:
            candidate_items_set.add(candidate_item)
        
        # Store the vote to avoid recalculate it later
        prior_votes.append(vote)

    # Now, for each candidate, get votes by each prior, and ponderate them with the decay
    #n_priors = len(prior_items)
    candidates_probabilities = {}
    for candidate_item in candidate_items_set:
        probabilities_sum = 0.0
        decay = 1.0
        weights_sum = 0
        # Do a weighted probabilities sum. Most recent items will have a higher probability weight
        for prior_vote in reversed(prior_votes):
            probabilities_sum += prior_vote[candidate_item] * decay
            weights_sum += decay
            decay *= PROBABILITY_DECAY
        candidates_probabilities[candidate_item] = probabilities_sum / weights_sum
    
    # Get most voted candidates
    probable_items = sorted(candidates_probabilities.items(), key=lambda probable_item: probable_item[1], reverse=True)
    probable_items = probable_items[0:n_top]
    # Return (items, probabilities)
    return tuple(zip(*probable_items))

prediction_function = predict_with_voting

# print()
# print( predict(skipgrams, ['21131'], 10) )
# print()
# print( predict(skipgrams, ['21131', '21131', 'achilipu'], 10) )
# print()
# print( predict(skipgrams, ['21177', '21131'], 10) )

# Number of top predicted items to count
N_TOP_PREDICTIONS = 32
OUT_OF_RANKINGS = N_TOP_PREDICTIONS + 1

def rank_prediction(predicted_items, predicted_probabilities, expected_item, prediction_rankings: Counter) -> float:
    if expected_item in predicted_items:
        expected_item_idx = predicted_items.index(expected_item)
        probability = predicted_probabilities[expected_item_idx]
        ranking = expected_item_idx + 1
    else:
        probability = 0.0
        ranking = OUT_OF_RANKINGS
    prediction_rankings[ranking] += 1
    return probability

def print_ranking(prediction_rankings: Counter, n_predictions: int, ranking_top: int):
     # Print results rank
    sum = 0
    for i in range(1, ranking_top + 1):
        if i in prediction_rankings:
            sum += prediction_rankings[i]
    txt_result = "* N. times next item in top " + str(ranking_top) + " predictions: " + str(sum) + " of " + str(n_predictions)
    if n_predictions > 0:
        txt_result += " / Ratio: " + str(sum / n_predictions)
    print(txt_result)

def print_all_rankings():
    print("WINDOW_SIZE =", WINDOW_SIZE, "PROBABILITY_DECAY =", PROBABILITY_DECAY, "Prediction function = " + prediction_function.__name__)
    # Print rankings
    if n_predictions > 0:
        mean_ranking = sum(ranking * count for ranking, count in prediction_rankings.items()) / n_predictions
        print("Mean ranking:", mean_ranking)
    for rank in [1, 8, 16, 32]:
        print_ranking(prediction_rankings, n_predictions, rank)
    if OUT_OF_RANKINGS in prediction_rankings:
        n_out_of_rank = prediction_rankings[OUT_OF_RANKINGS]
        print("Predictions out of rank: " + str(n_out_of_rank) + " / Ratio: " + str(n_out_of_rank / n_predictions))

    if n_predictions > 0:
        print("Mean probability: " + str(probs_sum / n_predictions))
    print()

# Do evaluation
print("Evaluating")
prediction_rankings = Counter()
probs_sum = 0.0
n_predictions = 0
trn: transaction.Transaction
for trn in eval_transactions:
    items = trn.item_labels
    for i in range(1, len(items)):
        prior_items = items[0:i] if WINDOW_SIZE <= 0 else items[max(0, i-WINDOW_SIZE):i]
        item_to_predict = items[i]

        predicted_items, probabilities = prediction_function(skipgrams, prior_items, 64)
        
        probs_sum += rank_prediction(predicted_items, probabilities, item_to_predict, prediction_rankings)
        n_predictions += 1
        if n_predictions % 2000 == 0:
            print_all_rankings()


print_all_rankings()
