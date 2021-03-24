# Code from https://github.com/gabrielspmoreira/chameleon_recsys/blob/master/nar_module/nar/metrics.py

from __future__ import absolute_import, division, print_function

import math
from collections import defaultdict

import numpy as np
from sklearn.metrics import pairwise


def cosine_distance(v1, v2):
    # As cosine similarity interval is [-1.0, 1.0], the cosine distance interval is [0.0, 2.0].
    # This normalizes the cosine distance to interval [0.0, 1.0]
    return pairwise.cosine_distances(v1, v2) / 2.0


# For ranks index starting from 0
def log_rank_discount(k):
    return 1.0 / math.log2(k + 2)


class StreamingMetric:
    name = "undefined"

    def __init__(self, topn):
        self.topn = topn
        self.reset()

    def reset(self):
        pass

    def add(self, predictions, labels):
        pass

    def result(self):
        pass


class MRR(StreamingMetric):

    name = "mrr_at_n"

    def __init__(self, topn):
        super().__init__(topn)

    def reset(self):
        self.mrr_results = []

    def add(self, predictions, labels):
        measures = []
        for row_idx, session_labels in enumerate(labels):
            for col_idx, item_label in enumerate(session_labels):
                if item_label != 0:
                    correct_preds = (item_label == predictions[row_idx, col_idx])[
                        : self.topn
                    ].astype(np.int32)
                    correct_preds_pos = np.where(correct_preds)[0]

                    reciprocal_rank = 0
                    if len(correct_preds_pos) > 0:
                        reciprocal_rank = 1.0 / (1 + correct_preds_pos[0])
                    measures.append(reciprocal_rank)
        self.mrr_results.extend(measures)

    def result(self):
        avg_mrr = np.mean(self.mrr_results)
        return avg_mrr


class NDCG(StreamingMetric):

    name = "ndcg_at_n"

    def __init__(self, topn):
        super().__init__(topn)

    def reset(self):
        self.ndcg_results = []

    def add(self, predictions, labels):
        measures = []
        for row_idx, session_labels in enumerate(labels):
            for col_idx, item_label in enumerate(session_labels):
                if item_label != 0:
                    correct_preds = (
                        item_label == predictions[row_idx, col_idx]
                    ).astype(np.int32)
                    ndcg = NDCG._ndcg_at_k(correct_preds, self.topn)
                    measures.append(ndcg)
        self.ndcg_results.extend(measures)

    def result(self):
        avg_ndcg = np.mean(self.ndcg_results)
        return avg_ndcg

    @staticmethod
    def _ndcg_at_k(r, k):
        # Based on https://gist.github.com/bwhite/3726239, but with alternative formulation of DCG
        # which places stronger emphasis on retrieving relevant documents (used in Kaggle)
        def dcg_at_k(r, k):
            r = np.asfarray(r)[:k]
            if r.size:
                return np.sum((np.power(2, r) - 1) / np.log2(np.arange(2, r.size + 2)))
            return 0.0

        dcg_max = dcg_at_k(sorted(r, reverse=True), k)
        if not dcg_max:
            return 0.0
        return dcg_at_k(r, k) / dcg_max


class HitRate(StreamingMetric):

    name = "hitrate_at_n"

    def __init__(self, topn):
        super().__init__(topn)

    def reset(self):
        self.hitrate_total = 0
        self.hitrate_matches = 0

    def add(self, predictions, labels):
        total = 0
        matches = 0
        for row_idx, session_labels in enumerate(labels):
            for col_idx, item_label in enumerate(session_labels):
                if item_label != 0:
                    total += 1
                    if item_label in predictions[row_idx, col_idx][: self.topn]:
                        matches += 1
        self.hitrate_total += total
        self.hitrate_matches += matches

    def result(self):
        hitrate = self.hitrate_matches / float(self.hitrate_total)
        return hitrate


class HitRateBySessionPosition(StreamingMetric):

    name = "hitrate_at_n_by_pos"

    def __init__(self, topn):
        super().__init__(topn)

    def reset(self):
        self.hitrate_matches_by_session_pos = defaultdict(int)
        self.hitrate_total_by_session_pos = defaultdict(int)
        self.norm_pop_by_pos = defaultdict(int)

    def add(self, predictions, labels, labels_norm_pop):
        for row_idx, session_labels in enumerate(labels):
            for col_idx, item_label in enumerate(session_labels):
                if item_label != 0:
                    self.hitrate_total_by_session_pos[col_idx + 1] += 1
                    self.norm_pop_by_pos[col_idx + 1] += labels_norm_pop[
                        row_idx, col_idx
                    ]
                    if item_label in predictions[row_idx, col_idx][: self.topn]:
                        self.hitrate_matches_by_session_pos[col_idx + 1] += 1

    def result(self):
        hitrate_by_session_pos = dict(
            [
                (
                    key,
                    (
                        self.hitrate_matches_by_session_pos[key]
                        if key in self.hitrate_matches_by_session_pos
                        else 0
                    )
                    / float(self.hitrate_total_by_session_pos[key]),
                )
                for key in self.hitrate_total_by_session_pos
            ]
        )

        avg_norm_pop_by_session_pos = dict(
            [
                (
                    key,
                    (self.norm_pop_by_pos[key] if key in self.norm_pop_by_pos else 0)
                    / float(self.hitrate_total_by_session_pos[key]),
                )
                for key in self.hitrate_total_by_session_pos
            ]
        )

        return (
            hitrate_by_session_pos,
            avg_norm_pop_by_session_pos,
            self.hitrate_total_by_session_pos,
        )


class PopularityBias(StreamingMetric):

    name = "pop_bias_at_n"

    def __init__(self, topn):
        super().__init__(topn)

    def reset(self):
        self.results = []

    def add(self, predictions, labels, predictions_norm_pop):
        measures = []
        for row_idx, session_preds_norm_pop in enumerate(predictions_norm_pop):
            for col_idx, item_preds_norm_pop in enumerate(session_preds_norm_pop):
                # If this is not a padded item
                if labels[row_idx, col_idx] != 0:
                    click_top_predictions_pop_norm = item_preds_norm_pop[: self.topn]
                    measures.extend(click_top_predictions_pop_norm)

        self.results.extend(measures)

    def result(self):
        avg_pop = np.mean(self.results)
        return avg_pop


class Novelty(StreamingMetric):

    name = "novelty_at_n"

    def __init__(self, topn):
        super().__init__(topn)

    def reset(self):
        self.results = []

    def add(self, predictions, labels, predictions_norm_pop):
        measures = []
        for row_idx, session_preds_norm_pop in enumerate(predictions_norm_pop):
            for col_idx, item_preds_norm_pop in enumerate(session_preds_norm_pop):
                # If this is not a padded item
                if labels[row_idx, col_idx] != 0:
                    # From "Novelty and Diversity Metrics for Recommender Systems - Choice, Discovery and Relevance" (2011)
                    novelty = -np.log2(item_preds_norm_pop[: self.topn])
                    measures.extend(novelty)

        self.results.extend(measures)

    def result(self):
        avg_novelty = np.mean(self.results)
        return avg_novelty


class ExpectedRankSensitiveNovelty(StreamingMetric):

    name = "esi-r_at_n"

    def __init__(self, topn):
        super().__init__(topn)

    def reset(self):
        self.results = []

    def add(self, predictions, labels, predictions_norm_pop):

        for row_idx, session_preds_norm_pop in enumerate(predictions_norm_pop):
            for col_idx, item_preds_norm_pop in enumerate(session_preds_norm_pop):
                # If this is not a padded item
                if labels[row_idx, col_idx] != 0:
                    top_preds_norm_pop = item_preds_norm_pop[: self.topn]

                    disc_weights = []
                    items_novelty = []
                    for i in range(0, len(top_preds_norm_pop) - 1):
                        # Logarithmic rank discount, to prioritize more diverse items in the top of the list
                        discount = log_rank_discount(i)

                        # From "Novelty and Diversity Metrics for Recommender Systems - Choice, Discovery and Relevance" (2011)
                        item_novelty = -np.log2(top_preds_norm_pop[i])

                        items_novelty.append(item_novelty * discount)
                        disc_weights.append(discount)

                    # Expected Novelty with logarithmic rank discount
                    # Adapted from "Incorporating Diversity in a Learning to Rank Recommender System" (2016)
                    avg_novelty = sum(items_novelty) / float(sum(disc_weights))

                    self.results.append(avg_novelty)

    def result(self):
        avg_novelty = np.mean(self.results)
        return avg_novelty


class ExpectedRankRelevanceSensitiveNovelty(StreamingMetric):

    name = "esi-rr_at_n"

    def __init__(self, topn, relevance_positive_sample, relevance_negative_samples):
        super().__init__(topn)
        self.relevance_positive_sample = relevance_positive_sample
        self.relevance_negative_samples = relevance_negative_samples

    def reset(self):
        self.results = []

    def add(self, predictions, labels, predictions_norm_pop):
        for row_idx in range(len(predictions)):
            for col_idx in range(len(predictions[row_idx])):
                # If this is not a padded item
                if labels[row_idx, col_idx] != 0:
                    click_top_predictions = predictions[row_idx, col_idx][: self.topn]
                    click_top_preds_norm_pop = predictions_norm_pop[row_idx, col_idx][
                        : self.topn
                    ]

                    items_novelty = []
                    weights = []
                    for i in range(0, len(click_top_preds_norm_pop) - 1):
                        # Logarithmic rank discount, to prioritize more diverse items in the top of the list
                        discount = log_rank_discount(i)

                        # From "Novelty and Diversity Metrics for Recommender Systems - Choice, Discovery and Relevance" (2011)
                        item_novelty = -np.log2(click_top_preds_norm_pop[i])

                        relevance = (
                            self.relevance_positive_sample
                            if click_top_predictions[i] == labels[row_idx, col_idx]
                            else self.relevance_negative_samples
                        )

                        items_novelty.append(item_novelty * discount * relevance)
                        weights.append(discount)

                    # Expected Novelty with logarithmic rank discount
                    # Adapted from "Incorporating Diversity in a Learning to Rank Recommender System" (2016)
                    avg_novelty = sum(items_novelty) / float(sum(weights))

                    self.results.append(avg_novelty)

    def result(self):
        avg_novelty = np.mean(self.results)
        return avg_novelty


# TODO: Initialize with the clicks in buffer (when evaluation starts)
class ItemCoverage(StreamingMetric):

    name = "item_coverage_at_n"

    def __init__(self, topn, recent_clicks_buffer):
        self.recent_clicks_buffer = recent_clicks_buffer
        super().__init__(topn)

    def reset(self):
        self.clicked_items = set(self.recent_clicks_buffer)
        self.recommended_items = set()

    def add(self, predictions, labels, clicked_items):
        for row_idx, session_predictions in enumerate(predictions):
            for col_idx, item_predictions in enumerate(session_predictions):
                # If this is not a padded item
                if labels[row_idx, col_idx] != 0:
                    click_top_predictions = item_predictions[: self.topn]
                    self.recommended_items.update(click_top_predictions)

        ####Including both the clicked item and all recommended items to ensure that we have a comprehensive set of all recommendable items at a given time
        batch_clicked_items = set(
            np.hstack(
                [labels[np.nonzero(labels)], clicked_items[np.nonzero(clicked_items)]]
            )
        )
        self.clicked_items.update(batch_clicked_items)

    def result(self):
        item_coverage = len(self.recommended_items) / float(len(self.clicked_items))
        return item_coverage


class ContentAverageIntraListDiversity(StreamingMetric):

    name = "content_avg_ild_at_n"

    def __init__(self, topn, content_article_embeddings_matrix):
        super().__init__(topn)
        self.content_article_embeddings_matrix = content_article_embeddings_matrix

    def reset(self):
        self.results = []

    def add(self, predictions, labels):
        for row_idx, session_predictions in enumerate(predictions):
            for col_idx, item_predictions in enumerate(session_predictions):
                # If this is not a padded item
                if labels[row_idx, col_idx] != 0:

                    click_top_predictions = item_predictions[: self.topn]

                    distances = cosine_distance(
                        self.content_article_embeddings_matrix[click_top_predictions],
                        self.content_article_embeddings_matrix[click_top_predictions],
                    )

                    dists = []
                    for i in range(0, len(click_top_predictions) - 1):
                        for j in range(i + 1, len(click_top_predictions)):

                            cos_dist = distances[i, j]

                            dists.append(cos_dist)

                    avg_cos_dist = sum(dists) / float(len(dists))
                    self.results.append(avg_cos_dist)

    def result(self):
        avg_cos_dist = np.mean(self.results)
        return avg_cos_dist


# Inspired in https://lsrs2017.files.wordpress.com/2017/08/lsrs_2017_lamingchen.pdf
class ContentMedianIntraListDiversity(StreamingMetric):

    name = "content_median_ild_at_n"

    def __init__(self, topn, content_article_embeddings_matrix):
        super().__init__(topn)
        self.content_article_embeddings_matrix = content_article_embeddings_matrix

    def reset(self):
        self.results = []

    def add(self, predictions, labels):
        for row_idx, session_predictions in enumerate(predictions):
            for col_idx, item_predictions in enumerate(session_predictions):
                # If this is not a padded item
                if labels[row_idx, col_idx] != 0:
                    click_top_predictions = item_predictions[: self.topn]

                    distances = cosine_distance(
                        self.content_article_embeddings_matrix[click_top_predictions],
                        self.content_article_embeddings_matrix[click_top_predictions],
                    )

                    dists = []
                    for i in range(0, len(click_top_predictions) - 1):
                        for j in range(i + 1, len(click_top_predictions)):

                            cos_dist = distances[i, j]

                            dists.append(cos_dist)

                    median_cos_dist = np.median(dists)
                    self.results.append(median_cos_dist)

    def result(self):
        avg_cos_dist = np.mean(self.results)
        return avg_cos_dist


# Inspired in https://lsrs2017.files.wordpress.com/2017/08/lsrs_2017_lamingchen.pdf
class ContentMinIntraListDiversity(StreamingMetric):

    name = "content_min_ild_at_n"

    def __init__(self, topn, content_article_embeddings_matrix):
        super().__init__(topn)
        self.content_article_embeddings_matrix = content_article_embeddings_matrix

    def reset(self):
        self.results = []

    def add(self, predictions, labels):
        for row_idx, session_predictions in enumerate(predictions):
            for col_idx, item_predictions in enumerate(session_predictions):
                # If this is not a padded item
                if labels[row_idx, col_idx] != 0:

                    click_top_predictions = item_predictions[: self.topn]

                    distances = cosine_distance(
                        self.content_article_embeddings_matrix[click_top_predictions],
                        self.content_article_embeddings_matrix[click_top_predictions],
                    )

                    dists = []
                    for i in range(0, len(click_top_predictions) - 1):
                        for j in range(i + 1, len(click_top_predictions)):

                            cos_dist = distances[i, j]

                            dists.append(cos_dist)

                    min_cos_dist = np.min(dists)
                    self.results.append(min_cos_dist)

    def result(self):
        avg_cos_dist = np.mean(self.results)
        return avg_cos_dist


class ContentExpectedRankSensitiveIntraListDiversity(StreamingMetric):

    name = "content_eild-r_v2_at_n"

    def __init__(self, topn, content_article_embeddings_matrix):
        super().__init__(topn)
        self.content_article_embeddings_matrix = content_article_embeddings_matrix

    def reset(self):
        self.results = []

    def add(self, predictions, labels):
        for row_idx, session_predictions in enumerate(predictions):
            for col_idx, item_predictions in enumerate(session_predictions):
                # If this is not a padded item
                if labels[row_idx, col_idx] != 0:

                    click_top_predictions = item_predictions[: self.topn]

                    distances = cosine_distance(
                        self.content_article_embeddings_matrix[click_top_predictions],
                        self.content_article_embeddings_matrix[click_top_predictions],
                    )

                    avg_dists = []
                    disc_weights = []
                    for i in range(0, len(click_top_predictions) - 1):
                        # Logarithmic rank discount, to prioritize more diverse items in the top of the list
                        discount = 1.0 / math.log2(i + 2)
                        disc_weights.append(discount)

                        dists = []
                        for j in range(i + 1, len(click_top_predictions)):

                            dist = distances[i, j]

                            dists.append(dist)

                        avg_dist = sum(dists) / float(len(dists))
                        avg_dists.append(avg_dist * discount)

                    # Expected Intra-List Diversity (EILD) with logarithmic rank discount
                    # From "Incorporating Diversity in a Learning to Rank Recommender System" (2016)
                    avg_cos_dist = sum(avg_dists) / float(sum(disc_weights))

                    self.results.append(avg_cos_dist)

    def result(self):
        avg_cos_dist = np.mean(self.results)
        return avg_cos_dist


class ContentExpectedRankRelativeSensitiveIntraListDiversity(StreamingMetric):

    name = "content_eild-r_at_n"

    def __init__(self, topn, content_article_embeddings_matrix):
        super().__init__(topn)
        self.content_article_embeddings_matrix = content_article_embeddings_matrix

    def reset(self):
        self.results = []

    def add(self, predictions, labels):
        for row_idx, session_predictions in enumerate(predictions):
            for col_idx, item_predictions in enumerate(session_predictions):
                # If this is not a padded item
                if labels[row_idx, col_idx] != 0:

                    click_top_predictions = item_predictions[: self.topn]

                    distances = cosine_distance(
                        self.content_article_embeddings_matrix[click_top_predictions],
                        self.content_article_embeddings_matrix[click_top_predictions],
                    )

                    avg_dists = []
                    disc_weights = []
                    for i in range(0, len(click_top_predictions) - 1):

                        dists = []
                        weights = []
                        for j in range(0, len(click_top_predictions)):
                            # Ignoring self-similarity
                            if j == i:
                                continue

                            dist = distances[i, j]

                            # Under the assumption that diversity of the items is more perceived by users when items are near in the ranked list
                            rel_discount = log_rank_discount(max(0, j - i - 1))
                            dists.append(dist * rel_discount)
                            weights.append(rel_discount)

                        weighted_avg_dists = sum(dists) / float(sum(weights))

                        # Logarithmic rank discount, to prioritize more diverse items in the top of the list
                        discount = log_rank_discount(i)

                        avg_dists.append(weighted_avg_dists * discount)
                        disc_weights.append(discount)

                    # Expected Intra-List Diversity (EILD) with logarithmic rank discount
                    # From "Incorporating Diversity in a Learning to Rank Recommender System" (2016)
                    avg_cos_dist = sum(avg_dists) / float(sum(disc_weights))

                    self.results.append(avg_cos_dist)

    def result(self):
        avg_cos_dist = np.mean(self.results)
        return avg_cos_dist


class ContentExpectedRankRelativeRelevanceSensitiveIntraListDiversity(StreamingMetric):

    name = "content_eild-rr_at_n"

    def __init__(
        self,
        topn,
        content_article_embeddings_matrix,
        relevance_positive_sample,
        relevance_negative_samples,
    ):
        super().__init__(topn)
        self.content_article_embeddings_matrix = content_article_embeddings_matrix
        self.relevance_positive_sample = relevance_positive_sample
        self.relevance_negative_samples = relevance_negative_samples

    def reset(self):
        self.results = []

    def add(self, predictions, labels):
        for row_idx, session_predictions in enumerate(predictions):
            for col_idx, item_predictions in enumerate(session_predictions):
                # If this is not a padded item
                if labels[row_idx, col_idx] != 0:

                    click_top_predictions = item_predictions[: self.topn]

                    distances = cosine_distance(
                        self.content_article_embeddings_matrix[click_top_predictions],
                        self.content_article_embeddings_matrix[click_top_predictions],
                    )

                    avg_dists = []
                    disc_weights = []
                    for i in range(0, len(click_top_predictions) - 1):

                        dists = []
                        weights = []
                        for j in range(i + 1, len(click_top_predictions)):
                            # Ignoring self-similarity
                            if j == i:
                                continue

                            dist = distances[i, j]

                            # Weights item by relevance
                            relevance_j = (
                                self.relevance_positive_sample
                                if click_top_predictions[j] == labels[row_idx, col_idx]
                                else self.relevance_negative_samples
                            )

                            # Under the assumption that diversity of the items is more perceived by users when items are near in the ranked list
                            rel_discount = log_rank_discount(max(0, j - i - 1))

                            dists.append(dist * rel_discount * relevance_j)
                            weights.append(rel_discount * relevance_j)

                        avg_dists_i = sum(dists) / float(sum(weights))

                        # Weights item by relevance
                        relevance_i = (
                            self.relevance_positive_sample
                            if click_top_predictions[i] == labels[row_idx, col_idx]
                            else self.relevance_negative_samples
                        )

                        # Logarithmic rank discount, to prioritize more diverse items in the top of the list
                        rank_discount_i = log_rank_discount(i)

                        avg_dists.append(avg_dists_i * rank_discount_i * relevance_i)
                        disc_weights.append(rank_discount_i)

                    # Expected Intra-List Diversity (EILD) with logarithmic rank discount
                    # From "Incorporating Diversity in a Learning to Rank Recommender System" (2016)
                    avg_cos_dist = sum(avg_dists) / float(sum(disc_weights))

                    self.results.append(avg_cos_dist)

    def result(self):
        avg_cos_dist = np.mean(self.results)
        return avg_cos_dist


class ContentExpectedRankRelevanceSensitiveIntraListDiversity(StreamingMetric):

    name = "content_eild-rr_v2_at_n"

    def __init__(
        self,
        topn,
        content_article_embeddings_matrix,
        relevance_positive_sample,
        relevance_negative_samples,
    ):
        super().__init__(topn)
        self.content_article_embeddings_matrix = content_article_embeddings_matrix
        self.relevance_positive_sample = relevance_positive_sample
        self.relevance_negative_samples = relevance_negative_samples

    def reset(self):
        self.results = []

    def add(self, predictions, labels):
        for row_idx, session_predictions in enumerate(predictions):
            for col_idx, item_predictions in enumerate(session_predictions):
                # If this is not a padded item
                if labels[row_idx, col_idx] != 0:

                    click_top_predictions = item_predictions[: self.topn]

                    distances = cosine_distance(
                        self.content_article_embeddings_matrix[click_top_predictions],
                        self.content_article_embeddings_matrix[click_top_predictions],
                    )

                    avg_dists = []
                    disc_weights = []
                    for i in range(0, len(click_top_predictions) - 1):

                        dists = []
                        # weights = []
                        for j in range(i + 1, len(click_top_predictions)):
                            # Ignoring self-similarity
                            if j == i:
                                continue

                            """
                                embedding_item_i = self.content_article_embeddings_matrix[click_top_predictions[i]].reshape(1,-1)
                                embedding_item_j = self.content_article_embeddings_matrix[click_top_predictions[j]].reshape(1,-1)
                                dist = cosine_distance(embedding_item_i, embedding_item_j)
                                """

                            dist = distances[i, j]

                            # Weights item by relevance
                            # relevance_j = self.relevance_positive_sample if click_top_predictions[j] == labels[row_idx, col_idx] else self.relevance_negative_samples

                            # Under the assumption that diversity of the items is more perceived by users when items are near in the ranked list
                            # rel_discount = log_rank_discount(max(0, j-i-1))

                            dists.append(dist)
                            # weights.append(rel_discount * relevance_j)

                        # avg_dists_i = sum(dists)/float(sum(weights))
                        avg_dists_i = sum(dists) / float(len(dists))

                        # Weights item by relevance
                        relevance_i = (
                            self.relevance_positive_sample
                            if click_top_predictions[i] == labels[row_idx, col_idx]
                            else self.relevance_negative_samples
                        )

                        # Logarithmic rank discount, to prioritize more diverse items in the top of the list
                        rank_discount_i = log_rank_discount(i)

                        avg_dists.append(avg_dists_i * rank_discount_i * relevance_i)
                        disc_weights.append(rank_discount_i)

                    # Expected Intra-List Diversity (EILD) with logarithmic rank discount
                    # From "Incorporating Diversity in a Learning to Rank Recommender System" (2016)
                    avg_cos_dist = sum(avg_dists) / float(sum(disc_weights))

                    self.results.append(avg_cos_dist)

    def result(self):
        avg_cos_dist = np.mean(self.results)
        return avg_cos_dist


class CategoryExpectedIntraListDiversity(StreamingMetric):

    name = "category_eild_at_n"

    def __init__(self, topn, categories):
        super().__init__(topn)
        self.categories = categories

    def reset(self):
        self.results = []

    def add(self, predictions, labels):
        for row_idx, session_predictions in enumerate(predictions):
            for col_idx, item_predictions in enumerate(session_predictions):
                # If this is not a padded item
                if labels[row_idx, col_idx] != 0:

                    click_top_predictions = item_predictions[: self.topn]

                    # TODO: Vectorize this inner loop (too slow)

                    avg_dists = []
                    disc_weights = []
                    for i in range(0, len(click_top_predictions) - 1):

                        dists = []
                        weights = []
                        for j in range(0, len(click_top_predictions)):
                            # Ignoring self-similarity
                            if j == i:
                                continue

                            category_item_i = self.categories[
                                click_top_predictions[i]
                            ].reshape(1, -1)
                            category_item_j = self.categories[
                                click_top_predictions[j]
                            ].reshape(1, -1)
                            dist = 0.0 if category_item_i == category_item_j else 1.0

                            # Under the assumption that diversity of the items is more perceived by users when items are near in the ranked list
                            rel_discount = log_rank_discount(max(0, j - i - 1))
                            dists.append(dist * rel_discount)
                            weights.append(rel_discount)

                        avg_dist = sum(dists) / float(sum(weights))

                        # Logarithmic rank discount, to prioritize more diverse items in the top of the list
                        discount = log_rank_discount(i)

                        avg_dists.append(avg_dist * discount)
                        disc_weights.append(discount)

                    # Expected Intra-List Diversity (EILD) with logarithmic rank discount
                    # From "Incorporating Diversity in a Learning to Rank Recommender System" (2016)
                    avg_cos_dist = sum(avg_dists) / float(sum(disc_weights))
                    self.results.append(avg_cos_dist)

    def result(self):
        avg_cos_dist = np.mean(self.results)
        return avg_cos_dist
