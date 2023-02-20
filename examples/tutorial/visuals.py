import matplotlib.pyplot as plt
import numpy as np


def create_bar_chart(text_file_name):
    ndcg_10 = []
    ndcg_20 = []
    recall_10 = []
    recall_20 = []

    with open(text_file_name, "r") as infile:
        for line in infile:
            if "ndcg_at_10" in line:
                data = [line.rstrip().split(":")]
                key, value = zip(*data)
                ndcg_10.append(float(value[0]))
            elif "ndcg_at_20" in line:
                data = [line.rstrip().split(":")]
                key, value = zip(*data)
                ndcg_20.append(float(value[0]))
            elif "recall_at_10" in line:
                data = [line.rstrip().split(":")]
                key, value = zip(*data)
                recall_10.append(float(value[0]))
            elif "recall_at_20" in line:
                data = [line.rstrip().split(":")]
                key, value = zip(*data)
                recall_20.append(float(value[0]))

    models = ["GRU", "XLNET-MLM", "XLNET-MLM_with_side_info"]

    X_axis = np.arange(len(models))

    plt.subplot(2, 1, 1)
    plt.title("Models' accuracy metrics comparison", pad=20)
    plt.bar(X_axis - 0.2, ndcg_10, 0.4, label="ndcg@10")
    plt.bar(X_axis + 0.2, ndcg_20, 0.4, label="ndcg@20")

    plt.xticks(X_axis, models)
    plt.xlabel("Models")
    plt.ylabel("NDCG")
    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()

    plt.subplot(2, 1, 2)
    plt.bar(X_axis - 0.2, recall_10, 0.4, label="recall@10", color="blue")
    plt.bar(X_axis + 0.2, recall_20, 0.4, label="recall@20", color="lightgreen")

    plt.xticks(X_axis, models)
    plt.xlabel("Models")
    plt.ylabel("Recall")
    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()
