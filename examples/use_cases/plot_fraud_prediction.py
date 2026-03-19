"""
Using skore with a skrub DataOp
===============================

TODO
"""

# %%
import skore
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier

dataset = skrub.datasets.fetch_credit_fraud(split="all")

products = skrub.var("products", dataset.products)
baskets = skrub.var("baskets", dataset.baskets)

basket_ids = baskets[["ID"]].skb.mark_as_X()
fraud_flags = baskets["fraud_flag"].skb.mark_as_y()


def filter_products(products, basket_ids):
    return products[products["basket_ID"].isin(basket_ids["ID"])]


vectorized_products = products.skb.apply_func(filter_products, basket_ids).skb.apply(
    skrub.TableVectorizer(), exclude_cols="basket_ID"
)


def join_product_info(basket_ids, vectorized_products):
    return basket_ids.merge(
        vectorized_products.groupby("basket_ID").agg("mean").reset_index(),
        left_on="ID",
        right_on="basket_ID",
    ).drop(columns=["ID", "basket_ID"])


pred = basket_ids.skb.apply_func(join_product_info, vectorized_products).skb.apply(
    HistGradientBoostingClassifier(), y=fraud_flags
)

# This would generate a report with previous of intermediate results & fitted
# estimators:
#
# pred.skb.full_report()

pred

# %%
report = skore.EstimatorReport(pred, pos_label=1)
report.metrics.roc_auc()

# %%
report.metrics.precision_recall().plot()
