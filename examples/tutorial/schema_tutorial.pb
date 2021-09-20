feature {
  name: "user_session"
  type: INT
  int_domain {
    name: "user_session"
    min: 1
    max: 1877365
    is_categorical: false
  }
  annotation {
    tag: "groupby_col"
  }
}
feature {
  name: "category_id-list_seq"
  value_count {
    min: 2
    max: 20
  }
  type: INT
  int_domain {
    name: "category_id-list_seq"
    min: 1
    max: 566
    is_categorical: true
  }
  annotation {
    tag: "list"
    tag: "categorical"
    tag: "item"
  }
}
feature {
  name: "category_code-list_seq"
  value_count {
    min: 2
    max: 20
  }
  type: INT
  int_domain {
    name: "category_code-list_seq"
    min: 1
    max: 124
    is_categorical: true
  }
  annotation {
    tag: "list"
    tag: "categorical"
    tag: "item"
  }
}
feature {
  name: "brand-list_seq"
  value_count {
    min: 2
    max: 20
  }
  type: INT
  int_domain {
    name: "brand-list_seq"
    min: 1
    max: 2640
    is_categorical: true
  }
  annotation {
    tag: "list"
    tag: "categorical"
    tag: "item"
  }
}
feature {
  name: "product_id-list_seq"
  value_count {
    min: 2
    max: 20
  }
  type: INT
  int_domain {
    name: "product_id-list_seq"
    min: 1
    max: 118334
    is_categorical: true
  }
  annotation {
    tag: "item_id"
    tag: "list"
    tag: "categorical"
    tag: "item"
  }
}
feature {
  name: "product_recency_days_log_norm-list_seq"
  value_count {
    min: 2
    max: 20
  }
  type: FLOAT
  float_domain {
    name: "product_recency_days_log_norm-list_seq"
    min: -2.9177291
    max: 1.5231701
  }
  annotation {
    tag: "continuous"
    tag: "list"
  }
}
feature {
  name: "et_dayofweek_cos-list_seq"
  value_count {
    min: 2
    max: 20
  }
  type: FLOAT
  float_domain {
    name: "et_dayofweek_cos-list_seq"
    min: -0.90096927
    max: 1.0
  }
  annotation {
    tag: "continuous"
    tag: "time"
    tag: "list"
  }
}
feature {
  name: "et_dayofweek_sin-list_seq"
  value_count {
    min: 2
    max: 20
  }
  type: FLOAT
  float_domain {
    name: "et_dayofweek_sin-list_seq"
    min: -0.9749281
    max: 0.9749277
  }
  annotation {
    tag: "continuous"
    tag: "time"
    tag: "list"
  }
}
feature {
  name: "relative_price_to_avg_categ_id-list_seq"
  value_count {
    min: 2
    max: 20
  }
  type: FLOAT
  float_domain {
    name: "relative_price_to_avg_categ_id-list_seq"
    min: -1.0
    max: 38.91276787189527
  }
  annotation {
    tag: "continuous"
    tag: "list"
  }
}
feature {
  name: "price_log_norm-list_seq"
  value_count {
    min: 2
    max: 20
  }
  type: FLOAT
  float_domain {
    name: "price_log_norm-list_seq"
    min: -4.011559
    max: 2.246093
  }
  annotation {
    tag: "continuous"
    tag: "list"
  }
}