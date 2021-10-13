feature {
  name: "session_id"
  type: INT
  int_domain {
    name: "session_id"
    min: 1
    max: 9249733 
    is_categorical: false
  }
  annotation {
    tag: "groupby_col"
  }
}
feature {
  name: "item_id-list_seq"
  value_count {
    min: 2
    max: 185
  }
  type: INT
  int_domain {
    name: "item_id/list"
    min: 1
    max: 52742
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
  name: "category-list_seq"
  value_count {
    min: 2
    max: 185
  }
  type: INT
  int_domain {
    name: "category-list_seq"
    min: 1
    max: 337
    is_categorical: true
  }
  annotation {
    tag: "list"
    tag: "categorical"
    tag: "item"
  }
}
feature {
  name: "product_recency_days_log_norm-list_seq"
  value_count {
    min: 2
    max: 185
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
  name: "et_dayofweek_sin-list_seq"
  value_count {
    min: 2
    max: 185
  }
  type: FLOAT
  float_domain {
    name: "et_dayofweek_sin-list_seq"
    min: 0.7421683
    max: 0.9995285
  }
  annotation {
    tag: "continuous"
    tag: "time"
    tag: "list"
  }
}
