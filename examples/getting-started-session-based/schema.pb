feature {
  name: "session_id"
  type: INT
  int_domain {
    name: "session_id"
    min: 1
    max: 11562158
    is_categorical: false
  }
  annotation {
    tag: "groupby_col"
  }
}
feature {
  name: "category-list_trim"
  value_count {
    min: 2
    max: 20
  }
  type: INT
  int_domain {
    name: "category-list_trim"
    min: 1
    max: 334
    is_categorical: true
  }
  annotation {
    tag: "list"
    tag: "categorical"
    tag: "item"
  }
}
feature {
  name: "item_id-list_trim"
  value_count {
    min: 2
    max: 20
  }
  type: INT
  int_domain {
    name: "item_id/list"
    min: 1
    max: 51998
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
  name: "timestamp/age_days-list_trim"
  value_count {
    min: 2
    max: 20
  }
  type: FLOAT
  float_domain {
    name: "timestamp/age_days-list_trim"
    min: -2.9177291
    max: 1.5231701
  }
  annotation {
    tag: "continuous"
    tag: "list"
  }
}
feature {
  name: "timestamp/weekday/sin-list_trim"
  value_count {
    min: 2
    max: 20
  }
  type: FLOAT
  float_domain {
    name: "timestamp/weekday-sin_trim"
    min: 0.7421683
    max: 0.9995285
  }
  annotation {
    tag: "time"
    tag: "list"
  }
}
feature {
  name: "day-first"
  annotation {
  }
}
