feature {
  name: "session_id"
  type: INT
  int_domain {
    name: "session_id"
    min: 1
    max: 100001
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
    max: 400
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
    max: 50005
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
    min: 0.0000003
    max: 0.9999999
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
    min: 0.0000003
    max: 0.9999999
  }
  annotation {
    tag: "time"
    tag: "list"
  }
}