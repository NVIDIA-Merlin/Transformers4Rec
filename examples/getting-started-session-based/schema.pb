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
  name: "category-list"
  value_count {
    min: 2
    max: 20
  }
  type: INT
  int_domain {
    name: "category-list"
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
  name: "item_id-list"
  value_count {
    min: 2
    max: 20
  }
  type: INT
  int_domain {
    name: "item_id-list"
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
  name: "age_days-list"
  value_count {
    min: 2
    max: 20
  }
  type: FLOAT
  float_domain {
    name: "age_days-list"
    min: 0.0000003
    max: 0.9999999
  }
  annotation {
    tag: "continuous"
    tag: "list"
  }
}
feature {
  name: "weekday_sin-list"
  value_count {
    min: 2
    max: 20
  }
  type: FLOAT
  float_domain {
    name: "weekday_sin-list"
    min: 0.0000003
    max: 0.9999999
  }
  annotation {
    tag: "continuous"
    tag: "time"
    tag: "list"
  }
}