# Customer Analytics: Dtype Optimization with pandas

> **Guided project** completed on [DataCamp](https://www.datacamp.com/) as part of my data analytics learning path.

## Overview

A data science training provider needs to prepare a large customer dataset for a machine learning model that predicts whether students are looking for a new job. The dataset is too heavy to run predictions efficiently — the goal is to optimize storage without losing information.

**Result: memory reduced from 2.0 MB → 69.5 KB (~96% reduction)**

## What I did

Analyzed each column's data type and applied the most efficient storage format:

| Transformation | Columns | Reasoning |
|---|---|---|
| `object` → `bool` | `job_change`, `relevant_experience` | Only 2 possible values |
| `int64` → `int32` | `student_id`, `training_hours` | Values fit in 32-bit range |
| `float64` → `float16` | `city_development_index` | Precision sufficient for index values |
| `object` → `category` (nominal) | `gender`, `education_level`, `major_discipline`, `enrolled_university`, `company_type` | Repeated string values |
| `object` → `category` (ordered) | `experience`, `company_size`, `last_new_job`, `city` | Natural order exists |

### Looking at the dataset
The first thing i do is to use info, describe and valuecounts so i can know which way the data is distributed at first sight

### Notable challenge: ordering `city`

The city codes (`city_1`, `city_10`, `city_100`...) do not sort numerically. Instead of stripping and re-appending the prefix, I generated the correct order directly, how did i came up to this conclusion of knowing the range of numbers needed? what i did was checking the size of the column with .info so i knew what was the number of possible categories the column takes, i double checked with value_counts to see if there was any attypical or error in the column :

```python
ordered_cats = ["city_" + str(i) for i in range(1, 124)]
ds_jobs_transformed["city"] = pd.Categorical(
    ds_jobs_transformed["city"],
    categories=ordered_cats,
    ordered=True
)
```

### Filtering with ordered categories

Once `experience` and `company_size` were stored as ordered categories, filtering became clean and readable:

```python
ds_jobs_transformed = ds_jobs_transformed[
    (ds_jobs_transformed["experience"] >= "10") &
    (ds_jobs_transformed["company_size"] >= "1000-4999")
]
```

This works because pandas respects the category order for comparisons — no need for `.isin()` with a manual list.

## Dataset

`customer_train.csv` — 19,158 rows, 14 columns. Provided by DataCamp. Contains anonymized student data including demographics, work experience, education level, and job-seeking status.

## Skills demonstrated

- pandas dtype casting (`astype`, `map`)
- Ordered and unordered categorical data
- Memory optimization
- DataFrame filtering with categorical comparisons

## Tech

- Python 3
- pandas

## Notes

This is a guided project from DataCamp's curriculum. The problem statement and dataset were provided; the analysis, reasoning, and code are my own work.
