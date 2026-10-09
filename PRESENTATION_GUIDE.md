# Lab 3 Presentation and Python Study Guide

This guide follows `kmeans_rent_classification.ipynb` in order. Each section has two parts:

1. **What to tell the TA** - a short explanation of the purpose, method, and interpretation.
2. **Line-by-line Python explanation** - what every statement in the corresponding notebook cell does.

**How line numbers are counted:** the numbers below match PyCharm's line numbers inside each individual notebook code cell. PyCharm starts again at line 1 for every cell. Blank lines and comment-only lines still count, and two commands written on one physical line share one line number.

The project studies whether Sweden's 21 counties form natural groups based on:

- annual rent per square metre; and
- average yearly income in thousands of Swedish kronor (KSEK).

The clustering algorithm and silhouette calculation are implemented from scratch. Pandas is used for data handling, NumPy for numerical operations, and Matplotlib for plots.

---

## Section 1: Imports and reproducible setup

### What to tell the TA

I first import the libraries needed for paths, arrays, tables, and visualization. I use a fixed random seed so K-means chooses the same initial centroids every time the notebook runs. I also define the dataset path. The fallback path makes the notebook convenient to run if the CSV is temporarily placed beside the notebook instead of under `data/`.

### Code

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.precision', 2)
RANDOM_STATE = 42
DATA_PATH = Path('data/inc_vs_rent.csv')
if not DATA_PATH.exists():
    DATA_PATH = Path('inc_vs_rent.csv')
```

### Line-by-line explanation

- **Line 1:** `from pathlib import Path` imports `Path`, a convenient object-oriented way to work with file paths.
- **Line 2:** `import numpy as np` imports NumPy and gives it the standard short name `np`. NumPy handles arrays, distances, means, and random sampling.
- **Line 3:** `import pandas as pd` imports Pandas as `pd`. Pandas loads and summarizes the CSV as a DataFrame.
- **Line 4:** `import matplotlib.pyplot as plt` imports Matplotlib's plotting interface as `plt`.
- **Line 6:** `plt.style.use(...)` selects a clean plot style with a white grid.
- **Line 7:** `pd.set_option(...)` tells Pandas to display floating-point values with two digits after the decimal point.
- **Line 8:** `RANDOM_STATE = 42` stores a fixed seed. The particular number is arbitrary; using the same number makes the result reproducible.
- **Line 9:** `Path(...)` creates a path object pointing to the CSV in the repository's `data` directory.
- **Line 10:** `DATA_PATH.exists()` checks whether that file is present. `not` makes the condition true when it is absent.
- **Line 11:** If the first path is absent, this assigns a fallback path beside the notebook.

---

## Section 2: Load the dataset

### What to tell the TA

I load the supplied CSV into a Pandas DataFrame. The first CSV column is only a saved row index, so I use `index_col=0` instead of treating it as a feature. I then display the first ten rows to check the structure and values. The PDF calls the file `rent_vs_inc.csv`, but the supplied file is actually named `inc_vs_rent.csv`.

### Code

```python
df = pd.read_csv(DATA_PATH, index_col=0)
df.head(10)
```

### Line-by-line explanation

- **Line 1:** `pd.read_csv` reads the comma-separated file. `DATA_PATH` says where it is, and `index_col=0` uses the first column as row labels rather than data.
- **Line 2:** `df.head(10)` returns the first ten rows. Because it is the last expression in a Jupyter cell, Jupyter renders it as a formatted table.

### What we learn from the first table

- Every row is a county.
- All observations are from 2020.
- `region` is descriptive text and is not used in the distance calculation.
- `Annual rent sqm` and `Avg yearly inc KSEK` are the two numerical clustering features.
- Rent values are numerically much larger than income values, so rent has more influence on raw Euclidean distance.

---

## Section 3: Check data quality and summary statistics

### What to tell the TA

Before modelling, I verify the dataset size and quality. There are 21 rows, no missing values, and no duplicate regions. The descriptive statistics show the range, centre, and spread of each numerical column. This is important because K-means requires numerical values and can be affected by feature scales and extreme observations.

### Code

```python
print(f'Rows: {len(df)}, columns: {df.shape[1]}')
print('Missing values:', int(df.isna().sum().sum()))
print('Duplicate regions:', int(df['region'].duplicated().sum()))
display(df.describe(include=[np.number]))
```

### Line-by-line explanation

- **Line 1:** `len(df)` counts rows. `df.shape[1]` gets the number of columns. The `f` before the string allows expressions inside `{}` to be inserted into the text.
- **Line 2:** `df.isna()` produces `True` wherever a value is missing. The first `.sum()` counts missing values per column, and the second `.sum()` totals them. `int(...)` converts the result to a normal integer.
- **Line 3:** `df['region']` selects the region column. `.duplicated()` marks repeated values, and `.sum()` counts the `True` values.
- **Line 4:** `df.describe(...)` calculates count, mean, standard deviation, minimum, quartiles, and maximum. `include=[np.number]` limits the summary to numerical columns. `display(...)` renders the result as a table in Jupyter.

### Why there is no train/validation split

This is unsupervised learning: the dataset has no correct cluster label for each county. A normal validation set would therefore have no target label against which to calculate classification accuracy. Instead, all observations are used to find structure, and the silhouette coefficient provides an internal measure of cohesion and separation. If the purpose were to test stability on future counties or years, resampling or an external test dataset could still be useful.

---

## Section 4: Create the initial scatter plot

### What to tell the TA

I select the two required features and convert them to a NumPy matrix because the from-scratch model works with numerical arrays. The scatter plot gives a visual check before clustering. Each point is a county, rent is on the x-axis, and income is on the y-axis. The plot suggests a dense lower-rent group and a few higher-rent observations, especially Stockholm and Uppsala.

### Code

```python
FEATURES = ['Annual rent sqm', 'Avg yearly inc KSEK']
X = df[FEATURES].to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(X[:, 0], X[:, 1], s=75, color='steelblue', edgecolor='white')
for _, row in df.iterrows():
    ax.annotate(row['region'].split(' ', 1)[1].replace(' county', ''),
                (row[FEATURES[0]], row[FEATURES[1]]), xytext=(4, 4),
                textcoords='offset points', fontsize=7)
ax.set(xlabel='Annual rent per m²', ylabel='Average yearly income (KSEK)',
       title='Swedish counties: rent versus income (2020)')
plt.tight_layout()
plt.show()
```

### Line-by-line explanation

- **Line 1:** Creates a list containing the exact names of the two columns used as features.
- **Line 2:** `df[FEATURES]` selects those columns in that order. `.to_numpy(dtype=float)` converts them to a two-dimensional floating-point NumPy array named `X`.
- **Line 4:** Creates a figure and one axes object. `figsize=(9, 6)` sets its size in inches.
- **Line 5:** Draws the points. `X[:, 0]` means every row from column 0, and `X[:, 1]` means every row from column 1. The remaining arguments control marker size and colour.
- **Line 6:** `df.iterrows()` loops over the DataFrame rows. `_` receives the unused row index, while `row` receives the row data.
- **Line 7:** Starts adding a text label. `.split(' ', 1)[1]` removes the numeric county code, and `.replace(...)` removes the word `county`.
- **Line 8:** Supplies the point coordinates and moves the label four screen points right and up so it does not sit directly on the marker.
- **Line 9:** Says that the offset is measured in display points and uses a small font.
- **Lines 10-11:** Set both axis labels and the plot title.
- **Line 12:** `plt.tight_layout()` adjusts margins to prevent clipping, and `plt.show()` displays the figure. The semicolon places both commands on the same physical line.

---

## Section 5: Define the K-means class

### What to tell the TA

I implement K-means from scratch as a class. The algorithm randomly chooses existing observations as initial centroids. It repeatedly assigns every observation to its nearest centroid and replaces each centroid with the mean of its assigned points. It stops when the largest centroid movement is at most the tolerance or when the maximum number of iterations is reached. The fitted object stores its centroids, labels, number of iterations, and inertia. It can also assign unseen points to the nearest learned centroid.

### Code and line-by-line explanation

```python
class KMeansScratch:
```

- **Line 1:** Defines a new class. A class groups data and behaviour into reusable model objects.

```python
    """K-means using Euclidean distance and sample-based random initialization."""
```

- **Line 2:** This docstring briefly documents the class.

```python
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-6, random_state=None):
```

- **Line 3:** Defines the constructor, which runs when a new model object is created. `self` refers to that object. The other arguments have default values.

```python
        if n_clusters < 1:
            raise ValueError('n_clusters must be at least 1')
```

- **Line 4:** Checks that the requested cluster count is valid.
- **Line 5:** Stops with a clear error if it is invalid.

```python
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
```

- **Lines 6-9:** Save the constructor arguments as attributes of the model object. `self.attribute` can be accessed by the other methods.

```python
    @staticmethod
    def _distances(X, centroids):
        return np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
```

- **Line 11:** `@staticmethod` indicates that this helper does not need a particular fitted object. Line 10 is blank.
- **Line 12:** Defines an internal distance method. The leading underscore signals that it is intended as a helper.
- **Line 13:** Broadcasting subtracts every centroid from every observation. `np.linalg.norm(..., axis=2)` calculates Euclidean distance across the feature dimension. The result has one row per observation and one column per centroid.

For two points \((x_1,y_1)\) and \((x_2,y_2)\), Euclidean distance is:

\[
d = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}
\]

```python
    def fit(self, X):
        X = np.asarray(X, dtype=float)
```

- **Line 15:** Defines the method that learns clusters from data. Line 14 is blank.
- **Line 16:** Converts the input to a floating-point NumPy array, even if the caller provided a list or DataFrame.

```python
        if X.ndim != 2 or self.n_clusters > len(X):
            raise ValueError('X must be 2D and n_clusters cannot exceed sample count')
```

- **Line 17:** Checks that `X` is a matrix and that there are enough observations for the requested number of clusters.
- **Line 18:** Raises an informative error when either condition is invalid.

```python
        rng = np.random.default_rng(self.random_state)
        self.centroids_ = X[rng.choice(len(X), self.n_clusters, replace=False)].copy()
```

- **Line 19:** Creates a modern NumPy random-number generator using the stored seed.
- **Line 20:** Randomly chooses `n_clusters` different row positions. Those observations become the initial centroids. `.copy()` creates independent centroid data.

```python
        for iteration in range(self.max_iter):
            distances = self._distances(X, self.centroids_)
            labels = distances.argmin(axis=1)
```

- **Line 21:** Begins the update loop, with at most `max_iter` repetitions.
- **Line 22:** Calculates every observation-to-centroid distance.
- **Line 23:** `argmin(axis=1)` returns the column position of the smallest distance in each row. That position is the observation's cluster label.

```python
            new_centroids = np.empty_like(self.centroids_)
            for cluster in range(self.n_clusters):
                members = X[labels == cluster]
```

- **Line 24:** Allocates an array with the same shape and type as the current centroids.
- **Line 25:** Loops through each cluster number.
- **Line 26:** Boolean indexing selects only observations whose assigned label equals the current cluster.

```python
                if len(members):
                    new_centroids[cluster] = members.mean(axis=0)
```

- **Line 27:** Tests whether the cluster contains at least one observation. A nonzero length is treated as `True`.
- **Line 28:** Calculates the feature-wise mean of the members and stores it as the new centroid.

```python
                else:
                    # Re-seed an empty cluster with the currently worst represented point.
                    new_centroids[cluster] = X[distances.min(axis=1).argmax()]
```

- **Line 29:** Handles the rare case in which no point is assigned to a centroid.
- **Line 30:** This comment explains the empty-cluster recovery rule; it is counted by PyCharm even though it is not executed.
- **Line 31:** Finds the point farthest from its nearest existing centroid and uses it to restart the empty cluster.

```python
            shift = np.linalg.norm(new_centroids - self.centroids_, axis=1).max()
            self.centroids_ = new_centroids
            if shift <= self.tol:
                break
```

- **Line 32:** Calculates how far every centroid moved, then takes the largest movement.
- **Line 33:** Replaces the old centroids with the newly calculated centroids.
- **Line 34:** Compares the largest movement with the convergence tolerance.
- **Line 35:** Ends the loop early when the centroids have effectively stopped changing.

```python
        self.n_iter_ = iteration + 1
        self.labels_ = self.predict(X)
        self.inertia_ = float(np.sum((X - self.centroids_[self.labels_]) ** 2))
        return self
```

- **Line 36:** Saves how many iterations ran. `+1` converts the zero-based loop counter into a human-readable count.
- **Line 37:** Uses the final centroids to calculate and save the final labels.
- **Line 38:** Calculates inertia: the sum of squared differences between each observation and its assigned centroid. Lower inertia means more compact clusters, but inertia always tends to decrease as `k` increases.
- **Line 39:** Returns the fitted object, enabling syntax such as `model = KMeansScratch(...).fit(X)`.

```python
    def predict(self, X):
        if not hasattr(self, 'centroids_'):
            raise RuntimeError('Call fit before predict')
```

- **Line 41:** Defines the method that assigns data to learned clusters. Line 40 is blank.
- **Line 42:** Checks whether fitting has created the `centroids_` attribute.
- **Line 43:** Produces a clear error if someone tries to predict before fitting.

```python
        X = np.asarray(X, dtype=float)
        return self._distances(X, self.centroids_).argmin(axis=1)
```

- **Line 44:** Converts new input data into a floating-point NumPy array.
- **Line 45:** Calculates distances to all fitted centroids and returns the nearest centroid index for every input row.

```python
    def fit_predict(self, X):
        return self.fit(X).labels_
```

- **Line 47:** Defines a convenience method that fits and returns labels in one call. Line 46 is blank.
- **Line 48:** `self.fit(X)` returns the fitted object, and `.labels_` retrieves its saved labels.

---

## Section 6: Fit and visualize an initial three-cluster model

### What to tell the TA

Before tuning `k`, I demonstrate the class using three clusters. The model converges when its centroids stop moving beyond the tolerance. Points are coloured by their assigned cluster, and large X markers show the learned centroids. Cluster numbers are arbitrary IDs, not rankings.

### Code

```python
initial_model = KMeansScratch(n_clusters=3, max_iter=100, random_state=RANDOM_STATE).fit(X)
print(f'Converged in {initial_model.n_iter_} iterations; inertia = {initial_model.inertia_:.2f}')

fig, ax = plt.subplots(figsize=(9, 6))
points = ax.scatter(X[:, 0], X[:, 1], c=initial_model.labels_, cmap='tab10',
                    s=80, edgecolor='white')
ax.scatter(initial_model.centroids_[:, 0], initial_model.centroids_[:, 1],
           marker='X', s=240, c=np.arange(3), cmap='tab10', edgecolor='black',
           linewidth=1.2, label='Centroids')
ax.set(xlabel='Annual rent per m²', ylabel='Average yearly income (KSEK)',
       title='Initial K-means solution (k=3)')
ax.legend(); plt.tight_layout(); plt.show()
```

### Line-by-line explanation

- **Line 1:** Constructs a three-cluster model and immediately fits it to `X`. The fixed seed makes initialization repeatable.
- **Line 2:** Prints the completed iteration count and inertia. `:.2f` formats inertia to two decimal places.
- **Line 3:** Blank line separating calculation from plotting.
- **Line 4:** Creates a figure and axes.
- **Lines 5-6:** Plots observations and stores the returned plot object in `points`. `c=labels` assigns colours from `tab10` according to cluster labels.
- **Lines 7-9:** Plots centroids as large X symbols. `np.arange(3)` generates `[0, 1, 2]` so colours match the cluster colour map.
- **Lines 10-11:** Sets axis labels and title.
- **Line 12:** Shows the legend, corrects the layout, and displays the plot. Semicolons keep the three commands on one physical line.

---

## Section 7: Calculate the silhouette coefficient from scratch

### What to tell the TA

The silhouette coefficient measures both cluster cohesion and separation. For observation `i`, `a(i)` is its average distance to the other members of its own cluster. `b(i)` is its smallest average distance to any different cluster. The silhouette is:

\[
s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}
\]

Values close to 1 indicate a well-separated point, values near 0 indicate a boundary point, and negative values suggest that the point may fit another cluster better. The dataset score is the mean of all point scores.

### Function 1: Intra-cluster distance `a(i)`

```python
def average_intra_cluster_distance(X, labels, i):
    """a(i): mean distance from point i to other points in its cluster."""
    same = np.flatnonzero(labels == labels[i])
    same = same[same != i]
    if len(same) == 0:
        return 0.0
    return float(np.linalg.norm(X[same] - X[i], axis=1).mean())
```

- **Line 1:** Defines a function receiving all points, labels, and the position `i` of one point.
- **Line 2:** The docstring documents that this function calculates `a(i)`.
- **Line 3:** `labels == labels[i]` marks points in the same cluster; `np.flatnonzero` returns their positions.
- **Line 4:** Removes `i` itself so its zero self-distance does not reduce the average.
- **Line 5:** Checks whether the point is alone in its cluster.
- **Line 6:** Returns zero for a singleton, following the usual silhouette convention.
- **Line 7:** Calculates and returns the mean Euclidean distance from point `i` to its cluster neighbours.

### Function 2: Nearest-cluster distance `b(i)`

```python
def average_nearest_cluster_distance(X, labels, i):
    """b(i): smallest mean distance from point i to another cluster."""
    other_clusters = np.unique(labels[labels != labels[i]])
    means = [np.linalg.norm(X[labels == c] - X[i], axis=1).mean() for c in other_clusters]
    return float(min(means))
```

- **Line 9:** Defines the function for the nearest different cluster. Line 8 is blank.
- **Line 10:** The docstring documents that this function calculates `b(i)`.
- **Line 11:** Selects labels different from point `i`'s label, then keeps each distinct cluster ID once.
- **Line 12:** The list comprehension calculates the mean distance from point `i` to every other cluster.
- **Line 13:** Selects the smallest mean, representing the closest alternative cluster.

### Function 3: Silhouette score for every point

```python
def silhouette_samples_scratch(X, labels):
    X, labels = np.asarray(X, float), np.asarray(labels)
    if len(np.unique(labels)) < 2:
        raise ValueError('Silhouette is undefined for a single cluster')
    scores = np.zeros(len(X))
    for i in range(len(X)):
        if np.sum(labels == labels[i]) == 1:
            scores[i] = 0.0
            continue
        a_i = average_intra_cluster_distance(X, labels, i)
        b_i = average_nearest_cluster_distance(X, labels, i)
        scores[i] = (b_i - a_i) / max(a_i, b_i)
    return scores
```

- **Line 15:** Defines the per-observation silhouette function. Line 14 is blank.
- **Line 16:** Converts both inputs to NumPy arrays; `X` is explicitly converted to floats.
- **Line 17:** Counts distinct labels and checks that at least two clusters exist.
- **Line 18:** Raises an error for one cluster because no alternative cluster exists for `b(i)`.
- **Line 19:** Creates one zero-initialized score position per observation.
- **Line 20:** Loops through every observation position.
- **Line 21:** Checks whether the current point is the only member of its cluster.
- **Line 22:** Assigns the conventional silhouette value zero to a singleton. The inline comment documents why.
- **Line 23:** Skips to the next point.
- **Line 24:** Calculates `a(i)`.
- **Line 25:** Calculates `b(i)`.
- **Line 26:** Applies the silhouette formula and stores the result.
- **Line 27:** Returns all individual scores.

### Function 4: Mean silhouette score

```python
def silhouette_score_scratch(X, labels):
    return float(silhouette_samples_scratch(X, labels).mean())
```

- **Line 29:** Defines the overall score function. Line 28 is blank.
- **Line 30:** Calculates every point's silhouette, averages the results, and returns a normal float.

---

## Section 8: Grid search over the number of clusters

### What to tell the TA

The number of clusters is a hyperparameter because K-means cannot learn it automatically. I perform the requested grid search from 1 to 10. For each candidate `k`, I fit a fresh model and save its silhouette score and inertia. The silhouette for `k=1` is undefined because there is no second cluster, so I store `NaN` and exclude it when selecting the best value.

### Code

```python
def grid_search_clusters(X, k_values=range(1, 11), random_state=42):
    results, models = [], {}
    for k in k_values:
        model = KMeansScratch(k, max_iter=100, random_state=random_state).fit(X)
        models[k] = model
        score = np.nan if k == 1 else silhouette_score_scratch(X, model.labels_)
        results.append({'k': k, 'silhouette': score, 'inertia': model.inertia_})
    return pd.DataFrame(results), models

grid_results, grid_models = grid_search_clusters(X)
display(grid_results.round(4))
```

### Line-by-line explanation

- **Line 32:** Defines the grid-search function. `range(1, 11)` generates integers 1 through 10. Line 31 is blank.
- **Line 33:** Creates an empty list for result rows and an empty dictionary for fitted models.
- **Line 34:** Loops over each candidate number of clusters.
- **Line 35:** Creates and fits a new K-means model for the current `k`.
- **Line 36:** Stores the fitted model in the dictionary using `k` as its key.
- **Line 37:** Stores `NaN` for one cluster; otherwise calculates the silhouette score.
- **Line 38:** Appends the current hyperparameter and metrics to the result list.
- **Line 39:** Converts the results into a DataFrame and returns it together with all fitted models.
- **Line 40:** Blank line separating the function definition from its use.
- **Line 41:** Runs the grid search and unpacks its two returned objects.
- **Line 42:** Rounds displayed numbers to four decimal places and renders the table.

---

## Section 9: Select and plot the optimal `k`

### What to tell the TA

I remove the undefined one-cluster row and select the `k` with the maximum silhouette coefficient. For this dataset and initialization, the best result is `k=2` with a silhouette coefficient of approximately `0.6521`. The plot makes the comparison visible. The red point marks the selected value.

### Code

```python
valid = grid_results.dropna(subset=['silhouette'])
optimal_k = int(valid.loc[valid['silhouette'].idxmax(), 'k'])
optimal_score = float(valid['silhouette'].max())
print(f'Best k = {optimal_k}, silhouette coefficient = {optimal_score:.4f}')

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(valid['k'], valid['silhouette'], marker='o', linewidth=2)
ax.scatter([optimal_k], [optimal_score], s=130, color='crimson', zorder=3, label='Best k')
ax.set(xticks=range(1, 11), xlabel='Number of clusters (k)',
       ylabel='Mean silhouette coefficient', title='Grid search over k=1...10')
ax.legend(); plt.tight_layout(); plt.show()
```

### Line-by-line explanation

- **Line 1:** Removes rows whose silhouette value is missing, which excludes `k=1`.
- **Line 2:** `.idxmax()` finds the row label containing the largest silhouette. `.loc[row, 'k']` retrieves that row's `k`, and `int(...)` converts it to an integer.
- **Line 3:** Finds and saves the maximum silhouette value.
- **Line 4:** Prints both values; `:.4f` displays four decimal places.
- **Line 6:** Creates a figure and axes.
- **Line 7:** Draws silhouette against `k`, placing a circular marker at each tested value.
- **Line 8:** Adds a larger red marker at the optimum. `zorder=3` draws it above the line.
- **Lines 9-10:** Set integer x-axis ticks, labels, and title.
- **Line 11:** Shows the legend, adjusts margins, and displays the plot; all three commands share one physical line.

---

## Section 10: Visualize and summarize the optimal clusters

### What to tell the TA

I retrieve the already fitted model corresponding to the optimal `k`. The point colours show cluster membership, the X markers are centroids, and county labels help interpret the groups. I also create a summary table containing cluster size, mean rent, mean income, and member counties. In this run, one cluster contains Stockholm, Uppsala, and Skåne as the higher-rent group; the other contains the remaining 18 counties. The cluster numbers themselves have no ordered meaning.

### Plot code explained line by line

```python
optimal_model = grid_models[optimal_k]
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(X[:, 0], X[:, 1], c=optimal_model.labels_, cmap='tab10',
           s=85, edgecolor='white')
ax.scatter(optimal_model.centroids_[:, 0], optimal_model.centroids_[:, 1],
           c=np.arange(optimal_k), cmap='tab10', marker='X', s=260,
           edgecolor='black', linewidth=1.2, label='Centroids')
```

- **Line 1:** Looks up the fitted model stored under the best `k`.
- **Line 2:** Creates the figure.
- **Lines 3-4:** Plot all county points, coloured by optimal cluster label.
- **Lines 5-7:** Plot each centroid with a large X and colours consistent with the clusters.

```python
for _, row in df.iterrows():
    ax.annotate(row['region'].split(' ', 1)[1].replace(' county', ''),
                (row[FEATURES[0]], row[FEATURES[1]]), xytext=(4, 4),
                textcoords='offset points', fontsize=7)
```

- **Line 8:** Loop through the county rows.
- **Line 9:** Remove the county code and the word `county` to make a shorter label.
- **Line 10:** Place the label at the row's rent and income coordinates with a small offset.
- **Line 11:** Configure the offset units and font size.

```python
ax.set(xlabel='Annual rent per m²', ylabel='Average yearly income (KSEK)',
       title=f'Optimal K-means clustering (k={optimal_k})')
ax.legend(); plt.tight_layout(); plt.show()
```

- **Lines 12-13:** Set axes and insert `optimal_k` into the title with an f-string.
- **Line 14:** Shows the centroid legend, fixes layout spacing, and displays the figure on one physical line.

### Summary-table code explained line by line

```python
cluster_summary = (df.assign(cluster=optimal_model.labels_)
                   .groupby('cluster')
                   .agg(count=('region', 'size'),
                        mean_rent=('Annual rent sqm', 'mean'),
                        mean_income=('Avg yearly inc KSEK', 'mean'),
                        regions=('region', lambda s: ', '.join(s.str.replace(r'^\d+ ', '', regex=True)))))
display(cluster_summary)
```

- **Line 15:** Blank line separating the plot from the summary calculation.
- **Line 16:** `.assign(...)` creates a temporary DataFrame with the fitted label added as a `cluster` column.
- **Line 17:** Groups rows that have the same cluster label.
- **Line 18:** Begins named aggregations; `count` is the number of regions in each group.
- **Line 19:** Calculates mean annual rent for each group.
- **Line 20:** Calculates mean income for each group.
- **Line 21:** Builds a comma-separated string of region names per cluster. The regular expression `^\d+ ` removes the numeric code at the start.
- **Line 22:** Displays the resulting summary table.

---

## Section 11: Predict the clusters of three new regions

### What to tell the TA

I represent the three supplied regions in exactly the same feature order as the training data: rent first and income second. Prediction does not rerun K-means. It measures each new point's distance to the already learned centroids and returns the nearest centroid's cluster ID. Region A and Region C are assigned to cluster 1, while the higher-rent Region B is assigned to cluster 0.

### Prediction-table code

```python
new_regions = np.array([[1010, 320.12], [1258, 320.00], [980, 292.40]])
new_labels = optimal_model.predict(new_regions)
predictions = pd.DataFrame(new_regions, columns=FEATURES)
predictions.insert(0, 'new_region', ['Region A', 'Region B', 'Region C'])
predictions['predicted_cluster'] = new_labels
display(predictions)
```

### Line-by-line explanation

- **Line 1:** Creates a 3-by-2 NumPy array. Every inner list is one region with `[annual rent, average income]`.
- **Line 2:** Assigns each new region to its nearest centroid using the fitted model.
- **Line 3:** Converts the values into a readable DataFrame with the same feature names.
- **Line 4:** Inserts the human-readable region names at column position zero.
- **Line 5:** Adds the predicted cluster labels as a new column.
- **Line 6:** Displays the prediction table.
- **Line 7:** Blank line separating the table code from the plotting code.

### Prediction-plot code

```python
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(X[:, 0], X[:, 1], c=optimal_model.labels_, cmap='tab10',
           s=75, alpha=.65, edgecolor='white', label='Original counties')
ax.scatter(optimal_model.centroids_[:, 0], optimal_model.centroids_[:, 1],
           c=np.arange(optimal_k), cmap='tab10', marker='X', s=250,
           edgecolor='black', linewidth=1.2, label='Centroids')
ax.scatter(new_regions[:, 0], new_regions[:, 1], c=new_labels, cmap='tab10',
           vmin=0, vmax=max(optimal_k - 1, 1), marker='*', s=330,
           edgecolor='black', linewidth=1.2, label='New regions')
```

- **Line 8:** Creates the figure.
- **Lines 9-10:** Plot original counties with partial transparency (`alpha=.65`) so the new points stand out.
- **Lines 11-13:** Plot the learned centroids as X markers.
- **Lines 14-16:** Plot the new regions as large stars. `vmin` and `vmax` keep colour mapping aligned with cluster IDs.

```python
for name, (x, y) in zip(predictions['new_region'], new_regions):
    ax.annotate(name, (x, y), xytext=(7, 7), textcoords='offset points', weight='bold')
ax.set(xlabel='Annual rent per m²', ylabel='Average yearly income (KSEK)',
       title='Optimal clusters and predictions for new regions')
ax.legend(); plt.tight_layout(); plt.show()
```

- **Line 17:** `zip(...)` pairs each region name with its coordinate pair. `(x, y)` unpacks each pair into two variables.
- **Line 18:** Adds a bold, offset label beside each new star.
- **Lines 19-20:** Set plot labels and title.
- **Line 21:** Shows the legend, fixes margins, and displays the plot on one physical line.

### How to evaluate these predictions

The predictions are successful in the limited geometric sense required by K-means: each new point is assigned to its nearest centroid and appears near the corresponding group in the plot. Region B has high rent and joins the higher-rent cluster. Region C has low rent and joins the larger lower-rent cluster. Region A has low rent but unusually high income, so it is less typical of its assigned group and should be interpreted cautiously.

This is not accuracy in the supervised-learning sense because there are no true labels for these regions. We can discuss plausibility, distance to centroids, and silhouette quality, but we cannot calculate prediction accuracy.

---

## Section 12: Important interpretation and limitations

### What to tell the TA

1. **K-means is unsupervised.** It discovers groups rather than learning known categories.
2. **Cluster IDs are arbitrary.** Cluster 0 is not inherently better or lower than cluster 1.
3. **The selected value is `k=2`.** It has the highest tested silhouette coefficient, about `0.6521`.
4. **The algorithm uses Euclidean distance.** Both assignment and prediction depend on distance to centroids.
5. **Scaling matters.** Rent spans a larger numerical range than income, so raw-distance clustering is influenced more by rent. Raw values were intentionally used to follow the assignment and directly accept the supplied new points.
6. **Initialization matters.** K-means can reach different local solutions from different initial centroids. The fixed seed makes this particular experiment reproducible.
7. **No train/validation split is used.** There are no target labels. Silhouette is an internal validation metric, not proof of a real-world ground truth.
8. **The dataset is small.** It contains only 21 counties from one year, so conclusions should not be generalized too strongly.

---

## Short presentation script

> I clustered Sweden's 21 counties using annual rent per square metre and average yearly income. I first inspected the data and plotted the observations. I then implemented K-means from scratch as a Python class. The class randomly initializes centroids from existing samples, assigns each point to its nearest centroid using Euclidean distance, updates each centroid to the mean of its members, and repeats until convergence.
>
> Because the number of clusters is a hyperparameter, I also implemented the silhouette coefficient from scratch. For every point, I compare its average distance within its own cluster, called `a(i)`, with its average distance to the nearest alternative cluster, called `b(i)`. I searched values of `k` from 1 through 10. Since silhouette is undefined for one cluster, selection uses `k=2` through `k=10`.
>
> The best result was two clusters with a mean silhouette of about 0.6521. One cluster contains the higher-rent counties Stockholm, Uppsala, and Skåne, while the other contains the remaining counties. Finally, I assigned the three new regions to their nearest learned centroids and plotted them as stars. The assignments are geometrically consistent, although Region A is less typical because it combines low rent with relatively high income.
>
> These are unsupervised clusters, so there are no true labels and no normal classification accuracy. The result should be understood as a useful segmentation of this dataset. A major limitation is that raw Euclidean distance gives more influence to rent because its numeric range is larger than income's.

---

## Likely TA questions and answers

### Why is the silhouette coefficient undefined for `k=1`?

`b(i)` requires another cluster. With only one cluster, there is no alternative cluster, so the formula cannot be evaluated meaningfully.

### Why not choose the `k` with the lowest inertia?

Inertia almost always decreases when more clusters are added, reaching zero if every point becomes its own cluster. It therefore cannot select `k` by its minimum alone. Silhouette balances compactness with separation.

### Why use `axis=1` in the distance calculation?

After subtraction, the final dimension contains the two feature differences for one observation-centroid pair. `axis=2` combines those features in the full distance matrix, while later uses of `axis=1` combine the two features in a simpler point matrix. The chosen axis always indicates which dimension NumPy should reduce.

### Why use the mean for centroid updates?

K-means minimizes squared Euclidean distance. For a fixed set of cluster members, their arithmetic mean is the point that minimizes the sum of squared distances.

### What is inertia?

It is the sum of squared distances from every observation to its assigned centroid. It describes within-cluster compactness but not separation between different clusters.

### Does K-means really classify the new points?

It assigns them to discovered clusters, but this is not supervised classification with learned class labels. The notebook follows the assignment's wording while making this distinction explicit.

### Why can initialization change the result?

The K-means objective can have local minima. Different starting centroids can lead the update process to different final centroids. A fixed seed guarantees repeatability; multiple initializations would be a useful robustness extension.

### Why might standardized features produce different clusters?

Standardization puts both features on comparable scales. Without it, a difference of 100 rent units affects Euclidean distance much more than a difference of 10 income units. Standardizing would make relative variation in both features contribute more equally.

---

## Final facts to memorize

- Dataset: 21 Swedish counties, year 2020.
- Features: annual rent per square metre and average yearly income in KSEK.
- Algorithm: K-means implemented from scratch with Euclidean distance.
- Update: assign to nearest centroid, then replace centroids with cluster means.
- Stopping rule: centroid movement is at most the tolerance, or maximum iterations are reached.
- Hyperparameter grid: `k=1` through `k=10`.
- Selection metric: mean silhouette coefficient.
- Best result: `k=2`, silhouette approximately `0.6521`.
- New-region predictions: Region A -> cluster 1, Region B -> cluster 0, Region C -> cluster 1.
- Main limitation: raw feature scales make rent more influential than income.
