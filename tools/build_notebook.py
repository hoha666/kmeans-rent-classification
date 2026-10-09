"""Build the Lab 3 Jupyter notebook in a reproducible way."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "kmeans_rent_classification.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

cells = []
cells.append(nbf.v4.new_markdown_cell(
    "# Lab 3: K-means rent classification and hyperparameter optimization\n\n"
    "This notebook implements K-means **from scratch** with object-oriented Python, applies it to "
    "Swedish county rent and income data, chooses the number of clusters with a from-scratch "
    "silhouette grid search, and assigns three new unnamed regions to clusters.\n\n"
    "> The assignment calls the file `rent_vs_inc.csv`; the supplied file is `inc_vs_rent.csv`. "
    "The latter is used here."
))
cells.append(nbf.v4.new_code_cell(
    "from pathlib import Path\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n\n"
    "plt.style.use('seaborn-v0_8-whitegrid')\n"
    "pd.set_option('display.precision', 2)\n"
    "RANDOM_STATE = 42\n"
    "DATA_PATH = Path('data/inc_vs_rent.csv')\n"
    "if not DATA_PATH.exists():\n"
    "    DATA_PATH = Path('inc_vs_rent.csv')  # convenient fallback\n"
))
cells.append(nbf.v4.new_markdown_cell("## Task 1 - Classification with K-means\n\n### 1.1 Load and inspect the data"))
cells.append(nbf.v4.new_code_cell(
    "df = pd.read_csv(DATA_PATH, index_col=0)\n"
    "df.head(10)"
))
cells.append(nbf.v4.new_code_cell(
    "print(f'Rows: {len(df)}, columns: {df.shape[1]}')\n"
    "print('Missing values:', int(df.isna().sum().sum()))\n"
    "print('Duplicate regions:', int(df['region'].duplicated().sum()))\n"
    "display(df.describe(include=[np.number]))"
))
cells.append(nbf.v4.new_markdown_cell(
    "**Interpretation.** Each row represents one of Sweden's 21 counties in 2020. The two features "
    "used for clustering are annual rent per square metre and average yearly income in KSEK. The "
    "table contains no missing values or duplicate counties. Stockholm has visibly high rent and "
    "income relative to most counties, while most observations are concentrated at lower values. "
    "Because the task is unsupervised and has no ground-truth cluster labels, a conventional "
    "train/validation split is not used; cluster quality is instead compared internally using the "
    "silhouette coefficient."
))
cells.append(nbf.v4.new_markdown_cell("### 1.2 Scatter plot"))
cells.append(nbf.v4.new_code_cell(
    "FEATURES = ['Annual rent sqm', 'Avg yearly inc KSEK']\n"
    "X = df[FEATURES].to_numpy(dtype=float)\n\n"
    "fig, ax = plt.subplots(figsize=(9, 6))\n"
    "ax.scatter(X[:, 0], X[:, 1], s=75, color='steelblue', edgecolor='white')\n"
    "for _, row in df.iterrows():\n"
    "    ax.annotate(row['region'].split(' ', 1)[1].replace(' county', ''),\n"
    "                (row[FEATURES[0]], row[FEATURES[1]]), xytext=(4, 4),\n"
    "                textcoords='offset points', fontsize=7)\n"
    "ax.set(xlabel='Annual rent per m²', ylabel='Average yearly income (KSEK)',\n"
    "       title='Swedish counties: rent versus income (2020)')\n"
    "plt.tight_layout(); plt.show()"
))
cells.append(nbf.v4.new_markdown_cell(
    "The scatter plot suggests a dense lower-rent group, a middle-rent group, and a few high-rent "
    "counties. Stockholm is the clearest high-value point. Since the features have different numeric "
    "ranges, raw-unit Euclidean distance is influenced more by rent. This notebook deliberately uses "
    "the assignment's raw coordinates so the supplied new points can be evaluated directly."
))
cells.append(nbf.v4.new_markdown_cell("### 1.3 K-means implemented from scratch"))
cells.append(nbf.v4.new_code_cell(
    "class KMeansScratch:\n"
    "    \"\"\"K-means using Euclidean distance and sample-based random initialization.\"\"\"\n"
    "    def __init__(self, n_clusters=3, max_iter=100, tol=1e-6, random_state=None):\n"
    "        if n_clusters < 1:\n"
    "            raise ValueError('n_clusters must be at least 1')\n"
    "        self.n_clusters = n_clusters\n"
    "        self.max_iter = max_iter\n"
    "        self.tol = tol\n"
    "        self.random_state = random_state\n\n"
    "    @staticmethod\n"
    "    def _distances(X, centroids):\n"
    "        return np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)\n\n"
    "    def fit(self, X):\n"
    "        X = np.asarray(X, dtype=float)\n"
    "        if X.ndim != 2 or self.n_clusters > len(X):\n"
    "            raise ValueError('X must be 2D and n_clusters cannot exceed sample count')\n"
    "        rng = np.random.default_rng(self.random_state)\n"
    "        self.centroids_ = X[rng.choice(len(X), self.n_clusters, replace=False)].copy()\n"
    "        for iteration in range(self.max_iter):\n"
    "            distances = self._distances(X, self.centroids_)\n"
    "            labels = distances.argmin(axis=1)\n"
    "            new_centroids = np.empty_like(self.centroids_)\n"
    "            for cluster in range(self.n_clusters):\n"
    "                members = X[labels == cluster]\n"
    "                if len(members):\n"
    "                    new_centroids[cluster] = members.mean(axis=0)\n"
    "                else:\n"
    "                    # Re-seed an empty cluster with the currently worst represented point.\n"
    "                    new_centroids[cluster] = X[distances.min(axis=1).argmax()]\n"
    "            shift = np.linalg.norm(new_centroids - self.centroids_, axis=1).max()\n"
    "            self.centroids_ = new_centroids\n"
    "            if shift <= self.tol:\n"
    "                break\n"
    "        self.n_iter_ = iteration + 1\n"
    "        self.labels_ = self.predict(X)\n"
    "        self.inertia_ = float(np.sum((X - self.centroids_[self.labels_]) ** 2))\n"
    "        return self\n\n"
    "    def predict(self, X):\n"
    "        if not hasattr(self, 'centroids_'):\n"
    "            raise RuntimeError('Call fit before predict')\n"
    "        X = np.asarray(X, dtype=float)\n"
    "        return self._distances(X, self.centroids_).argmin(axis=1)\n\n"
    "    def fit_predict(self, X):\n"
    "        return self.fit(X).labels_"
))
cells.append(nbf.v4.new_code_cell(
    "initial_model = KMeansScratch(n_clusters=3, max_iter=100, random_state=RANDOM_STATE).fit(X)\n"
    "print(f'Converged in {initial_model.n_iter_} iterations; inertia = {initial_model.inertia_:.2f}')\n\n"
    "fig, ax = plt.subplots(figsize=(9, 6))\n"
    "points = ax.scatter(X[:, 0], X[:, 1], c=initial_model.labels_, cmap='tab10',\n"
    "                    s=80, edgecolor='white')\n"
    "ax.scatter(initial_model.centroids_[:, 0], initial_model.centroids_[:, 1],\n"
    "           marker='X', s=240, c=np.arange(3), cmap='tab10', edgecolor='black',\n"
    "           linewidth=1.2, label='Centroids')\n"
    "ax.set(xlabel='Annual rent per m²', ylabel='Average yearly income (KSEK)',\n"
    "       title='Initial K-means solution (k=3)')\n"
    "ax.legend(); plt.tight_layout(); plt.show()"
))
cells.append(nbf.v4.new_markdown_cell("## Task 2 - Hyperparameter optimization\n\n### 2.1 From-scratch silhouette coefficient and grid search"))
cells.append(nbf.v4.new_code_cell(
    "def average_intra_cluster_distance(X, labels, i):\n"
    "    \"\"\"a(i): mean distance from point i to other points in its cluster.\"\"\"\n"
    "    same = np.flatnonzero(labels == labels[i])\n"
    "    same = same[same != i]\n"
    "    if len(same) == 0:\n"
    "        return 0.0\n"
    "    return float(np.linalg.norm(X[same] - X[i], axis=1).mean())\n\n"
    "def average_nearest_cluster_distance(X, labels, i):\n"
    "    \"\"\"b(i): smallest mean distance from point i to another cluster.\"\"\"\n"
    "    other_clusters = np.unique(labels[labels != labels[i]])\n"
    "    means = [np.linalg.norm(X[labels == c] - X[i], axis=1).mean() for c in other_clusters]\n"
    "    return float(min(means))\n\n"
    "def silhouette_samples_scratch(X, labels):\n"
    "    X, labels = np.asarray(X, float), np.asarray(labels)\n"
    "    if len(np.unique(labels)) < 2:\n"
    "        raise ValueError('Silhouette is undefined for a single cluster')\n"
    "    scores = np.zeros(len(X))\n"
    "    for i in range(len(X)):\n"
    "        if np.sum(labels == labels[i]) == 1:\n"
    "            scores[i] = 0.0  # standard convention for singleton clusters\n"
    "            continue\n"
    "        a_i = average_intra_cluster_distance(X, labels, i)\n"
    "        b_i = average_nearest_cluster_distance(X, labels, i)\n"
    "        scores[i] = (b_i - a_i) / max(a_i, b_i)\n"
    "    return scores\n\n"
    "def silhouette_score_scratch(X, labels):\n"
    "    return float(silhouette_samples_scratch(X, labels).mean())\n\n"
    "def grid_search_clusters(X, k_values=range(1, 11), random_state=42):\n"
    "    results, models = [], {}\n"
    "    for k in k_values:\n"
    "        model = KMeansScratch(k, max_iter=100, random_state=random_state).fit(X)\n"
    "        models[k] = model\n"
    "        score = np.nan if k == 1 else silhouette_score_scratch(X, model.labels_)\n"
    "        results.append({'k': k, 'silhouette': score, 'inertia': model.inertia_})\n"
    "    return pd.DataFrame(results), models\n\n"
    "grid_results, grid_models = grid_search_clusters(X)\n"
    "display(grid_results.round(4))"
))
cells.append(nbf.v4.new_code_cell(
    "valid = grid_results.dropna(subset=['silhouette'])\n"
    "optimal_k = int(valid.loc[valid['silhouette'].idxmax(), 'k'])\n"
    "optimal_score = float(valid['silhouette'].max())\n"
    "print(f'Best k = {optimal_k}, silhouette coefficient = {optimal_score:.4f}')\n\n"
    "fig, ax = plt.subplots(figsize=(8, 5))\n"
    "ax.plot(valid['k'], valid['silhouette'], marker='o', linewidth=2)\n"
    "ax.scatter([optimal_k], [optimal_score], s=130, color='crimson', zorder=3, label='Best k')\n"
    "ax.set(xticks=range(1, 11), xlabel='Number of clusters (k)',\n"
    "       ylabel='Mean silhouette coefficient', title='Grid search over k=1...10')\n"
    "ax.legend(); plt.tight_layout(); plt.show()"
))
cells.append(nbf.v4.new_markdown_cell(
    "#### Elbow plot using inertia\n\n"
    "The elbow method plots within-cluster sum of squares (inertia), not silhouette. "
    "It provides a complementary visual check: we look for the point after which adding clusters "
    "produces only smaller reductions in inertia."
))
cells.append(nbf.v4.new_code_cell(
    "fig, ax = plt.subplots(figsize=(8, 5))\n"
    "ax.plot(grid_results['k'], grid_results['inertia'], marker='o', linewidth=2, color='darkorange')\n"
    "ax.axvline(optimal_k, color='crimson', linestyle='--', alpha=.75, label=f'Silhouette choice: k={optimal_k}')\n"
    "ax.set(xticks=range(1, 11), xlabel='Number of clusters (k)',\n"
    "       ylabel='Inertia (within-cluster sum of squares)', title='Elbow plot')\n"
    "ax.legend(); plt.tight_layout(); plt.show()"
))
cells.append(nbf.v4.new_markdown_cell(
    "The silhouette coefficient is undefined for `k=1` because there is no other cluster from which "
    "to calculate `b(i)`; it is therefore shown as `NaN` and excluded from selection. The maximum "
    "score determines the optimal `k` among 2 through 10. A larger score means points are, on "
    "average, more cohesive within their own cluster and better separated from their nearest other cluster. "
    "Inertia always falls as `k` increases, so the elbow is interpreted by diminishing improvement rather "
    "than by choosing the absolute minimum. Here the visual bend is around `k=3`, whereas silhouette "
    "has a slightly stronger preference for `k=2`; the final model follows the assignment's silhouette criterion."
))
cells.append(nbf.v4.new_markdown_cell("### 2.2 Plot the optimal clustering"))
cells.append(nbf.v4.new_code_cell(
    "optimal_model = grid_models[optimal_k]\n"
    "fig, ax = plt.subplots(figsize=(9, 6))\n"
    "ax.scatter(X[:, 0], X[:, 1], c=optimal_model.labels_, cmap='tab10',\n"
    "           s=85, edgecolor='white')\n"
    "ax.scatter(optimal_model.centroids_[:, 0], optimal_model.centroids_[:, 1],\n"
    "           c=np.arange(optimal_k), cmap='tab10', marker='X', s=260,\n"
    "           edgecolor='black', linewidth=1.2, label='Centroids')\n"
    "for _, row in df.iterrows():\n"
    "    ax.annotate(row['region'].split(' ', 1)[1].replace(' county', ''),\n"
    "                (row[FEATURES[0]], row[FEATURES[1]]), xytext=(4, 4),\n"
    "                textcoords='offset points', fontsize=7)\n"
    "ax.set(xlabel='Annual rent per m²', ylabel='Average yearly income (KSEK)',\n"
    "       title=f'Optimal K-means clustering (k={optimal_k})')\n"
    "ax.legend(); plt.tight_layout(); plt.show()\n\n"
    "cluster_summary = (df.assign(cluster=optimal_model.labels_)\n"
    "                   .groupby('cluster')\n"
    "                   .agg(count=('region', 'size'),\n"
    "                        mean_rent=('Annual rent sqm', 'mean'),\n"
    "                        mean_income=('Avg yearly inc KSEK', 'mean'),\n"
    "                        regions=('region', lambda s: ', '.join(s.str.replace(r'^\\d+ ', '', regex=True)))))\n"
    "display(cluster_summary)"
))
cells.append(nbf.v4.new_markdown_cell(
    "Cluster numbers are arbitrary identifiers rather than ordered classes. The summary above gives "
    "them meaning through their centroid levels and member counties. The solution separates counties "
    "mainly along annual rent, which has the larger numerical scale."
))
cells.append(nbf.v4.new_markdown_cell("### 2.3 Classify and plot the three unnamed regions"))
cells.append(nbf.v4.new_code_cell(
    "new_regions = np.array([[1010, 320.12], [1258, 320.00], [980, 292.40]])\n"
    "new_labels = optimal_model.predict(new_regions)\n"
    "predictions = pd.DataFrame(new_regions, columns=FEATURES)\n"
    "predictions.insert(0, 'new_region', ['Region A', 'Region B', 'Region C'])\n"
    "predictions['predicted_cluster'] = new_labels\n"
    "display(predictions)\n\n"
    "fig, ax = plt.subplots(figsize=(9, 6))\n"
    "ax.scatter(X[:, 0], X[:, 1], c=optimal_model.labels_, cmap='tab10',\n"
    "           s=75, alpha=.65, edgecolor='white', label='Original counties')\n"
    "ax.scatter(optimal_model.centroids_[:, 0], optimal_model.centroids_[:, 1],\n"
    "           c=np.arange(optimal_k), cmap='tab10', marker='X', s=250,\n"
    "           edgecolor='black', linewidth=1.2, label='Centroids')\n"
    "ax.scatter(new_regions[:, 0], new_regions[:, 1], c=new_labels, cmap='tab10',\n"
    "           vmin=0, vmax=max(optimal_k - 1, 1), marker='*', s=330,\n"
    "           edgecolor='black', linewidth=1.2, label='New regions')\n"
    "for name, (x, y) in zip(predictions['new_region'], new_regions):\n"
    "    ax.annotate(name, (x, y), xytext=(7, 7), textcoords='offset points', weight='bold')\n"
    "ax.set(xlabel='Annual rent per m²', ylabel='Average yearly income (KSEK)',\n"
    "       title='Optimal clusters and predictions for new regions')\n"
    "ax.legend(); plt.tight_layout(); plt.show()"
))
cells.append(nbf.v4.new_markdown_cell(
    "**Evaluation.** The assignments are geometrically consistent with the nearest-centroid rule: "
    "each star appears in or near the cloud represented by its assigned centroid. Region B has high "
    "rent and is placed with the high-rent group; Region C is close to the low-rent group; and Region "
    "A combines low rent with unusually high income. That last point is less typical of its nearby "
    "counties and should be interpreted cautiously. Because there are no true labels, this visual "
    "and silhouette-based assessment measures plausibility, not predictive accuracy."
))
cells.append(nbf.v4.new_markdown_cell(
    "## Conclusion\n\n"
    "The complete workflow loads and explores the county data, implements K-means and the silhouette "
    "coefficient without using a library clustering implementation, selects the best cluster count "
    "over the requested grid, and assigns new points by their nearest learned centroid. The result is "
    "reproducible through a fixed random seed. A useful extension would compare raw-unit clustering "
    "with standardized features, because feature scaling changes the meaning of Euclidean distance."
))

nb["cells"] = cells
nbf.write(nb, OUTPUT)
print(f"Wrote {OUTPUT}")
