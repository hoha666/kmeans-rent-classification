# K-means Rent Classification

Lab 3 project for clustering Swedish counties by annual rent per square metre and average yearly income.

The notebook includes:

- exploratory inspection and scatter plots;
- an object-oriented K-means implementation from scratch;
- from-scratch intra-cluster, inter-cluster, and silhouette calculations;
- grid search over `k=1...10`;
- visualization and interpretation of the optimal clusters; and
- cluster predictions for the three new regions specified in the assignment.

## Run

Create an environment with the packages in `requirements.txt`, start Jupyter, and open `kmeans_rent_classification.ipynb`. Run all cells from the repository root so `data/inc_vs_rent.csv` resolves correctly.

```powershell
python -m pip install -r requirements.txt
jupyter notebook kmeans_rent_classification.ipynb
```

The assignment refers to `rent_vs_inc.csv`, while the file supplied with the lab is named `inc_vs_rent.csv`; the supplied filename is retained under `data/`.

