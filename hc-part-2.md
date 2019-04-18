# Hierarchical Clustering Implementation Assignment

#### Overview

Now that you have worked with the `scipy` hierarchical clustering functions. It would be useful to try and implement them yourself - to appreciate what's happening beneath the surface. Also to strengthen your command of numpy and linear algebra concepts.

#### Data

You could use any dataset you'd like, for example the Iris dataset. Or you can just create your own data as you see fit.

#### Workflow

The basic workflow for hierarchical clustering is:

- Load data
- Scale
- `pdist`
- `squareform`
- `linkage`
- `dendrogram`
- `fcluster`

In this assignment, I want you to implement - from scratch - three functions from the above list:

- `pdist`
- `squareform`
- `linkage`

#### Begin

Fork and clone this repository

#### Function 1, `pdist`

https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

```python
# run this code after importing pdist, look at its output
vector = pdist([[1,2],[9,-4],[5,6],[21,8]], metric='euclidean')

# now, make your own function
def my_pdist(arguments):
  pass
```

The output from `my_pdist` should be the same as `pdist`.

#### Function 2, `squareform`

https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html

```python
# run this code after importing squareform, look at its output
matrix = squareform(vector)

# now, make your own function
def my_squareform(arguments):
  pass
```

The output from `my_squareform` should be the same as `squareform`.

#### Function 3, `linkage`

https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

```python
# run the line below after importing linkage, look at its output
linkage(matrix, method='single', metric='euclidean')

# now, make your own function
def my_linkage(arguments):
  pass
```

The output from `my_linkage` should be the same as `linkage`.

#### Complete

When complete commit and push your code back to your repository on GitHub.

