# Graph Neural Networks

Graph neural networks are a type of neural network designed to work with graph-structured data.

**Definition.** A graph $G = (V, E)$ consists of a set of **vertices** $V$ and **edges** $E = \{(u, v) : u, v \in V\}$. We often define **labels** on $E$ and/or $V$: $w_E(e) : E \mapsto \mathbb{R}^d$, $w(v) : V \mapsto \mathbb{R}^d$.

The purpose of a GNN is to infer some value from the graph's labels. In practice, this task manifests in two forms:

- **Graph-Level Tasks**. Inferring a value from the entire graph.
- **Vertex/Edge-Level Tasks**. Inferring a value for each edge/vertex.

Where our task is to learn $\theta$ given training data.

> Note that sometimes we have only one single graph to work with - for instance, a social network. In this case, we split the graph into $(V_\text{test}, E_\text{test})$ and $(V_\text{train}, E_\text{train})$.

## Aggregation Functions

GNN layers have the inductive bias of spatial locality, that all relevant information about a node is contained in its local neighborhood. Formally, we wish to learn an **aggregation function** $f_\theta$:

$$
\begin{align}
    f_\theta(v, N) : v \in V, N = \{ u : (v, u) \in E \}
\end{align}
$$

Where $N$ is the **neighborhood** of $v$. Importantly, $f_\theta$ must be **permutation invariant** in $N$, as we have the inductive bias that $E, V$ are unordered.

Possible aggregation functions:

1. Sum aggregation:

$$
\begin{align}
f(v, N) = w_0^Tv + \sum_{u \in N} w^Tu
\end{align}
$$

2. Mean aggregation:

$$
\begin{align}
f(v, N) = w_0^Tv + \frac{1}{|N|}\sum_{u \in N} w^Tu
\end{align}
$$

3. Max aggregation:

$$
\begin{align}
f(v, N) = w_0^Tv + \max_{u \in N} w^Tu
\end{align}
$$

Note the similarity to convolutional neural networks in that we learn a single function / filter / kernel that is then applied to every position.

## GNN Layers

A single GNN layer consists of multiple learned aggregation functions, mapping multiple input channels to multiple output channels in much the same way was CNNs. Additionally, we incorperate **nonlinearities** to increase the expressivity of our network.

Depth in a GNN has an important implication. As a GNN grows deeper, the **receptive field** of the network increases. For example, a 2-layer GNN can take into account information from a node's "two-hop" neighbors.

We also incorporate **residual connections**. Formally:

$$
\begin{align}
x^{(\ell)} = f^{(\ell)}(x^{(\ell - 1)}, N) + x^{(\ell - 1)} : x \in V
\end{align}
$$

Residual connections assist in allowing the gradient to flow to earlier layers in the network without "dying" or "blowing up".

## Graph Representation

**Defintion**. The adjacency matrix of a graph is defined as

$$
\begin{align}
[A]_{ij} = \begin{cases}
1 & \text{if there is an edge from i to j} \\
0 & \text{otherwise}
\end{cases}
\end{align}
$$

> $A$ need not be symmetric, such as when $G$ is directed

**Definition.** The degree matrix of a graph is defined as

$$
\begin{align}
[D]_{ij} = \begin{cases}
\text{deg}(v_i) & \text{if }i = j \\
0 & \text{otherwise}
\end{cases}
\end{align}
$$

We use the degree matrix to define $A^\text{Normalized} = AD^{-1}$ and $A^\text{SymNorm} = D^{-1/2}AD^{-1/2}$

**Definition.** The Laplacian matrix of $G$ is $L \doteq D - A$. Note that $L_i$ sums to zero. Additionally, we define $L^\text{SymNorm} = I - A^\text{SymNorm}$.

The Laplacian provides a convenient way of representing the graph as a matrix. Importantly, $A^k$ has $[A^k]_{ij} = n$ where $n$ is the number of ways $j$ can be reached with exactly $k$ hops from $i$.

> With our definition of $A \in \mathbb{R}^{N \times N}$ and vertex labels $X \in \mathbb{R}^{N \times C}$, applying the sum aggregation function to $G$ is similar to a convolution of dimension $1 \times N$ on $AX$. This establishes the parallel between GNNs and CNNs.

### Normalization

Layer norms in GNNs are similar to layer norms in CNNs. In GNNs, we normalize across batch entries and nodes, keeping channels separate.

### Pooling

In GNNs, pooling is a technique applied to downsample the graph structure and reduce the computational requirements of the network. Usually, it involves merging a node into its neighbors. However, it is often domain-specific, so for our purposes we assume the dimensionality of the graph stays constant.
