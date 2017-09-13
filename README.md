# deep-image-analogy
A python implementation of the deep image analogy model from https://arxiv.org/abs/1705.01088

#### Progress
1. [X] **Preprocessing**:
    1. Extract feature pyramids $\{F_A^L\}_{L=1}^5$ and $\{F_{B\prime}^L\}_{L=1}^5$ from images A and and B' with VGG-19
    2. set $F^5_{A\prime} = F^5_A, F^5_B = F^5_{B\prime}$
    3. fill $\phi_{a\rightarrow b}^5, \phi_{b\rightarrow a}^5$ with random offsets
2. [ ] NNF search (patchmatch):
    
    **Patchmatch**:
    1. [ ] make patch indices
    
    *repeat 5 times*:
    2. [ ] *Propagation*: for each patch $N(p)$ in $F_A^L$ and $F_{A^\prime}^L$ get corresponding local patches offset by (x-1, y) and (x, y-1) in $F_B^L$ and $F_{B^\prime}^L$. compute the distance $\sum_{x\in N\left( p\right), y\in N\left( q\right)}\left( \Vert \bar F_A^L\left( x\right) - \bar F_B^L\left( y\right) \Vert^2 + \Vert \bar F_{A^\prime}^L\left( x\right) - \bar F_{B^\prime}^L\left( y\right) \Vert^2 \right)$ for each $a \rightarrow b$ pair. store the best patch offset in the corresponding $\phi_{a\rightarrow b}^L$ patch index. on even iterations get the (x+1, y) and (x, y+1) offsets instead.

    3. [ ] *Random search*: get a random set of patches at exponentially decreasing distance from the current best offset. the random offsets are determined by:
    
    $\mathbf{u}_i = \mathbf{v}_0 + w\alpha^i\mathbf{R}_i$, where $\mathbf{v}_0$ is the current best offset, $w$ is the max image dim, $\alpha = 1/2$, $\mathbf{R}_i$ is a uniform random in [-1, 1], [-1, 1], and $i = 0, 1,2...$ until the search radius is less than 1. for example the values of $w\alpha^i$ for a 16 X 16 image  would be: $(16/2^0, 16/2^1,...,16/2^4)$
    4. [ ] compute the distance (defined above!) for all offsets in the random sequence. update the offset if any of the distances are less than the current best.
3. [ ] **Reconstruction**:

4. [ ] **NNF upsampling**:

5. [ ] **Patch aggregation**:

## Contributors
[Jacob Polloreno](https://github.com/JacobPolloreno)
