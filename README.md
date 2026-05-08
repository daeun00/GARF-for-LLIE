# Gate-aware recurrent framework for progressive low-light image enhancement
The code will be uploaded after the paper is published.
 </br> *- currently under review at IEEE Access*
## Abstract
This study proposes a gate-aware recurrent framework for low-light image enhancement. By augmenting the conventional long short-term memory architecture with an additional gating mechanism, a gate-aware recurrent illumination block is proposed to selectively retain and update information across multiple iterations, thereby preserving contextual dependencies while progressively improving illumination. To realize this process, two specialized blocks are incorporated: a gate-aware feature amplification block that selectively enhances brightness-related features through global maximum pooling, and a global illumination refinement block that regulates the overall brightness levels to achieve stable enhancement. Furthermore, initializing hidden states with saturation features from the input low-light image stabilizes color restoration. In addition, an auxiliary refinement step ensures sharper details and more natural brightness reproduction. Extensive experiments, including thorough ablation studies, validate the contributions of each design component. The results demonstrate that the proposed network consistently achieves state-of-the-art performance on both reference-based and no-reference datasets, providing a lightweight yet effective solution for practical low-light image enhancement applications.

## Results
* **Qualitative Comparison**<br/>
<img src="/figures/qualitative_comparison.png" width="90%" height="90%" title="Qualitative Comparison"></img><br/>
<br/>
