# Gate-Aware Recurrent Framework for Progressive Low-Light Image Enhancement

## Abstract
This study proposes a gate-aware recurrent framework for low-light image enhancement.
By augmenting the conventional long short-term memory architecture with an additional gating mechanism,
a gate-aware recurrent illumination block is proposed to selectively retain and update information across
multiple iterations, thereby preserving contextual dependencies while progressively improving illumination.
To realize this process, two specialized blocks are incorporated: a gate-aware feature amplification block that
selectively enhances brightness-related features through global maximum pooling, and a global illumination
refinement block that regulates the overall brightness levels to achieve stable enhancement. Furthermore,
initializing hidden states with saturation features from the input low-light image stabilizes color restoration.
In addition, an auxiliary refinement step ensures sharper details and more natural brightness reproduction.
Extensive experiments, including thorough ablation studies, validate the contributions of each design component.
The results demonstrate that the proposed network consistently achieves state-of-the-art performance
on both reference-based and no-reference datasets, providing a lightweight yet effective solution for practical
low-light image enhancement applications.  :book: : [[Paper]](https://ieeexplore.ieee.org/document/11527169)

## Architecture
* **Overall Architecture**<br/>
<img src="/figures/overall architecture.jpg" width="90%" height="90%" title="Qualitative Comparison"></img><br/>
<br/>

## Requirements
* Python 3.8
* PyTorch 2.4
* CUDA 11.8
```
pip install pillow, opencv-python, scikit-image, sacred, pymongo
```

## Test
* Put test images under *./test_img* folder.
* Put the trained model under  *./models* folder.
<br/> - You can use best pretrained model file : *./models/LIENet.pth*
* Run test.py
```
python test.py --modelfile models/LIENet.pth
```
* The test results will be saved to the folder: ./output.
<br/>

## Results
* You can check example results about *DICM*, *Fusion*, *LIME*, *LOL*, *MEF* datasets in each folder.<br/>
* **Qualitative Comparison**<br/>
<img src="/figures/qualitative_comparison.png" width="90%" height="90%" title="Qualitative Comparison"></img><br/>
<br/>

## Citation Information
```
@article{GARF2026lee,
  title={Gate-Aware Recurrent Framework for Progressive Low-Light Image Enhancement},
  author={Lee Da Eun, Jun Young Park, and Il Kyu Eom},
  journal={IEEE Access},
  year={2026},
}
```
