This is a repo of core implementation of our research "FRADE: Forgery-aware Audio-distilled Multimodal Learning for Deepfake Detection" (ACM MM 2024, Poster) [Link](https://dl.acm.org/doi/abs/10.1145/3664647.3681672). this code version is based on ViT-base of timm.
  
## Dataset Pipeline
  The pipeline is identical to the RealForensics' pipeline: [Link](https://github.com/ahaliassos/RealForensics/tree/main/stage1/data). Note: The augmented transformations of Realforensics are not involved in our model training.

## Details
  The reduction factor is set to 8 for all learnable modules. Unless the paper specifies the settings of hyperparameters, all settings are followed as our implementation.
  As our experiments show, our ViT-based Frade would converge after around six training epochs on FakeAVCeleb.
## Citation
```
@inproceedings{nie2024frade,
  title={FRADE: Forgery-aware Audio-distilled Multimodal Learning for Deepfake Detection},
  author={Nie, Fan and Ni, Jiangqun and Zhang, Jian and Zhang, Bin and Zhang, Weizhe},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={6297--6306},
  year={2024}
}
```
