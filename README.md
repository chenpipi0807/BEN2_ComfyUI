我给这个输出加了个MASK用来支持在更多地方使用，
此外我不知道他的requirements.txt里面的image是啥反正我没装也能用


# ComfyUI BEN - Background Erase Network

****


Remove backgrounds from images with [BEN2](https://huggingface.co/PramaLLC/BEN2) in [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## Installation

```
git clone https://github.com/PramaLLC/BEN2_ComfyUI.git
```
```
cd BEN2_ComfyUI
```
```
pip install -r requirements.txt
```
Go to the [BEN2 huggingface page](https://huggingface.co/PramaLLC/BEN2/tree/main) and download `BEN2_Base.pth`and place them inside:
```
...ComfyUI/custom-nodes/BEN2_ComfyUI/
```
The folder should look like this:
```
└── ComfyUI/custom-nodes/BEN2_ComfyUI/
    ├── __init__.py
    ├── background_erase_network.py
    ├── ben.png
    ├── BEN2_Base.pth
    ├── BEN2.py
    ├── README.md
    └── requirements.txt
```
## Nodes

### Background Erase Network
![BEN2](BEN2.png)




## Credits

- [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)

- [DoctorDiffusion/ComfyUI-BEN](https://github.com/DoctorDiffusion/ComfyUI-BEN)

⭐ If you like the project, please give it a star! ⭐

