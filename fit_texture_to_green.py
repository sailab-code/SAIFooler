import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset

from saifooler.texture_module import TextureModule
import matplotlib.pyplot as plt


mesh_path = "./candle/candle_test.obj"




if __name__ == '__main__':
    tex_module = TextureModule(mesh_path)


    def show_figs():
        plt.figure(figsize=(7, 7))
        texture_image = tex_module.apply_filter().maps_padded()
        plt.imshow(texture_image.squeeze().detach().cpu().numpy())
        plt.grid("off")
        plt.axis("off")

        image = tex_module.render().cpu().detach().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.grid("off")
        plt.axis("off")
        plt.show()

    show_figs()

    trainer = pl.Trainer(max_epochs=50)
    trainer.fit(tex_module, None, None)

    show_figs()
