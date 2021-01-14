from math import sqrt

import matplotlib.pyplot as plt
import torchvision

class Viewer3D:
    def __init__(self, module, figure_size=7):
        self.module = module
        self.figure_size = figure_size

    def __create_figure(self, image, title=""):
        fig = plt.figure(figsize=(self.figure_size, self.figure_size))
        plt.imshow(image)
        plt.title(title)
        plt.grid("off")
        plt.axis("off")
        return fig

    @staticmethod
    def make_grid(images):
        n_imgs_per_row = int(sqrt(len(images)))
        images = [image.permute(2, 0, 1) for image in images]
        grid = torchvision.utils.make_grid(images, nrow=n_imgs_per_row, scale_each=True)
        return grid.permute(1, 2, 0)

    def multi_view_grid(self, views):
        images = []
        for view in views:
            self.module.apply_input(*view)
            image = self.module.render()
            images.append(image.cpu())
        grid = self.make_grid(images)
        fig = self.__create_figure(grid, "Multi-view renders")
        fig.show()

    def textures(self):
        textures = {tex_name: texture.cpu() for tex_name, texture in self.module.get_textures().items()}
        for tex_name, tex in textures.items():
            fig = self.__create_figure(tex, tex_name)
            fig.show()


