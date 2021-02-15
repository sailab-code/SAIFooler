import torch
from tqdm import tqdm

class SaliencyEstimator:
    def __init__(self, mesh, classifier, p3d_module, unity_module, data_module, use_cache=False):
        self.mesh = mesh
        self.classifier = classifier
        self.p3d_module = p3d_module
        self.unity_module = unity_module
        self.data_module = data_module
        self.use_cache = use_cache
        self.device = self.classifier.device
        self.tex_shape = self.mesh.textures.maps_padded().shape[1:-1]

    def compute_batch_view_saliency_maps(self, images):
        images.requires_grad_(True)
        scores = self.classifier.classify(images)
        max_score, _ = scores.max(1, keepdim=True)
        max_score.mean().backward()
        view_saliency_maps = torch.mean(images.grad.data.abs(), dim=3, keepdim=True)
        return view_saliency_maps

    def convert_view2tex_saliency_maps(self, view_saliency_maps, view2tex_maps):
        tex_saliency = torch.zeros((view_saliency_maps.shape[0], *
        self.tex_shape), device=self.device)
        view2tex_maps = view2tex_maps * torch.tensor(self.tex_shape, device=self.device)
        view2tex_maps = view2tex_maps.to(dtype=torch.long)

        for idx in range(view_saliency_maps.shape[0]):
            tex_saliency[(idx, view2tex_maps[idx, ..., 1], view2tex_maps[idx, ..., 0])] = view_saliency_maps.squeeze(3)[idx]

        return tex_saliency

    def render_batch(self, render_module, batch, view2tex_maps=None):
        images = []
        view2tex_maps_list = []
        render_inputs, targets = batch
        for render_input in render_inputs:
            distance, camera_azim, camera_elev = render_input[:3]
            render_module.look_at_mesh(distance, camera_azim, camera_elev)
            lights_azim, lights_elev = render_input[3:]
            render_module.set_lights_direction(lights_azim, lights_elev)
            image = render_module.render(self.mesh)

            if view2tex_maps is None:
                view2tex_map = render_module.get_view2tex_map(self.mesh)
                view2tex_maps_list.append(view2tex_map)

            images.append(image.to(self.classifier.device))
        images = torch.cat(images, 0)
        view2tex_maps = view2tex_maps if view2tex_maps is not None else torch.cat(view2tex_maps_list, 0)
        return images, view2tex_maps

    def estimate_view_saliency_map(self):
        view_saliencies = [[], []]

        view2tex_maps = None
        for idx, render_module in tqdm(enumerate([self.p3d_module, self.unity_module]), position=0, desc="Module"):
            if render_module is None:
                del view_saliencies[idx]  # if unity module is not provided, just skip it
                continue
            for batch in tqdm(self.data_module.test_dataloader(), position=1, desc="Batch"):
                images, view2tex_maps = self.render_batch(render_module, batch, view2tex_maps)
                view_saliency_maps = self.compute_batch_view_saliency_maps(images)
                view_saliencies[idx].append(view_saliency_maps)

            view_saliencies[idx] = torch.cat(view_saliencies[idx], 0)

        # saliency_map = sum(view_saliencies) / len(view_saliencies)
        # saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        return view_saliencies


    def estimate_saliency_map(self):
        tex_saliencies = [[], []]

        view2tex_maps = None
        for idx, render_module in tqdm(enumerate([self.p3d_module, self.unity_module]), position=0, desc="Module"):
            if render_module is None:
                del tex_saliencies[idx]  # if unity module is not provided, just skip it
                continue
            for batch in tqdm(self.data_module.test_dataloader(), position=1, desc="Batch"):
                images, view2tex_maps = self.render_batch(render_module, batch, view2tex_maps)
                view_saliency_maps = self.compute_batch_view_saliency_maps(images)
                tex_saliency_maps = self.convert_view2tex_saliency_maps(view_saliency_maps, view2tex_maps)
                tex_saliencies[idx].append(tex_saliency_maps)

            tex_saliencies[idx] = torch.cat(tex_saliencies[idx], 0)

        # saliency_map = sum(tex_saliencies) / len(tex_saliencies)
        #saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        return tex_saliencies

    def to(self, device):
        self.mesh = self.mesh.to(device)
        self.mesh.textures = self.mesh.textures.to(device)
        self.p3d_module.to(device)
        self.classifier.to(device)
        self.device = device