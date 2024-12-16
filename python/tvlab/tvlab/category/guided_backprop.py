'''
Copyright (C) 2023 TuringVision

Produces gradients generated with guided back propagation from the
given image for model visualizations.
'''

import numpy as np

# https://github.com/utkuozbulak/pytorch-cnn-visualizations.git

def convert_to_grayscale(im_as_arr):
    """
    Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    return grayscale_im


class GuidedBackprop:
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        self.hooks = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def __enter__(self, *args): return self
    def __exit__ (self, *args): self.remove_hooks()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def hook_layers(self):
        from torch.nn import Sequential

        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        def get_first_layer(module):
            if isinstance(module, Sequential):
                return get_first_layer(module[0])
            return module
        # Register hook to the first layer
        first_layer = get_first_layer(self.model)
        self.hooks.append(first_layer.register_backward_hook(hook_function))

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        import torch
        from torch.nn import Sequential, ReLU

        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        def update_relus_seq(seq):
            for module in seq:
                if isinstance(module, Sequential):
                    update_relus_seq(module)
                if isinstance(module, ReLU):
                    self.hooks.append(module.register_backward_hook(relu_backward_hook_function))
                    self.hooks.append(module.register_forward_hook(relu_forward_hook_function))

        update_relus_seq(self.model)

    def generate_gradients(self, input_image, target_class):
        from torch import FloatTensor
        # Forward pass
        input_image.unsqueeze_(0)
        input_image = input_image.cuda()
        input_image.requires_grad = True
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output = one_hot_output.cuda()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,xxx,xxx)
        gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        return gradients_as_arr

    def get_heatmap(self, input_image, target_class):
        gradients = self.generate_gradients(input_image, target_class)
        heatmap = convert_to_grayscale(gradients)
        return heatmap
