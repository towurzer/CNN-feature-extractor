import torch
import torchvision.models as models
import torch.nn as nn

class EfficientNet(nn.Module):
    """
    A wrapper class for EfficientNet-B0 that functions as both an image  classifier and a feature extractor.
    https://medium.com/@kdk199604/efficientnet-smarter-not-just-bigger-neural-networks-94db3e2f8699

    """
    def __init__(self, train=False):
        super(EfficientNet, self).__init__()
        # download model and weights
        self.weights = models.EfficientNet_B0_Weights.DEFAULT
        self.model = models.efficientnet_b0(weights=self.weights)
        self.setMode(train)
        # ---
        self.extractedFeatures = [] # Buffer to store data captured by the forward hook
        self.hookHandle = self.registerHook() # Register the hook to extract the intermediate Data

    def setMode(self, train):
        """Sets the model to training or evaluation mode."""
        if train:
            self.model.train()
        else:
            self.model.eval()

    def registerHook(self):
        """
        Registers a forward hook on the global average pooling layer.
        This captures the feature vector before it reaches the final classification step.
        """
        # replace the classifier with an Identity layer, discarded since it would make the model a feature extractor but break its image classification capabilites.
        # self.model.classifier = nn.Identity()
        def featureHook(module, input, output):
            # output is the result of the avg pool layer ([N, 1280, 1, 1] -> [N, 1280])
            self.extractedFeatures = torch.flatten(output, 1)

        hook = self.model.avgpool.register_forward_hook(featureHook)  # Attach the hook to the avgpool layer
        return hook

    def extract(self, batch):
        """
        Performs a forward pass to extract both the top prediction and intermediate feature vector.

        :arg batch: Input image batch.

        :return tuple: (predictions, features)
        """
        with torch.no_grad():
            classifications = self.model(batch)
            predictions = torch.argmax(classifications, dim=1)
            features = self.extractedFeatures


        return predictions, features


    def inspect(self, showDetails=False):
        """
        Utility function to print the model architecture.
        :arg showDetails: Print Every Layer or only the first level modules
        """
        if showDetails:
            print(self.model)
        else:
            for name, _ in self.model.named_children():
                print(name)