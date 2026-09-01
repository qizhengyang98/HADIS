import torch
import torch.nn as nn


class CLIPDiscriminator(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_image_encoder = clip_model.visual
        self.fc = nn.Sequential(
            nn.Linear(clip_model.visual.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, image):
        with torch.no_grad():
            x = self.clip_image_encoder(image)
        return self.fc(x), x
