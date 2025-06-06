import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from slot_attention import SlotAttention
from adaptive_slot_wrapper import AdaptiveSlotWrapper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)

"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        # print('grid.shape = ' + str(grid.shape))
        # print('inputs.shape = b' + str(inputs.shape))
        return inputs + grid

class Encoder(nn.Module):
    def __init__(self, resolution, hid_dim):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, hid_dim, 5, padding = 2)
        # self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        # self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        # self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)

        self.conv1 = nn.Conv2d(1, hid_dim, 5, padding=2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding=2)
        # self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, stride=2)
        # self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, stride=2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding=2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, padding=2)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=1)

        self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)

        # print('x.shape(Encoder) = ' + str(x.shape))
        x = x.permute(0,2,3,1)
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        return x

class Decoder(nn.Module):
    def __init__(self, hid_dim, resolution):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2).to(device)
        self.conv6 = nn.ConvTranspose2d(hid_dim, 2, 3, stride=(1, 1), padding=1).to(device)

        # self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        # self.conv6 = nn.ConvTranspose2d(hid_dim, 2, 3, stride=(2, 2), padding=2, output_padding=1).to(device)

        self.decoder_initial_size = (10, 15)
        self.decoder_pos = SoftPositionEmbed(hid_dim, self.decoder_initial_size)
        self.resolution = resolution

    def forward(self, x):
        x = self.decoder_pos(x)
        x = x.permute(0,3,1,2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        # print('x.shape(Decoder) = ' + str(x.shape))
        x = x[:,:,:self.resolution[0], :self.resolution[1]]
        x = x.permute(0,2,3,1)
        return x

"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution, num_slots, num_iterations, hid_dim, base_sigma=0, temperature=None):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.encoder_cnn = Encoder((81, 121), self.hid_dim)
        self.decoder_cnn = Decoder(self.hid_dim, self.resolution)

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=hid_dim,
            iters = self.num_iterations,
            eps = 1e-8, 
            hidden_dim = 128,
            base_sigma=base_sigma,
            temperature=temperature,
            sink_reg=0.01,
            noise_std=0)
        
        self.adaptive_slot_wrapper = AdaptiveSlotWrapper(
            slot_attn=self.slot_attention,
            temperature=1.0
        )

    def forward(self, image):
        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        x = nn.LayerNorm(x.shape[1:]).to(device)(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        # slots = self.slot_attention(x)
        slots, keep_slots = self.adaptive_slot_wrapper(x)
        # `slots` has shape: [batch_size, num_slots, slot_size].
        # Keep only the slots with indices given in keep_slots
        slots = slots[keep_slots.bool()]
        # print('slots.shape = ' + str(slots.shape))

        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        # slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots = slots.unsqueeze(1).unsqueeze(2)
        slots = slots.repeat((1, 10, 15, 1))
        
        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        x = self.decoder_cnn(slots)
        # print('x.shape = ' + str(x.shape))
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        out = torch.full((image.shape[0], keep_slots.shape[1], x.shape[1], x.shape[2], x.shape[3]), 0.0).to(device)
        out[:,:,:,:,1:2] = -1e9
        # `out` has shape: [batch_size, num_slots, width, height, num_channels+1].
        
        b_idx, s_idx = keep_slots.nonzero(as_tuple=True)
        out[b_idx, s_idx] = x

        # Undo combination of slot and batch dimension; split alpha masks.
        # recons, masks = x.reshape(image.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([3,1], dim=-1)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        recons = out[:,:,:,:,0:1]
        masks = out[:,:,:,:,1:2]

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0,3,1,2)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].
        
        eps = 1e-8                         # avoid log(0)
        entropy_map = masks * (masks.clamp_min(eps).log())
        loss_entropy = -entropy_map.sum(dim=1).mean()      # scalar

        num_slots = masks.shape[1]
        flat = masks.view(masks.shape[0], num_slots, -1)            # [batch_size, num_slots, width * height]
        gram = torch.bmm(flat, flat.transpose(1, 2))       # [batch_size, num_slots, num_slots]
        # zero diagonal, average the off-diagonals
        loss_repulsion = gram.sum() - gram.diagonal(dim1=1, dim2=2).sum()
        loss_repulsion = loss_repulsion / (masks.size(0) * num_slots * (num_slots - 1))

        return recon_combined, recons, masks, slots, loss_entropy, loss_repulsion
