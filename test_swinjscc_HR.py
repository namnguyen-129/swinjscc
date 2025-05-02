import torch
from models.swinjscc import SWINJSCC
from dataset.getds import get_div2k    # Import hàm lấy dataset
from channels.channel_base import Channel  # Import lớp Channel
from torch import nn
class Args:
    base_snr = 20  # Example SNR value
    bs = 1  # Batch size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #ds = "cifar10"  # Dataset name
    snr_list = [10]  # Danh sách SNR
    ratio = [1/12]   # R = C(Bottleneck Channel Dimension)/(2 x C_in(3) x 2^i x 2^i)
    algo = "swinjscc"  # Tên thuật toán
    image_dims = (3, 256 , 256)
    downsample = 4
    #channel_number = snr_list[0] * (2 * 3 * 2**downsample * 2**downsample) # bottleneck channel   
    channel_number = 32 
    channel_type = "AWGN"
    inv_cdim = var_cdim = channel_number #khong biết đúng khóng
    encoder_kwargs = dict(
                img_size=(256,256), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 2, 2], num_heads=[4, 6, 8, 10], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
    decoder_kwargs = dict(
                img_size=(256,256),
                embed_dims=[320, 256, 192, 128], depths=[2, 2, 2, 2], num_heads=[10, 8, 6, 4], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
    pass_channel = True 

# Initialize arguments
args = Args()

# Initialize the model configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model with only 3 arguments
model = SWINJSCC(args, 3, 10).to(device)   # 3 channels (RGB), 10 classes (CIFAR-10)
channel = Channel(channel_type="AWGN", snr = args.base_snr).to(device)

# Load model from checkpoint
checkpoint_path = "/home/namdeptrai/djscc/SemCom_new (copy)/SemCom-Pytorch/out/checkpoints/DIV2K_20_[0.16666666666666666]_AWGN_19h23m15s_on_May_01_2025/epoch_0.pkl"  # Thay đường dẫn checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)

# Kiểm tra nội dung checkpoint
#print("Checkpoint keys:", checkpoint.keys())

# Load state_dict into the model
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint, strict=False)  # Nếu checkpoint chứa trực tiếp state_dict

# Set model to evaluation mode
model.eval()

# Load CIFAR-10 dataset using get_cifar10
(train_dl, test_dl, valid_dl), _ = get_div2k(args)

# Get a batch of data from the test DataLoader
data_iter = iter(test_dl)
images = next(data_iter)

# Chuyển dữ liệu sang đúng thiết bị
images = images.to(device)
print("Shape of images:", images.shape)
# Kiểm tra thiết bị của các tensor
print(f"Model device: {next(model.parameters()).device}")
print(f"Images device: {images.device}")
#print(f"Labels device: {labels.device}")

# Pass the input through the encoder
snr = args.snr_list[0]  # Lấy giá trị SNR đầu tiên từ danh sách
rate = args.channel_number

z, mask = model.encoder(images, snr, rate)  # Tách tuple trả về thành z và mask
print("Size of input images:", images.shape)
print("Size of z after encoder:", z.shape)
print("Size of mask after encoder:", mask.shape)

# Chuyển đổi kích thước z thành (B, C, H, W)
B, L, C = z.shape
H =  W = int(L**0.5)          # Giả định L là số lượng patch (H * W)
z_4D = z.reshape(B, H, W, C).permute(0, 3, 1, 2)  # Chuyển đổi về (B, C, H, W)
print("Size of z after reshape:", z_4D.shape)

# Chuyển đổi kích thước mask thành (B, C, H, W)
B_mask, L_mask, C_mask = mask.shape
mask_4D = mask.reshape(B_mask, H, W, C_mask).permute(0, 3, 1, 2)  # Chuyển đổi về (B, C, H, W)
print("Size of mask after reshape:", mask_4D.shape)

# Thêm noise vào z qua channel
z_noise = channel.forward(z_4D, snr)  # Truyền qua channel
print("Size of z after channel:", z_noise.shape)

# Chuyển đổi z_noise thành (B, L, C) để truyền vào decoder
z_noise_3D = z_noise.flatten(2).permute(0, 2, 1)  # Chuyển đổi về (B, L, C)
print("Size of z_noise after flatten:", z_noise_3D.shape)

# Pass the noisy latent representation through the decoder
recon_image = model.decoder(z_noise_3D, snr)
print("Recon_image ", recon_image)  

# Tính toán loss sử dụng MSE
mse_loss = torch.nn.MSELoss()
loss = mse_loss(images * 255., recon_image.clamp(0., 1.) * 255.)
print("MSE Loss:", loss.item())
print("PSNR:", 10 * torch.log10(255**2 / loss))
# Chuẩn bị hiển thị
recon_image = recon_image.clamp(0, 1).cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
orig_image = images.cpu().squeeze(0).permute(1, 2, 0).numpy()

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(orig_image)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(recon_image)
axes[1].set_title("Reconstruction")
axes[1].axis("off")

plt.tight_layout()
plt.show()

