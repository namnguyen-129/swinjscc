import torch
from models.djsccn import DJSCCN_CIFAR

from dataset.getds import get_cifar10  # Import hàm lấy dataset
from channels.channel_base import Channel

class Args:
    base_snr = 20  # Example SNR value
    device = 0  # Change to 'cuda' if using GPU
    inv_cdim = 32  # Cập nhật giá trị này để khớp với checkpoint
    var_cdim = 32  # Cập nhật giá trị này để khớp với checkpoint
    bs = 1  # Batch size
    ds = "cifar10"  # Dataset name
    channel_type = "AWGN" 
    
# Initialize arguments
args = Args()

channel = Channel(channel_type=args.channel_type, snr=args.base_snr)

# Move model to the correct device
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
# Load model from checkpoint
checkpoint_path = "/home/namdeptrai/djscc/SemCom_new (copy)/SemCom-Pytorch/out/checkpoints/CIFAR10_1_0.16666666666666666_AWGN_21h53m25s_on_Apr_20_2025/epoch_9.pkl"
checkpoint = torch.load(checkpoint_path, map_location=device)



# Initialize the model
model = DJSCCN_CIFAR(args, 3, 10).to(device)

model.load_state_dict(checkpoint, strict=False)  # Bỏ qua các tham số không khớp

# Set model to evaluation mode
model.eval()

# Load CIFAR-10 dataset using get_cifar10
(train_dl, test_dl, valid_dl), _ = get_cifar10(args)

# Get a batch of data from the test DataLoader
data_iter = iter(test_dl)
images, labels = next(data_iter)

# Move data to the correct device
images, labels = images.to(device), labels.to(device)

# Disable gradient computation for testing
with torch.no_grad():
    # Pass the input through the encoder
    print("Size of input images:", images.shape)
    z = model.encoder(images)
    print("Size of z after encoder:", z.size())
    print("z before channel:", z)

    # Pass the latent representation through the channel
    z_noise = channel.forward(z, args.base_snr)  # Truyền qua channel
    print("Size of z after channel:", z_noise.size())
    print("z after channel:", z_noise)

    # Pass the noisy latent representation through the decoder
    x_hat = model.decoder(z_noise)
    print("Size of x_hat after decoder:", x_hat.size())
    print("Input image shape:", images.shape)
    print("Reconstructed image shape:", x_hat.shape)

    # Tính toán loss sử dụng MSE
    mse_loss = torch.nn.MSELoss()  # Corrected from torch.nn.mse()
    loss = mse_loss(images * 255., x_hat.clamp(0., 1.) * 255.)
    print("MSE Loss:", loss.item())

    # Chuẩn bị hiển thị
    recon_image = x_hat.clamp(0, 1).cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
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
