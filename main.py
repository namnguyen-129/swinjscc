import os
import argparse
from train.train_djsccn import DJSCCNTrainer
from train.train_djsccf import DJSCCFTrainer
from train.train_swindjscc import SWINJSCCTrainer
from torch import nn
trainer_map = {
    "djsccf": DJSCCFTrainer,
    "djsccn": DJSCCNTrainer,
    "swinjscc": SWINJSCCTrainer
    }

# ratio_list = [1/6, 1/12]
# snr_list = [19, 13, 7, 4, 1]
ratio_list = [1/6]
snr_list = [20]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='./out',
                         help="Path to save outputs")
    parser.add_argument("--ds", type=str, default='cifar10',
                        choices=['cifar10', 'DIV2K', 'mnist'],
                        help="Dataset")
    parser.add_argument("--base_snr", type=float, default=10,
                        help="SNR during train")
    parser.add_argument('--channel_type', default='AWGN', type=str,
                        choices=['AWGN', 'Rayleigh'], help='channel')
    parser.add_argument("--recl", type=str, default='mse',
                        help="Reconstruction Loss")
    parser.add_argument("--clsl", type=str, default='ce',
                        help="Classification Loss")
    parser.add_argument("--disl", type=str, default='kl',
                        help="Invariance and Variance Loss")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Inner learning Rate")

    # Thêm tham số distortion_metric
    parser.add_argument("--distortion_metric", type=str, default="MSE",
                        choices=["MSE", "MS-SSIM"],
                        help="Distortion metric to use (MSE or MS-SSIM)")

    # Loss Setting
    parser.add_argument("--cls-coeff", type=float, default=0.5,
                        help="Coefficient for Classification Loss")
    parser.add_argument("--rec-coeff", type=float, default=1,
                        help="Coefficient for Reconstruction Loss")
    parser.add_argument("--inv-coeff", type=float, default=0.2,
                        help="Coefficient for Invariant Loss")
    parser.add_argument("--var-coeff", type=float, default=0.2,
                        help="Coefficient for Variant Loss")

    # Model Setting
    parser.add_argument("--inv-cdim", type=int, default=32,
                        help="Channel dimension for invariant features")
    parser.add_argument("--var-cdim", type=int, default=32,
                        help="Channel dimension for variant features")

    # VAE Setting
    parser.add_argument("--vae", action="store_true",
                        help="vae switch")
    parser.add_argument("--kld-coeff", type=float, default=0.00025,
                        help="VAE Weight Coefficient")

    # Meta Setting
    parser.add_argument("--bs", type=int, default=128,
                        help="#batch size")
    parser.add_argument("--wk", type=int, default=os.cpu_count(),
                        help="#number of workers")
    parser.add_argument("--out-e", type=int, default=10,
                        help="#number of epochs")
    parser.add_argument("--dv", type=int, default=0,
                        help="Index of GPU")
    parser.add_argument("--device", type=bool, default=True,
                        help="Return device or not")
    parser.add_argument("--operator", type=str, default='window',
                        help="Operator for Pycharm")

    # LOGGING
    parser.add_argument('--wandb', action='store_true',
                        help='toggle to use wandb for online saving')
    parser.add_argument('--log', action='store_true',
                        help='toggle to use tensorboard for offline saving')
    parser.add_argument('--wandb_prj', type=str, default="SemCom-",
                        help='toggle to use wandb for online saving')
    parser.add_argument('--wandb_entity', type=str, default="scalemind",
                        help='toggle to use wandb for online saving')
    parser.add_argument("--verbose", action="store_true",
                        help="printout mode")
    parser.add_argument("--algo", type=str, default="swinjscc",
                        help="necst/djsccf mode")
    
    # RUNNING
    parser.add_argument('--train_flag', type=str, default="True",
                        help='Training mode')
    # Model size of swinjscc
    parser.add_argument('--model_size', type=str, default="small",
                        choices=['small', 'base', 'large'],
                        help='Model size of swinjscc')
    args = parser.parse_args()
    
    if args.algo not in trainer_map:
        raise ValueError("Invalid trainer")
    
    TrainerClass = trainer_map[args.algo]
    
    "Để tính snr và channel number"
    args.snr_list = snr_list
    args.ratio = ratio_list
    args.pass_channel = True
    if args.ds == 'cifar10':
        args.image_dims = (3, 32, 32)
        args.downsample = 2
        args.bs = 128
    elif args.ds == 'DIV2K':
        args.image_dims = (3, 256, 256)
        args.downsample = 4
        args.bs = 4
    else:  # ví dụ mnist
        args.image_dims = (1, 28, 28)
        args.downsample = 2

    # Kích thước latent channels
    args.channel_number = int(args.var_cdim)

    # Unpack spatial dims
    _, H, W = args.image_dims

    # Thiết lập encoder_kwargs
    if args.ds == 'cifar10':
        args.encoder_kwargs = dict(
        img_size=(H, W), patch_size=2, in_chans=args.image_dims[0],
        embed_dims=[64, 128], depths=[2, 4], num_heads=[4, 8],
        C=args.channel_number,
        window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm,  # Sử dụng nn.LayerNorm thay vì None
        patch_norm=True
    )

    # Thiết lập decoder_kwargs
        args.decoder_kwargs = dict(
        img_size=(H, W),
        embed_dims=[128, 64], depths=[4, 2], num_heads=[8, 4],
        C=args.channel_number,
        window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm,  # Sử dụng nn.LayerNorm thay vì None
        patch_norm=True
    )
    elif args.ds == 'DIV2K':
        #image_dims = (3, 256, 256)
        # train_data_dir = ["/media/D/Dataset/HR_Image_dataset/"]
        # base_path = "/home/namdeptrai/djscc/data"
        # if args.testset == 'kodak':
        #     test_data_dir = ["/home/namdeptrai/djscc/data/kodak/"]
        
        # train_data_dir = [base_path + '/clic2020/**',
        #                   base_path + '/clic2021/train',
        #                   base_path + '/clic2021/valid',
        #                   base_path + '/clic2022/val',
        #                   base_path + '/DIV2K_train_HR',
        #                   base_path + '/DIV2K_valid_HR']
        # batch_size = 16
        # if args.model == 'SwinJSCC_w/o_SAandRA' or args.model == 'SwinJSCC_w/_SA':
        #     channel_number = int(args.C)
        # else:
        #     channel_number = None
        #channel_number =int (args.c)
        if args.model_size == 'small':
            args.encoder_kwargs = dict(
                img_size=(H,W), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 2, 2], num_heads=[4, 6, 8, 10], C=args.channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
            args.decoder_kwargs = dict(
                img_size=(H,W),
                embed_dims=[320, 256, 192, 128], depths=[2, 2, 2, 2], num_heads=[10, 8, 6, 4], C=args.channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
        elif args.model_size == 'base':
            args.encoder_kwargs = dict(
                img_size=(H,W), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10], C=args.channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
            args.decoder_kwargs = dict(
                img_size=(H,W),
                embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4], C=args.channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
        elif args.model_size =='large':
            args.encoder_kwargs = dict(
                img_size=(H,W), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 18, 2], num_heads=[4, 6, 8, 10], C=args.channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
            args.decoder_kwargs = dict(
                img_size=(H,W),
                embed_dims=[320, 256, 192, 128], depths=[2, 18, 2, 2], num_heads=[10, 8, 6, 4], C=args.channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
    if args.train_flag == "True":
        print("Training mode")
        for ratio in ratio_list:
            for snr in snr_list:
                args.ratio = [ratio]  # Đảm bảo args.ratio luôn là một danh sách
                args.base_snr = snr

                trainer = TrainerClass(args=args)
                trainer.train()
    
    else:
        print("Evaluation mode")
        trainer = TrainerClass(args=args)

        config_dir = os.path.join(args.out, 'configs')
        config_files = [os.path.join(config_dir, name) for name in os.listdir(config_dir)
                        if (args.ds in name or args.ds.upper() in name) and args.channel_type in name and name.endswith('.yaml')]
        output_dir = args.out

        for config_path in config_files:
            trainer.evaluate(
                config_path=config_path,
                output_dir=output_dir
            )
