import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

# Import core innovations (DPA, DDR/IMEX)
from .core_innovations import DPA_Transform, IMEX_Block, DPA_Estimator
from .utils import NormDownsample, NormUpsample


class CATFormer(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
                 ):
        super(CATFormer, self).__init__()

        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads

        # --- Stage 1: Dynamic White Point Adaptation (DPA) ---
        # The physical anchor: Transforms RGB -> CIE XYZ -> Adapted XYZ -> CIELAB (Decoupled)
        self.dpa_trans = DPA_Transform()
        # The estimator network: Predicts the Von Kries transform matrix
        self.dpa_estimator = DPA_Estimator()

        # --- Stage 2: Dual-branch Decoupled Restoration (DDR) ---

        # Encoder (Chrominance Path)
        self.Chrom_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(5, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.Chrom_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.Chrom_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.Chrom_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        # Encoder (Luminance Path)
        self.Lum_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(4, ch1, 3, stride=1, padding=0, bias=False),
        )
        self.Lum_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.Lum_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.Lum_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        # Interactive Mixture-of-Experts (IMEX) Blocks
        # These replace the old LCA blocks, handling cross-attention between Lum and Chrom
        self.IMEX_Chrom1 = IMEX_Block(ch2, head2)
        self.IMEX_Chrom2 = IMEX_Block(ch3, head3)
        self.IMEX_Chrom3 = IMEX_Block(ch4, head4)
        self.IMEX_Chrom4 = IMEX_Block(ch4, head4)
        self.IMEX_Chrom5 = IMEX_Block(ch3, head3)
        self.IMEX_Chrom6 = IMEX_Block(ch2, head2)

        self.IMEX_Lum1 = IMEX_Block(ch2, head2)
        self.IMEX_Lum2 = IMEX_Block(ch3, head3)
        self.IMEX_Lum3 = IMEX_Block(ch4, head4)
        self.IMEX_Lum4 = IMEX_Block(ch4, head4)
        self.IMEX_Lum5 = IMEX_Block(ch3, head3)
        self.IMEX_Lum6 = IMEX_Block(ch2, head2)

        # Decoders
        self.Chrom_Dblock3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.Chrom_Dblock2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.Chrom_Dblock1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.Chrom_Dblock0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 5, 3, stride=1, padding=0, bias=False)
        )

        self.Lum_Dblock3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.Lum_Dblock2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.Lum_Dblock1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.Lum_Dblock0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 4, 3, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        dtypes = x.dtype
        b, c, h, w = x.shape

        # 1. DPA Pipeline (Learnable Von Kries)
        xyz = self.dpa_trans.rgb_to_xyz(x)
        source_wp = self.dpa_estimator(xyz)
        source_wp = torch.clamp(source_wp, min=1e-8, max=1.088754)

        # Apply Von Kries Adaptation
        xyz_corrected = self.dpa_trans.von_kries_xyz_adaptation(xyz, source_wp)
        xyz_corrected = torch.clamp(xyz_corrected, min=1e-8, max=1.088754)

        # Convert to Decoupled Space (CIELAB-like)
        decoupled = self.dpa_trans.xyz_to_lab(xyz_corrected)

        # Split into Luminance (L) and Chrominance (ab)
        lum = decoupled[:, 2, :, :].unsqueeze(1).to(dtypes)
        chrom = decoupled[:, 0:2, :, :].to(dtypes)

        lum_in = torch.cat([lum, source_wp], dim=1)
        chrom_in = torch.cat([chrom, source_wp], dim=1)

        # 2. DDR: Encoder
        lum_enc0 = self.Lum_block0(lum_in)
        chrom_enc0 = self.Chrom_block0(chrom_in)
        lum_jump0 = lum_enc0
        chrom_jump0 = chrom_enc0

        lum_enc0 = self.Lum_block1(lum_enc0)
        chrom_enc0 = self.Chrom_block1(chrom_enc0)

        # 3. DDR: Interactive Processing (IMEX Blocks)
        # Cross-attention: Lum attends to Chrom, Chrom attends to Lum
        lum_enc1, _ = self.IMEX_Lum1(lum_enc0, chrom_enc0)
        chrom_enc1, _ = self.IMEX_Chrom1(chrom_enc0, lum_enc0)
        lum_jump1 = lum_enc1
        chrom_jump1 = chrom_enc1

        lum_enc1 = self.Lum_block2(lum_enc1)
        chrom_enc1 = self.Chrom_block2(chrom_enc1)

        lum_enc2, _ = self.IMEX_Lum2(lum_enc1, chrom_enc1)
        chrom_enc2, _ = self.IMEX_Chrom2(chrom_enc1, lum_enc1)
        lum_jump2 = lum_enc2
        chrom_jump2 = chrom_enc2

        lum_enc2 = self.Lum_block3(lum_enc2)
        chrom_enc2 = self.Chrom_block3(chrom_enc2)

        lum_enc3, _ = self.IMEX_Lum3(lum_enc2, chrom_enc2)
        chrom_enc3, _ = self.IMEX_Chrom3(chrom_enc2, lum_enc2)

        # 4. DDR: Decoder & Skip Connections
        lum_dec4, _ = self.IMEX_Lum4(lum_enc3, chrom_enc3)
        chrom_dec4, _ = self.IMEX_Chrom4(chrom_enc3, lum_enc3)

        chrom_dec4 = self.Chrom_Dblock3(chrom_dec4, chrom_jump2)
        lum_dec4 = self.Lum_Dblock3(lum_dec4, lum_jump2)

        lum_dec3, _ = self.IMEX_Lum5(lum_dec4, chrom_dec4)
        chrom_dec3, _ = self.IMEX_Chrom5(chrom_dec4, lum_dec4)

        chrom_dec3 = self.Chrom_Dblock2(chrom_dec3, chrom_jump1)
        lum_dec3 = self.Lum_Dblock2(lum_dec3, lum_jump1)

        lum_dec2, _ = self.IMEX_Lum6(lum_dec3, chrom_dec3)
        chrom_dec2, _ = self.IMEX_Chrom6(chrom_dec3, lum_dec3)

        lum_dec2 = self.Lum_Dblock1(lum_dec2, lum_jump0)
        chrom_dec2 = self.Chrom_Dblock1(chrom_dec2, chrom_jump0)

        lum_dec1 = self.Lum_Dblock0(lum_dec2)
        lum_residual = lum_dec1[:, 0:1, :, :]
        lum_prior_residual = lum_dec1[:, 1:, :, :]

        chrom_dec1 = self.Chrom_Dblock0(chrom_dec2)
        chrom_residual = chrom_dec1[:, 0:2, :, :]
        chrom_prior_residual = chrom_dec1[:, 2:, :, :]

        # 5. Reconstruction (Inverse Transform)
        output_decoupled = torch.cat([chrom_residual, lum_residual], dim=1) + decoupled
        output_rgb = self.dpa_trans.lab_to_rgb(output_decoupled)

        if self.training:
            return output_rgb, lum_prior_residual, chrom_prior_residual
        else:
            return output_rgb

    # Helper for loss calculation or visualization
    def get_decoupled_space(self, x):
        return self.dpa_trans.rgb_to_decoupled(x)