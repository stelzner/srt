from torch import nn

from srt.encoder import SRTEncoder, ImprovedSRTEncoder
from srt.decoder import SRTDecoder, ImprovedSRTDecoder, NerfDecoder

class SRT(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if 'encoder' in cfg and cfg['encoder'] == 'isrt':
            self.encoder = ImprovedSRTEncoder(**cfg['encoder_kwargs'])
        else:  # We leave the SRTEncoder as default for backwards compatibility
            self.encoder = SRTEncoder(**cfg['encoder_kwargs'])

        if cfg['decoder'] == 'lightfield':
            self.decoder = SRTDecoder(**cfg['decoder_kwargs'])
        elif cfg['encoder'] == 'isrt':
            self.decoder = ImprovedSRTDecoder(**cfg['decoder_kwargs'])
        elif cfg['decoder'] == 'nerf':
            self.decoder = NerfDecoder(**cfg['decoder_kwargs'])
        else:
            raise ValueError('Unknown decoder type', cfg['decoder'])

