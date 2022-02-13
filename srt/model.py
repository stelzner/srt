from torch import nn

from srt.encoder import SRTEncoder
from srt.decoder import SRTDecoder, NerfDecoder

class SRT(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.encoder = SRTEncoder(**cfg['encoder_kwargs'])
        if cfg['decoder'] == 'lightfield':
            self.decoder = SRTDecoder(**cfg['decoder_kwargs'])
        elif cfg['decoder'] == 'nerf':
            self.decoder = NerfDecoder(**cfg['decoder_kwargs'])
        else:
            raise ValueError('Unknown decoder type', cfg['decoder'])

