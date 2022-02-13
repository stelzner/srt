import os
import torch


class Checkpoint():
    """
    Handles saving and loading checkpoints.

    Args:
        checkpoint_dir (str): path where checkpoints are saved
        device: PyTorch device onto which loaded weights should be mapped
        kwargs: PyTorch modules whose state should be checkpointed
    """
    def __init__(self, checkpoint_dir='./chkpts', device=None, **kwargs):
        self.module_dict = kwargs
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def save(self, filename, **kwargs):
        """ Saves the current module states
        Args:
            filename (str): name of checkpoint file
            kwargs: Additional state to save
        """
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        outdict = kwargs
        for k, v in self.module_dict.items():
            if k in outdict:
                print(f"Warning: Checkpoint key '{k}' overloaded. Defaulting to saving state_dict {v}.")
            outdict[k] = v.state_dict()
        torch.save(outdict, filename)

    def load(self, filename):
        """Loads a checkpoint from file.
        Args:
            filename (str): Name of checkpoint file.
        Returns:
            Dictionary containing checkpoint data which does not correspond to registered modules.
        """

        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        print(f'Loading checkpoint from file {filename}...')
        state_dict = torch.load(filename, map_location=self.device)

        for k, v in self.module_dict.items():
            if k in state_dict:
                v.load_state_dict(state_dict[k])
            else:
                print(f'Warning: Could not find "{k}" in checkpoint!')

        remaining_state = {k: v for k, v in state_dict.items()
                           if k not in self.module_dict}
        return remaining_state

