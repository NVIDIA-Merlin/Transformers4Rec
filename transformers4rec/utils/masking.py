class MaskSequence:
    """
    Module to prepare masked data for LM tasks

    Parameters
    ----------
    pad_token:
        index of padding.
    hidden_size:
        dimension of the interaction embeddings
    """

    def __init__(self, hidden_size: int, device: str, pad_token: int = 0):
        self.pad_token = pad_token
        self.hidden_size = hidden_size
        self.device = device
