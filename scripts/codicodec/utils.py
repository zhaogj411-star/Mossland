import os
import torch
import numpy as np

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None


@torch.no_grad()
def distribute(model, x, max_batch_size, device, *args, mixed_precision_enabled=False, **kwargs):
    """Apply model to x by splitting to multiple batches with max_batch_size.

    Moves inputs to the specified device, optionally autocasts to fp16
    if mixed_precision is enabled, and stitches outputs back together.
    """
    data_device = x.device

    def split_tensor(t, batch_size):
        return torch.split(t, batch_size, dim=0)

    def split_arg(arg, batch_size):
        if isinstance(arg, torch.Tensor):
            return split_tensor(arg, batch_size)
        elif isinstance(arg, list):
            if any(isinstance(item, torch.Tensor) for item in arg):
                splits = [split_tensor(item, batch_size) if isinstance(item, torch.Tensor) else [item] * num_batches
                          for item in arg]
                return [[split[i] for split in splits] for i in range(num_batches)]
            return [arg] * num_batches
        return [arg] * num_batches

    def to_device(arg):
        if isinstance(arg, torch.Tensor):
            return arg.to(device)
        elif isinstance(arg, list):
            return [to_device(a) for a in arg]
        return arg

    if max_batch_size is None or x.shape[0] <= max_batch_size:
        x = x.to(device)
        args = tuple(to_device(arg) for arg in args)
        kwargs = {k: to_device(v) for k, v in kwargs.items()}

        with torch.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=torch.float16, enabled=mixed_precision_enabled):
            outputs = model(x, *args, **kwargs)
        if isinstance(outputs, list):
            outputs = [out.to(data_device) for out in outputs]
        else:
            outputs = outputs.to(data_device)
    else:
        num_batches = (x.shape[0] + max_batch_size - 1) // max_batch_size
        x_splits = split_tensor(x, max_batch_size)

        arg_splits = [split_arg(arg, max_batch_size) for arg in args]
        kwarg_splits = {k: split_arg(v, max_batch_size) for k, v in kwargs.items()}

        outputs = []
        for i in range(num_batches):
            batch_x = x_splits[i].to(device)
            batch_args = tuple(to_device(arg_split[i]) for arg_split in arg_splits)
            batch_kwargs = {k: to_device(kwarg_splits[k][i]) for k in kwargs}

            with torch.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=torch.float16, enabled=mixed_precision_enabled):
                batch_output = model(batch_x, *batch_args, **batch_kwargs)
            if isinstance(batch_output, list):
                batch_output = [out.to(data_device) for out in batch_output]
            else:
                batch_output = batch_output.to(data_device)
            outputs.append(batch_output)

        if isinstance(outputs[0], list):
            outputs = [torch.cat([out[j] for out in outputs], dim=0) for j in range(len(outputs[0]))]
        else:
            outputs = torch.cat(outputs, dim=0)

    return outputs


def is_integer(x):
    """Return True if x is an integer-valued np.ndarray or torch.Tensor."""
    if isinstance(x, np.ndarray):
        return np.issubdtype(x.dtype, np.integer)
    elif isinstance(x, torch.Tensor):
        return x.dtype in [torch.int32, torch.int64]
    return False


def is_path(variable):
    """Return True if variable is a filesystem path that exists."""
    return isinstance(variable, str) and os.path.exists(variable)


def download_model():
    """Download the codicodec checkpoint from the Hugging Face Hub if missing.

    Expects a file named 'codicodec.pt' in the local 'models' directory inside
    the installed package. The repo_id is assumed to be 'SonyCSLParis/codicodec'.
    """
    filepath = os.path.abspath(__file__)
    lib_root = os.path.dirname(filepath)
    local_dir = os.path.join(lib_root, "models")
    local_path = os.path.join(local_dir, "codicodec.pt")

    if os.path.exists(local_path):
        return

    os.makedirs(local_dir, exist_ok=True)
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is required to download the model.")
    print("Downloading model...")
    downloaded_path = hf_hub_download(
        repo_id="SonyCSLParis/codicodec",
        filename="codicodec.pt",
        cache_dir=local_dir,
        local_dir=local_dir,
    )
    # If the file was saved under a nested path, copy/move it into expected location
    if downloaded_path != local_path and os.path.exists(downloaded_path):
        try:
            # Avoid overwriting if same inode
            if not os.path.exists(local_path):
                import shutil
                shutil.copy2(downloaded_path, local_path)
        except Exception:
            pass
    print("Model was downloaded successfully!")
