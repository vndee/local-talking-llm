"""Convert Encodec checkpoint into the GGML format.

The bytes are packed in a binary file in the following order:
    - Magic (`ggml` in binary format)
    - Tensors

For each tensor, the bytes are packed as follows:
    - Number of dimensions    (int)
    - Name length             (int)
    - Dimensions              (int[n_dims])
    - Name                    (char[name_length])
    - Data                    (float[n_dims])

Note
----
Encodec uses weight normalization for its convolutional layers. All the weights are
decomposed into two tensors called with the suffixes _weight_v and _weight_g. A simple
call to the hook torch._weight_norm allows to get the final weight tensor of the
convolution from weight_v and weight_g. To drastically reduce the number of operations
at inference time, the ggml weights file only contain the final convolution weights but
does not store the decomposition into weight_v and weight_g.

Usage
-----

```bash
    python convert.py \
        --dir-model ./ggml_weights/ \
        --out-dir ./ggml_weights/ \
        --use-f16
```
"""
import argparse
from pathlib import Path
import struct

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--dir-model", type=str, required=True)
parser.add_argument("--out-dir", type=str, required=True)
parser.add_argument("--use-f16", action="store_true")


def parse_codec_model(checkpoint, outfile, use_f16):
    """Load encodec model checkpoint."""
    n_f16, n_f32 = 0, 0

    for name in checkpoint.keys():
        if "weight_g" in name:
            # the tensor has already been parsed with the corresponding "weight_v"
            # tensor to form the final weights tensor of the convolution, therefore
            # we skip it
            continue

        if "inited" in name or "cluster_size" in name or "embed_avg" in name:
            # "inited", "cluster_size" and "embed_avg" tensors in quantizer are not used
            # for the forward pass
            continue

        var_data = checkpoint[name]

        if not "weight_v" in name:
            # if conv kernel, do not squeeze because 3d tensor
            var_data = var_data.numpy().squeeze()
        else:
            # weight_v has its corresponding magnitude tensor to rescale the weights
            # of the convolutional layers. We parse both kinds of weights jointly to
            # build the final weight tensor of the convolution.
            base_name = name.split(".")[:-1]
            weight_g_name = ".".join(base_name + ["weight_g"])
            var_data_g = checkpoint[weight_g_name]

            final_var_data = torch._weight_norm(var_data, var_data_g, dim=0)
            var_data = final_var_data.numpy()

            name = ".".join(base_name + ["weight"])

        print(f"Processing variable: {name} with shape: {var_data.shape}")

        if use_f16:
            if "embed" in name:
                print("  Converting to float32")
                var_data = var_data.astype(np.float32)
                ftype_cur = 0
                n_f32 += 1
            elif "weight" in name:
                print("  Converting to float16")
                var_data = var_data.astype(np.float16)
                ftype_cur = 1
                n_f16 += 1
            else:
                print("  Converting to float32")
                var_data = var_data.astype(np.float32)
                ftype_cur = 0
                n_f32 += 1
        else:
            print("  Converting to float32")
            var_data = var_data.astype(np.float32)
            ftype_cur = 0
            n_f32 += 1

        n_dims = len(var_data.shape)
        encoded_name = name.encode("utf-8")
        outfile.write(struct.pack("iii", n_dims, len(encoded_name), ftype_cur))

        for i in range(n_dims):
            outfile.write(struct.pack("i", var_data.shape[n_dims - 1 - i]))
        outfile.write(encoded_name)

        var_data.tofile(outfile)

    outfile.close()

    print("\n")
    print(f"n_f16: {n_f16} ({n_f16/(n_f16 + n_f32)*100:.0f}%)")
    print(f"n_f32: {n_f32} ({n_f32/(n_f16 + n_f32)*100:.0f}%)")


def parse_hparams(outfile, use_f16):
    # for now this is hardcoded as we only support the 24Khz model
    in_channels = 1
    hidden_dim = 128
    n_filters = 32
    kernel_size = 7
    residual_kernel_size = 3
    n_bins = 1024
    bandwidth = 24
    sr = 24000
    ftype = int(use_f16)

    outfile.write(struct.pack("i", in_channels))
    outfile.write(struct.pack("i", hidden_dim))
    outfile.write(struct.pack("i", n_filters))
    outfile.write(struct.pack("i", kernel_size))
    outfile.write(struct.pack("i", residual_kernel_size))
    outfile.write(struct.pack("i", n_bins))
    outfile.write(struct.pack("i", bandwidth))
    outfile.write(struct.pack("i", sr))
    outfile.write(struct.pack("i", ftype))


if __name__ == "__main__":
    args = parser.parse_args()

    dir_model = Path(args.dir_model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    outfile = Path(out_dir / "ggml-model.bin")

    checkpoint = torch.load(dir_model / "encodec_24khz-d7cc33bc.th", map_location="cpu")

    # Step 1: insert ggml magic
    outfile = open(outfile, "wb")
    outfile.write(struct.pack("i", 0x67676d6c))

    # Step 2: insert hyperparameters
    parse_hparams(outfile, args.use_f16)

    # Step 3: insert weights
    parse_codec_model(checkpoint, outfile, args.use_f16)

    print("Done.")
