import torch
from timesfm_torch.model.model import *
from timesfm_torch.model.utils import *

class TimesFm(nn.Module):
    def __init__(self, context_len, horizon_len, input_patch_len=32, output_patch_len=128, num_layers=20, model_dims=1280, num_outputs=10, tolerance=1e-6, pad_val=1123581321.0):
        super().__init__()
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.input_patch_len = input_patch_len
        self.output_patch_len = output_patch_len
        self.num_layers = num_layers
        self.model_dims = model_dims
        self.num_outputs = num_outputs
        
        self.tolerance = tolerance
        self.pad_val = pad_val

        self.freq_emb = Embedding(num_classes=3, input_dims=model_dims)
        self.position_emb = PositionalEmbedding(embedding_dims=model_dims).to('cuda')
        self.horizon_ff_layer = ResidualBlock(input_dims=model_dims, hidden_dims=model_dims, output_dims=model_dims).to('cuda')
        self.input_ff_layer = ResidualBlock(input_dims=64, hidden_dims=model_dims, output_dims=model_dims).to('cuda')
        self.stacked_transformer_layer = Transformer(num_layers=20, d_model=model_dims, num_heads=16, hidden_dim=model_dims).to('cuda')

    def load_from_checkpoint(self, ckpt_dir):
        self.freq_emb.load_state_dict(torch.load(f'{ckpt_dir}/freq_emb.pt', weights_only=True))
        self.horizon_ff_layer.load_state_dict(torch.load(f'{ckpt_dir}/horizon_ff_layer.pt', weights_only=True))
        self.input_ff_layer.load_state_dict(torch.load(f'{ckpt_dir}/input_ff_layer.pt', weights_only=True))
        self.stacked_transformer_layer.load_state_dict(torch.load(f'{ckpt_dir}/stack_transformer.pt', weights_only=True))

    def _preprocess(self, inputs, horizon_len):
        bs, input_len = inputs.shape
        padding = torch.zeros([bs, input_len + horizon_len], dtype=float).to(inputs.device)
        
        if input_len < self.context_len:
            num_front_pad = self.context_len - input_len
            inputs = torch.concat([torch.zeros([bs, num_front_pad]), inputs], dim=-1) 
            padding = torch.concat([torch.zeros([bs, num_front_pad]), padding], dim=-1) 
        elif input_len > self.context_len:
            inputs = inputs[:, -self.context_len:]
            padding = padding[:, -(self.context_len + horizon_len): ]

        return inputs, padding
    
    def _preprocess_input(self, input_ts, input_padding):
        """Preprocess input for stacked transformer."""

        # Reshape into patches (using view for efficiency)
        bsize = input_ts.shape[0]
        
        patched_inputs = input_ts.view(bsize, -1, self.input_patch_len)
        patched_pads = input_padding.view(bsize, -1, self.input_patch_len)

        patched_inputs = torch.where(
            torch.abs(patched_pads - 1.0) < self.tolerance,
            torch.tensor(0.0,
                        dtype=patched_inputs.dtype,
                        device=patched_inputs.device),
            patched_inputs,
        )
        patched_pads = torch.where(
            torch.abs(patched_inputs - self.pad_val) < self.tolerance,
            torch.tensor(1.0, dtype=patched_pads.dtype, device=patched_pads.device),
            patched_pads,
        )
        patched_inputs, stats = forward_transform(patched_inputs,patched_pads, self.tolerance, self.pad_val)

        # B x N x D
        patched_inputs = patched_inputs * (1.0 - patched_pads)
        concat_inputs = torch.cat([patched_inputs, patched_pads], dim=-1).float()
        model_input = self.input_ff_layer(concat_inputs)

        # A patch should not be padded even if there is at least one zero.
        patched_padding = torch.min(patched_pads,
                                    dim=-1)[0]  # Get the values from the min result
        
        pos_emb = self.position_emb(model_input.shape[1]).to(model_input.device)
        pos_emb = torch.concat([pos_emb] * model_input.shape[0], dim=0)
        pos_emb = shift_padded_seq(patched_padding, pos_emb)
        model_input += pos_emb

        return model_input, patched_padding, stats, patched_inputs
    
    def _postprocess_output(
        self,
        model_output: torch.Tensor,
        stats: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Postprocess output of stacked transformer."""

        # B x N x (H.Q)
        output_ts = self.horizon_ff_layer(model_output)

        # Reshape using view
        b, n, _ = output_ts.shape
        output_ts = output_ts.view(b, n, self.output_patch_len, self.num_outputs)

        return reverse_transform(output_ts, stats)
    
    def forward(self, input_ts, paddings, freq):
        bs = input_ts.shape[0]
        
        model_input, patched_padding, stats, _ = self._preprocess_input(input_ts, paddings)

        f_emb = self.freq_emb(torch.tensor([freq] * bs).unsqueeze(-1).long()).to(input_ts.device)

        model_input = model_input + f_emb # [b, patch_len, model_dims]

        model_output = self.stacked_transformer_layer(model_input.float(), paddings=patched_padding.float()) 
        
        output_ts = self._postprocess_output(model_output, stats)
        
        return output_ts

    def decode(self, input_ts, input_padding, horizon_len=None, freq=0, max_len=512):
        """
        input_ts: a tensor of shape (bs, context_len)
        freq: 0 for high frequency (default), 1 for medium, and 2 for low
        """
        
        if horizon_len is None:
            horizon_len = self.output_patch_len

        context_len = input_ts.shape[1]
        full_outputs = []
        if input_padding.shape[1] != context_len + horizon_len:
            raise ValueError(
                "Length of paddings must match length of input + horizon_len:"
                f" {input_padding.shape[1]} != {context_len} + {horizon_len}")
            

        num_decode_patches = (horizon_len + self.output_patch_len -
                            1) // self.output_patch_len
        
        final_out = input_ts
        # decoding in an autoregressive manner
        for step_index in range(num_decode_patches):
            current_padding = input_padding[:, 0:final_out.shape[1]]
            input_ts = final_out[:, -max_len:]
            input_padding = current_padding[:, -max_len:]
            
            fprop_outputs = self(input_ts, current_padding, freq)
            
            # (full batch, last patch, output_patch_len, index of mean forecast = 0)
            new_ts = fprop_outputs[:, -1, :self.output_patch_len, 0]
            new_full_ts = fprop_outputs[:, -1, :self.output_patch_len, :]
            # (full batch, last patch, output_patch_len, all output indices)
            full_outputs.append(new_full_ts)
            final_out = torch.concatenate([final_out, new_ts], axis=-1)

        full_outputs = torch.concatenate(full_outputs, axis=1)[:,0:horizon_len, :]

        return (full_outputs[:, :, 0], full_outputs)

        
    def forecast(self, inputs, freq=0, normalize=True):
        input_ts, input_padding = self._preprocess(inputs, horizon_len=self.horizon_len)

        mean_output, full_output = self.decode(input_ts, input_padding, horizon_len=self.horizon_len, freq=freq)
        return mean_output, full_output