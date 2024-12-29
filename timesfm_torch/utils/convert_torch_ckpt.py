import timesfm
import torch
import os

context_len = 1024
horizon_len = 1


model = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="gpu",
        per_core_batch_size=32,
        horizon_len=horizon_len,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
)

torch_ckpt = model._model.state_dict()

if not os.path.exists('../ckpt'):
    os.makedirs('../ckpt')

freq_emb_ckpt = {}
freq_emb_ckpt['emb_var.weight'] = torch_ckpt['freq_emb.weight']
torch.save(freq_emb_ckpt, '../ckpt/freq_emb.pt')

horizon_ff_layer_ckpt = {}

horizon_ff_layer_ckpt['hidden_layer.weight'] = torch_ckpt['horizon_ff_layer.hidden_layer.0.weight']
horizon_ff_layer_ckpt['hidden_layer.bias'] = torch_ckpt['horizon_ff_layer.hidden_layer.0.bias']
horizon_ff_layer_ckpt['output_layer.weight'] = torch_ckpt['horizon_ff_layer.output_layer.weight']
horizon_ff_layer_ckpt['output_layer.bias'] = torch_ckpt['horizon_ff_layer.output_layer.bias']
horizon_ff_layer_ckpt['residual_layer.weight'] = torch_ckpt['horizon_ff_layer.residual_layer.weight']
horizon_ff_layer_ckpt['residual_layer.bias'] = torch_ckpt['horizon_ff_layer.residual_layer.bias']

torch.save(horizon_ff_layer_ckpt, '../ckpt/horizon_ff_layer.pt')

input_ff_layer_ckpt = {}

input_ff_layer_ckpt['hidden_layer.weight'] = torch_ckpt['input_ff_layer.hidden_layer.0.weight']
input_ff_layer_ckpt['hidden_layer.bias'] = torch_ckpt['input_ff_layer.hidden_layer.0.bias']
input_ff_layer_ckpt['output_layer.weight'] = torch_ckpt['input_ff_layer.output_layer.weight']
input_ff_layer_ckpt['output_layer.bias'] = torch_ckpt['input_ff_layer.output_layer.bias']
input_ff_layer_ckpt['residual_layer.weight'] = torch_ckpt['input_ff_layer.residual_layer.weight']
input_ff_layer_ckpt['residual_layer.bias'] = torch_ckpt['input_ff_layer.residual_layer.bias']

torch.save(input_ff_layer_ckpt, '../ckpt/input_ff_layer.pt')

stack_transformer_ckpt = {}
for i in range(20):
    
    stack_transformer_ckpt[f'layers.{i}.feed_forward.layer_norm.weight'] = torch_ckpt[f'stacked_transformer.layers.{i}.mlp.layer_norm.weight']
    stack_transformer_ckpt[f'layers.{i}.feed_forward.layer_norm.bias'] = torch_ckpt[f'stacked_transformer.layers.{i}.mlp.layer_norm.bias']
    stack_transformer_ckpt[f'layers.{i}.feed_forward.gate_proj.weight'] = torch_ckpt[f'stacked_transformer.layers.{i}.mlp.gate_proj.weight']
    stack_transformer_ckpt[f'layers.{i}.feed_forward.gate_proj.bias'] = torch_ckpt[f'stacked_transformer.layers.{i}.mlp.gate_proj.bias']
    stack_transformer_ckpt[f'layers.{i}.feed_forward.down_proj.weight'] = torch_ckpt[f'stacked_transformer.layers.{i}.mlp.down_proj.weight']
    stack_transformer_ckpt[f'layers.{i}.feed_forward.down_proj.bias'] = torch_ckpt[f'stacked_transformer.layers.{i}.mlp.down_proj.bias']
    
    stack_transformer_ckpt[f'layers.{i}.layer_norm.scale'] = torch_ckpt[f'stacked_transformer.layers.{i}.input_layernorm.weight']

    # stack_transformer_ckpt[f'layers.{i}.attention.scale_query.per_dim_scale'] = torch_ckpt[f'stacked_transformer.layers.{i}.self_attn.scaling']
    stack_transformer_ckpt[f'layers.{i}.attention.scaling'] = torch_ckpt[f'stacked_transformer.layers.{i}.self_attn.scaling']
    stack_transformer_ckpt[f'layers.{i}.attention.qkv_proj.weight'] = torch_ckpt[f'stacked_transformer.layers.{i}.self_attn.qkv_proj.weight']
    stack_transformer_ckpt[f'layers.{i}.attention.qkv_proj.bias'] = torch_ckpt[f'stacked_transformer.layers.{i}.self_attn.qkv_proj.bias']
    stack_transformer_ckpt[f'layers.{i}.attention.o_proj.weight'] = torch_ckpt[f'stacked_transformer.layers.{i}.self_attn.o_proj.weight']
    stack_transformer_ckpt[f'layers.{i}.attention.o_proj.bias'] = torch_ckpt[f'stacked_transformer.layers.{i}.self_attn.o_proj.bias']
    

torch.save(stack_transformer_ckpt, '../ckpt/stack_transformer.pt')
