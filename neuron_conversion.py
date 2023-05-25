import torch
import torch.neuron
import torchvision
ts_model = torch.jit.load("model_traced_320.pt")

nW = 320
nH = 320
image = torch.rand(size=(3,nW,nH))
allowed_ops = set(torch.neuron.get_supported_operations())
# allowed_ops.remove("torchvision::roi_align")
neuron_model = torch.neuron.trace(ts_model, example_inputs=image, op_whitelist=allowed_ops)
torch.jit.save(neuron_model,'model_traced_320_neuron.pt')
