"""Input-gradient saliency for qualitative ECG model inspection."""
import torch

def input_saliency(model, beat, target_class=None):
    model.eval(); x=torch.as_tensor(beat,dtype=torch.float32).reshape(1,1,-1).detach().requires_grad_(True)
    logits=model(x); cls=int(logits.argmax(1).item()) if target_class is None else int(target_class)
    model.zero_grad(set_to_none=True); logits[0,cls].backward()
    sal=x.grad.detach().abs().reshape(-1).cpu().numpy()
    return sal/(float(sal.max())+1e-8)
