import re, os, torch_sim, inspect
f = os.path.join(os.path.dirname(inspect.getfile(torch_sim)), "transforms.py")
txt = open(f).read()
txt2 = re.sub(
    r"torch\.nonzero\(\s*(\w+)\s*,\s*as_tuple=True\s*,?\s*\)",
    r"torch.where(\1)",
    txt,
    flags=re.DOTALL,
)
n = txt.count("as_tuple") - txt2.count("as_tuple")
open(f, "w").write(txt2)
print(f"Patched {n} occurrence(s) in {f}")
