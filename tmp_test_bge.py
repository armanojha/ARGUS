import time
t0 = time.monotonic()
from FlagEmbedding import BGEM3FlagModel
print(f"Import: {time.monotonic()-t0:.1f}s")
t1 = time.monotonic()
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
print(f"Load: {time.monotonic()-t1:.1f}s")
t2 = time.monotonic()
out = model.encode(["test query"], return_dense=True, return_sparse=True)
print(f"Encode: {time.monotonic()-t2:.1f}s")
print(f"Dense shape: {out['dense_vecs'].shape}")
print(f"Sparse type: {type(out['lexical_weights'])}")
if isinstance(out["lexical_weights"], list):
    print(f"Sparse len: {len(out['lexical_weights'])}")
    print(f"Sparse[0] type: {type(out['lexical_weights'][0])}")
    print(f"Sparse[0] len: {len(out['lexical_weights'][0])}")
