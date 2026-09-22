# linnet-onnx

Imports an ONNX graph as Linnet source.

```python
from linnet_onnx import import_onnx

import_onnx("model.onnx", output="src/model.linnet", weights="weights/")
```

Initializers become parameters (their dotted names become the block
hierarchy), graph inputs become the entry's inputs with named symbolic
dimensions as generic parameters, and each node becomes the Linnet primitive
or standard-library operation with the same meaning. `linnet emit` prints
the plan as source and the result is checked before it is written; see
`docs/onnx.md`.
