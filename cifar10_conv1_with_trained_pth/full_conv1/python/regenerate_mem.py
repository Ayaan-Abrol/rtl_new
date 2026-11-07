import sys
from pathlib import Path
import torch
import numpy as np
checkpoint=Path(sys.argv[1]) if len(sys.argv)>1 else Path("../vgg6_cifar10.pth")
out=Path("mem");out.mkdir(exist_ok=True)
s=torch.load(checkpoint,map_location="cpu",weights_only=True)
Wq=np.rint(s["conv1.weight"].numpy()*65536).astype(np.int32)
Bq=np.rint(s["conv1.bias"].numpy()*65536).astype(np.int32)
rng=np.random.default_rng(20261107)
Xq=np.rint(rng.uniform(-1.5,1.5,size=(3,32,32))*65536).astype(np.int32)
def save(a,name):
 with open(out/name,"w") as f:
  for v in a.reshape(-1):f.write(f"{int(v)&0xffffffff:08x}\n")
save(Wq,"conv1_all_weights_q16.mem");save(Bq,"conv1_all_bias_q16.mem");save(Xq,"input_q16.mem")
Xpad=np.pad(Xq,((0,0),(1,1),(1,1)))
G=np.empty((64,32,32),dtype=np.int32)
for f in range(64):
 for y in range(32):
  for x in range(32):
   a=int(np.sum(Xpad[:,y:y+3,x:x+3].astype(np.int64)*Wq[f].astype(np.int64)))
   G[f,y,x]=(a>>16)+int(Bq[f])
save(G,"golden_conv1_q16.mem")
print("Wrote Conv1 64-filter golden and memory files to",out)
