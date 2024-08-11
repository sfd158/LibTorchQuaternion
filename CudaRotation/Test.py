import copy
import numpy as np
import time
import torch
from torch import nn
from scipy.spatial.transform import Rotation
import RotationCuda

from VclSimuBackend.DiffODE import DiffQuat
import RotationLibTorch as TorchQuat

def test_quat_multiply():  # check OK
    batch = int(1e7)
    dtype = torch.float64
    a = torch.randn(batch, 4, device="cuda", dtype=dtype)
    a /= torch.linalg.norm(a, dim=-1, keepdim=True)
    b = torch.randn(batch, 4, device="cuda", dtype=dtype)
    b /= torch.linalg.norm(b, dim=-1, keepdim=True)
    # gt_res = (Rotation(a.cpu().numpy()) * Rotation(b.cpu().numpy())).as_quat()
    param_a, param_b = nn.Parameter(a.clone(), requires_grad=True), nn.Parameter(b.clone(), requires_grad=True)
    
    start  = time.time()
    res_cuda = RotationCuda.quat_multiply(param_a, param_b)
    loss_cuda = (1 + res_cuda).cos().sum()
    loss_cuda.backward()
    end = time.time()
    print(end - start)

    param_c, param_d = nn.Parameter(a.clone(), requires_grad=True), nn.Parameter(b.clone(), requires_grad=True)
    start = time.time()
    res_torch = DiffQuat.quat_multiply(param_c, param_d)
    loss_torch = (1 + res_torch).cos().sum()
    loss_torch.backward()
    end = time.time()
    print(end - start)

    with torch.no_grad():
        print(torch.max(torch.abs(param_c.grad - param_a.grad)))
        print(torch.max(torch.abs(param_d.grad - param_b.grad)))

def test_quat_from_matrix():
    for i in range(2):
        batch = int(233333)
        dtype = torch.float32
        a = torch.randn(batch, 4, device="cpu", dtype=dtype)
        a /= torch.linalg.norm(a, dim=-1, keepdim=True)
        mat = Rotation(a.cpu().numpy()).as_matrix()

        mat_0 = nn.Parameter(torch.from_numpy(mat.copy()).to("cuda"), True)
        t0 = time.time()
        qa = DiffQuat.quat_from_matrix(mat_0)
        la = (qa.cos()).sum()
        la.backward()
        t1 = time.time()
        print(f"{t1 - t0:.4f}", flush=True)

        mat_1 = nn.Parameter(torch.from_numpy(mat.copy()).to("cuda"), True)
        t2 = time.time()
        qb = RotationCuda.quat_from_matrix(mat_1)
        lb = (qb.cos()).sum()
        lb.backward()
        t3 = time.time()
        print(f"{t3 - t2:.4f}", flush=True)

        with torch.no_grad():
            print(i, (la - lb).item())
    

def test_quat_apply():
    batch = int(5000000)
    dtype = torch.float32
    a = torch.randn((batch, 4), device="cuda", dtype=dtype)
    a /= torch.linalg.norm(a, dim=-1, keepdim=True)
    b = torch.randn((batch, 3), device="cuda", dtype=dtype)
    for i in range(20):
        param_a0 = nn.Parameter(a.clone(), True)
        param_b0 = nn.Parameter(b.clone(), True)

        t0 = time.time()
        r0 = DiffQuat.quat_apply(param_a0, param_b0)
        l0 = r0.cos().sum()
        l0.backward()
        t1 = time.time()
        print(f"{t1 - t0:.4f}", flush=True)

        param_a1 = nn.Parameter(a.clone(), True)
        param_b1 = nn.Parameter(b.clone(), True)

        t2 = time.time()
        r1 = RotationCuda.quat_apply(param_a1, param_b1)
        l1 = r1.cos().sum()
        l1.backward()
        t3 = time.time()
        print(f"{t3 - t2:.4f}", flush=True)

        print((param_b0.grad - param_b1.grad).abs().max())

if __name__ == "__main__":
    test_quat_apply()