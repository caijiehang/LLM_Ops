import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(current_dir, "../scripts")
sys.path.append(scripts_dir)

import llm_ops

def warm_up():
    N = 102400
    a = torch.randn(N,dtype=torch.float32,device='cuda').contiguous()
    b = torch.randn(N,dtype=torch.float32,device='cuda').contiguous()

    for _ in range(100):
        _ = torch.add(a,b)

def test_elementwise_f32():
    N = 102400
    a = torch.randn(N,dtype=torch.float32,device='cuda').contiguous()
    b = torch.randn(N,dtype=torch.float32,device='cuda').contiguous()
    c_custom = torch.zeros_like(a).contiguous()
    c_pytorch = torch.zeros_like(a)

    warm_up()

    c_pytorch = torch.add(a,b)
    torch.cuda.synchronize()

    llm_ops.elementwise.elementwise_f32(a,b,c_custom,N)
    torch.cuda.synchronize()
    


    max_diff = torch.max(torch.abs(c_custom-c_pytorch)).item()

    is_correct = torch.allclose(c_pytorch, c_custom, atol=1e-3, rtol=1e-3)

    print(f"elementwise_f32单元最大绝对误差 (Max Diff): {max_diff:.6f}")
    if not is_correct:
        print("elementwise_f32单元精度不对！\n")
        return
    else:
        print("elementwise_f32单元测试通过！\n")
        return

def test_elementwise_f32x4():
    N = 102400
    a = torch.randn(N,dtype=torch.float32,device='cuda').contiguous()
    b = torch.randn(N,dtype=torch.float32,device='cuda').contiguous()
    c_custom = torch.zeros_like(a).contiguous()
    c_pytorch = torch.zeros_like(a)

    warm_up()

    c_pytorch = torch.add(a,b)
    torch.cuda.synchronize()

    llm_ops.elementwise.elementwise_f32x4(a,b,c_custom,N)
    torch.cuda.synchronize()

    max_diff = torch.max(torch.abs(c_custom-c_pytorch)).item()

    is_correct = torch.allclose(c_pytorch, c_custom, atol=1e-3, rtol=1e-3)

    print(f"elementwise_f32x4单元最大绝对误差 (Max Diff): {max_diff:.6f}")
    if not is_correct:
        print("elementwise_f32x4单元精度不对！\n")
        return
    else:
        print("elementwise_f32x4单元测试通过！\n")
        return

def test_elementwise_f16():
    N = 102400
    a = torch.randn(N,dtype=torch.float16,device='cuda').contiguous()
    b = torch.randn(N,dtype=torch.float16,device='cuda').contiguous()
    c_custom = torch.zeros_like(a).contiguous()
    c_pytorch = torch.zeros_like(a)

    warm_up()

    c_pytorch = torch.add(a,b)
    torch.cuda.synchronize()

    llm_ops.elementwise.elementwise_f16(a,b,c_custom,N)
    torch.cuda.synchronize()

    max_diff = torch.max(torch.abs(c_custom-c_pytorch)).item()

    is_correct = torch.allclose(c_pytorch, c_custom, atol=1e-3, rtol=1e-3)

    print(f"elementwise_f16单元最大绝对误差 (Max Diff): {max_diff:.6f}")
    if not is_correct:
        print("elementwise_f16单元精度不对！\n")
        return
    else:
        print("elementwise_f16单元测试通过！\n")
        return

def test_elementwise_f16x2():
    N = 102400
    a = torch.randn(N,dtype=torch.float16,device='cuda').contiguous()
    b = torch.randn(N,dtype=torch.float16,device='cuda').contiguous()
    c_custom = torch.zeros_like(a).contiguous()
    c_pytorch = torch.zeros_like(a)

    warm_up()

    c_pytorch = torch.add(a,b)
    torch.cuda.synchronize()

    llm_ops.elementwise.elementwise_f16x2(a,b,c_custom,N)
    torch.cuda.synchronize()

    max_diff = torch.max(torch.abs(c_custom-c_pytorch)).item()

    is_correct = torch.allclose(c_pytorch, c_custom, atol=1e-3, rtol=1e-3)

    print(f"elementwise_f16x2单元最大绝对误差 (Max Diff): {max_diff:.6f}")
    if not is_correct:
        print("elementwise_f16x2单元精度不对！\n")
        return
    else:
        print("elementwise_f16x2单元测试通过！\n")
        return

def test_elementwise_f16x8_pack():
    N = 102400
    a = torch.randn(N,dtype=torch.float16,device='cuda').contiguous()
    b = torch.randn(N,dtype=torch.float16,device='cuda').contiguous()
    c_custom = torch.zeros_like(a).contiguous()
    c_pytorch = torch.zeros_like(a)

    warm_up()

    c_pytorch = torch.add(a,b)
    torch.cuda.synchronize()

    llm_ops.elementwise.elementwise_f16x8_pack(a,b,c_custom,N)
    torch.cuda.synchronize()

    max_diff = torch.max(torch.abs(c_custom-c_pytorch)).item()

    is_correct = torch.allclose(c_pytorch, c_custom, atol=1e-3, rtol=1e-3)

    print(f"elementwise_f16x8单元最大绝对误差 (Max Diff): {max_diff:.6f}")
    if not is_correct:
        print("elementwise_f16x8单元精度不对！\n")
        return
    else:
        print("elementwise_f16x8单元测试通过！\n")
        return

if __name__=='__main__':
    test_elementwise_f32()
    test_elementwise_f32x4()
    test_elementwise_f16()
    test_elementwise_f16x2()
    test_elementwise_f16x8_pack()