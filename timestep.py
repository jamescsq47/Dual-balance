import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os

def greedy_partition_and_rearrange_ulysses8(sparse: torch.Tensor, num_groups: int = 8, group_size: int = 5):
    """
    将 [B, H, W] 的 sparse（B=40）按贪心法分到 num_groups 组，并重排为
    [B, H, W]，使得前 group_size 个属于组0，接着 group_size 个属于组1，依此类推。
    每组强制恰好 group_size 个元素。
    返回 deperm_idx，可用于恢复原顺序。
    """
    B = sparse.shape[0]
    assert num_groups * group_size == B, "总数必须能被 num_groups * group_size 整除"

    weights = sparse.sum(dim=(1, 2))
    order = torch.argsort(weights, descending=True)
    w_list = weights[order].detach().cpu().tolist()
    idx_list = order.detach().cpu().tolist()

    groups = [[] for _ in range(num_groups)]
    group_sums = [0.0] * num_groups
    group_counts = [0] * num_groups

    for idx, w in zip(idx_list, w_list):
        gid = min(
            (g for g in range(num_groups) if group_counts[g] < group_size),
            key=lambda g: group_sums[g]
        )
        groups[gid].append(idx)
        group_sums[gid] += float(w)
        group_counts[gid] += 1

    new_order = [i for g in groups for i in g]
    perm_idx = torch.tensor(new_order, device=sparse.device, dtype=torch.long)

    # 生成 deperm_idx
    deperm_idx = torch.empty_like(perm_idx)
    deperm_idx[perm_idx] = torch.arange(len(perm_idx), device=perm_idx.device)

    sparse_reordered = sparse.index_select(0, perm_idx)

    return sparse_reordered, groups, group_sums, perm_idx, deperm_idx

def greedy_partition_and_rearrange_ulysses8_multi(sparse: torch.Tensor, old_perm_idx: list, old_deperm_idx:list, num_groups: int = 8, group_size: int = 6):
    """
    输入: sparse [num_blocks, num_heads]
    对每个 block 的 num_heads 按贪心法分组，返回重排后的 sparse、groups、group_sums、perm_idx、deperm_idx
    """
    num_blocks, B = sparse.shape
    assert num_groups * group_size == B, "总数必须能被 num_groups * group_size 整除"

    sparse_reordered = []
    all_groups = []
    all_group_sums = []
    all_perm_idx = []
    all_deperm_idx = []

    for block in range(num_blocks):
        weights = sparse[block]
        order = torch.argsort(weights, descending=True)
        w_list = weights[order].detach().cpu().tolist()
        idx_list = order.detach().cpu().tolist()

        groups = [[] for _ in range(num_groups)]
        group_sums = [0.0] * num_groups
        group_counts = [0] * num_groups

        for idx, w in zip(idx_list, w_list):
            gid = min(
                (g for g in range(num_groups) if group_counts[g] < group_size),
                key=lambda g: group_sums[g]
            )
            groups[gid].append(idx)
            group_sums[gid] += float(w)
            group_counts[gid] += 1

        new_order = [i for g in groups for i in g]
        perm_idx = torch.tensor(new_order, device=sparse.device, dtype=torch.long)
        deperm_idx = torch.empty_like(perm_idx)
        deperm_idx[perm_idx] = torch.arange(len(perm_idx), device=perm_idx.device)

        sparse_reordered.append(weights.index_select(0, perm_idx))
        all_groups.append(groups)
        all_group_sums.append(group_sums)
        if old_perm_idx is not None and old_deperm_idx is not None:
            all_perm_idx.append(old_perm_idx[block].index_select(0, perm_idx))
            all_deperm_idx.append(deperm_idx.index_select(0, old_deperm_idx[block]))
        else:
            all_perm_idx.append(perm_idx)
            all_deperm_idx.append(deperm_idx)

    sparse_reordered = torch.stack(sparse_reordered, dim=0)
    
    return sparse_reordered, all_groups, all_group_sums, all_perm_idx, all_deperm_idx

def zig_partition_and_rearrange_ulysses8_multi(sparse: torch.Tensor, num_groups: int = 8, group_size: int = 5):
    """
    输入: sparse [num_blocks, num_heads]
    对每个 block 的 num_heads 按贪心法分组，返回重排后的 sparse、groups、group_sums、perm_idx、deperm_idx
    """
    num_blocks, B = sparse.shape
    assert num_groups * group_size == B, "总数必须能被 num_groups * group_size 整除"

    sparse_reordered = []
    all_perm_idx = []
    all_deperm_idx = []

    for block in range(num_blocks):
        weights = sparse[block]
        order = torch.argsort(weights, descending=True)
        lut = [0,15,16,31,32,1,14,17,30,33,2,13,18,29,34,3,12,19,28,35,4,11,20,27,36,5,10,21,26,37,6,9,22,25,38,7,8,23,24,39]
        perm_idx = order.index_select(0, torch.tensor(lut, device=order.device, dtype=torch.long))
        deperm_idx = torch.empty_like(perm_idx)
        deperm_idx[perm_idx] = torch.arange(len(perm_idx), device=perm_idx.device)

        sparse_reordered.append(weights.index_select(0, perm_idx))
        all_perm_idx.append(perm_idx)
        all_deperm_idx.append(deperm_idx)

    sparse_reordered = torch.stack(sparse_reordered, dim=0)
    return sparse_reordered, all_perm_idx, all_deperm_idx

def x_ulysses8ring1 (sparse:torch.Tensor):
    sums = sparse.sum(dim=(1,2))
    group_sums = sums.view(8, 5).sum(dim=1)
    x = group_sums.float().max() / group_sums.float().mean()
    return x

# get head-wise reorder index from the first timestep
# data = torch.load("/mnt/public/chensiqi/ParaAttention/first_block_cache_examples/profile/head_density_276.5862121582031.pt", map_location='cpu', weights_only=True)
# data1,_,_,all_perm_idx,_ = greedy_partition_and_rearrange_ulysses8_multi(data)

def imbalance_ratio(sparse:torch.Tensor):
    group_sums = sparse.view(42, 8, 6).sum(dim=-1)
    x = group_sums.float().max(dim=1).values / group_sums.float().mean(dim=1)
    return x.float().mean()

profile_dir = "/mnt/public/chensiqi/xDiT/results/profile"
pt_files = sorted(
    [f for f in os.listdir(profile_dir) if f.endswith(".pt")],
    key=lambda f: os.path.getmtime(os.path.join(profile_dir, f))
)

x_reordered = 1
count = 0
all_perm_idx = None
all_deperm_idx = None
for fname in pt_files:
    fpath = os.path.join(profile_dir, fname)
    data = torch.load(fpath, map_location='cpu')   
    if x_reordered > 1.1:
        _,_,_,all_perm_idx, all_deperm_idx = greedy_partition_and_rearrange_ulysses8_multi(data_reordered, all_perm_idx, all_deperm_idx)
        count += 1

    x = imbalance_ratio(data)
    # _,_,_,new_perm_idx,_ = greedy_partition_and_rearrange_ulysses8_multi(data)
    if(all_perm_idx is not None):
        data_reordered = torch.stack([
            data[block].index_select(0, all_perm_idx[block].to(data.device))
            for block in range(data.shape[0])
        ], dim=0)
    else:
        data_reordered = data

    x_reordered = imbalance_ratio(data_reordered)
    print(f"file:{fname}, {x}, {x_reordered}")
print(f"count: {count}")


test = torch.randn(42,48)
test1=test.index_select(1, all_perm_idx[0].to(test.device))
test2=test1.index_select(1, all_deperm_idx[0].to(test.device))
print(test)
print(test2)
print(torch.allclose(test, test2))
# 以1.1作为分界线
# 1000.0
# 965.5517578125
# 758.862060546875
# 276.5862121582031
# 1.0
