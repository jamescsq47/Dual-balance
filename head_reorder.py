import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
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

def imbalance_ratio_ulysses8ring1 (sparse:torch.Tensor):
    sums = sparse.sum(dim=(1,2))
    group_sums = sums.view(8, 5).sum(dim=1)
    x = group_sums.float().max() / group_sums.float().mean()
    return x

def x_ulysses4ring2 (sparse:torch.Tensor):
    result = torch.zeros(4, 2, 2, dtype=torch.int32)
    head,height,width = sparse.shape

    for i in range(4):  # 第1维四等分
        head_start = i * head//4
        head_end = (i + 1) * head//4
        for j in range(2):  # 第2维二等分
            h_start = j * (height // 2)
            h_end = (j + 1) * (height // 2)
            for k in range(2):  # 第3维二等分
                w_start = k * (width // 2)
                w_end = (k + 1) * (width // 2)
                block = sparse[head_start:head_end, h_start:h_end, w_start:w_end]
                result[i, j, k] = block.sum().item()

    # print(result)
    # print(result.shape)  # [4, 2, 2]

    a = result[:, 0, 0].float()
    b = result[:, 1, 1].float()
    c = result[:, 0, 1].float()
    d = result[:, 1, 0].float()

    mean_a = a.mean().item()
    mean_b = b.mean().item()
    mean_c = c.mean().item()
    mean_d = d.mean().item()
    mean1 = (mean_a + mean_b) / 2
    mean2 = (mean_c + mean_d) / 2

    max1 = torch.maximum(a, b) #[4]
    max2 = torch.maximum(c, d) #[4]
    mmax = (max1+max2).max().item()
    mean = (mean1 + mean2) 
    x = mmax/mean
    return x


data2 = torch.load("/mnt/public/ns-t-te-b905754427352261-427-bk/fs/home/xieruiqi/diffuser-dev520/examples/wan/logs/calib_data/720p/sparse_plan_expanded.pth", map_location='cpu', weights_only=True)
sparse_expanded = data2['sparse']
print(sparse_expanded.shape)

perm_idx_list = []
deperm_idx_list = []
sparse_reordered_list = []
x_list = []
x_reordered_list = []
for block in range(40):
    sparse_expanded_slice = sparse_expanded[0,block, :, :,:]
    sparse_reordered,_,_,perm_idx, deperm_idx = greedy_partition_and_rearrange_ulysses8(sparse_expanded_slice)
    perm_idx_list.append(perm_idx)  # [1, 40]
    deperm_idx_list.append(deperm_idx)  # [1, 40]
    sparse_reordered_list.append(sparse_reordered.unsqueeze(0))
    x = imbalance_ratio_ulysses8ring1(sparse_expanded_slice)
    x_reordered = imbalance_ratio_ulysses8ring1(sparse_reordered)
    x_list.append(x)
    x_reordered_list.append(x_reordered)

    #print(f"block{block}每个 head 的有效元素数量：", sums.tolist())
    print(f"block{block}：", x, x_reordered)

print("平均 x:", sum(x_list)/len(x_list))
print("平均 x_reordered:", sum(x_reordered_list)/len(x_reordered_list))
# perm_idx_all = torch.cat(perm_idx_list, dim=0)  # [40, 40]
# deperm_idx_all = torch.cat(deperm_idx_list, dim=0)  # [40, 40]
sparse_reordered_all = torch.cat(sparse_reordered_list, dim=0)
torch.save(perm_idx_list, "/mnt/public/chensiqi/perm_idx_all.pt")
torch.save(deperm_idx_list, "/mnt/public/chensiqi/deperm_idx_all.pt")
torch.save(sparse_reordered_all, "/mnt/public/chensiqi/sparse_reordered_all.pt")
print(perm_idx_list[0].shape)
print("sparse_reordered_all shape", sparse_reordered_all.shape)

# 白色是1
