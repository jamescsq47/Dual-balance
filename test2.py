import torch 
from spas_sage_attn import customize_spas_sage_attn_meansim_cuda
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
            
        for i in range(num_groups):
            # 获取组内元素对应的权重
            group_weights = [weights[idx].item() for idx in groups[i]]
            # 将组内元素和权重配对，按权重升序排序
            sorted_group = sorted(zip(group_weights, groups[i]), key=lambda x: x[0])
            # 提取排序后的索引
            groups[i] = [idx for _, idx in sorted_group]

        new_order = [i for g in groups for i in g]
        perm_idx = torch.tensor(new_order, device=sparse.device, dtype=torch.long)
        deperm_idx = torch.empty_like(perm_idx)
        deperm_idx[perm_idx] = torch.arange(len(perm_idx), device=perm_idx.device)

        sparse_reordered.append(weights.index_select(0, perm_idx))
        all_groups.append(groups)
        all_group_sums.append(group_sums)
        if old_perm_idx is not None and old_deperm_idx is not None:
            # ensure indices are on the same device as the tensors being indexed
            target = old_perm_idx[block]
            all_perm_idx.append(target.index_select(0, perm_idx.to(target.device, dtype=torch.long)))
            target2 = deperm_idx
            all_deperm_idx.append(target2.index_select(0, old_deperm_idx[block].to(target2.device, dtype=torch.long)))
        else:
            all_perm_idx.append(perm_idx)
            all_deperm_idx.append(deperm_idx)

    sparse_reordered = torch.stack(sparse_reordered, dim=0)
    
    return sparse_reordered, all_groups, all_group_sums, all_perm_idx, all_deperm_idx

def imbalance_ratio(sparse:torch.Tensor, num_groups:int=8):
    group_sums = sparse.view(42, num_groups, -1).sum(dim=-1)
    x = group_sums.float().max(dim=1).values / group_sums.float().mean(dim=1)
    return x.float().mean()

sparse = torch.load("cogvideo_head_density_all.pt")
sparse_new = torch.load("cogvideo_head_density_perm.pt")
sparse_new2 = torch.load("cogvideo_head_density_perm2.pt")
perm_idx_all = torch.load("cogvideo_perm_idx_all.pt")
deperm_idx_all = torch.load("cogvideo_deperm_idx_all.pt")
print(sparse[1][0])
print(sparse_new[1][0])
print(sparse_new2[1][0])

# print(len(sparse))
# all_perm_idx = None
# all_deperm_idx = None
# for i in range(len(sparse_new)):
#     sparse_piece = sparse_new[i]
#     print(f"step {i}, imbalance ratio: {imbalance_ratio(sparse_piece)}")



# all_perm_idx = None
# all_deperm_idx = None
# for i in range(len(sparse_new)):
#     sparse_piece = sparse_new[i]
#     print(f"step {i}, imbalance ratio: {imbalance_ratio(sparse_piece,num_groups=8)}")
#     if i == 0:
#         print(f"step {i}, imbalance ratio: {imbalance_ratio(sparse_piece,num_groups=8)}")
#         sparse_reordered = sparse_piece
#     else:
#         test, all_groups, all_group_sums, all_perm_idx, all_deperm_idx = greedy_partition_and_rearrange_ulysses8_multi(sparse_reordered, all_perm_idx, all_deperm_idx, num_groups=8, group_size=6)
#         # print(f"{imbalance_ratio(test,num_groups=8)}")
#         for block in range(sparse_piece.shape[0]):
#             sparse_reordered[block] = sparse_piece[block].index_select(0, all_perm_idx[block])
#         print(f"step {i}, imbalance ratio: {imbalance_ratio(sparse_reordered,num_groups=8)}")