import torch
import matplotlib.pyplot as plt
import math
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity


def imbalance_ratio(sparse:torch.Tensor,num_groups:int=8):
    # for sparge input shape [blocks, heads]
    group_sums = sparse.view(sparse.shape[0], num_groups, -1).sum(dim=-1)
    x = group_sums.float().max(dim=1).values / group_sums.float().mean(dim=1)
    return x.float().mean()

def greedy_partition_and_rearrange_multi(sparse: torch.Tensor, num_groups: int = 8, old_perm_idx = None, old_deperm_idx = None):
    """
    输入: sparse [num_blocks, num_heads]
    对每个 block 的 num_heads 按贪心法分组，返回重排后的 sparse、groups、group_sums、perm_idx、deperm_idx
    """
    if sparse.dim() == 4:
        sparse = sparse.sum(dim=(-1,-2))
    num_blocks, B = sparse.shape
    group_size = B // num_groups

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

def hybrid_imbalance_ratio(sparse:torch.Tensor,ulysses_degree:int=2, ring_degree:int=4):
    num_devices = ulysses_degree*ring_degree
    # for paro, input shape [head, height, width]
    if sparse.dim() == 2:
        assert ulysses_degree == 1
        sparse = sparse.unsqueeze(0)
        head, height, width = sparse.shape
    elif sparse.dim() == 3:
        head, height, width = sparse.shape
    elif sparse.dim() == 4:
        batch, head, height, width = sparse.shape
    #pad the last two dimension to be divisible by ring_degree
    if height % ring_degree != 0:
        pad_h = ring_degree - (height % ring_degree)
    else:
        pad_h = 0
    if width % ring_degree != 0:
        pad_w = ring_degree - (width % ring_degree)
    else:
        pad_w = 0
    if pad_h != 0 or pad_w != 0:
        if sparse.dim() == 3:
            sparse = torch.nn.functional.pad(sparse, (0, pad_w, 0, pad_h), "constant", 0)
        elif sparse.dim() == 4:
            sparse = torch.nn.functional.pad(sparse, (0, pad_w, 0, pad_h), "constant", 0)
    block_h = height // ring_degree
    block_w = width // ring_degree
    sums = torch.zeros((ulysses_degree, ring_degree, ring_degree), device=sparse.device)
    for k in range(ulysses_degree):    
        for i in range(ring_degree):
            for j in range(ring_degree):
                start_h = i * block_h
                end_h = (i + 1) * block_h
                start_w = j * block_w
                end_w = (j + 1) * block_w
                start_head = k * (head // ulysses_degree)
                end_head = (k + 1) * (head // ulysses_degree)
                if sparse.dim() == 3:
                    sums[k,i,j] += sparse[start_head:end_head, start_h:end_h, start_w:end_w].sum()
                elif sparse.dim() == 4:
                    sums[k,i,j] += sparse[:, start_head:end_head, start_h:end_h, start_w:end_w].sum()
    if ring_degree == 1:
        sums = sums.squeeze(-1).squeeze(-1)
        x = sums.float().max() / sums.float().mean()
        return x.item()
    elif ring_degree == 2:
        iter_1 = torch.maximum(sums[:,0,0], sums[:,1,1])
        iter_2 = torch.maximum(sums[:,0,1], sums[:,1,0])
        max = (iter_1 + iter_2).max()
        mean = sums.sum() / num_devices
        x = max.float() / mean.float()
        return x.item()
    elif ring_degree == 4:
        iter_1 = torch.stack([sums[:,0,0], sums[:,1,1], sums[:,2,2], sums[:,3,3]], dim=0).max(dim=0).values
        iter_2 = torch.stack([sums[:,0,1], sums[:,1,2], sums[:,2,3], sums[:,3,0]], dim=0).max(dim=0).values
        iter_3 = torch.stack([sums[:,0,2], sums[:,1,3], sums[:,2,0], sums[:,3,1]], dim=0).max(dim=0).values
        iter_4 = torch.stack([sums[:,0,3], sums[:,1,0], sums[:,2,1], sums[:,3,2]], dim=0).max(dim=0).values
        max = (iter_1 + iter_2 + iter_3 + iter_4).max()
        mean = sums.sum() / num_devices
        x = max.float() / mean.float()
        return x.item()
    elif ring_degree == 8:
        iter_1 = torch.stack([sums[:,0,0], sums[:,1,1], sums[:,2,2], sums[:,3,3], sums[:,4,4], sums[:,5,5], sums[:,6,6], sums[:,7,7]], dim=0).max(dim=0).values
        iter_2 = torch.stack([sums[:,0,1], sums[:,1,2], sums[:,2,3], sums[:,3,4], sums[:,4,5], sums[:,5,6], sums[:,6,7], sums[:,7,0]], dim=0).max(dim=0).values
        iter_3 = torch.stack([sums[:,0,2], sums[:,1,3], sums[:,2,4], sums[:,3,5], sums[:,4,6], sums[:,5,7], sums[:,6,0], sums[:,7,1]], dim=0).max(dim=0).values
        iter_4 = torch.stack([sums[:,0,3], sums[:,1,4], sums[:,2,5], sums[:,3,6], sums[:,4,7], sums[:,5,0], sums[:,6,1], sums[:,7,2]], dim=0).max(dim=0).values
        iter_5 = torch.stack([sums[:,0,4], sums[:,1,5], sums[:,2,6], sums[:,3,7], sums[:,4,0], sums[:,5,1], sums[:,6,2], sums[:,7,3]], dim=0).max(dim=0).values
        iter_6 = torch.stack([sums[:,0,5], sums[:,1,6], sums[:,2,7], sums[:,3,0], sums[:,4,1], sums[:,5,2], sums[:,6,3], sums[:,7,4]], dim=0).max(dim=0).values
        iter_7 = torch.stack([sums[:,0,6], sums[:,1,7], sums[:,2,0], sums[:,3,1], sums[:,4,2], sums[:,5,3], sums[:,6,4], sums[:,7,5]], dim=0).max(dim=0).values
        iter_8 = torch.stack([sums[:,0,7], sums[:,1,0], sums[:,2,1], sums[:,3,2], sums[:,4,3], sums[:,5,4], sums[:,6,5], sums[:,7,6]], dim=0).max(dim=0).values
        max = (iter_1 + iter_2 + iter_3 + iter_4 + iter_5 + iter_6 + iter_7 + iter_8).max()
        mean = sums.sum() / num_devices
        x = max.float() / mean.float()
        return x.item()

def hybrid_permute(
    sparse: torch.Tensor,
    ulysses_degree: int = 4,
    ring_degree: int = 2
):
    num_heads, H, W = sparse.shape
    # 1. head 维度贪心分组重排
    if ulysses_degree == 1:
        head_perm_idx = torch.arange(num_heads, device=sparse.device)
        head_deperm_idx = torch.arange(num_heads, device=sparse.device)
        sparse_reordered = sparse
    else:
        head_group_size = num_heads // ulysses_degree
        head_weights = sparse.sum(dim=(1,2))
        head_order = torch.argsort(head_weights, descending=True)
        head_w_list = head_weights[head_order].detach().cpu().tolist()
        head_idx_list = head_order.detach().cpu().tolist()
        head_groups = [[] for _ in range(ulysses_degree)]
        head_group_sums = [0.0] * ulysses_degree
        head_group_counts = [0] * ulysses_degree
        for idx, w in zip(head_idx_list, head_w_list):
            gid = min(
                (g for g in range(ulysses_degree) if head_group_counts[g] < head_group_size),
                key=lambda g: head_group_sums[g]
            )
            head_groups[gid].append(idx)
            head_group_sums[gid] += float(w)
            head_group_counts[gid] += 1
        head_new_order = [i for g in head_groups for i in g]
        head_perm_idx = torch.tensor(head_new_order, device=sparse.device, dtype=torch.long)
        head_deperm_idx = torch.empty_like(head_perm_idx)
        head_deperm_idx[head_perm_idx] = torch.arange(len(head_perm_idx), device=head_perm_idx.device)
        sparse_reordered = sparse.index_select(0, head_perm_idx)

    # 2. 对每个head做H/W贪心分组重排
    if ring_degree == 1:
        row_perm_idx_list = [torch.arange(H, device=sparse.device) for _ in range(num_heads)]
        col_perm_idx_list = [torch.arange(W, device=sparse.device) for _ in range(num_heads)]
        row_deperm_idx_list = [torch.arange(H, device=sparse.device) for _ in range(num_heads)]
        col_deperm_idx_list = [torch.arange(W, device=sparse.device) for _ in range(num_heads)]
        sparse_final = sparse_reordered
    else:
        assert H % ring_degree == 0 and W % ring_degree == 0, "H和W必须能被ring_degree整除"
        row_perm_idx_list = []
        col_perm_idx_list = []
        row_deperm_idx_list = []
        col_deperm_idx_list = []
        permuted_list = []
        for h in range(num_heads):
            mat = sparse_reordered[h]
            group_size_h = H // ring_degree
            row_sum = mat.sum(dim=1)
            row_order = torch.argsort(row_sum, descending=True)
            row_w_list = row_sum[row_order].detach().cpu().tolist()
            row_idx_list = row_order.detach().cpu().tolist()
            row_groups = [[] for _ in range(ring_degree)]
            row_group_sums = [0.0] * ring_degree
            row_group_counts = [0] * ring_degree
            for idx, w in zip(row_idx_list, row_w_list):
                gid = min(
                    (g for g in range(ring_degree) if row_group_counts[g] < group_size_h),
                    key=lambda g: row_group_sums[g]
                )
                row_groups[gid].append(idx)
                row_group_sums[gid] += float(w)
                row_group_counts[gid] += 1
            row_new_order = [i for g in row_groups for i in g]
            row_perm_idx = torch.tensor(row_new_order, device=sparse.device, dtype=torch.long)
            row_deperm_idx = torch.empty_like(row_perm_idx)
            row_deperm_idx[row_perm_idx] = torch.arange(len(row_perm_idx), device=row_perm_idx.device)
            group_size_w = W // ring_degree
            col_sum = mat.sum(dim=0)
            col_order = torch.argsort(col_sum, descending=True)
            col_w_list = col_sum[col_order].detach().cpu().tolist()
            col_idx_list = col_order.detach().cpu().tolist()
            col_groups = [[] for _ in range(ring_degree)]
            col_group_sums = [0.0] * ring_degree
            col_group_counts = [0] * ring_degree
            for idx, w in zip(col_idx_list, col_w_list):
                gid = min(
                    (g for g in range(ring_degree) if col_group_counts[g] < group_size_w),
                    key=lambda g: col_group_sums[g]
                )
                col_groups[gid].append(idx)
                col_group_sums[gid] += float(w)
                col_group_counts[gid] += 1
            col_new_order = [i for g in col_groups for i in g]
            col_perm_idx = torch.tensor(col_new_order, device=sparse.device, dtype=torch.long)
            col_deperm_idx = torch.empty_like(col_perm_idx)
            col_deperm_idx[col_perm_idx] = torch.arange(len(col_perm_idx), device=col_perm_idx.device)
            mat_permuted = mat.index_select(0, row_perm_idx).index_select(1, col_perm_idx)
            permuted_list.append(mat_permuted)
            row_perm_idx_list.append(row_perm_idx)
            col_perm_idx_list.append(col_perm_idx)
            row_deperm_idx_list.append(row_deperm_idx)
            col_deperm_idx_list.append(col_deperm_idx)
        sparse_final = torch.stack(permuted_list, dim=0)
    return sparse_final, head_perm_idx, row_perm_idx_list, col_perm_idx_list, head_deperm_idx, row_deperm_idx_list, col_deperm_idx_list 

def hybrid_permute_v2(
    sparse: torch.Tensor,
    ulysses_degree: int = 2,
    ring_degree: int = 4
):
    """
    先对head做贪心分组重排，然后将每组ulysses的head累加为一个head（降维），
    再对每个组累加后的mask做H/W贪心分组重排。
    返回 [ulysses_degree, H, W]，以及各维度的perm/deperm idx。
    """
    num_heads, H, W = sparse.shape
    # 1. head 维度贪心分组重排
    if ulysses_degree == 1:
        head_perm_idx = torch.arange(num_heads, device=sparse.device)
        head_deperm_idx = torch.arange(num_heads, device=sparse.device)
        sparse_reordered = sparse
        head_group_size = num_heads
    else:
        head_group_size = num_heads // ulysses_degree
        head_weights = sparse.sum(dim=(1,2))
        head_order = torch.argsort(head_weights, descending=True)
        head_w_list = head_weights[head_order].detach().cpu().tolist()
        head_idx_list = head_order.detach().cpu().tolist()
        head_groups = [[] for _ in range(ulysses_degree)]
        head_group_sums = [0.0] * ulysses_degree
        head_group_counts = [0] * ulysses_degree
        for idx, w in zip(head_idx_list, head_w_list):
            gid = min(
                (g for g in range(ulysses_degree) if head_group_counts[g] < head_group_size),
                key=lambda g: head_group_sums[g]
            )
            head_groups[gid].append(idx)
            head_group_sums[gid] += float(w)
            head_group_counts[gid] += 1
        head_new_order = [i for g in head_groups for i in g]
        head_perm_idx = torch.tensor(head_new_order, device=sparse.device, dtype=torch.long)
        head_deperm_idx = torch.empty_like(head_perm_idx)
        head_deperm_idx[head_perm_idx] = torch.arange(len(head_perm_idx), device=head_perm_idx.device)
        sparse_reordered = sparse.index_select(0, head_perm_idx)

    # 2. 将每组ulysses的head累加为一个head

    mat = sparse_reordered.sum(dim=0) # [ulysses_degree, H, W]

    # 3. 对每个组累加后的mask做H/W贪心分组重排
    if ring_degree == 1:
        row_perm_idx = torch.arange(H, device=sparse.device) 
        col_perm_idx = torch.arange(W, device=sparse.device)
        row_deperm_idx = torch.arange(H, device=sparse.device)
        col_deperm_idx = torch.arange(W, device=sparse.device)
        sparse_final = sparse_reordered
    else:
        assert H % ring_degree == 0 and W % ring_degree == 0, "H和W必须能被ring_degree整除"

        group_size_h = H // ring_degree
        row_sum = mat.sum(dim=1)
        row_order = torch.argsort(row_sum, descending=True)
        row_w_list = row_sum[row_order].detach().cpu().tolist()
        row_idx_list = row_order.detach().cpu().tolist()
        row_groups = [[] for _ in range(ring_degree)]
        row_group_sums = [0.0] * ring_degree
        row_group_counts = [0] * ring_degree
        for idx, w in zip(row_idx_list, row_w_list):
            gid = min(
                (g for g in range(ring_degree) if row_group_counts[g] < group_size_h),
                key=lambda g: row_group_sums[g]
            )
            row_groups[gid].append(idx)
            row_group_sums[gid] += float(w)
            row_group_counts[gid] += 1
        row_new_order = [i for g in row_groups for i in g]
        row_perm_idx = torch.tensor(row_new_order, device=sparse.device, dtype=torch.long)
        row_deperm_idx = torch.empty_like(row_perm_idx)
        row_deperm_idx[row_perm_idx] = torch.arange(len(row_perm_idx), device=row_perm_idx.device)
        group_size_w = W // ring_degree
        col_sum = mat.sum(dim=0)
        col_order = torch.argsort(col_sum, descending=True)
        col_w_list = col_sum[col_order].detach().cpu().tolist()
        col_idx_list = col_order.detach().cpu().tolist()
        col_groups = [[] for _ in range(ring_degree)]
        col_group_sums = [0.0] * ring_degree
        col_group_counts = [0] * ring_degree
        for idx, w in zip(col_idx_list, col_w_list):
            gid = min(
                (g for g in range(ring_degree) if col_group_counts[g] < group_size_w),
                key=lambda g: col_group_sums[g]
            )
            col_groups[gid].append(idx)
            col_group_sums[gid] += float(w)
            col_group_counts[gid] += 1
        col_new_order = [i for g in col_groups for i in g]
        col_perm_idx = torch.tensor(col_new_order, device=sparse.device, dtype=torch.long)
        col_deperm_idx = torch.empty_like(col_perm_idx)
        col_deperm_idx[col_perm_idx] = torch.arange(len(col_perm_idx), device=col_perm_idx.device)
        sparse_final = sparse_reordered.index_select(1, row_perm_idx).index_select(2, col_perm_idx)

    return sparse_final, head_perm_idx, row_perm_idx, col_perm_idx, head_deperm_idx, row_deperm_idx, col_deperm_idx

def hybrid_permute_v3(
    sparse: torch.Tensor,
    ulysses_degree: int = 2,
    ring_degree: int = 4
):
    if sparse.dim() == 3:
        num_heads, H, W = sparse.shape
        # 1. head 维度贪心分组重排
        if ulysses_degree == 1:
            head_perm_idx = torch.arange(num_heads, device=sparse.device)
            head_deperm_idx = torch.arange(num_heads, device=sparse.device)
            sparse_reordered = sparse
            head_group_size = num_heads
        else:
            head_group_size = num_heads // ulysses_degree
            head_weights = sparse.sum(dim=(1,2))
            head_order = torch.argsort(head_weights, descending=True)
            head_w_list = head_weights[head_order].detach().cpu().tolist()
            head_idx_list = head_order.detach().cpu().tolist()
            head_groups = [[] for _ in range(ulysses_degree)]
            head_group_sums = [0.0] * ulysses_degree
            head_group_counts = [0] * ulysses_degree
            for idx, w in zip(head_idx_list, head_w_list):
                gid = min(
                    (g for g in range(ulysses_degree) if head_group_counts[g] < head_group_size),
                    key=lambda g: head_group_sums[g]
                )
                head_groups[gid].append(idx)
                head_group_sums[gid] += float(w)
                head_group_counts[gid] += 1
            head_new_order = [i for g in head_groups for i in g]
            head_perm_idx = torch.tensor(head_new_order, device=sparse.device, dtype=torch.long)
            head_deperm_idx = torch.empty_like(head_perm_idx)
            head_deperm_idx[head_perm_idx] = torch.arange(len(head_perm_idx), device=head_perm_idx.device)
            sparse_reordered = sparse.index_select(0, head_perm_idx)

        # 2. 将每组ulysses的head累加为一个head

        mat = sparse_reordered.sum(dim=0) # [H, W]

        # 3. 对每个组累加后的mask做H/W贪心分组重排
        if ring_degree == 1:
            new_row_perm_idx = None
            new_col_perm_idx = None
            transpose_matrix_q = None
            transpose_matrix_k = None
            new_row_deperm_idx = None
            new_col_deperm_idx = None
            sparse_final = sparse_reordered
        else:
            assert H % ring_degree == 0 and W % ring_degree == 0, "H和W必须能被ring_degree整除"

            group_size_h = H // ring_degree
            row_sum = mat.sum(dim=1)
            row_order = torch.argsort(row_sum, descending=True)
            row_w_list = row_sum[row_order].detach().cpu().tolist()
            row_idx_list = row_order.detach().cpu().tolist()
            row_groups = [[] for _ in range(ring_degree)]
            row_group_sums = [0.0] * ring_degree
            row_group_counts = [0] * ring_degree
            for idx, w in zip(row_idx_list, row_w_list):
                gid = min(
                    (g for g in range(ring_degree) if row_group_counts[g] < group_size_h),
                    key=lambda g: row_group_sums[g]
                )
                row_groups[gid].append(idx)
                row_group_sums[gid] += float(w)
                row_group_counts[gid] += 1
            row_new_order = [i for g in row_groups for i in g]
            row_perm_idx = torch.tensor(row_new_order, device=sparse.device, dtype=torch.long)
            group_size_w = W // ring_degree
            col_sum = mat.sum(dim=0)
            col_order = torch.argsort(col_sum, descending=True)
            col_w_list = col_sum[col_order].detach().cpu().tolist()
            col_idx_list = col_order.detach().cpu().tolist()
            col_groups = [[] for _ in range(ring_degree)]
            col_group_sums = [0.0] * ring_degree
            col_group_counts = [0] * ring_degree
            for idx, w in zip(col_idx_list, col_w_list):
                gid = min(
                    (g for g in range(ring_degree) if col_group_counts[g] < group_size_w),
                    key=lambda g: col_group_sums[g]
                )
                col_groups[gid].append(idx)
                col_group_sums[gid] += float(w)
                col_group_counts[gid] += 1
            col_new_order = [i for g in col_groups for i in g]
            col_perm_idx = torch.tensor(col_new_order, device=sparse.device, dtype=torch.long)
            # sparse_final = sparse_reordered.index_select(1, row_perm_idx).index_select(2, col_perm_idx)

            num_groups = ring_degree
            group_size = row_perm_idx.shape[0] // num_groups
            row_perm_idx_groups_sorted = torch.sort(row_perm_idx.view(num_groups, group_size), dim=1)[0]
            col_perm_idx_groups_sorted = torch.sort(col_perm_idx.view(num_groups, group_size), dim=1)[0]

            transpose_matrix_q = torch.stack([
                torch.stack([
                    ((g >= j * group_size) & (g < (j + 1) * group_size)).sum()
                    for j in range(num_groups)
                ])
                for g in row_perm_idx_groups_sorted
            ]).T.contiguous()

            transpose_matrix_k = torch.stack([
                torch.stack([
                    ((g >= j * group_size) & (g < (j + 1) * group_size)).sum()
                    for j in range(num_groups)
                ])
                for g in col_perm_idx_groups_sorted
            ]).T.contiguous()

            new_row_perm_idx = torch.cat([
                row_perm_idx_groups_sorted[(row_perm_idx_groups_sorted >= i * group_size) & (row_perm_idx_groups_sorted < (i + 1) * group_size)]
                for i in range(num_groups)
            ]).reshape(ring_degree, -1)
            # sparse_final = sparse_reordered.index_select(1, new_row_perm_idx.view(-1))
            new_row_perm_idx = new_row_perm_idx - new_row_perm_idx.min(dim=1, keepdim=True)[0]

            new_col_perm_idx = torch.cat([
                col_perm_idx_groups_sorted[(col_perm_idx_groups_sorted >= i * group_size) & (col_perm_idx_groups_sorted < (i + 1) * group_size)]
                for i in range(num_groups)
            ]).reshape(ring_degree, -1)
            # sparse_final = sparse_reordered.index_select(2, new_col_perm_idx.view(-1))
            new_col_perm_idx = new_col_perm_idx - new_col_perm_idx.min(dim=1, keepdim=True)[0]

            new_row_deperm_idx = torch.empty_like(new_row_perm_idx)
            for i in range(new_row_perm_idx.shape[0]):
                new_row_deperm_idx[i][new_row_perm_idx[i]] = torch.arange(new_row_perm_idx.shape[1], device=new_row_perm_idx.device)

            new_col_deperm_idx = torch.empty_like(new_col_perm_idx)
            for i in range(new_col_perm_idx.shape[0]):
                new_col_deperm_idx[i][new_col_perm_idx[i]] = torch.arange(new_col_perm_idx.shape[1], device=new_col_perm_idx.device)

            sparse_final = sparse_reordered.index_select(1, row_perm_idx_groups_sorted.view(-1)).index_select(2, col_perm_idx_groups_sorted.view(-1)) 
            #TODO: fix
        
        return sparse_final, head_perm_idx, new_row_perm_idx, new_col_perm_idx, transpose_matrix_q, transpose_matrix_k, head_deperm_idx, new_row_deperm_idx, new_col_deperm_idx

    elif sparse.dim() == 4:
        sparse_final_list = []
        head_perm_idx_list = []
        new_row_perm_idx_list = []
        new_col_perm_idx_list = []
        transpose_matrix_q_list = []
        transpose_matrix_k_list = []
        head_deperm_idx_list = []
        new_row_deperm_idx_list = []
        new_col_deperm_idx_list = []

        for block in range(sparse.shape[0]):
            sparse_final, head_perm_idx, new_row_perm_idx, new_col_perm_idx, transpose_matrix_q, transpose_matrix_k, head_deperm_idx, new_row_deperm_idx, new_col_deperm_idx = hybrid_permute_v3(sparse[block], ulysses_degree, ring_degree)
            sparse_final_list.append(sparse_final)
            head_perm_idx_list.append(head_perm_idx)
            new_row_perm_idx_list.append(new_row_perm_idx)
            new_col_perm_idx_list.append(new_col_perm_idx)
            transpose_matrix_q_list.append(transpose_matrix_q)
            transpose_matrix_k_list.append(transpose_matrix_k)
            head_deperm_idx_list.append(head_deperm_idx)
            new_row_deperm_idx_list.append(new_row_deperm_idx)
            new_col_deperm_idx_list.append(new_col_deperm_idx)

        sparse_final = torch.stack(sparse_final_list, dim=0)
        # head_perm_idx = torch.stack(head_perm_idx_list, dim=0)
        # new_row_perm_idx = torch.stack(new_row_perm_idx_list, dim=0)
        # new_col_perm_idx = torch.stack(new_col_perm_idx_list, dim=0)
        # transpose_matrix_q = torch.stack(transpose_matrix_q_list, dim=0)
        # transpose_matrix_k = torch.stack(transpose_matrix_k_list, dim=0)
        # head_deperm_idx = torch.stack(head_deperm_idx_list, dim=0)
        # new_row_deperm_idx = torch.stack(new_row_deperm_idx_list, dim=0)
        # new_col_deperm_idx = torch.stack(new_col_deperm_idx_list, dim=0)

    return sparse_final, head_perm_idx_list, new_row_perm_idx_list, new_col_perm_idx_list, transpose_matrix_q_list, transpose_matrix_k_list, head_deperm_idx_list, new_row_deperm_idx_list, new_col_deperm_idx_list

def hybrid_permute_v4(
    sparse: torch.Tensor,
    ulysses_degree: int = 2,
    ring_degree: int = 4,
    reward: float = 10,
):
    if sparse.dim() == 3:
        num_heads, H, W = sparse.shape
        # 1. head 维度贪心分组重排
        if ulysses_degree == 1:
            head_perm_idx = torch.arange(num_heads, device=sparse.device)
            head_deperm_idx = torch.arange(num_heads, device=sparse.device)
            sparse_reordered = sparse
            head_group_size = num_heads
        else:
            head_group_size = num_heads // ulysses_degree
            head_weights = sparse.sum(dim=(1,2))
            head_order = torch.argsort(head_weights, descending=True)
            head_w_list = head_weights[head_order].detach().cpu().tolist()
            head_idx_list = head_order.detach().cpu().tolist()
            head_groups = [[] for _ in range(ulysses_degree)]
            head_group_sums = [0.0] * ulysses_degree
            head_group_counts = [0] * ulysses_degree
            for idx, w in zip(head_idx_list, head_w_list):
                gid = min(
                    (g for g in range(ulysses_degree) if head_group_counts[g] < head_group_size),
                    key=lambda g: head_group_sums[g]
                )
                head_groups[gid].append(idx)
                head_group_sums[gid] += float(w)
                head_group_counts[gid] += 1
            head_new_order = [i for g in head_groups for i in g]
            head_perm_idx = torch.tensor(head_new_order, device=sparse.device, dtype=torch.long)
            head_deperm_idx = torch.empty_like(head_perm_idx)
            head_deperm_idx[head_perm_idx] = torch.arange(len(head_perm_idx), device=head_perm_idx.device)
            sparse_reordered = sparse.index_select(0, head_perm_idx)

        # 2. 将每组ulysses的head累加为一个head

        mat = sparse_reordered.sum(dim=0) # [H, W]

        # 3. 对每个组累加后的mask做H/W贪心分组重排
        if ring_degree == 1:
            new_row_perm_idx = None
            new_col_perm_idx = None
            transpose_matrix_q = None
            transpose_matrix_k = None
            new_row_deperm_idx = None
            new_col_deperm_idx = None
            sparse_final = sparse_reordered
        else:
            assert H % ring_degree == 0 and W % ring_degree == 0, "H和W必须能被ring_degree整除"

            group_size_h = H // ring_degree
            row_sum = mat.sum(dim=1)
            row_order = torch.argsort(row_sum, descending=True)
            row_w_list = row_sum[row_order].detach().cpu().tolist()
            row_idx_list = row_order.detach().cpu().tolist()
            row_groups = [[] for _ in range(ring_degree)]
            row_group_sums = [0.0] * ring_degree
            row_group_counts = [0] * ring_degree

            for idx, w in zip(row_idx_list, row_w_list):
                # 计算该行原本属于哪个块
                original_block = idx // group_size_h

                # 优先考虑原本的块，如果该块还有空间且负载相对均衡
                candidate_groups = []
                for g in range(ring_degree):
                    if row_group_counts[g] < group_size_h:
                        # 如果是原本的块，给予优先级（负载稍高也可以接受）
                        if g == original_block:
                            candidate_groups.append((g, row_group_sums[g] - reward * w))  # 降低原本块的负载计算
                        else:
                            candidate_groups.append((g, row_group_sums[g]))

                if candidate_groups:
                    gid = min(candidate_groups, key=lambda x: x[1])[0]
                    row_groups[gid].append(idx)
                    row_group_sums[gid] += float(w)
                    row_group_counts[gid] += 1

            row_new_order = [i for g in row_groups for i in g]
            row_perm_idx = torch.tensor(row_new_order, device=sparse.device, dtype=torch.long)
            
            group_size_w = W // ring_degree
            col_sum = mat.sum(dim=0)
            col_order = torch.argsort(col_sum, descending=True)
            col_w_list = col_sum[col_order].detach().cpu().tolist()
            col_idx_list = col_order.detach().cpu().tolist()
            col_groups = [[] for _ in range(ring_degree)]
            col_group_sums = [0.0] * ring_degree
            col_group_counts = [0] * ring_degree

            for idx, w in zip(col_idx_list, col_w_list):
                original_block = idx // group_size_w
                
                candidate_groups = []
                for g in range(ring_degree):
                    if col_group_counts[g] < group_size_w:
                        if g == original_block:
                            candidate_groups.append((g, col_group_sums[g] -  reward * w))  # 降低原本块的负载计算
                        else:
                            candidate_groups.append((g, col_group_sums[g]))
                
                if candidate_groups:
                    gid = min(candidate_groups, key=lambda x: x[1])[0]
                    col_groups[gid].append(idx)
                    col_group_sums[gid] += float(w)
                    col_group_counts[gid] += 1

            col_new_order = [i for g in col_groups for i in g]
            col_perm_idx = torch.tensor(col_new_order, device=sparse.device, dtype=torch.long)
            # sparse_final = sparse_reordered.index_select(1, row_perm_idx).index_select(2, col_perm_idx)

            num_groups = ring_degree
            group_size = row_perm_idx.shape[0] // num_groups
            row_perm_idx_groups_sorted = torch.sort(row_perm_idx.view(num_groups, group_size), dim=1)[0]
            col_perm_idx_groups_sorted = torch.sort(col_perm_idx.view(num_groups, group_size), dim=1)[0]

            transpose_matrix_q = torch.stack([
                torch.stack([
                    ((g >= j * group_size) & (g < (j + 1) * group_size)).sum()
                    for j in range(num_groups)
                ])
                for g in row_perm_idx_groups_sorted
            ]).T.contiguous()

            transpose_matrix_k = torch.stack([
                torch.stack([
                    ((g >= j * group_size) & (g < (j + 1) * group_size)).sum()
                    for j in range(num_groups)
                ])
                for g in col_perm_idx_groups_sorted
            ]).T.contiguous()

            new_row_perm_idx = torch.cat([
                row_perm_idx_groups_sorted[(row_perm_idx_groups_sorted >= i * group_size) & (row_perm_idx_groups_sorted < (i + 1) * group_size)]
                for i in range(num_groups)
            ]).reshape(ring_degree, -1)
            # sparse_final = sparse_reordered.index_select(1, new_row_perm_idx.view(-1))
            new_row_perm_idx = new_row_perm_idx - new_row_perm_idx.min(dim=1, keepdim=True)[0]

            new_col_perm_idx = torch.cat([
                col_perm_idx_groups_sorted[(col_perm_idx_groups_sorted >= i * group_size) & (col_perm_idx_groups_sorted < (i + 1) * group_size)]
                for i in range(num_groups)
            ]).reshape(ring_degree, -1)
            # sparse_final = sparse_reordered.index_select(2, new_col_perm_idx.view(-1))
            new_col_perm_idx = new_col_perm_idx - new_col_perm_idx.min(dim=1, keepdim=True)[0]

            new_row_deperm_idx = torch.empty_like(new_row_perm_idx)
            for i in range(new_row_perm_idx.shape[0]):
                new_row_deperm_idx[i][new_row_perm_idx[i]] = torch.arange(new_row_perm_idx.shape[1], device=new_row_perm_idx.device)

            new_col_deperm_idx = torch.empty_like(new_col_perm_idx)
            for i in range(new_col_perm_idx.shape[0]):
                new_col_deperm_idx[i][new_col_perm_idx[i]] = torch.arange(new_col_perm_idx.shape[1], device=new_col_perm_idx.device)

            sparse_final = sparse_reordered.index_select(1, row_perm_idx_groups_sorted.view(-1)).index_select(2, col_perm_idx_groups_sorted.view(-1)) 
            # sparse_final = sparse_reordered
        
        return sparse_final, head_perm_idx, new_row_perm_idx, new_col_perm_idx, transpose_matrix_q, transpose_matrix_k, head_deperm_idx, new_row_deperm_idx, new_col_deperm_idx

    elif sparse.dim() == 4:
        sparse_final_list = []
        head_perm_idx_list = []
        new_row_perm_idx_list = []
        new_col_perm_idx_list = []
        transpose_matrix_q_list = []
        transpose_matrix_k_list = []
        head_deperm_idx_list = []
        new_row_deperm_idx_list = []
        new_col_deperm_idx_list = []

        for block in range(sparse.shape[0]):
            sparse_final, head_perm_idx, new_row_perm_idx, new_col_perm_idx, transpose_matrix_q, transpose_matrix_k, head_deperm_idx, new_row_deperm_idx, new_col_deperm_idx = hybrid_permute_v3(sparse[block], ulysses_degree, ring_degree)
            sparse_final_list.append(sparse_final)
            head_perm_idx_list.append(head_perm_idx)
            new_row_perm_idx_list.append(new_row_perm_idx)
            new_col_perm_idx_list.append(new_col_perm_idx)
            transpose_matrix_q_list.append(transpose_matrix_q)
            transpose_matrix_k_list.append(transpose_matrix_k)
            head_deperm_idx_list.append(head_deperm_idx)
            new_row_deperm_idx_list.append(new_row_deperm_idx)
            new_col_deperm_idx_list.append(new_col_deperm_idx)

        sparse_final = torch.stack(sparse_final_list, dim=0)
        # head_perm_idx = torch.stack(head_perm_idx_list, dim=0)
        # new_row_perm_idx = torch.stack(new_row_perm_idx_list, dim=0)
        # new_col_perm_idx = torch.stack(new_col_perm_idx_list, dim=0)
        # transpose_matrix_q = torch.stack(transpose_matrix_q_list, dim=0)
        # transpose_matrix_k = torch.stack(transpose_matrix_k_list, dim=0)
        # head_deperm_idx = torch.stack(head_deperm_idx_list, dim=0)
        # new_row_deperm_idx = torch.stack(new_row_deperm_idx_list, dim=0)
        # new_col_deperm_idx = torch.stack(new_col_deperm_idx_list, dim=0)

    return sparse_final, head_perm_idx_list, new_row_perm_idx_list, new_col_perm_idx_list, transpose_matrix_q_list, transpose_matrix_k_list, head_deperm_idx_list, new_row_deperm_idx_list, new_col_deperm_idx_list

def hybrid_permute_v5(
    sparse: torch.Tensor,
    ulysses_degree: int = 2,
    ring_degree: int = 4,
    max_iters: int = 100,
    k: int = 5
):
    if sparse.dim() == 3:
        num_heads, H, W = sparse.shape
        # 1. head 维度贪心分组重排（与 v3/v4 一致）
        if ulysses_degree == 1:
            head_perm_idx = torch.arange(num_heads, device=sparse.device)
            head_deperm_idx = torch.arange(num_heads, device=sparse.device)
            sparse_reordered = sparse
            head_group_size = num_heads
        else:
            head_group_size = num_heads // ulysses_degree
            head_weights = sparse.sum(dim=(1, 2))
            head_order = torch.argsort(head_weights, descending=True)
            head_w_list = head_weights[head_order].detach().cpu().tolist()
            head_idx_list = head_order.detach().cpu().tolist()
            head_groups = [[] for _ in range(ulysses_degree)]
            head_group_sums = [0.0] * ulysses_degree
            head_group_counts = [0] * ulysses_degree
            for idx, w in zip(head_idx_list, head_w_list):
                gid = min(
                    (g for g in range(ulysses_degree) if head_group_counts[g] < head_group_size),
                    key=lambda g: head_group_sums[g]
                )
                head_groups[gid].append(idx)
                head_group_sums[gid] += float(w)
                head_group_counts[gid] += 1
            head_new_order = [i for g in head_groups for i in g]
            head_perm_idx = torch.tensor(head_new_order, device=sparse.device, dtype=torch.long)
            head_deperm_idx = torch.empty_like(head_perm_idx)
            head_deperm_idx[head_perm_idx] = torch.arange(len(head_perm_idx), device=head_perm_idx.device)
            sparse_reordered = sparse.index_select(0, head_perm_idx)

        # 2. 将每组 ulysses 的 head 累加为一个 head
        mat = sparse_reordered.sum(dim=0)  # [H, W]

        # 3. 对每个组累加后的 mask 做 H/W 贪心分组重排；对行做局部交换优化
        if ring_degree == 1:
            new_row_perm_idx = None
            new_col_perm_idx = None
            transpose_matrix_q = None
            transpose_matrix_k = None
            new_row_deperm_idx = None
            new_col_deperm_idx = None
            sparse_final = sparse_reordered
        else:
            assert H % ring_degree == 0 and W % ring_degree == 0, "H和W必须能被ring_degree整除"

            group_size_h = H // ring_degree
            row_sum = mat.sum(dim=1)  # [H]
            
            # 初始化：连续分组并计算各组和
            row_groups = [list(range(i * group_size_h, (i + 1) * group_size_h)) for i in range(ring_degree)]
            row_group_sums = [row_sum[group].sum().item() for group in row_groups]
            
            # 创建初始排列索引
            current_perm = list(range(H))
            
            # 平衡行分组：交换最大组中的最大值和最小组中的最小值
            # 改进的平衡算法
            for iter in range(max_iters):
                # 找到和最大和最小的组
                max_group_idx = torch.tensor(row_group_sums).argmax().item()
                min_group_idx = torch.tensor(row_group_sums).argmin().item()
                
                if max_group_idx == min_group_idx:
                    break  # 已经平衡
                
                # 找到最大组中行和最大的k个元素
                max_group_rows = row_groups[max_group_idx]
                max_rows_in_max = sorted(max_group_rows, key=lambda idx: row_sum[idx].item(), reverse=True)[:k]
                
                # 找到最小组中行和最小的k个元素
                min_group_rows = row_groups[min_group_idx]
                min_rows_in_min = sorted(min_group_rows, key=lambda idx: row_sum[idx].item())[:k]
                
                # 尝试所有可能的交换组合，选择最优的
                best_swap = None
                best_improvement = 0
                
                for max_row in max_rows_in_max:
                    for min_row in min_rows_in_min:
                        # 计算交换后的组和变化
                        current_max_sum = row_group_sums[max_group_idx]
                        current_min_sum = row_group_sums[min_group_idx]
                        new_max_sum = current_max_sum - row_sum[max_row].item() + row_sum[min_row].item()
                        new_min_sum = current_min_sum - row_sum[min_row].item() + row_sum[max_row].item()
                        
                        # 计算不平衡度改善
                        current_imbalance = abs(current_max_sum - current_min_sum)
                        new_imbalance = abs(new_max_sum - new_min_sum)
                        improvement = current_imbalance - new_imbalance
                        
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_swap = (max_row, min_row)
                
                # 如果找到改善的交换，执行它
                if best_swap and best_improvement > 0:
                    max_row, min_row = best_swap
                    
                    # 执行交换：更新分组
                    row_groups[max_group_idx].remove(max_row)
                    row_groups[max_group_idx].append(min_row)
                    row_groups[min_group_idx].remove(min_row)
                    row_groups[min_group_idx].append(max_row)
                    
                    # 更新组和
                    row_group_sums[max_group_idx] -= row_sum[max_row].item() - row_sum[min_row].item()
                    row_group_sums[min_group_idx] -= row_sum[min_row].item() - row_sum[max_row].item()
                    
                    # 记录交换：在排列中交换这两个位置
                    pos1 = current_perm.index(max_row)
                    pos2 = current_perm.index(min_row)
                    current_perm[pos1], current_perm[pos2] = current_perm[pos2], current_perm[pos1]
                else:
                    break  # 无法进一步改善
            
            row_perm_idx = torch.tensor(current_perm, device=sparse.device, dtype=torch.long)
            row_deperm_idx = torch.empty_like(row_perm_idx)
            row_deperm_idx[row_perm_idx] = torch.arange(H, device=row_perm_idx.device)

            group_size_w = W // ring_degree
            col_sum = mat.sum(dim=0)  # [W]

            # 初始化：连续分组并计算各组和
            col_groups = [list(range(i * group_size_w, (i + 1) * group_size_w)) for i in range(ring_degree)]
            col_group_sums = [col_sum[group].sum().item() for group in col_groups]

            # 创建初始排列索引
            current_perm = list(range(W))

            for iter in range(max_iters):
                # 找到和最大和最小的组
                max_group_idx = torch.tensor(col_group_sums).argmax().item()
                min_group_idx = torch.tensor(col_group_sums).argmin().item()
                
                if max_group_idx == min_group_idx:
                    break  # 已经平衡
                
                # 找到最大组中列和最大的k个元素
                max_group_cols = col_groups[max_group_idx]
                max_cols_in_max = sorted(max_group_cols, key=lambda idx: col_sum[idx].item(), reverse=True)[:k]
                
                # 找到最小组中列和最小的k个元素
                min_group_cols = col_groups[min_group_idx]
                min_cols_in_min = sorted(min_group_cols, key=lambda idx: col_sum[idx].item())[:k]
                
                # 尝试所有可能的交换组合，选择最优的
                best_swap = None
                best_improvement = 0
                
                for max_col in max_cols_in_max:
                    for min_col in min_cols_in_min:
                        # 计算交换后的组和变化
                        current_max_sum = col_group_sums[max_group_idx]
                        current_min_sum = col_group_sums[min_group_idx]
                        new_max_sum = current_max_sum - col_sum[max_col].item() + col_sum[min_col].item()
                        new_min_sum = current_min_sum - col_sum[min_col].item() + col_sum[max_col].item()
                        
                        # 计算不平衡度改善
                        current_imbalance = abs(current_max_sum - current_min_sum)
                        new_imbalance = abs(new_max_sum - new_min_sum)
                        improvement = current_imbalance - new_imbalance
                        
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_swap = (max_col, min_col)
                
                # 如果找到改善的交换，执行它
                if best_swap and best_improvement > 0:
                    max_col, min_col = best_swap
                    
                    # 执行交换：更新分组
                    col_groups[max_group_idx].remove(max_col)
                    col_groups[max_group_idx].append(min_col)
                    col_groups[min_group_idx].remove(min_col)
                    col_groups[min_group_idx].append(max_col)
                    
                    # 更新组和
                    col_group_sums[max_group_idx] = col_group_sums[max_group_idx] - col_sum[max_col].item() + col_sum[min_col].item()
                    col_group_sums[min_group_idx] = col_group_sums[min_group_idx] - col_sum[min_col].item() + col_sum[max_col].item()
                    
                    # 记录交换：在排列中交换这两个位置
                    pos1 = current_perm.index(max_col)
                    pos2 = current_perm.index(min_col)
                    current_perm[pos1], current_perm[pos2] = current_perm[pos2], current_perm[pos1]
                else:
                    break  # 无法进一步改善

            col_perm_idx = torch.tensor(current_perm, device=sparse.device, dtype=torch.long)
            col_deperm_idx = torch.empty_like(col_perm_idx)
            col_deperm_idx[col_perm_idx] = torch.arange(W, device=col_perm_idx.device)

            # change to the form to adapt to multi GPU
            num_groups = ring_degree
            group_size = row_perm_idx.shape[0] // num_groups
            row_perm_idx_groups_sorted = torch.sort(row_perm_idx.view(num_groups, group_size), dim=1)[0]
            col_perm_idx_groups_sorted = torch.sort(col_perm_idx.view(num_groups, group_size), dim=1)[0]

            transpose_matrix_q = torch.stack([
                torch.stack([
                    ((g >= j * group_size) & (g < (j + 1) * group_size)).sum()
                    for j in range(num_groups)
                ])
                for g in row_perm_idx_groups_sorted
            ]).T.contiguous()

            transpose_matrix_k = torch.stack([
                torch.stack([
                    ((g >= j * group_size) & (g < (j + 1) * group_size)).sum()
                    for j in range(num_groups)
                ])
                for g in col_perm_idx_groups_sorted
            ]).T.contiguous()

            new_row_perm_idx = torch.cat([
                row_perm_idx_groups_sorted[(row_perm_idx_groups_sorted >= i * group_size) & (row_perm_idx_groups_sorted < (i + 1) * group_size)]
                for i in range(num_groups)
            ]).reshape(ring_degree, -1)
            new_row_perm_idx = new_row_perm_idx - new_row_perm_idx.min(dim=1, keepdim=True)[0]

            new_col_perm_idx = torch.cat([
                col_perm_idx_groups_sorted[(col_perm_idx_groups_sorted >= i * group_size) & (col_perm_idx_groups_sorted < (i + 1) * group_size)]
                for i in range(num_groups)
            ]).reshape(ring_degree, -1)
            new_col_perm_idx = new_col_perm_idx - new_col_perm_idx.min(dim=1, keepdim=True)[0]

            new_row_deperm_idx = torch.empty_like(new_row_perm_idx)
            for i in range(new_row_perm_idx.shape[0]):
                new_row_deperm_idx[i][new_row_perm_idx[i]] = torch.arange(new_row_perm_idx.shape[1], device=new_row_perm_idx.device)

            new_col_deperm_idx = torch.empty_like(new_col_perm_idx)
            for i in range(new_col_perm_idx.shape[0]):
                new_col_deperm_idx[i][new_col_perm_idx[i]] = torch.arange(new_col_perm_idx.shape[1], device=new_col_perm_idx.device)

            sparse_final = sparse_reordered.index_select(1, row_perm_idx_groups_sorted.view(-1)).index_select(2, col_perm_idx_groups_sorted.view(-1)) 

            return sparse_final, head_perm_idx, new_row_perm_idx, new_col_perm_idx, transpose_matrix_q, transpose_matrix_k, head_deperm_idx, new_row_deperm_idx, new_col_deperm_idx

    elif sparse.dim() == 4:
        sparse_final_list = []
        head_perm_idx_list = []
        new_row_perm_idx_list = []
        new_col_perm_idx_list = []
        transpose_matrix_q_list = []
        transpose_matrix_k_list = []
        head_deperm_idx_list = []
        new_row_deperm_idx_list = []
        new_col_deperm_idx_list = []

        for block in range(sparse.shape[0]):
            sparse_final, head_perm_idx, new_row_perm_idx, new_col_perm_idx, transpose_matrix_q, transpose_matrix_k, head_deperm_idx, new_row_deperm_idx, new_col_deperm_idx = hybrid_permute_v3(sparse[block], ulysses_degree, ring_degree)
            sparse_final_list.append(sparse_final)
            head_perm_idx_list.append(head_perm_idx)
            new_row_perm_idx_list.append(new_row_perm_idx)
            new_col_perm_idx_list.append(new_col_perm_idx)
            transpose_matrix_q_list.append(transpose_matrix_q)
            transpose_matrix_k_list.append(transpose_matrix_k)
            head_deperm_idx_list.append(head_deperm_idx)
            new_row_deperm_idx_list.append(new_row_deperm_idx)
            new_col_deperm_idx_list.append(new_col_deperm_idx)

        sparse_final = torch.stack(sparse_final_list, dim=0)
        # head_perm_idx = torch.stack(head_perm_idx_list, dim=0)
        # new_row_perm_idx = torch.stack(new_row_perm_idx_list, dim=0)
        # new_col_perm_idx = torch.stack(new_col_perm_idx_list, dim=0)
        # transpose_matrix_q = torch.stack(transpose_matrix_q_list, dim=0)
        # transpose_matrix_k = torch.stack(transpose_matrix_k_list, dim=0)
        # head_deperm_idx = torch.stack(head_deperm_idx_list, dim=0)
        # new_row_deperm_idx = torch.stack(new_row_deperm_idx_list, dim=0)
        # new_col_deperm_idx = torch.stack(new_col_deperm_idx_list, dim=0)

    return sparse_final, head_perm_idx_list, new_row_perm_idx_list, new_col_perm_idx_list, transpose_matrix_q_list, transpose_matrix_k_list, head_deperm_idx_list, new_row_deperm_idx_list, new_col_deperm_idx_list




device = "cuda" if torch.cuda.is_available() else "cpu"
sparse_data = torch.load("/mnt/public/ns-t-te-b905754427352261-427-bk/fs/home/xieruiqi/diffuser-dev520/examples/wan/logs/calib_data/rebuttal_720p/sparse_expanded.pth", map_location='cpu', weights_only=True)
sparse = sparse_data['sparse'][0, :, :, :, :].to(device)  # [40, 40, 1182, 1182] 0.414
print(sparse.shape)

# _, all_groups, all_group_sums, all_perm_idx, all_deperm_idx = greedy_partition_and_rearrange_multi(sparse,num_groups=8)
# print(sparse.shape)
# sparse_reordered = torch.stack(
#     [sparse[block].index_select(0, all_perm_idx[block]) for block in range(sparse.shape[0])]
# )
# print(sparse_reordered.shape)
ulysses_degree = 4
ring_degree = 2
imbalance_ratio_list = []
new_imbalance_ratio_list_v4 = []
new_imbalance_ratio_list_v5 = []
total_transpose_count = 0
for block in range(40):
    sparse_piece = sparse[block,:,:,:]
    H, W = sparse_piece.shape[-2], sparse_piece.shape[-1]
    pad_h = (8 - H % 8) if H % 8 != 0 else 0
    pad_w = (8 - W % 8) if W % 8 != 0 else 0
    if pad_h != 0 or pad_w != 0:
        sparse_piece = torch.nn.functional.pad(sparse_piece, (0, pad_w, 0, pad_h), "constant", 0)

    print(sparse_piece.float().sum()/sparse_piece.numel())
    sparse_final4, _, _, _, transpose_matrix_q1, _, _, _, _ = hybrid_permute_v4(sparse_piece, ulysses_degree=ulysses_degree, ring_degree=ring_degree,reward=2)
    total_transpose_count += transpose_matrix_q1[0].sum()-transpose_matrix_q1[0][0]

    # sparse_final5, head_perm_idx, new_row_perm_idx, new_col_perm_idx, transpose_matrix_q, transpose_matrix_k, head_deperm_idx, new_row_deperm_idx, new_col_deperm_idx = hybrid_permute_v5(sparse_piece, ulysses_degree=ulysses_degree, ring_degree=ring_degree,max_iters=500)

    imbalance_ratio_list.append(hybrid_imbalance_ratio(sparse_piece, ulysses_degree=ulysses_degree, ring_degree=ring_degree))
    new_imbalance_ratio_list_v4.append(hybrid_imbalance_ratio(sparse_final4, ulysses_degree=ulysses_degree, ring_degree=ring_degree))
    # new_imbalance_ratio_list_v5.append(hybrid_imbalance_ratio(sparse_final5, ulysses_degree=ulysses_degree, ring_degree=ring_degree))

    print(f"{block}: {hybrid_imbalance_ratio(sparse_piece, ulysses_degree=ulysses_degree, ring_degree=ring_degree)}")
    print(f"v4: {hybrid_imbalance_ratio(sparse_final4, ulysses_degree=ulysses_degree, ring_degree=ring_degree)},{transpose_matrix_q1[0].sum()-transpose_matrix_q1[0][0]}")
    # print(f"v5: {hybrid_imbalance_ratio(sparse_final5, ulysses_degree=ulysses_degree, ring_degree=ring_degree)}, {transpose_matrix_q[0].sum()-transpose_matrix_q[0][0]}")

print(f"average: {sum(imbalance_ratio_list) / len(imbalance_ratio_list)}")
print(f"average v4: {sum(new_imbalance_ratio_list_v4) / len(new_imbalance_ratio_list_v4)}")
print(f"total transpose count v4: {total_transpose_count/40}")
# print(f"average v5: {sum(new_imbalance_ratio_list_v5) / len(new_imbalance_ratio_list_v5)}")

# sparse = sparse[block,:,:,:]
# H, W = sparse.shape[-2], sparse.shape[-1]
# pad_h = (8 - H % 8) if H % 8 != 0 else 0
# pad_w = (8 - W % 8) if W % 8 != 0 else 0
# if pad_h != 0 or pad_w != 0:
#     sparse = torch.nn.functional.pad(sparse, (0, pad_w, 0, pad_h), "constant", 0)
# sparse, head_perm_idx, new_row_perm_idx, new_col_perm_idx, transpose_matrix_q, transpose_matrix_k, head_deperm_idx, new_row_deperm_idx, new_col_deperm_idx = hybrid_permute_v3(sparse, ulysses_degree=ulysses_degree, ring_degree=ring_degree)
# print(new_row_perm_idx[0].shape)

# visualization
# block = 2
# sparse_piece = sparse[block]
# num_heads = sparse_piece.shape[0]
# ncols = 5
# nrows = math.ceil(num_heads / ncols)
# fig, axes = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows), squeeze=False)

# num_heads = sparse_piece.shape[0]
# ncols = 5
# nrows = math.ceil(num_heads / ncols)
# fig, axes = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows), squeeze=False)

# for i in range(num_heads):
#     row, col = divmod(i, ncols)
#     ax = axes[row][col]
#     ax.imshow(sparse_piece[i].cpu().numpy(), cmap='gray', aspect='auto')
#     ax.set_title(f'head {i}')
#     ax.axis('off')

# # 去除多余子图
# for j in range(num_heads, nrows * ncols):
#     row, col = divmod(j, ncols)
#     axes[row][col].axis('off')

# plt.tight_layout()
# plt.savefig(f'/mnt/public/chensiqi/emulation/sparse_0.414_block{block}_ulysses{ulysses_degree}_ring_{ring_degree}.png')


# num_heads = sparse_final.shape[0]
# ncols = 5
# nrows = math.ceil(num_heads / ncols)
# fig, axes = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows), squeeze=False)

# for i in range(num_heads):
#     row, col = divmod(i, ncols)
#     ax = axes[row][col]
#     ax.imshow(sparse_final[i].cpu().numpy(), cmap='gray', aspect='auto')
#     ax.set_title(f'head {i}')
#     ax.axis('off')

# # 去除多余子图
# for j in range(num_heads, nrows * ncols):
#     row, col = divmod(j, ncols)
#     axes[row][col].axis('off')

# plt.tight_layout()
# plt.savefig(f'/mnt/public/chensiqi/emulation/sparse_final_block{block}_ulysses{ulysses_degree}_ring_{ring_degree}.png')


# with profile(
#         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True
#     ) as prof:
#         torch.cuda.synchronize()
#         for i in range(3):
#             q_permuted = hybrid_q_permute_v2(q, head_perm_idx, row_perm_idx)
#             k_permuted = hybrid_q_permute_v2(k, head_perm_idx, col_perm_idx)
#             v_permuted = hybrid_q_permute_v2(v, head_perm_idx, col_perm_idx)
#         with record_function("reorder_overhead"):
#             q_permuted = hybrid_q_permute_v2(q, head_perm_idx, row_perm_idx)
#             k_permuted = hybrid_q_permute_v2(k, head_perm_idx, col_perm_idx)
#             v_permuted = hybrid_q_permute_v2(v, head_perm_idx, col_perm_idx)
#         torch.cuda.synchronize()

# prof.export_chrome_trace(f"overhead_profile/overhead_hybrid_v2_test.json")
