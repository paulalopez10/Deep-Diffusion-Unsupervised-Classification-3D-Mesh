import torch
import numpy as np
from scipy.stats import rankdata




def pw_cosine_distance(input_a, input_b):
   normalized_input_a = torch.nn.functional.normalize(input_a)  
   normalized_input_b = torch.nn.functional.normalize(input_b)
   res = torch.mm(normalized_input_a, normalized_input_b.T)
   res *= -1 # 1-res without copy
   res += 1
   return res

   
def retrieval(embedding, label, calc_rpcurve=False):

    recall_step = 0.05

    mean_average_precision = 0.0
    mean_recall = np.zeros(int(1.0/recall_step))
    mean_precision = np.zeros(int(1.0/recall_step))

    n_data = embedding.shape[0]

    #D = pairwise_distances(embedding.cpu(), metric="cosine")
    D = pw_cosine_distance(embedding.cpu(),embedding.cpu())

    for i in range(n_data): # for each query

        dist_vec = D[i]
        gt_vec = (label.cpu()==label.cpu()[i]).int() # 1 if the retrieval target belongs to the same category with the query, 0 otherwise

        dist_vec_woq = torch.cat([dist_vec[:i], dist_vec[i+1:]])  # distance vector without query

        gt_vec_woq = torch.cat((gt_vec[:i], gt_vec[i+1:])) # groundtruth vector without query
        nonzero_indices = torch.nonzero(gt_vec_woq).t() # indices of nonzero elements
        values = gt_vec_woq[nonzero_indices[0]] # corresponding values
        gt_vec_woq_sp = torch.sparse_coo_tensor(nonzero_indices, values, size=(n_data-1,))

        gt_vec_woq_sp = gt_vec_woq_sp.coalesce()
        relevant = gt_vec_woq_sp.indices()[0]
        n_correct = gt_vec_woq_sp._values().shape[0] 

        relevant = gt_vec_woq_sp.indices()
        n_correct = gt_vec_woq_sp._nnz()  # number of correct targets for the query
        rank = rankdata(dist_vec_woq.cpu().detach().numpy(), 'max')[relevant]  # positions where correct data appear in a retrieval ranking
        if rank.size>1:
            rank_sorted = np.sort( rank )
        else:
            rank_sorted = rank


        # average precision
        if n_correct == 0:
            ap = 1.0
        else:
            L = rankdata(dist_vec_woq[relevant].cpu().detach().numpy(), 'max')
            ap = (L / rank).mean()
        mean_average_precision += ap

        # recall-precision curve
        if calc_rpcurve:
            one_to_n = torch.arange(n_correct).float() + 1
            precision = one_to_n / rank_sorted
            recall = one_to_n / n_correct
            recall_interp = torch.arange(recall_step, 1.01, recall_step)
    

    mean_average_precision /= n_data

    if calc_rpcurve:
        mean_precision /= n_data
    else:
        mean_recall = None
        mean_precision = None

    return mean_average_precision

