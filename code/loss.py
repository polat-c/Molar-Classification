import torch
import numpy as np
from utils import cuda

class NTXentLoss(torch.nn.Module):

    def __init__(self, use_cuda, batch_size, temperature):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.use_cuda = use_cuda
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = torch.nn.CosineSimilarity(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_correlated_mask(self): # mask values that correspond to the same image
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return cuda(mask, self.use_cuda)

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations.unsqueeze(1), representations.unsqueeze(0))

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1) # value pairs that correspond to the same image (original, augmented)

        mask = self._get_correlated_mask()
        negatives = similarity_matrix[mask].view(2 * self.batch_size, -1) # value pairs that correspind to different images

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size)
        labels = cuda(labels, self.use_cuda).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)