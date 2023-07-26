import torch
from torchmetrics import Metric
from torchmetrics.classification import BinaryAUROC
from torchmetrics.classification import BinaryConfusionMatrix
from abc import abstractmethod

class LoggableMetric(Metric):
    @abstractmethod
    def log(self, logger, dataset_name, log_prefix):
        ...

class ClassificationMetric(LoggableMetric):

    def __init__(self):
        super().__init__()
        self.CM_metric = BinaryConfusionMatrix(threshold=0.5)
        self.AUROC_metric = BinaryAUROC()

    def update(self, prob_of_1: torch.Tensor, target: torch.Tensor):
        self.CM_metric.update(prob_of_1.cpu(), target.cpu())
        self.AUROC_metric.update(prob_of_1.cpu(), target.cpu())

    def reset(self):
        self.CM_metric.reset()
        self.AUROC_metric.reset()

    def compute(self):
        return self.CM_metric.compute(), self.AUROC_metric.compute()

    def log(self, logger, dataset_name, log_prefix):
        ((TN, FP),
         (FN, TP)), AUROC = self.compute()
        total_train = TP + FP + FN + TN

        logger.info(log_prefix + f"Mean {dataset_name} Accuracy: {TP + TN}/{total_train} = {(TP + TN)/total_train:.6f}")
        logger.info(log_prefix + f"Mean {dataset_name} Positive Precision: {TP}/{TP + FP} = {TP/(TP + FP):.6f}")
        logger.info(log_prefix + f"Mean {dataset_name} Positive Recall: {TP}/{TP + FN} = {TP/(TP + FN):.6f}")
        logger.info(log_prefix + f"Mean {dataset_name} Negative Precision: {TN}/{TN + FN} = {TN/(TN + FN):.6f}")
        logger.info(log_prefix + f"Mean {dataset_name} Negative Recall: {TN}/{TN + FP} = {TN/(TN + FP):.6f}")
        logger.info(log_prefix + f"Mean {dataset_name} AUROC: {AUROC:.6f}")