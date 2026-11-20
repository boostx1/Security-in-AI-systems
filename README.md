# Security in AI Systems (DV2607)

How does input preprocessing affect an image classifier's predictions, and how well does federated learning tolerate clients with corrupted labels? Bilal Abdul and Subhi Alloush studied these questions at Blekinge Institute of Technology using ResNet-50 on ImageNet images and a five-client simulation on CIFAR-10.

[English report (PDF)](report/dv2607-ai-security-report.pdf) · [Image classification notebook](notebooks/image_classifier_robustness.ipynb) · [Federated learning notebook](notebooks/federated_learning_robustness.ipynb)

## Image classification

ResNet-50 classified a perturbed koala image as a tractor. After feature squeezing and JPEG compression, the model classified the same image as a koala with 94.9% confidence. On the original koala image, the model reported 99.1% confidence after preprocessing. These measurements describe one image pair. A broader accuracy estimate would require a test set.

## Federated learning

The authors trained an image classifier for 35 rounds on CIFAR-10, using five clients with equally sized training partitions. The table shows the final test accuracy for each aggregation method as the number of clients with corrupted labels increased.

| Aggregation method | 0 of 5 clients | 1 of 5 clients | 2 of 5 clients |
| --- | ---: | ---: | ---: |
| Federated averaging | 73.59% | 64.49% | 44.76% |
| Coordinate-wise median | 73.52% | 72.23% | 59.67% |
| Averaging with server momentum | 56.97% | 49.32% | — |
| Coordinate-wise trimmed mean | 73.31% | 72.32% | 52.77% |

The trimmed mean achieved the highest accuracy with one affected client. With two affected clients, the median achieved 59.67%, or 14.91 percentage points above federated averaging. The authors reported no two-client result for server momentum.

[Evaluation figures](images/) · [Python dependencies](requirements.txt)

## Reading the materials

The report includes all eleven federated learning runs, the input preprocessing results, and six figures. The image classification notebook contains the original evaluation code with English labels. Its reported measurements are preserved in the report. The federated learning notebook includes its recorded outputs.

[Original submitted report (Swedish)](https://github.com/boostx1/Security-in-AI-systems/blob/1e5d4dfe3ab16f2a2a89d6ae9816b917f31df0b7/report/Rapport_DV2607_Bilal_Subhi.pdf)
