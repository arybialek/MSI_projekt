from scipy import stats

svc_test_acc = [
    0.61806,
    0.59177,
    0.59945,
    0.61384,
    0.59003,
    0.42932,
    0.61186,
    0.57192,
    0.61384,
    0.61434,
]

knn_test_acc = [
    0.7247,
    0.72197,
    0.71875,
    0.71726,
    0.71602,
    0.2877,
    0.72545,
    0.73785,
    0.72222,
    0.73165,
]

resnet_test_acc = [
    0.8469742063492064,
    0.7457837301587301,
    0.796875,
    0.7514880952380952,
    0.8395337301587301,
    0.8308531746031746,
    0.8273809523809523,
    0.8139880952380952,
    0.8253968253968254,
    0.8115079365079365,

]

t_statistic_resnet_knn, pvalue_resnet_knn = stats.ttest_ind(resnet_test_acc,
                                                            knn_test_acc,
                                                            nan_policy='omit')

t_statistic_resnet_svc, pvalue_resnet_svc = stats.ttest_ind(resnet_test_acc,
                                                            svc_test_acc,
                                                            nan_policy='omit')

t_statistic_knn_svc, pvalue_knn_svc = stats.ttest_ind(knn_test_acc,
                                                      svc_test_acc,
                                                      nan_policy='omit')

print(t_statistic_resnet_knn, pvalue_resnet_knn)
print(t_statistic_resnet_svc, pvalue_resnet_svc)
print(t_statistic_knn_svc, pvalue_knn_svc)
