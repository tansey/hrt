library(knockoff)

ntrials = 100
S = 40
fdr = 0.1
avg_tpr = 0
avg_fdr = 0
for (trial in 0:(ntrials-1)){
    print(sprintf("Trial %i", trial))
    X = as.matrix(read.csv(file.path(sprintf("data/%i/X.csv", trial)), header=FALSE))
    y = scan(file.path(sprintf("data/%i/Y.csv", trial)), sep=",")
    results = knockoff.filter(X, y, fdr=fdr)
    write.table(results$selected, file=sprintf("data/%i/knockoffs.csv", trial), row.names=FALSE, col.names=FALSE, sep=',')
    print(results$selected)
    tp = sum(results$selected <= S) * 100 / S # True positive rate
    fdp = sum(results$selected > S) * 100 / max(1,length(results$selected)) # False discovery proportion
    print(sprintf("TPR: %f FDR: %f", tp, fdp))
    avg_tpr = avg_tpr + tp / ntrials
    avg_fdr = avg_fdr + fdp / ntrials
}
print(sprintf("*** Average over %i trials ***", ntrials))
print(sprintf("TPR: %f", avg_tpr))
print(sprintf("FDR: %f", avg_fdr))
