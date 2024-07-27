suppressMessages(library(dplyr))
suppressMessages(library(DESeq2))
suppressMessages(library(tibble))
#install.packages("svMisc")
# library(data.table)

#setwd("./Desktop/CancerResponseData/src/")
#getwd()
args = commandArgs(trailingOnly=TRUE)

drug = args[1]
tissue = args[2]
countFile = args[3]
colDataFile = args[4]
thresh = args[5]
manif = args[6]
trim  = as.logical(args[7])




counts = as.matrix(read.csv(countFile,row.names = 'Gene_ID',check.names=FALSE))
counts = round(counts)

mode(counts) <- "integer"


col_data = read.csv(colDataFile)

col_data$Community <- factor(col_data$Community)


N = length(colnames(counts))


dds <- DESeqDataSetFromMatrix(countData = counts,
                              colData = col_data,
                              design = ~ Community)
 

smallestGroupSize <- 3
keep <- rowSums(counts(dds) >= 5) >= smallestGroupSize
dds <- dds[keep,]

dds <- estimateSizeFactors(dds)

dds <- DESeq(dds,quiet = TRUE)
# print(dds)
res <- results(dds)
# print(res)
res <- res[!is.na(res$padj),]
res <- res[res$padj <= thresh,]




res_df = data.frame(res)
res_df <- tibble::rownames_to_column(res_df, 'Gene')

res_keep = res_df[,c('Gene','padj')]

res_keep <- res_keep[order(res_keep$padj),]


if(trim){
    write.csv(x= res_keep,file = paste0("../results/",drug,"/",tissue,"/trimmed/",manif,"/DE_genes.csv"))
} else{
     write.csv(x= res_keep,file = paste0("../results/",drug,"/",tissue,"/",manif,"/DE_genes.csv"))
}
    




